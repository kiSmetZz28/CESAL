"""Evaluate the RAG-enhanced LLM open-set incident classifier on HDFS abnormal sequences.

Evaluates paper Table 7 macro-averaged P/R/F1 for four LLM backbones.
Also exports per-class metrics. Summary CSV scores are fractions (0-1);
terminal scores and the per-class F1 pivot are percentages.

Usage (from project root):
  python -m incident_response.evaluate --config configs/llm/hdfs.yaml
  python -m incident_response.evaluate --config configs/llm/hdfs.yaml --models qwen2.5-14b-instruct

Outputs in <output_dir>/:
  results_<model>.csv          per-sequence predictions, retrieval evidence, raw LLM output
  report_<model>.txt           sklearn classification report
  model_summary.csv            accuracy, macro/weighted P/R/F1, timing, effective thresholds
  per_class_metrics_long.csv   per-class P/R/F1/support for every model
  per_class_f1_table.csv       per-class F1 (%) pivot, one column per model
"""


import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.steps import StepReporter
from incident_response.classifier import (
    ALL_EVAL_LABELS, KNOWN_LABELS, OTHER_LABEL,
    LexicalSequenceRetriever, build_kb_docs_from_csv, load_llm, normalize_open_set_test_label,
    normalize_sequence, predict_one, resolve_model_name, safe_name,
)


_ABOUT = """
Benchmark a language model on naming the incident type behind an abnormal
stretch of logs.

Detection flags anomalous events. This stage classifies abnormal sequences using
retrieved reference incidents, open-set decision rules and, when needed, a
language model. Scores are macro-averaged classification metrics; the unknown
category is recorded as Other anomaly type for analyst review.
"""


def load_data(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text_col, label_col = cfg["text_col"], cfg["label_col"]
    test_df = pd.read_csv(cfg["test_csv"])
    kb_df = pd.read_csv(cfg["kb_csv"])

    test_df = test_df[[text_col, label_col]].copy()
    test_df[text_col] = test_df[text_col].apply(normalize_sequence)

    # Unknown labels map to Other; known labels stay as-is.
    test_df[label_col] = test_df[label_col].apply(normalize_open_set_test_label)
    test_df = test_df[test_df[text_col].astype(str).str.len() > 0].reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=cfg["random_seed"]).reset_index(drop=True)
    return test_df, kb_df


def compute_metrics(results_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    y_true = results_df["true_label"]
    y_pred = results_df["pred_label"]
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    labels_present = [lab for lab in ALL_EVAL_LABELS if (lab in set(y_true) or lab in set(y_pred))]
    per_p, per_r, per_f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels_present, zero_division=0)
    per_class_rows = [
        {"model": model_name, "label": lab, "precision": p, "recall": r, "f1": f1, "support": int(sup)}
        for lab, p, r, f1, sup in zip(labels_present, per_p, per_r, per_f1, support)
    ]
    return {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,
        "per_class": per_class_rows,
        "report": classification_report(y_true, y_pred, labels=labels_present, zero_division=0, digits=4),
    }


def effective_params(model_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Per-model overrides (config model_overrides) take precedence over the config defaults."""
    overrides = (cfg.get("model_overrides") or {}).get(model_name, {})
    return {
        "open_set_threshold": overrides.get("open_set_threshold", cfg["open_set_threshold"]),
        "high_confidence_threshold": overrides.get("high_confidence_threshold", cfg["high_confidence_threshold"]),
        "max_new_tokens": overrides.get("max_new_tokens", cfg["max_new_tokens"]),
        "min_bypass_majority": overrides.get("min_bypass_majority", 1),
        "length_penalty": overrides.get("length_penalty", cfg.get("length_penalty", True)),
    }


def evaluate_model(model_name: str, cfg: Dict[str, Any], retriever, test_df: pd.DataFrame,
                   load_in_4bit: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    text_col, label_col = cfg["text_col"], cfg["label_col"]
    steps.current().phase(f"loading {model_name}")
    tokenizer, model = load_llm(model_name, load_in_4bit=load_in_4bit)

    params = effective_params(model_name, cfg)
    logging.debug("Effective params: %s", params)
    steps.current().phase(f"classifying {len(test_df):,} sequences with {model_name.split('/')[-1]}")

    step = steps.current()
    step.progress_note(model_name.split("/")[-1])

    rows = []
    total_infer_time = 0.0
    start_all = time.perf_counter()

    for i, row in test_df.iterrows():
        infer_start = time.perf_counter()
        out = predict_one(
            row[text_col], retriever, tokenizer, model,
            top_k=cfg["top_k_retrieve"],
            open_set_threshold=params["open_set_threshold"],
            high_confidence_threshold=params["high_confidence_threshold"],
            threshold_first=cfg.get("threshold_first", True),
            max_new_tokens=params["max_new_tokens"],
            min_bypass_majority=params["min_bypass_majority"],
        )
        infer_time = time.perf_counter() - infer_start
        total_infer_time += infer_time

        rows.append({
            "model": model_name,
            "template_sequence": row[text_col],
            "true_label": row[label_col],
            "pred_label": out["pred_label"],
            "decision_source": out["decision_source"],
            "best_score": out["best_score"],
            "candidate_labels": json.dumps(out["candidate_labels"], ensure_ascii=False),
            "retrieved_labels": json.dumps(out["retrieved_labels"], ensure_ascii=False),
            "retrieved_doc_types": json.dumps(out["retrieved_doc_types"], ensure_ascii=False),
            "retrieved_scores": json.dumps(out["retrieved_scores"], ensure_ascii=False),
            "retrieved_sequences": json.dumps(out["retrieved_sequences"], ensure_ascii=False),
            "raw_output": out["raw_output"],
            "inference_time_sec": infer_time,
        })

        if i < cfg.get("debug_first_n", 10):
            logging.debug("INPUT: %s | TRUE: %s | SRC: %s | RAW: %r | PRED: %s | %.4fs",
                         row[text_col], row[label_col], out["decision_source"],
                         out["raw_output"], out["pred_label"], infer_time)

        step.tick("sequences")
        logging.debug("Processed %d/%d | Avg inference: %.4f sec/sample",
                      i + 1, len(test_df), total_infer_time / (i + 1))

    total_eval_time = time.perf_counter() - start_all
    results_df = pd.DataFrame(rows)
    metrics = compute_metrics(results_df, model_name)
    metrics["total_inference_time_sec"] = total_infer_time
    metrics["average_inference_time_sec"] = total_infer_time / max(len(test_df), 1)
    metrics["total_evaluation_time_sec"] = total_eval_time
    metrics["open_set_threshold"] = params["open_set_threshold"]
    metrics["high_confidence_threshold"] = params["high_confidence_threshold"]
    metrics["max_new_tokens"] = params["max_new_tokens"]
    metrics["length_penalty"] = params["length_penalty"]

    logging.info("%s", steps.leader(
        model_name.split("/")[-1],
        "macro  P {:.2f}   R {:.2f}   F1 {:.2f}".format(
            metrics["macro_precision"] * 100, metrics["macro_recall"] * 100,
            metrics["macro_f1"] * 100),
        steps.body_indent() or 3))
    logging.debug("\n%s", metrics["report"])

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CESAL LLM-based open-set incident classification.")
    parser.add_argument("--config", default="configs/llm/hdfs.yaml")
    parser.add_argument("--models", default=None,
                        help="Comma-separated HF model IDs or aliases (default: models listed in the config).")
    parser.add_argument("--output_dir", default=None, help="Override the config output_dir.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use bitsandbytes 4-bit loading if installed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    model_names = [resolve_model_name(m) for m in (args.models.split(",") if args.models else cfg["models"]) if m.strip()]

    rep = StepReporter("classify", dataset=cfg.get("dataset", "HDFS"),
                       steps=steps.CLASSIFY_STEPS, about=_ABOUT)

    # ── Step 1: load the sequences and the reference library ──────────────
    with rep.step("prepare") as st:
        seed = cfg["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.makedirs(cfg["output_dir"], exist_ok=True)

        st.phase("loading the abnormal sequences")
        test_df, kb_df = load_data(cfg)
        st.phase("indexing the retrieval corpus")
        kb_docs = build_kb_docs_from_csv(kb_df)
        logging.debug("Test label distribution:\n%s", test_df[cfg["label_col"]].value_counts())

        st.detail("sequences to classify", len(test_df))
        st.detail("reference incidents", len(kb_docs))
        st.detail("known incident types", len(KNOWN_LABELS))
        st.detail("fallback label", OTHER_LABEL)
        st.outcome(**{
            "language models to try": len(model_names),
            "models": ", ".join(m.split("/")[-1] for m in model_names),
        })

    # ── Step 2: label every sequence with every model ─────────────────────
    retrievers: Dict[bool, LexicalSequenceRetriever] = {}
    summary_rows, per_class_rows = [], []

    with rep.step("classify") as st:
        # One tick per (model, sequence): the bar tracks the whole benchmark,
        # which for four backbones over 4,124 sequences runs for hours.
        st.expect("sequences", len(model_names) * len(test_df))

        for model_name in model_names:
            length_penalty = effective_params(model_name, cfg)["length_penalty"]
            if length_penalty not in retrievers:
                retrievers[length_penalty] = LexicalSequenceRetriever(kb_docs, length_penalty=length_penalty)
            results_df, metrics = evaluate_model(model_name, cfg, retrievers[length_penalty], test_df,
                                                 load_in_4bit=args.load_in_4bit)
            model_safe = safe_name(model_name)
            result_path = os.path.join(cfg["output_dir"], f"results_{model_safe}.csv")
            report_path = os.path.join(cfg["output_dir"], f"report_{model_safe}.txt")
            results_df.to_csv(result_path, index=False)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(metrics["report"])

            summary_rows.append({k: v for k, v in metrics.items() if k not in {"per_class", "report"}})
            per_class_rows.extend(metrics["per_class"])
            logging.debug("Saved %s and %s", result_path, report_path)

        best = max(summary_rows, key=lambda r: r["macro_f1"]) if summary_rows else None
        st.outcome(**{
            "models evaluated": len(summary_rows),
            "labels assigned": len(model_names) * len(test_df),
            "best model": (f"{best['model'].split('/')[-1]} "
                           f"(F1 {best['macro_f1'] * 100:.2f})") if best else "—",
        })

    # ── Step 3: write the per-class tables ────────────────────────────────
    with rep.step("score") as st:
        summary_df = pd.DataFrame(summary_rows)
        per_class_df = pd.DataFrame(per_class_rows)
        summary_df.to_csv(os.path.join(cfg["output_dir"], "model_summary.csv"), index=False)
        per_class_df.to_csv(os.path.join(cfg["output_dir"], "per_class_metrics_long.csv"), index=False)

        f1_table = per_class_df.pivot(index="label", columns="model", values="f1") * 100.0
        f1_table = f1_table.reindex(ALL_EVAL_LABELS).dropna(how="all")
        f1_table.to_csv(os.path.join(cfg["output_dir"], "per_class_f1_table.csv"))

        for row in summary_rows:
            rep.metric(row["model"].split("/")[-1],
                       row["accuracy"] * 100, row["macro_precision"] * 100,
                       row["macro_recall"] * 100, row["macro_f1"] * 100)
        st.outcome(**{
            "incident types scored": len(f1_table),
            "files written": "model_summary.csv, per_class_metrics_long.csv, per_class_f1_table.csv",
        })

    rep.finish(outputs=cfg["output_dir"])


if __name__ == "__main__":
    setup_logging("llm_classify")
    main()
