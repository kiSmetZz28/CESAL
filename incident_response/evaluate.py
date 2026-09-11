"""Evaluate the RAG-enhanced LLM open-set incident classifier on HDFS abnormal sequences.

Reproduces paper Table 7 (per-class P/R/F1 of four LLM backbones).

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

from cesal_core.utils.config import load_config, setup_logging
from incident_response.classifier import (
    ALL_EVAL_LABELS, KNOWN_LABELS, OTHER_LABEL,
    LexicalSequenceRetriever, build_kb_docs_from_csv, load_llm, normalize_open_set_test_label,
    normalize_sequence, predict_one, resolve_model_name, safe_name,
)


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
    logging.info("========== Loading model: %s ==========", model_name)
    tokenizer, model = load_llm(model_name, load_in_4bit=load_in_4bit)

    params = effective_params(model_name, cfg)
    logging.info("Effective params: %s", params)

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
            logging.info("INPUT: %s | TRUE: %s | SRC: %s | RAW: %r | PRED: %s | %.4fs",
                         row[text_col], row[label_col], out["decision_source"],
                         out["raw_output"], out["pred_label"], infer_time)

        if (i + 1) % cfg.get("print_every", 20) == 0:
            logging.info("Processed %d/%d | Avg inference: %.4f sec/sample",
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

    logging.info("Accuracy %.4f | Macro P %.4f | Macro R %.4f | Macro F1 %.4f",
                 metrics["accuracy"], metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"])
    logging.info("\n%s", metrics["report"])

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

    seed = cfg["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    test_df, kb_df = load_data(cfg)
    kb_docs = build_kb_docs_from_csv(kb_df)
    logging.info("Known labels: %d | Open-set label: %s | KB sequence docs: %d",
                 len(KNOWN_LABELS), OTHER_LABEL, len(kb_docs))
    logging.info("Test label distribution:\n%s", test_df[cfg["label_col"]].value_counts())

    retrievers: Dict[bool, LexicalSequenceRetriever] = {}
    summary_rows, per_class_rows = [], []

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
        logging.info("Saved %s and %s", result_path, report_path)

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    summary_df.to_csv(os.path.join(cfg["output_dir"], "model_summary.csv"), index=False)
    per_class_df.to_csv(os.path.join(cfg["output_dir"], "per_class_metrics_long.csv"), index=False)

    f1_table = per_class_df.pivot(index="label", columns="model", values="f1") * 100.0
    f1_table = f1_table.reindex(ALL_EVAL_LABELS).dropna(how="all")
    f1_table.to_csv(os.path.join(cfg["output_dir"], "per_class_f1_table.csv"))
    logging.info("Saved aggregate files to %s", cfg["output_dir"])


if __name__ == "__main__":
    setup_logging("llm_classify")
    main()
