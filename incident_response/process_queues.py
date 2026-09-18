"""Process CESAL anomaly queues with the LLM-based open-set classification and response module.

Every queued incident (incident_response.queues) is classified into a known anomaly type or
"Other anomaly type", then mapped to its predefined response workflow (paper Table 1). Identical
template-event sequences are classified once; each classification is appended to a per-model cache
as soon as it finishes, so an interrupted run resumes where it stopped.

Outputs in <queue_dir>/:
  classified_<model>.csv    one row per unique sequence: label, decision source, retrieval evidence, raw LLM output
  incidents_<model>.csv     one row per queued incident: queue, detection metadata, label, selected workflow
  evaluation_<model>.csv    per-class metrics on detected abnormal sessions whose anomaly type is known

Usage (from project root):
  python -m incident_response.process_queues --config configs/llm/hdfs.yaml
  python -m incident_response.process_queues --config configs/llm/hdfs.yaml --model qwen2.5-7b-instruct
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.steps import StepReporter
from incident_response.classifier import (
    KNOWN_LABELS, OTHER_LABEL,
    LexicalSequenceRetriever, build_kb_docs_from_csv, load_llm, normalize_open_set_test_label,
    normalize_sequence, predict_one, resolve_model_name, safe_name,
)
from incident_response.evaluate import compute_metrics, effective_params
from incident_response.queues import QUEUE_FILES
from incident_response.workflows import select_workflow


def load_queues(queue_dir: str, queues) -> pd.DataFrame:
    frames = []
    for queue in queues:
        path = os.path.join(queue_dir, QUEUE_FILES[queue])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run `python -m incident_response.queues` first.")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def classify_sequences(sequences, model_name: str, cfg, cache_path: str, load_in_4bit: bool) -> pd.DataFrame:
    done = pd.read_csv(cache_path, keep_default_na=False) if os.path.exists(cache_path) else pd.DataFrame()
    finished = set(done["template_sequence"]) if len(done) else set()
    todo = [s for s in sequences if s not in finished]
    step = steps.current()
    if len(sequences) != len(todo):
        step.detail("already present", f"{len(sequences) - len(todo):,} (resuming)")
    if not todo:
        step.note("Every sequence was already classified in an earlier run.")
        return done

    params = effective_params(model_name, cfg)
    logging.debug("Model %s | effective params: %s", model_name, params)
    step.phase("indexing the retrieval corpus")
    retriever = LexicalSequenceRetriever(build_kb_docs_from_csv(pd.read_csv(cfg["kb_csv"])),
                                         length_penalty=params["length_penalty"])
    step.phase(f"loading {model_name}")
    tokenizer, model = load_llm(model_name, load_in_4bit=load_in_4bit)
    step.phase(f"classifying {len(todo):,} distinct sequences")
    step.expect("sequences", len(todo))

    start, n_llm = time.perf_counter(), 0
    for i, seq in enumerate(todo, 1):
        out = predict_one(seq, retriever, tokenizer, model,
                          top_k=cfg["top_k_retrieve"],
                          open_set_threshold=params["open_set_threshold"],
                          high_confidence_threshold=params["high_confidence_threshold"],
                          threshold_first=cfg.get("threshold_first", True),
                          max_new_tokens=params["max_new_tokens"],
                          min_bypass_majority=params["min_bypass_majority"])
        n_llm += bool(out["raw_output"])
        row = {
            "template_sequence": seq,
            "pred_label": out["pred_label"],
            "decision_source": out["decision_source"],
            "best_score": out["best_score"],
            "candidate_labels": json.dumps(out["candidate_labels"], ensure_ascii=False),
            "retrieved_labels": json.dumps(out["retrieved_labels"], ensure_ascii=False),
            "retrieved_scores": json.dumps(out["retrieved_scores"], ensure_ascii=False),
            "retrieved_sequences": json.dumps(out["retrieved_sequences"], ensure_ascii=False),
            "raw_output": out["raw_output"],
        }
        pd.DataFrame([row]).to_csv(cache_path, mode="a", header=not os.path.exists(cache_path), index=False)
        step.tick("sequences")
        elapsed = time.perf_counter() - start
        logging.debug("Classified %d/%d | LLM calls %d | %.2f s/sequence",
                      i, len(todo), n_llm, elapsed / i)
    return pd.read_csv(cache_path, keep_default_na=False)


def attach_workflows(incidents: pd.DataFrame) -> pd.DataFrame:
    wf = {label: select_workflow(label) for label in incidents["pred_label"].unique()}
    incidents["workflow"] = incidents["pred_label"].map(lambda lab: wf[lab].anomaly_type)
    incidents["automated_response"] = incidents["pred_label"].map(lambda lab: wf[lab].automated)
    incidents["approval_steps"] = incidents["pred_label"].map(lambda lab: sum(s.requires_approval for s in wf[lab].steps))
    incidents["escalation_steps"] = incidents["pred_label"].map(lambda lab: sum(s.escalation for s in wf[lab].steps))
    return incidents


def evaluate_incidents(incidents: pd.DataFrame, cfg, model_name: str, out_path: str) -> None:
    """Score detected abnormal sessions whose anomaly type is known from the open-set test set."""
    test = pd.read_csv(cfg["test_csv"])
    seq_to_label = dict(zip(test[cfg["text_col"]].map(normalize_sequence),
                            test[cfg["label_col"]].map(normalize_open_set_test_label)))
    step, run = steps.current(), steps.current_run()
    abnormal = incidents[incidents["ground_truth"] == 1].copy()
    abnormal["true_label"] = abnormal["template_sequence"].map(seq_to_label)
    typed = abnormal[abnormal["true_label"].notna()]

    # Sessions whose true type is unknown cannot be scored; say so rather than
    # quietly reporting a number computed over a subset.
    step.detail("detected abnormal sessions", len(abnormal))
    step.detail("with ground-truth labels", f"{len(typed):,} "
                                          f"({100 * len(typed) / max(len(abnormal), 1):.1f}% — "
                                          f"the rest cannot be scored)")

    results = {}
    if len(typed):
        metrics = compute_metrics(typed, model_name)
        pd.DataFrame(metrics["per_class"]).to_csv(out_path, index=False)
        logging.debug("Typed abnormal sessions — accuracy %.4f | macro P %.4f | macro R %.4f | macro F1 %.4f\n%s",
                     metrics["accuracy"], metrics["macro_precision"], metrics["macro_recall"],
                     metrics["macro_f1"], metrics["report"])
        if run is not None:
            run.metric(model_name.split("/")[-1],
                       metrics["accuracy"] * 100, metrics["macro_precision"] * 100,
                       metrics["macro_recall"] * 100, metrics["macro_f1"] * 100)
        results["type identified correctly"] = f"{metrics['accuracy'] * 100:.2f}% of scorable sessions"

    fp = incidents[incidents["ground_truth"] == 0]
    if len(fp):
        pct_other = 100 * (fp["pred_label"] == OTHER_LABEL).mean()
        logging.debug("False positives labelled a known type: %s",
                      fp.loc[fp["pred_label"].isin(KNOWN_LABELS), "pred_label"].value_counts().to_dict())
        results["false alarms"] = (f"{len(fp):,} normal sessions, {pct_other:.1f}% correctly "
                                   f"left as '{OTHER_LABEL}'")
    step.outcome(**results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify queued CESAL incidents and select response workflows.")
    parser.add_argument("--config", default="configs/llm/hdfs.yaml")
    parser.add_argument("--model", default=None, help="HF model ID or alias (default: config respond_model).")
    parser.add_argument("--queues", default="edge,cloud", help="Comma-separated queues to process.")
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Classify only the first N unique sequences (for quick checks).")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use bitsandbytes 4-bit loading if installed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = resolve_model_name(args.model or cfg["respond_model"])
    queue_dir = cfg["queue_dir"]
    model_safe = safe_name(model_name)

    # Continue the run queues.py started, so steps 1-4 read as one sequence.
    rep = StepReporter("respond", steps=steps.RESPOND_STEPS,
                       adopt=os.environ.get("CESAL_STEP_HANDOFF")
                       or os.path.join(queue_dir, ".run_steps.json"))

    # ── Step 2: identify each incident's type ─────────────────────────────
    with rep.step("classify") as st:
        incidents = load_queues(queue_dir, [q.strip() for q in args.queues.split(",") if q.strip()])
        sequences = list(dict.fromkeys(incidents["template_sequence"]))
        if args.max_sequences:
            sequences = sequences[:args.max_sequences]
            incidents = incidents[incidents["template_sequence"].isin(sequences)]

        by_queue = incidents["queue"].value_counts()
        st.detail("incidents queued",
                  ", ".join(f"{n:,} from the {q}" for q, n in by_queue.items()))
        st.detail("distinct sequences", len(sequences))
        st.detail("language model", model_name)

        classified = classify_sequences(sequences, model_name, cfg,
                                        os.path.join(queue_dir, f"classified_{model_safe}.csv"),
                                        args.load_in_4bit)
        incidents = incidents.merge(
            classified[["template_sequence", "pred_label", "decision_source", "best_score"]],
            on="template_sequence", how="inner")

        n_unknown = int((incidents["pred_label"] == OTHER_LABEL).sum())
        st.outcome(**{
            "incidents identified": len(incidents),
            "given a known type": len(incidents) - n_unknown,
            "kept as unknown": f"{n_unknown:,} — flagged for a person to review",
        })

    # ── Step 3: choose a response for each ────────────────────────────────
    with rep.step("workflow") as st:
        incidents = attach_workflows(incidents)
        incidents_path = os.path.join(queue_dir, f"incidents_{model_safe}.csv")
        incidents.to_csv(incidents_path, index=False)
        logging.debug("Incidents per workflow:\n%s",
                      incidents.groupby(["queue", "workflow"]).size().to_string())

        for workflow, count in incidents["workflow"].value_counts().items():
            logging.info("   %-42s %s incidents", workflow, f"{count:,}")

        st.outcome(**{
            "response plans chosen": incidents["workflow"].nunique(),
            "steps needing approval": int(incidents["approval_steps"].sum()),
            "steps needing escalation": int(incidents["escalation_steps"].sum()),
            "saved to": incidents_path,
        })

    # ── Step 4: score the identifications ─────────────────────────────────
    with rep.step("score"):
        evaluate_incidents(incidents, cfg, model_name,
                           os.path.join(queue_dir, f"evaluation_{model_safe}.csv"))

    rep.finish(outputs=queue_dir)


if __name__ == "__main__":
    setup_logging("process_queues")
    main()
