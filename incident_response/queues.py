"""Build CESAL's anomaly queues from the HDFS collaborative LAD outputs (paper Sec. 3.1).

A test session (one line of the LAD test files) is detected as abnormal when CESAL's final
prediction — edge Q-BAT, with routed events replaced by cloud BAT, before point adjustment —
marks any of its events anomalous. Each detected session becomes an incident record:

  Q_C  queue_cloud.csv   at least one anomalous event was verified by cloud-side BAT
  Q_E  queue_edge.csv    every anomalous event was decided locally by edge-side Q-BAT

Records hold the session's template-event sequence and detection metadata. Block IDs and
timestamps are not part of the LAD test data. `ground_truth` (1 = session from the abnormal
test file) is kept for evaluation only.

Usage (from project root):
  python -m incident_response.queues --config configs/llm/hdfs.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.steps import StepReporter

_ABOUT = """
Turning raw detections into incidents, working out what each one is, and
choosing how to respond.
Sessions the detector flagged are filed into two queues — one for what the edge
device decided alone, one for what the cloud confirmed. Each queued sequence is
then matched against a library of known incident types; anything that matches
none of them is kept as unknown and left for a person. Known types map to a
predefined response plan, whose riskier actions need an administrator's approval.
"""

# LAD test files in the order HDFSSegLoader concatenates them (normal first, then abnormal)
TEST_FILES = (("hdfs_test_normal.txt", 0), ("hdfs_test_abnormal.txt", 1))
QUEUE_FILES = {"edge": "queue_edge.csv", "cloud": "queue_cloud.csv"}


def load_sessions(data_path: str) -> Tuple[List[List[str]], np.ndarray]:
    """Event-ID sequences of all test sessions and their session-level ground truth."""
    sequences, labels = [], []
    for fname, label in TEST_FILES:
        with open(os.path.join(data_path, fname)) as f:
            for line in f:
                sequences.append(line.split())
                labels.append(label)
    return sequences, np.array(labels, dtype=np.int8)


def final_predictions(out_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event edge predictions, final hybrid predictions, and the routed-to-cloud mask.

    Mirrors cesal_inference_pipeline/cloud_runner.py without the ground-truth point adjustment.
    """
    edge = np.load(os.path.join(out_dir, "edge_preds_raw.npy")).astype(np.int8)
    routed = np.zeros(len(edge), dtype=bool)
    hybrid = edge.copy()
    ri_path, cp_path = os.path.join(out_dir, "routed_indices.npy"), os.path.join(out_dir, "cloud_preds.npy")
    if os.path.exists(ri_path) and os.path.exists(cp_path):
        cloud = np.load(cp_path).astype(np.int8)
        used = np.load(ri_path)[:len(cloud)]
        hybrid[used] = cloud
        routed[used] = True
    else:
        logging.warning("No cloud predictions in %s — queues use edge predictions only.", out_dir)
    return edge, hybrid, routed


def build_queues(sequences: List[List[str]], ground_truth: np.ndarray, edge: np.ndarray,
                 hybrid: np.ndarray, routed: np.ndarray, energy: np.ndarray) -> pd.DataFrame:
    n = len(hybrid)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    ends = starts + lengths
    s_clip, e_clip = np.minimum(starts, n), np.minimum(ends, n)   # events past the last full window are unscored

    def per_session(mask: np.ndarray) -> np.ndarray:
        csum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        return csum[e_clip] - csum[s_clip]

    n_anomalous = per_session(hybrid == 1)
    n_cloud = per_session((hybrid == 1) & routed)
    n_edge = per_session((hybrid == 1) & ~routed)
    n_routed = per_session(routed)

    rows = []
    for i in np.flatnonzero(n_anomalous > 0):
        rows.append({
            "session_id": int(i),
            "queue": "cloud" if n_cloud[i] > 0 else "edge",
            "template_sequence": " ".join(f"E{e}" for e in sequences[i]),
            "n_events": int(lengths[i]),
            "n_scored_events": int(e_clip[i] - s_clip[i]),
            "n_anomalous_events": int(n_anomalous[i]),
            "n_edge_anomalous": int(n_edge[i]),
            "n_cloud_anomalous": int(n_cloud[i]),
            "n_routed_events": int(n_routed[i]),
            "edge_energy_max": json.dumps([round(float(v), 6) for v in energy[s_clip[i]:e_clip[i]].max(axis=0)]),
            "ground_truth": int(ground_truth[i]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CESAL anomaly queues from HDFS detection outputs.")
    parser.add_argument("--config", default="configs/llm/hdfs.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    det_cfg = load_config(cfg["detection_config"])
    data_path, out_dir = det_cfg["data_path"], det_cfg["output_dir"]

    rep = StepReporter("respond", dataset=det_cfg.get("dataset", "HDFS"),
                       steps=steps.RESPOND_STEPS, about=_ABOUT)

    with rep.step("queue") as st:
        st.phase("reading the detector predictions")
        sequences, ground_truth = load_sessions(data_path)
        edge, hybrid, routed = final_predictions(out_dir)
        energy = np.load(os.path.join(out_dir, "energy_matrix.npy"), mmap_mode="r")
        st.detail("sessions in the test set", len(sequences))
        st.detail("events scored", f"{len(hybrid):,} of {sum(map(len, sequences)):,}")
        # routed counts every event sent to the cloud, not just the flagged ones.
        st.detail("events flagged", f"{int(hybrid.sum()):,}")
        st.detail("events re-verified in cloud", int(routed.sum()))

        st.phase("queueing each detected session by originating tier")
        records = build_queues(sequences, ground_truth, edge, hybrid, routed, energy)
        os.makedirs(cfg["queue_dir"], exist_ok=True)
        counts = {}
        for queue, fname in QUEUE_FILES.items():
            part = records[records["queue"] == queue]
            path = os.path.join(cfg["queue_dir"], fname)
            part.to_csv(path, index=False)
            counts[queue] = len(part)
            logging.info(
                "   %-5s queue · %s incidents · %s truly abnormal · %s false alarms",
                queue, f"{len(part):,}", f"{int(part['ground_truth'].sum()):,}",
                f"{int((part['ground_truth'] == 0).sum()):,}",
            )
        starts = np.concatenate([[0], np.cumsum([len(s) for s in sequences])[:-1]])
        unscored = (starts >= len(hybrid)) & (ground_truth == 1)
        missed = int((ground_truth == 1).sum() - unscored.sum() - records["ground_truth"].sum())
        if missed or int(unscored.sum()):
            st.warn(f"{missed} abnormal sessions were not detected, and "
                    f"{int(unscored.sum())} fell past the last full window and were never scored.")

        st.outcome(**{
            "incidents queued": len(records),
            "caught at the edge": counts.get("edge", 0),
            "verified in the cloud": counts.get("cloud", 0),
            "unique sequences": records["template_sequence"].nunique(),
        })

    # Steps 2-4 run in process_queues.py; hand this run over so the two
    # processes report as one numbered sequence with a single summary.
    handoff = os.environ.get("CESAL_STEP_HANDOFF") or os.path.join(
        cfg["queue_dir"], ".run_steps.json")
    rep.export(handoff)
    logging.info("")
    logging.info("   Queues built — handing over to the classification step.")


if __name__ == "__main__":
    setup_logging("queues")
    main()
