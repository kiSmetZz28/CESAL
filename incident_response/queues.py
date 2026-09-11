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

from cesal_core.utils.config import load_config, setup_logging

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

    Mirrors dashboard/cloud_runner.py without the ground-truth point adjustment.
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

    sequences, ground_truth = load_sessions(data_path)
    edge, hybrid, routed = final_predictions(out_dir)
    energy = np.load(os.path.join(out_dir, "energy_matrix.npy"), mmap_mode="r")
    logging.info("Sessions: %d | scored events: %d of %d | anomalous events: %d (routed to cloud: %d)",
                 len(sequences), len(hybrid), sum(map(len, sequences)), int(hybrid.sum()), int(routed.sum()))

    records = build_queues(sequences, ground_truth, edge, hybrid, routed, energy)
    os.makedirs(cfg["queue_dir"], exist_ok=True)
    for queue, fname in QUEUE_FILES.items():
        part = records[records["queue"] == queue]
        path = os.path.join(cfg["queue_dir"], fname)
        part.to_csv(path, index=False)
        logging.info("%-5s queue: %6d incidents (%d abnormal sessions, %d normal false positives), "
                     "%d unique sequences -> %s", queue, len(part), int(part["ground_truth"].sum()),
                     int((part["ground_truth"] == 0).sum()), part["template_sequence"].nunique(), path)
    starts = np.concatenate([[0], np.cumsum([len(s) for s in sequences])[:-1]])
    unscored = (starts >= len(hybrid)) & (ground_truth == 1)
    missed = int((ground_truth == 1).sum() - unscored.sum() - records["ground_truth"].sum())
    logging.info("Detected %d sessions (%d unique sequences); abnormal sessions missed: %d "
                 "(plus %d past the last full window, never scored)",
                 len(records), records["template_sequence"].nunique(), missed, int(unscored.sum()))


if __name__ == "__main__":
    setup_logging("queues")
    main()
