"""Routing-ratio sweep: run the pipeline at several ratios and tabulate the result.

The routing ratio is the fraction of events the edge escalates to the cloud. The
paper reports detection quality as that ratio varies; this reproduces that table.

Only the routing, cloud and merge stages depend on the ratio — the edge scan does
not. So the edge scan runs once and every subsequent ratio reuses it, which is the
difference between one Q-BAT pass and one per ratio (hours, on HDFS).

Usage (from project root):
    python -m cesal_inference_pipeline.sweep --config configs/inference/os.yaml \\
        --ratios 0.05,0.1,0.2,0.3
    python run.py sweep os 0.05,0.1,0.2,0.3
"""
import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_inference_pipeline.run import run_inference

_ABOUT = """
Measuring what the cloud actually buys you, as the amount sent to it changes.
The routing ratio is the share of log events the device escalates for a second
opinion. A higher ratio should catch more, at the cost of sending more
security-sensitive data off the device. This runs the pipeline at each ratio and
puts the results side by side. The expensive on-device scan is done once and
reused, since it does not depend on the ratio.
"""


def _scores(out_dir: str):
    """Precision / recall / F1 of a finished run, or None if it produced nothing."""
    gt_p = os.path.join(out_dir, 'ground_truth.npy')
    hy_p = os.path.join(out_dir, 'hybrid_preds.npy')
    if not (os.path.exists(gt_p) and os.path.exists(hy_p)):
        return None
    from sklearn.metrics import precision_recall_fscore_support
    gt = np.load(gt_p).astype(int)
    hy = np.load(hy_p).astype(int)
    p, r, f, _ = precision_recall_fscore_support(gt, hy, average='binary', zero_division=0)
    routed = os.path.join(out_dir, 'routed_indices.npy')
    n_routed = len(np.load(routed)) if os.path.exists(routed) else 0
    return p * 100, r * 100, f * 100, n_routed, len(gt)


def sweep(config_path: str, ratios: List[float]) -> None:
    cfg = load_config(config_path)
    dataset = cfg['dataset']
    base = cfg.get('output_dir', str(Path('outputs') / dataset.lower()))
    mkdir(base)

    # Each ratio's run reports its own four steps, so the sweep only frames them.
    logging.info("")
    logging.info("═" * 66)
    logging.info(" CESAL · routing-ratio sweep · %s", dataset)
    logging.info("═" * 66)
    for line in _ABOUT.strip().splitlines():
        logging.info("   %s", line.strip())
    logging.info("")
    logging.info("   ratios to run ............... %s",
                 ", ".join(f"{r:.0%}" for r in ratios))
    logging.info("   results under ............... %s", base)

    edge_source = None
    rows = []
    for i, r in enumerate(ratios):
        out_dir = os.path.join(base, f"ratio_{int(round(r * 100)):02d}")
        logging.info("")
        logging.info("═══ routing ratio %.0f%%  →  %s ═══", r * 100, out_dir)
        run_inference(config_path, ratio=r, output_dir=out_dir,
                      reuse_edge_from=edge_source)
        # Every later ratio reuses the first run's edge scan.
        edge_source = edge_source or out_dir
        rows.append((r, _scores(out_dir)))

    # ── The table ─────────────────────────────────────────────────────────
    logging.info("")
    logging.info("═" * 66)
    logging.info(" ROUTING RATIO SWEEP · %s", dataset)
    logging.info("═" * 66)
    logging.info("   %-8s %10s %8s %8s %8s", "ratio", "escalated", "P", "R", "F1")
    logging.info("   %s", "─" * 60)
    for r, sc in rows:
        if sc is None:
            logging.info("   %-8s %10s %8s %8s %8s", f"{r:.0%}", "—", "—", "—", "(no result)")
            continue
        p, rec, f, n_routed, n_total = sc
        logging.info("   %-8s %10s %8.2f %8.2f %8.2f", f"{r:.0%}",
                     f"{n_routed:,}", p, rec, f)
    logging.info("═" * 66)

    csv_path = os.path.join(base, "routing_ratio_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["routing_ratio", "events_escalated", "events_total",
                    "precision", "recall", "f1"])
        for r, sc in rows:
            if sc is None:
                w.writerow([r, "", "", "", "", ""])
            else:
                p, rec, f, n_routed, n_total = sc
                w.writerow([r, n_routed, n_total, f"{p:.4f}", f"{rec:.4f}", f"{f:.4f}"])
    logging.info("   table written to %s", csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CESAL routing-ratio sweep.")
    parser.add_argument('--config', default='configs/inference/os.yaml',
                        help='Path to the inference YAML config.')
    parser.add_argument('--ratios', default='0.05,0.1,0.2,0.3',
                        help='Comma-separated routing ratios, e.g. 0.05,0.1,0.2,0.3')
    args, _ = parser.parse_known_args()

    ratios = [float(x) for x in args.ratios.split(',') if x.strip()]
    bad = [r for r in ratios if not 0 < r <= 1]
    if bad:
        parser.error(f"routing ratios must be in (0, 1]; got {bad}")
    sweep(args.config, ratios)


if __name__ == '__main__':
    setup_logging('sweep')
    main()
