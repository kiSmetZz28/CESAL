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

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_inference_pipeline.run import run_inference

_ABOUT = """
Measure what cloud escalation contributes, as a function of how much is sent.

The routing ratio is the share of log events the edge tier escalates for cloud
verification. A higher ratio sends more events for cloud processing; its effect
on detection scores is measured here. This runs the pipeline at each ratio and
reports the results side by side; the expensive on-device scan is performed once
and reused, since it does not depend on the ratio.
"""


def _scores(out_dir: str):
    """Point-adjusted precision / recall / F1, or None if no run outputs exist."""
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


def _file_stamp(path):
    """Identify a file version without reading large saved prediction arrays."""
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def sweep(config_path: str, ratios: List[float]) -> bool:
    """Run every ratio; return True only when each produced a fresh result."""
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
    logging.info("%s", steps.leader(
        "ratios to run", ", ".join(f"{r:.0%}" for r in ratios)))
    logging.info("%s", steps.leader("results under", base))

    edge_source = None
    rows = []
    edge_files = ('edge_preds.npy', 'edge_preds_raw.npy', 'ground_truth.npy', 'energy_matrix.npy')
    for r in ratios:
        out_dir = os.path.join(base, f"ratio_{int(round(r * 100)):02d}")
        logging.info("")
        logging.info("═══ routing ratio %.0f%%  →  %s ═══", r * 100, out_dir)
        hybrid_path = os.path.join(out_dir, 'hybrid_preds.npy')
        before_hybrid = _file_stamp(hybrid_path)
        before_edge = {name: _file_stamp(os.path.join(out_dir, name)) for name in edge_files}
        score = None
        try:
            run_inference(config_path, ratio=r, output_dir=out_dir,
                          reuse_edge_from=edge_source)
            after_hybrid = _file_stamp(hybrid_path)
            if after_hybrid is not None and after_hybrid != before_hybrid:
                score = _scores(out_dir)
            if score is None:
                logging.error("Ratio %.0f%% produced no new hybrid result.", r * 100)
        except Exception as exc:
            # Keep successful ratios, but never score an older result left in
            # this directory by a previous sweep (or an interrupted rerun).
            logging.error("Ratio %.0f%% failed: %s", r * 100, exc)

        # A completed edge scan is reusable even if its subsequent cloud stage
        # failed. Require all four files to be fresh to reject partial scans.
        if edge_source is None:
            after_edge = {name: _file_stamp(os.path.join(out_dir, name)) for name in edge_files}
            if all(after_edge[name] is not None and after_edge[name] != before_edge[name]
                   for name in edge_files):
                edge_source = out_dir
        rows.append((r, score))

    # ── The table ─────────────────────────────────────────────────────────
    logging.info("")
    logging.info("═" * 66)
    logging.info(" ROUTING RATIO SWEEP · %s", dataset)
    logging.info("═" * 66)
    logging.info("   %-10s%12s%10s%10s%10s",
                 "ratio", "escalated", "P", "R", "F1")
    logging.info("   %s", "─" * 60)
    for r, sc in rows:
        if sc is None:
            logging.info("   %-10s%12s%10s%10s%10s",
                         f"{r:.0%}", "—", "—", "—", "(no result)")
            continue
        p, rec, f, n_routed, n_total = sc
        logging.info("   %-10s%12s%10.2f%10.2f%10.2f",
                     f"{r:.0%}", f"{n_routed:,}", p, rec, f)
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
    logging.info("%s", steps.leader("table written to", csv_path))
    complete = bool(rows) and all(sc is not None for _, sc in rows)
    if not complete:
        logging.error("Sweep incomplete: %d of %d ratios produced results. See the failures above.",
                      sum(sc is not None for _, sc in rows), len(rows))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description="CESAL routing-ratio sweep.")
    parser.add_argument('--config', default='configs/inference/os.yaml',
                        help='Path to the inference YAML config.')
    parser.add_argument('--ratios', default='0.05,0.1,0.2,0.3',
                        help='Comma-separated routing ratios, e.g. 0.05,0.1,0.2,0.3')
    args, _ = parser.parse_known_args()

    ratios = [float(x) for x in args.ratios.split(',') if x.strip()]
    if not ratios:
        parser.error("provide at least one routing ratio")
    bad = [r for r in ratios if not 0 < r <= 1]
    if bad:
        parser.error(f"routing ratios must be in (0, 1]; got {bad}")
    if not sweep(args.config, ratios):
        raise SystemExit(1)


if __name__ == '__main__':
    setup_logging('sweep')
    main()
