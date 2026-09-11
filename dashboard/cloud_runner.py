#!/usr/bin/env python3
"""Cloud inference phase — runs in the cloud conda env (ceco-lad-cloud).

Reads intermediate files saved by the edge phase and runs the BAT ensemble.
Called automatically by the dashboard when using split environments.

Usage (from project root):
    python dashboard/cloud_runner.py --config configs/inference/os.yaml
"""
import argparse
import logging
import os
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _point_adjust(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Fill entire GT anomaly segments once any window in the segment is detected."""
    gt   = gt.astype(int)
    pred = pred.astype(int).copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="CECO-LAD cloud inference phase.")
    parser.add_argument("--config", required=True, help="Inference YAML config path.")
    args = parser.parse_args()

    os.chdir(str(ROOT))

    from ceco_core.utils.config import load_config, setup_logging
    from ceco_core.utils.metrics import evaluate
    from ceco_lad_inference_pipeline import lad_bat_cloud

    setup_logging("cloud")
    cfg      = load_config(args.config)
    dataset  = cfg["dataset"]
    out_base = cfg.get("output_dir", f"outputs/{dataset.lower()}")

    def _load(fname: str) -> np.ndarray:
        path = os.path.join(out_base, fname)
        if not os.path.exists(path):
            logging.error(
                "Required file not found: %s\n"
                "Run the edge phase first (Phase 1/2).", path
            )
            sys.exit(1)
        return np.load(path)

    ground_truth   = _load("ground_truth.npy")
    routed_indices = _load("routed_indices.npy")
    edge_preds_raw = _load("edge_preds_raw.npy")

    cloud_cfg = cfg.get("cloud")
    if not cloud_cfg:
        logging.info("No 'cloud' section in config — nothing to do.")
        return

    win_size = cfg.get("win_size", 100)

    lines_path   = os.path.join(out_base, "routed_lines.npy")
    windows_path = os.path.join(out_base, "routed_windows.npy")

    if os.path.exists(lines_path):
        routed_lines = np.load(lines_path)
        input_c = routed_lines.shape[1]
        n_lines = len(routed_lines)
        # Drop the tail that doesn't fill a complete window.
        n_full   = (n_lines // win_size) * win_size
        windows  = routed_lines[:n_full].reshape(-1, win_size, input_c)
        use_indices = routed_indices[:n_full]
    elif os.path.exists(windows_path):
        windows = np.load(windows_path)
        input_c = windows.shape[2]
        use_indices = routed_indices
    else:
        logging.error("Neither routed_lines.npy nor routed_windows.npy found in %s.\n"
                      "Run the edge phase first (Phase 1/2).", out_base)
        sys.exit(1)

    if len(windows) == 0:
        logging.info("No routed lines — skipping cloud inference.")
        return

    cloud_cfg.setdefault("dataset", dataset)
    cloud_cfg.setdefault("win_size", win_size)
    cloud_cfg.setdefault("input_c", input_c)

    logging.info("=== Cloud BAT Inference ===")
    # lad_bat_cloud returns one prediction per line [N_win * win_size]
    cloud_preds = lad_bat_cloud.run(windows, cloud_cfg)
    np.save(os.path.join(out_base, "cloud_preds.npy"), cloud_preds)

    logging.info("=== Stage 4: Hybrid Evaluation ===")

    hybrid_preds = edge_preds_raw.copy()
    if os.path.exists(lines_path):
        # Per-line predictions mapped directly back via index.
        hybrid_preds[use_indices] = cloud_preds
    else:
        # Legacy window format — stamp entire window with its prediction.
        for k, win_idx in enumerate(routed_indices):
            start = int(win_idx) * win_size
            end   = min(start + win_size, len(hybrid_preds))
            hybrid_preds[start:end] = cloud_preds[k * win_size:(k + 1) * win_size]

    logging.info(
        "Cloud preds: %d / %d routed %s anomalous.",
        int(cloud_preds.sum()), len(routed_indices),
        "lines" if os.path.exists(lines_path) else "windows",
    )
    logging.info(
        "Hybrid raw preds: %d anomalous / %d total timesteps.",
        int(hybrid_preds.sum()), len(hybrid_preds),
    )
    hybrid_adj = _point_adjust(ground_truth, hybrid_preds)
    np.save(os.path.join(out_base, "hybrid_preds.npy"), hybrid_adj)
    evaluate(ground_truth, hybrid_adj, prefix="Hybrid")

    logging.info("=== Cloud inference complete. Outputs in '%s' ===", out_base)


if __name__ == "__main__":
    main()
