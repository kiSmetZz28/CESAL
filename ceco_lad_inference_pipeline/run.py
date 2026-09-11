"""Inference pipeline orchestrator.

Stages
------
1. edge   : Q-BAT models compute per-model energy scores  [ceco-lad-edge env]
2. route  : Mahalanobis routing selects uncertain windows  [ceco-lad-edge env]
3. cloud  : BAT ensemble re-predicts routed windows        [ceco-lad-cloud env — subprocess]
4. hybrid : Merge edge and cloud predictions, log metrics  [ceco-lad-cloud env — subprocess]

Stage 3+4 always run inside the ceco-lad-cloud conda environment by calling
cloud_runner.py as a subprocess.  The Python interpreter is located via
_detect_cloud_python() which checks (in order):
  1. CECO_CLOUD_PYTHON environment variable
  2. ~/miniconda3/envs/ceco-lad-cloud/bin/python
  3. sys.executable  (same-env fallback)
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from ceco_core.utils.config import load_config, setup_logging
from ceco_core.utils.io import mkdir
from ceco_core.utils.metrics import evaluate
from ceco_lad_inference_pipeline import lad_qbat_edge
from ceco_lad_inference_pipeline.routing import compute_inv_cov, select_indices_by_distance

# lad_bat_cloud is NOT imported here — BAT models always run in the ceco-lad-cloud env.


def _detect_cloud_python() -> str:
    """Return the path to the ceco-lad-cloud env Python interpreter.

    Override by setting the CECO_CLOUD_PYTHON environment variable.
    Falls back to sys.executable when the ceco-lad-cloud env is not found
    (single-environment setups where both edge and cloud share one env).
    """
    env_var = os.environ.get("CECO_CLOUD_PYTHON")
    if env_var:
        return env_var
    cloud_py = Path.home() / "miniconda3" / "envs" / "ceco-lad-cloud" / "bin" / "python"
    if cloud_py.exists():
        return str(cloud_py)
    return sys.executable


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


def run_inference(inference_config_path: str) -> None:
    """Run the full inference pipeline for one dataset.

    Parameters
    ----------
    inference_config_path : str
        Path to per-dataset inference YAML (configs/inference/*.yaml).
    """
    cfg      = load_config(inference_config_path)
    dataset  = cfg['dataset']
    out_base = cfg.get('output_dir', str(Path('outputs') / dataset.lower()))
    mkdir(out_base)

    # ── Stage 1: Edge Q-BAT inference ─────────────────────────────────────
    logging.info("=== Stage 1: Edge Q-BAT Inference ===")
    result = lad_qbat_edge.run(cfg)

    edge_preds_adj = _point_adjust(result.ground_truth, result.predictions)
    np.save(os.path.join(out_base, 'edge_preds.npy'),     edge_preds_adj)
    np.save(os.path.join(out_base, 'edge_preds_raw.npy'), result.predictions)
    np.save(os.path.join(out_base, 'ground_truth.npy'),   result.ground_truth)
    np.save(os.path.join(out_base, 'energy_matrix.npy'),  result.energy_matrix)
    evaluate(result.ground_truth, edge_preds_adj, prefix="Edge")

    # ── Stage 2: Mahalanobis routing ──────────────────────────────────────
    logging.info("=== Stage 2: Mahalanobis Routing ===")
    tolerance     = cfg.get('routing_tolerance', 0.1)
    distance_type = cfg.get('routing_distance', 'ma')
    win_size      = cfg.get('win_size', 100)

    # Routing on per-line energy scores → routed_indices are line indices.
    routed_indices: list = []
    if result.energy_matrix.shape[1] >= 2:
        try:
            _, inv_cov = compute_inv_cov(result.train_energy_matrix)
            routed_indices = select_indices_by_distance(
                test_scores=result.energy_matrix,
                thresholds=result.thresholds,
                inv_covmat=inv_cov,
                distance_type=distance_type,
                tolerance=tolerance,
            )
        except np.linalg.LinAlgError:
            logging.warning("Covariance matrix is singular; routing all predicted anomalies.")
            routed_indices = list(np.where(result.predictions == 1)[0])
    else:
        margin  = result.energy_matrix[:, 0] - result.thresholds[0]
        n_route = max(1, int(len(margin) * tolerance))
        routed_indices = sorted(np.argsort(margin)[-n_route:].tolist())

    logging.info(
        "Routing: %d / %d lines selected to cloud (tolerance=%.0f%%).",
        len(routed_indices), len(result.predictions), tolerance * 100,
    )
    np.save(
        os.path.join(out_base, 'routed_indices.npy'),
        np.array(routed_indices, dtype=int),
    )

    # Extract the feature vector of each routed line for cloud processing.
    # cloud_runner pads each to win_size and runs EMAT — one prediction per line.
    if routed_indices:
        routed_lines = np.array([
            result.test_windows[i // win_size][i % win_size]
            for i in routed_indices
        ], dtype=np.float32)
        np.save(os.path.join(out_base, 'routed_lines.npy'), routed_lines)

    # ── Stages 3+4: Cloud BAT inference + hybrid merge ────────────────────
    # Always run in the ceco-lad-cloud conda env via cloud_runner.py subprocess so that
    # BAT checkpoints are never loaded inside the ceco-lad environment.

    cloud_cfg = cfg.get('cloud')
    if not cloud_cfg:
        logging.info("No 'cloud' section in config — edge inference complete.")
        logging.info("=== Inference complete. Outputs in '%s' ===", out_base)
        return

    if not routed_indices:
        logging.info("No lines routed to cloud — skipping cloud and hybrid stages.")
        logging.info("=== Inference complete. Outputs in '%s' ===", out_base)
        return

    logging.info("=== Stage 3+4: Cloud BAT Inference + Hybrid Merge ===")
    cloud_py    = _detect_cloud_python()
    runner      = str(Path(__file__).parent.parent / "dashboard" / "cloud_runner.py")
    project_root = str(Path(__file__).parent.parent)

    logging.info("Delegating to cloud env: %s", cloud_py)
    proc = subprocess.run(
        [cloud_py, runner, "--config", inference_config_path],
        cwd=project_root,
    )

    if proc.returncode != 0:
        logging.error(
            "Cloud inference subprocess exited with code %d.", proc.returncode
        )
    else:
        logging.info("=== Inference complete. Outputs in '%s' ===", out_base)


if __name__ == '__main__':
    setup_logging('inference')

    parser = argparse.ArgumentParser(description="CECO-LAD inference pipeline.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference/bgl.yaml',
        help='Path to inference YAML config.',
    )
    args, _ = parser.parse_known_args()
    run_inference(args.config)
