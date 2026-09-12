"""Inference pipeline orchestrator.

Stages
------
1. edge   : Q-BAT models compute per-model energy scores  [cesal-edge env]
2. route  : Mahalanobis routing selects uncertain windows  [cesal-edge env]
3. cloud  : BAT ensemble re-predicts routed windows        [cesal-cloud env — subprocess]
4. hybrid : Merge edge and cloud predictions, log metrics  [cesal-cloud env — subprocess]

Stage 3+4 always run inside the cesal-cloud conda environment by calling
cloud_runner.py as a subprocess.  The Python interpreter is located via
_detect_cloud_python() which checks (in order):
  1. CESAL_CLOUD_PYTHON environment variable
  2. ~/miniconda3/envs/cesal-cloud/bin/python
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

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_core.utils.metrics import evaluate
from cesal_core.utils.steps import StepReporter
from cesal_inference_pipeline import lad_qbat_edge
from cesal_inference_pipeline.routing import compute_inv_cov, select_indices_by_distance

# lad_bat_cloud is NOT imported here — BAT models always run in the cesal-cloud env.


def _detect_cloud_python() -> str:
    """Return the path to the cesal-cloud env Python interpreter.

    Override by setting the CESAL_CLOUD_PYTHON environment variable.
    Falls back to sys.executable when the cesal-cloud env is not found
    (single-environment setups where both edge and cloud share one env).
    """
    env_var = os.environ.get("CESAL_CLOUD_PYTHON")
    if env_var:
        return env_var
    cloud_py = Path.home() / "miniconda3" / "envs" / "cesal-cloud" / "bin" / "python"
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

    rep = StepReporter("infer", dataset=dataset, steps=steps.INFER_STEPS)

    # ── Step 1: Edge Q-BAT inference ──────────────────────────────────────
    with rep.step("edge") as st:
        st.detail("config", inference_config_path)
        result = lad_qbat_edge.run(cfg)

        edge_preds_adj = _point_adjust(result.ground_truth, result.predictions)
        np.save(os.path.join(out_base, 'edge_preds.npy'),     edge_preds_adj)
        np.save(os.path.join(out_base, 'edge_preds_raw.npy'), result.predictions)
        np.save(os.path.join(out_base, 'ground_truth.npy'),   result.ground_truth)
        np.save(os.path.join(out_base, 'energy_matrix.npy'),  result.energy_matrix)
        evaluate(result.ground_truth, edge_preds_adj, prefix="Edge")

    # ── Step 2: Mahalanobis routing ───────────────────────────────────────
    tolerance     = cfg.get('routing_tolerance', 0.1)
    distance_type = cfg.get('routing_distance', 'ma')
    win_size      = cfg.get('win_size', 100)
    routed_indices: list = []

    with rep.step("route") as st:
        st.detail("distance", "Mahalanobis" if distance_type == "ma" else "Euclidean")
        st.detail("tolerance", f"{tolerance * 100:.0f}% of events")

        # Routing on per-line energy scores → routed_indices are line indices.
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
                st.warn("Covariance matrix is singular; routing all predicted anomalies instead.")
                routed_indices = list(np.where(result.predictions == 1)[0])
        else:
            st.note("Only one edge model — falling back to a single-score margin.")
            margin  = result.energy_matrix[:, 0] - result.thresholds[0]
            n_route = max(1, int(len(margin) * tolerance))
            routed_indices = sorted(np.argsort(margin)[-n_route:].tolist())

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

        n_total = len(result.predictions)
        st.outcome(**{
            "events considered": n_total,
            "routed to cloud": f"{len(routed_indices):,} "
                               f"({len(routed_indices) / max(n_total, 1) * 100:.1f}%)",
            "kept at edge": n_total - len(routed_indices),
        })

    # ── Steps 3+4: Cloud BAT inference + hybrid merge ──────────────────────
    # Always run in the cesal-cloud conda env via cloud_runner.py subprocess so that
    # BAT checkpoints are never loaded inside the cesal-edge environment.

    # An external orchestrator (the dashboard's two-env script) sets
    # CESAL_STEP_HANDOFF and runs cloud_runner.py itself against a config that
    # still has its cloud section. Hand our steps over and let that process
    # close the run, rather than reporting steps 3-4 as skipped here.
    external_handoff = os.environ.get("CESAL_STEP_HANDOFF")

    cloud_cfg = cfg.get('cloud')
    if not cloud_cfg:
        if external_handoff:
            rep.export(external_handoff)
            logging.info("")
            logging.info("   Steps 1-2 complete — handing over to the cloud environment.")
            return
        reason = "No 'cloud' section in the config — this is an edge-only run."
        rep.skip("cloud", reason)
        rep.skip("hybrid", reason)
        rep.finish(outputs=out_base)
        return

    if not routed_indices:
        reason = "No events were routed to the cloud, so there is nothing to re-check."
        rep.skip("cloud", reason)
        rep.skip("hybrid", reason)
        rep.finish(outputs=out_base)
        return

    # Hand the completed steps to the cloud process so that it can number its own
    # steps 3 and 4 and print one summary covering the whole pipeline.
    handoff = external_handoff or os.path.join(out_base, '.run_steps.json')
    rep.export(handoff)

    cloud_py     = _detect_cloud_python()
    runner       = str(Path(__file__).parent.parent / "dashboard" / "cloud_runner.py")
    project_root = str(Path(__file__).parent.parent)

    logging.info("")
    logging.info("   Handing steps 3-4 to the cloud environment:")
    logging.info("   %s", cloud_py)

    proc = subprocess.run(
        [cloud_py, runner, "--config", inference_config_path],
        cwd=project_root,
        env={**os.environ, "CESAL_STEP_HANDOFF": handoff},
    )

    if proc.returncode != 0:
        logging.error(
            "Cloud inference subprocess exited with code %d.", proc.returncode
        )
        rep.finish(outputs=out_base)


if __name__ == '__main__':
    setup_logging('inference')

    parser = argparse.ArgumentParser(description="CESAL inference pipeline.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference/os.yaml',
        help='Path to inference YAML config.',
    )
    args, _ = parser.parse_known_args()
    run_inference(args.config)
