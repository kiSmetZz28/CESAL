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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from cesal_core.data.loaders import get_loader_segment
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


def _load_edge_result(cfg: dict, source_dir: str) -> lad_qbat_edge.EdgeResult:
    """Rebuild an EdgeResult from a previous run's saved arrays.

    The edge scan does not depend on the routing ratio, so a ratio sweep can
    reuse it instead of re-scoring every window with Q-BAT — by far the most
    expensive stage. Only the test windows are re-read (parsing, no inference),
    because the routed feature vectors are cut from them.
    """
    import yaml as _yaml

    energy = np.load(os.path.join(source_dir, 'energy_matrix.npy'))
    preds  = np.load(os.path.join(source_dir, 'edge_preds_raw.npy'))
    gt     = np.load(os.path.join(source_dir, 'ground_truth.npy'))

    thresh_path = cfg.get('threshold_output',
                          str(Path('outputs') / cfg['dataset'].lower() / 'thresholds_edge.yaml'))
    stored = {e['name']: float(e['threshold'])
              for e in (_yaml.safe_load(open(thresh_path)) or {}).get('models', [])}
    thresholds = np.array([stored[m['name']] for m in cfg.get('edge_models', [])
                           if m['name'] in stored], dtype=float)

    loader = get_loader_segment(
        [cfg.get('num_epochs', 3), cfg.get('k', 3),
         cfg.get('e_layer_num', 3), cfg['batch_size']],
        cfg['data_path'], batch_size=cfg['batch_size'], win_size=cfg['win_size'],
        step=cfg['win_size'], mode='test', dataset=cfg['dataset'],
    )
    windows = np.concatenate([x.numpy() for x, _ in loader], axis=0)

    return lad_qbat_edge.EdgeResult(
        predictions=preds, ground_truth=gt, energy_matrix=energy,
        train_energy_matrix=energy, thresholds=thresholds, test_windows=windows,
    )


def run_inference(
    inference_config_path: str,
    ratio: Optional[float] = None,
    distance: Optional[str] = None,
    output_dir: Optional[str] = None,
    reuse_edge_from: Optional[str] = None,
) -> None:
    """Run the full inference pipeline for one dataset.

    Parameters
    ----------
    inference_config_path : str
        Path to per-dataset inference YAML (configs/inference/*.yaml).
    ratio : float, optional
        Fraction of events escalated to the cloud; overrides the config's
        ``routing_tolerance``. This is the paper's routing ratio.
    distance : {'ma', 'eu'}, optional
        Overrides the config's ``routing_distance``.
    output_dir : str, optional
        Overrides the config's ``output_dir`` — used to keep each ratio of a
        sweep in its own directory.
    reuse_edge_from : str, optional
        Directory of a previous run whose edge outputs should be reused
        instead of re-running the Q-BAT scan.
    """
    cfg      = load_config(inference_config_path)
    dataset  = cfg['dataset']
    if ratio is not None:
        cfg['routing_tolerance'] = ratio
    if distance is not None:
        cfg['routing_distance'] = distance
    if output_dir is not None:
        cfg['output_dir'] = output_dir
    out_base = cfg.get('output_dir', str(Path('outputs') / dataset.lower()))
    mkdir(out_base)

    # The cloud stage runs as a subprocess and reads the config itself, so the
    # overrides have to reach it on disk; this also records exactly what ran.
    effective_cfg_path = os.path.join(out_base, 'effective_config.yaml')
    with open(effective_cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    rep = StepReporter("infer", dataset=dataset, steps=steps.INFER_STEPS)

    # ── Step 1: Edge Q-BAT inference ──────────────────────────────────────
    if reuse_edge_from:
        rep.skip("edge", f"Reusing the edge scan already computed in {reuse_edge_from} "
                         f"— it does not depend on the routing ratio.")
        result = _load_edge_result(cfg, reuse_edge_from)
        for name in ('edge_preds.npy', 'edge_preds_raw.npy', 'ground_truth.npy',
                     'energy_matrix.npy'):
            src = os.path.join(reuse_edge_from, name)
            if os.path.exists(src) and os.path.abspath(reuse_edge_from) != os.path.abspath(out_base):
                shutil.copyfile(src, os.path.join(out_base, name))
    else:
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
                st.phase("learning how the scores normally spread")
                _, inv_cov = compute_inv_cov(result.train_energy_matrix)
                st.phase("measuring how certain each window was")
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

        st.phase("collecting the events to send to the cloud")
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
        [cloud_py, runner, "--config", effective_cfg_path],
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
    parser.add_argument('--ratio', type=float, default=None,
                        help="Fraction of events escalated to the cloud (the paper's "
                             "routing ratio); overrides routing_tolerance in the config.")
    parser.add_argument('--distance', choices=['ma', 'eu'], default=None,
                        help='Routing distance; overrides routing_distance in the config.')
    parser.add_argument('--output-dir', default=None,
                        help='Where to write results; overrides output_dir in the config.')
    parser.add_argument('--reuse-edge-from', default=None,
                        help='Reuse the edge scan saved in this directory instead of '
                             're-running it (it does not depend on the routing ratio).')
    args, _ = parser.parse_known_args()
    run_inference(args.config, ratio=args.ratio, distance=args.distance,
                  output_dir=args.output_dir, reuse_edge_from=args.reuse_edge_from)
