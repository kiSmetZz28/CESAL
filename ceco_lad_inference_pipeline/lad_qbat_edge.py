import concurrent.futures
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.mixture import GaussianMixture

from ceco_core.data.loaders import get_loader_segment

try:
    from executorch.runtime import Verification, Runtime
    _EXECUTORCH_AVAILABLE = True
except ImportError:
    _EXECUTORCH_AVAILABLE = False

# C++ executor_runner binary (fallback when Python bindings are unavailable)
_RUNNER_NAME = "executor_runner.exe" if os.name == "nt" else "executor_runner"
_RUNNER_BIN = Path(__file__).parent / "executorch" / "cmake-out" / _RUNNER_NAME
_RUNNER_AVAILABLE = _RUNNER_BIN.is_file() and (os.name == "nt" or os.access(str(_RUNNER_BIN), os.X_OK))


class EdgeResult(NamedTuple):
    """Output of lad_qbat_edge.run()."""
    predictions: np.ndarray
    ground_truth: np.ndarray
    energy_matrix: np.ndarray
    train_energy_matrix: np.ndarray
    thresholds: np.ndarray
    test_windows: np.ndarray


def set_thresh_em(
    energy: np.ndarray,
    n_components: int = 7,
    covariance_type: str = 'tied',
    max_iter: int = 100,
    init_params: str = 'k-means++',
    n_init: int = 10,
) -> np.ndarray:
    """Fit a GMM on energy scores and return cluster labels."""
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        n_init=n_init,
        random_state=42,
    ).fit(energy)
    return gm.predict(energy)


def compute_threshold_from_energy(
    energy: np.ndarray,
    n_components: int = 7,
) -> Tuple[float, float, List[Tuple[int, float]]]:
    """Derive EM-GMM threshold from training energy scores.

    Returns
    -------
    thresh : float
        Energy percentile corresponding to the normal cluster ratio.
    normal_ratio : float
        Percentage of windows classified as normal (largest cluster).
    cluster_pcts : list of (label, pct) sorted by pct descending
    """
    labels = set_thresh_em(energy.reshape(-1, 1), n_components=n_components)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    label_pct = {lbl: (cnt / total) * 100 for lbl, cnt in zip(unique, counts)}
    cluster_pcts = sorted(label_pct.items(), key=lambda x: x[1], reverse=True)
    normal_ratio = cluster_pcts[0][1]
    thresh = float(np.percentile(energy, normal_ratio))
    return thresh, normal_ratio, cluster_pcts


def compute_binary_predictions(energy: np.ndarray, threshold: float) -> np.ndarray:
    """Return 1-D int array: 1 where energy > threshold, 0 otherwise."""
    return (energy > threshold).astype(int)


def _load_et_model(pte_path: str):
    """Load a .pte file via Python bindings."""
    if not _EXECUTORCH_AVAILABLE:
        raise RuntimeError("ExecuTorch Python bindings not available.")
    et_runtime = Runtime.get()
    program = et_runtime.load_program(
        Path(pte_path), verification=Verification.Minimal,
    )
    method = program.load_method("forward")
    return program, method


def _run_et_model(method, windows: np.ndarray) -> np.ndarray:
    """Run inference via Python bindings. Returns 1-D energy array."""
    energies = []
    for i in range(len(windows)):
        x = torch.from_numpy(windows[i : i + 1]).float()
        outputs = method.execute((x,))
        energies.append(outputs[0].numpy())
    return np.concatenate(energies)


# ── C++ executor_runner path ──────────────────────────────────────────────────

def _write_windows_csv(windows: np.ndarray, path: str) -> None:
    """Write [N, win_size, features] as one float per line (flat row-major).

    The C++ executor_runner reads values with a comma-delimiter parser, so
    lines with spaces are read one value at a time (first token per line).
    Writing one value per line gives exactly N * win_size * features values,
    which the runner groups into N windows of shape [1, win_size, features].
    """
    with open(path, 'w') as f:
        for v in windows.reshape(-1):
            f.write(f'{v:.6f}\n')


# CWD for the executor_runner — must be ceco_lad_inference_pipeline/executorch/
_EXECUTORCH_DIR = Path(_RUNNER_BIN).parent.parent


def _run_via_runner(pte_path: str, windows: np.ndarray, model_name: str) -> np.ndarray:
    """Run test inference using the C++ executor_runner binary.

    Mirrors the user command:
        ./cmake-out/executor_runner
            --model_path ../../checkpoints/qbat/os/<name>.pte
            --data_path  ./dataset/<data>.csv
            --model_name <name>

    Writes a temporary data file inside the executorch directory so relative
    paths resolve correctly, runs the binary, reads the score file, and
    returns a 1-D float32 energy array of length N (one score per window).
    """
    pred_dir    = _EXECUTORCH_DIR / "prediction_results"
    dataset_dir = _EXECUTORCH_DIR / "dataset"
    pred_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False,
        dir=str(dataset_dir),
    ) as f:
        tmp_path = f.name
    _write_windows_csv(windows, tmp_path)

    try:
        rel_pte  = os.path.relpath(os.path.abspath(pte_path), str(_EXECUTORCH_DIR))
        rel_data = os.path.join("dataset", os.path.basename(tmp_path))

        runner = Path("cmake-out") / _RUNNER_NAME
        proc = subprocess.run(
            [
                str(runner),
                f"--model_path={rel_pte}",
                f"--data_path={rel_data}",
                f"--model_name={model_name}",
                "--mode=test",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=str(_EXECUTORCH_DIR),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"executor_runner exited with code {proc.returncode}"
            )

        score_file = pred_dir / f"{model_name}_test_score.txt"
        return np.loadtxt(str(score_file), dtype=np.float32)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _infer_model(pte_path: str, windows: np.ndarray, model_name: str) -> np.ndarray:
    """Run test inference — dispatches to Python bindings or C++ executor_runner."""
    if _EXECUTORCH_AVAILABLE:
        _, method = _load_et_model(pte_path)
        return _run_et_model(method, windows)
    if _RUNNER_AVAILABLE:
        logging.info(
            "Python bindings unavailable — using C++ executor_runner for %s", model_name
        )
        return _run_via_runner(pte_path, windows, model_name)
    raise RuntimeError(
        "No ExecuTorch runtime found.\n"
        "  Option A: install Python bindings (executorch.runtime).\n"
        f"  Option B: build the C++ runner at {_RUNNER_BIN}"
    )


def _run_one_edge_model(
    m_cfg: dict,
    stored_thresholds: dict,
    test_windows: np.ndarray,
    thresh_path: str,
) -> Optional[Tuple[np.ndarray, float]]:
    """Run one Q-BAT model against all test windows.

    Returns (energy_col, threshold) on success, or None if the checkpoint
    or threshold is missing. Called concurrently for all edge models.
    """
    name = m_cfg['name']
    ckpt = m_cfg['checkpoint']

    if not os.path.exists(ckpt):
        logging.warning("Checkpoint not found, skipping: %s", ckpt)
        return None
    if name not in stored_thresholds:
        logging.warning("No threshold for '%s' in '%s', skipping.", name, thresh_path)
        return None

    thresh = stored_thresholds[name]
    logging.info("Edge agent: model '%s'  threshold=%.6f", name, thresh)

    test_energy = _infer_model(ckpt, test_windows, name)
    logging.debug("Edge agent: model '%s' done.", name)
    return (test_energy.reshape(-1, 1), thresh)


def run(config: dict) -> EdgeResult:
    """Run all Q-BAT edge models in parallel and return an EdgeResult.

    All .pte models score the same test windows simultaneously. Since
    ExecuTorch runs on CPU and releases the GIL during inference, each model
    runs on its own CPU thread with true parallelism.

    Thresholds must be pre-computed and stored in the file referenced by
    config['threshold_output'] before calling this function.
    """
    dataset    = config['dataset']
    win_size   = config['win_size']
    batch_size = config['batch_size']
    data_path  = config['data_path']

    model_cfgs = config.get('edge_models', [])
    if not model_cfgs:
        raise ValueError("No edge models listed in inference config under 'edge_models'.")

    # Load pre-computed thresholds (must exist before running inference)
    thresh_path = config.get('threshold_output',
                             str(Path('outputs') / dataset.lower() / 'thresholds_edge.yaml'))
    stored_thresholds = _load_thresholds(thresh_path)

    ensemble_param = [config.get('num_epochs', 3), config.get('k', 3),
                      config.get('e_layer_num', 3), batch_size]

    test_loader = get_loader_segment(
        ensemble_param, data_path,
        batch_size=batch_size, win_size=win_size, step=win_size,
        mode='test', dataset=dataset,
    )

    test_windows_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []
    for x, labels in test_loader:
        test_windows_list.append(x.numpy())
        label_list.append(labels.numpy().reshape(-1))

    test_windows = np.concatenate(test_windows_list, axis=0)
    ground_truth = np.concatenate(label_list).astype(int)

    logging.info(
        "Edge agent: %d test windows, running %d Q-BAT models in parallel.",
        len(test_windows), len(model_cfgs),
    )

    # All models score the same read-only array concurrently.
    # ExecuTorch (C inference) releases the GIL, so threads run in true parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_cfgs)) as executor:
        futures = [
            executor.submit(
                _run_one_edge_model,
                m_cfg, stored_thresholds, test_windows, thresh_path,
            )
            for m_cfg in model_cfgs
        ]
        # Preserve submission order so energy_matrix columns match model_cfgs order.
        raw_results = [f.result() for f in futures]

    valid = [r for r in raw_results if r is not None]
    if not valid:
        raise RuntimeError(
            f"No valid Q-BAT models with thresholds found. "
            f"Check checkpoint paths and '{thresh_path}'."
        )

    logging.info(
        "Edge agent: all %d/%d Q-BAT models complete.",
        len(valid), len(model_cfgs),
    )

    test_energy_cols = [r[0] for r in valid]
    thresholds_list  = [r[1] for r in valid]

    n_models = len(thresholds_list)

    # ExecuTorch already outputs one energy score per timestep across all windows.
    # Stack models, compare to threshold, majority vote.
    energy_matrix = np.concatenate(test_energy_cols, axis=1)
    thresholds    = np.array(thresholds_list)

    per_model_preds = (energy_matrix > thresholds).astype(int)
    predictions     = (per_model_preds.sum(axis=1) > n_models / 2).astype(int)

    # Ground truth is already per-timestep from DataLoader — no reshape needed.
    logging.info(
        "Edge agent: %d anomalous / %d timesteps predicted (GT anomalous: %d).",
        int(predictions.sum()), len(predictions), int(ground_truth.sum()),
    )

    return EdgeResult(
        predictions=predictions,
        ground_truth=ground_truth,
        energy_matrix=energy_matrix,
        train_energy_matrix=energy_matrix,  # proxy for Mahalanobis covariance in routing
        thresholds=thresholds,
        test_windows=test_windows,
    )


def _load_thresholds(yaml_path: str) -> dict:
    """Load per-model thresholds from a pre-computed YAML file.

    The file is produced by the training / evaluation pipeline and must exist
    before edge inference is run.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Edge threshold file not found: '{yaml_path}'. "
            "Run the evaluation / calibration step first to generate it."
        )
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    result = {}
    for entry in data.get('models', []):
        name = entry.get('name')
        thr  = entry.get('threshold')
        if name is not None and thr is not None:
            result[name] = float(thr)
    if not result:
        raise RuntimeError(f"No thresholds found in '{yaml_path}'.")
    return result
