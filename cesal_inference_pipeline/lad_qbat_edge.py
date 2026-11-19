import concurrent.futures
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import yaml

from cesal_core.data.loaders import get_loader_segment
from cesal_core.utils import steps

try:
    from executorch.runtime import Verification, Runtime
    _EXECUTORCH_AVAILABLE = True
except ImportError:
    _EXECUTORCH_AVAILABLE = False

# executor_runner's per-window progress line, e.g. "[Q-BAT] window 17 — qbat_e6_k1_l3_b96".
_WINDOW_LINE = re.compile(r"^\[Q-BAT\]\s+window\s+\d+")

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
    """Write [N, win_size, features] as one float per line (row-major): the runner's
    comma-delimited parser reads one value per line and regroups them into N windows."""
    with open(path, 'w') as f:
        for v in windows.reshape(-1):
            f.write(f'{v:.6f}\n')


# CWD for the executor_runner — must be cesal_inference_pipeline/executorch/
_EXECUTORCH_DIR = Path(_RUNNER_BIN).parent.parent


def _run_via_runner(pte_path: str, windows: np.ndarray, model_name: str) -> np.ndarray:
    """Score windows with the C++ executor_runner (run from the executorch directory so
    relative paths resolve); returns one float32 energy per timestep."""
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
        # The runner prints one "[Q-BAT] window N" line per window: count them as progress
        # (raw lines still go to the log at DEBUG).
        proc = subprocess.Popen(
            [
                str(runner),
                f"--model_path={rel_pte}",
                f"--data_path={rel_data}",
                f"--model_name={model_name}",
                "--mode=test",
                # The runner defaults to win_size=100; pass the configured size or the
                # flat values are regrouped wrongly.
                f"--win_size={windows.shape[1]}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_EXECUTORCH_DIR),
            text=True,
            errors="replace",
        )
        step = steps.current()
        tail: List[str] = []
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                logging.debug("[%s] %s", model_name, line)
                if _WINDOW_LINE.match(line):
                    step.tick("Q-BAT scan")
                else:
                    # Keep a short tail of non-progress output to explain a failure.
                    tail.append(line)
                    del tail[:-20]
            proc.wait()
        except BaseException:
            # Kill on interrupt, or each orphaned runner keeps a CPU core busy.
            proc.kill()
            proc.wait()
            raise
        if proc.returncode != 0:
            detail = ("\n  " + "\n  ".join(tail)) if tail else ""
            raise RuntimeError(
                f"executor_runner exited with code {proc.returncode} "
                f"for model '{model_name}'.{detail}"
            )

        score_file = pred_dir / f"{model_name}_test_score.txt"
        energy = np.loadtxt(str(score_file), dtype=np.float32)

        # One score per timestep (N x win_size); a mismatch usually means the .pte was
        # exported for a different win_size, which would otherwise misalign silently.
        expected = windows.shape[0] * windows.shape[1]
        if energy.size != expected:
            raise RuntimeError(
                f"'{model_name}' returned {energy.size:,} energy values but "
                f"{expected:,} were expected ({windows.shape[0]:,} windows x "
                f"{windows.shape[1]} timesteps). The checkpoint was probably "
                f"exported for a different win_size than the config specifies."
            )
        return energy
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
        # Reported once per run as a step detail rather than once per model.
        logging.debug("Using C++ executor_runner for %s", model_name)
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
    """Score all test windows with one Q-BAT learner: (energy_col, threshold), or None
    if its checkpoint or threshold is missing."""
    name = m_cfg['name']
    ckpt = m_cfg['checkpoint']

    if not os.path.exists(ckpt):
        logging.warning("Checkpoint not found, skipping: %s", ckpt)
        return None
    if name not in stored_thresholds:
        logging.warning("No threshold for '%s' in '%s', skipping.", name, thresh_path)
        return None

    thresh = stored_thresholds[name]
    logging.debug("Edge agent: model '%s'  threshold=%.6f", name, thresh)

    test_energy = _infer_model(ckpt, test_windows, name)
    logging.debug("Edge agent: model '%s' done.", name)
    return (test_energy.reshape(-1, 1), thresh)


def select_test_windows(windows, labels, indices=None):
    """Optionally select complete windows after full-dataset preprocessing."""
    if indices is None:
        return windows, labels
    selected = np.asarray(indices)
    if (selected.ndim != 1 or selected.size == 0 or selected.dtype.kind not in 'iu'
            or (selected < 0).any() or (selected >= len(windows)).any()
            or (np.diff(selected.astype(np.int64)) <= 0).any()):
        raise ValueError('test_window_indices must be a nonempty, increasing list of valid window indices.')
    steps.current().detail('experiment window selection', f'{len(selected):,} of {len(windows):,} complete windows')
    steps.current().detail('selected input encoding', 'full-dataset event encoding and training scaling retained')
    return windows[selected], labels.reshape(len(windows), windows.shape[1])[selected].reshape(-1)


def run(config: dict) -> EdgeResult:
    """Score the test windows with all Q-BAT learners in parallel threads (ExecuTorch
    releases the GIL) and majority-vote them. Thresholds must already exist in
    config['threshold_output']."""
    dataset    = config['dataset']
    win_size   = config['win_size']
    batch_size = config['batch_size']
    data_path  = config['data_path']

    model_cfgs = config.get('edge_models', [])
    if not model_cfgs:
        raise ValueError("No edge models listed in inference config under 'edge_models'.")

    # Load pre-computed thresholds (must exist before running inference)
    step = steps.current()

    step.phase("loading calibrated thresholds")
    thresh_path = config.get('threshold_output',
                             str(Path('outputs') / dataset.lower() / 'thresholds_edge.yaml'))
    stored_thresholds = _load_thresholds(thresh_path)

    ensemble_param = [config.get('num_epochs', 3), config.get('k', 3),
                      config.get('e_layer_num', 3), batch_size]

    step.phase("parsing log sequences into windows")
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
    if config.get('test_window_indices') is not None:
        test_windows, ground_truth = select_test_windows(
            test_windows, ground_truth, config['test_window_indices'],
        )

    step.detail("test windows", len(test_windows))
    step.detail("window size", win_size)
    step.detail("Q-BAT learners", f"{len(model_cfgs)} quantized EM-AT")
    step.detail("runtime", "ExecuTorch Python bindings" if _EXECUTORCH_AVAILABLE
                else "ExecuTorch C++ executor_runner")
    # One combined percentage across learners (each scores every window).
    if not _EXECUTORCH_AVAILABLE and _RUNNER_AVAILABLE:
        step.expect("Q-BAT scan", len(test_windows) * len(model_cfgs))

    step.phase("scoring every window with Q-BAT")

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
            f"No valid Q-BAT learners with thresholds found. "
            f"Check checkpoint paths and '{thresh_path}'."
        )

    if len(valid) < len(model_cfgs):
        step.warn(
            f"{len(model_cfgs) - len(valid)} of {len(model_cfgs)} Q-BAT learners were "
            f"skipped (missing checkpoint or threshold) — scoring with {len(valid)}."
        )

    step.phase("aggregating learner votes")
    test_energy_cols = [r[0] for r in valid]
    thresholds_list  = [r[1] for r in valid]

    n_models = len(thresholds_list)

    # One energy per timestep per learner: threshold each, then majority vote.
    energy_matrix = np.concatenate(test_energy_cols, axis=1)
    thresholds    = np.array(thresholds_list)

    per_model_preds = (energy_matrix > thresholds).astype(int)
    predictions     = (per_model_preds.sum(axis=1) > n_models / 2).astype(int)

    # Ground truth is already per-timestep from DataLoader — no reshape needed.
    n_flagged = int(predictions.sum())
    step.outcome(**{
        "learners scored": f"{len(valid)}/{len(model_cfgs)}",
        "thresholds": ", ".join(f"{t:.4f}" for t in thresholds_list),
        "events scanned": len(predictions),
        "flagged anomalous": f"{n_flagged:,} "
                             f"({n_flagged / max(len(predictions), 1) * 100:.2f}%)",
        "ground-truth anomalous": int(ground_truth.sum()),
    })

    return EdgeResult(
        predictions=predictions,
        ground_truth=ground_truth,
        energy_matrix=energy_matrix,
        train_energy_matrix=energy_matrix,  # proxy for Mahalanobis covariance in routing
        thresholds=thresholds,
        test_windows=test_windows,
    )


def _load_thresholds(yaml_path: str) -> dict:
    """Per-model thresholds from the YAML written by eval/calibration (must exist)."""
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
