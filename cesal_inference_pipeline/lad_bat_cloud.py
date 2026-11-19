"""BAT cloud ensemble inference on routed windows."""
import concurrent.futures
import logging
import os
from itertools import product
from typing import List, Optional

import numpy as np
import torch
import yaml

from cesal_core.models.EMAT import EMAT
from cesal_core.utils import steps
from cesal_core.utils.energy import compute_energy_batch
from cesal_core.utils.voting import ensemble_method


def _load_thresholds(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    result = {}
    for entry in cfg.get('models', []):
        name, thr = entry.get('name'), entry.get('threshold')
        if name is not None and thr is not None:
            result[name] = float(thr)
    if not result:
        raise RuntimeError(f"No model thresholds found in {yaml_path!r}.")
    return result


def _run_one_bat(
    combo: tuple,
    dataset: str,
    win_size: int,
    input_c: int,
    output_c: int,
    model_save_path: str,
    thresholds: dict,
    x: torch.Tensor,
    device: torch.device,
    infer_batch_size: int = 64,
) -> Optional[np.ndarray]:
    """Score the routed windows with one BAT checkpoint in mini-batches: binary
    predictions [N_windows * win_size, 1], or None if its checkpoint or threshold is missing."""
    num_epochs, k_val, e_layer_num, batch_size = combo
    fileparam  = f"e{num_epochs}_k{k_val}_l{e_layer_num}_b{batch_size}"
    model_name = f"{dataset}_{fileparam}"
    ckpt = os.path.join(model_save_path, f"{model_name}_checkpoint.pth")

    if not os.path.exists(ckpt):
        logging.warning("Checkpoint not found, skipping: %s", ckpt)
        return None
    if model_name not in thresholds:
        logging.warning("Threshold missing for '%s', skipping.", model_name)
        return None

    model = EMAT(win_size=win_size, enc_in=input_c, c_out=output_c, e_layers=e_layer_num)
    # strict=False: old attn.py stores `distances` as a plain attribute; presence in
    # the checkpoint varies across versions but is always re-computed in __init__.
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    model.to(device).eval()

    n = x.shape[0]
    energy_parts = []
    with torch.no_grad():
        for start in range(0, n, infer_batch_size):
            energy_parts.append(
                compute_energy_batch(model, x[start:start + infer_batch_size], win_size)
            )
    energy = np.concatenate(energy_parts)

    del model

    thresh = thresholds[model_name]
    preds  = (energy > thresh).astype(int)

    # 81 near-identical lines would bury the run; the step reports progress instead.
    logging.debug("BAT model '%s' done.", model_name)
    steps.current().tick("learners")
    return preds.reshape(-1, 1)


def run(windows: np.ndarray, config: dict) -> np.ndarray:
    """Run all BAT checkpoints on windows [N_windows, win_size, features] (routed event
    vectors packed into complete windows) and return 1-D ensemble predictions.
    ``config`` is the inference config's cloud section. Models run sequentially on
    CUDA (one device stream) and in a small thread pool on CPU.
    """
    if len(windows) == 0:
        raise ValueError("No routed windows to process.")

    if config.get('seed') is not None:
        from cesal_core.utils.reproducibility import seed_training
        seed_training(config['seed'])

    try:
        _cuda = torch.cuda.is_available()
    except Exception:
        _cuda = False
    device = torch.device('cuda:0' if _cuda else 'cpu')

    dataset          = config['dataset']
    win_size         = config['win_size']
    input_c          = config['input_c']
    output_c         = config.get('output_c', input_c)
    model_save_path  = config['model_save_path']
    thresholds_yaml  = config['thresholds_yaml']
    voting           = config.get('voting', 'majority')
    infer_batch_size = config.get('infer_batch_size', 64)

    # CUDA: sequential (threads only add VRAM pressure); CPU: a small pool (> 4 gains little).
    if 'max_parallel_models' in config:
        max_workers = config['max_parallel_models']
    elif device.type == 'cuda':
        max_workers = 1
    else:
        max_workers = min(4, os.cpu_count() or 1)

    step = steps.current()

    step.phase("loading calibrated thresholds")
    thresholds   = _load_thresholds(thresholds_yaml)
    search_keys  = ['num_epochs', 'k', 'e_layer_num', 'batch_size']
    combinations = list(product(*[config[k] for k in search_keys]))

    step.detail("windows to verify", len(windows))
    step.detail("BAT checkpoints", len(combinations))
    step.detail("device", f"{device} ({max_workers} worker(s))")
    step.detail("voting", voting)
    step.expect("learners", len(combinations))

    # Transfer input to device once — shared read-only across all models.
    step.phase(f"loading the routed windows onto {device}")
    x = torch.from_numpy(windows).float().to(device)

    step.phase(f"re-scoring the escalated windows with {len(combinations)} BAT learners")

    if max_workers == 1:
        # Sequential — no thread pool overhead, one model in VRAM at a time.
        results = [
            _run_one_bat(
                combo, dataset, win_size, input_c, output_c,
                model_save_path, thresholds, x, device, infer_batch_size,
            )
            for combo in combinations
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_one_bat,
                    combo, dataset, win_size, input_c, output_c,
                    model_save_path, thresholds, x, device, infer_batch_size,
                )
                for combo in combinations
            ]
            results = [f.result() for f in futures]

    all_preds: List[np.ndarray] = [r for r in results if r is not None]

    if not all_preds:
        raise RuntimeError("No valid BAT checkpoints found for cloud inference.")

    if len(all_preds) < len(combinations):
        step.warn(
            f"{len(combinations) - len(all_preds)} of {len(combinations)} BAT checkpoints "
            f"were skipped (missing file or threshold) — voting with {len(all_preds)}."
        )

    step.phase(f"applying {voting} voting across the ensemble")
    verdict = ensemble_method(voting, np.concatenate(all_preds, axis=1))
    n_flagged = int(verdict.sum())
    step.outcome(**{
        "learners voted": f"{len(all_preds)}/{len(combinations)}",
        "events re-checked": len(verdict),
        "confirmed anomalous": f"{n_flagged:,} "
                               f"({n_flagged / max(len(verdict), 1) * 100:.2f}%)",
        "cleared as normal": len(verdict) - n_flagged,
    })
    return verdict
