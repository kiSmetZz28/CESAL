#!/usr/bin/env python3
"""Container demo inference pipeline (Docker / HF Spaces).

Replaces the full two-environment pipeline (ExecuTorch edge + conda cloud)
with a Python-only execution that uses the BAT .pth checkpoints for BOTH
the edge scan and the cloud re-check:

  Stage 1  Edge scan  — 3 fast BAT checkpoints on all test windows (parallel)
  Stage 2  Routing    — Mahalanobis distance selects uncertain windows
  Stage 3  Cloud      — full BAT ensemble on routed windows (parallel)
  Stage 4  Evaluation — hybrid metrics

Log messages deliberately mirror lad_qbat_edge.py / lad_bat_cloud.py format so
the dashboard's live-progress parser (parseLiveLine in index.html) works
without any changes to the frontend.

Called by dashboard/app.py when ceco_lad_inference_pipeline/run.py is absent:
    python dashboard/demo_runner.py --config configs/inference/<ds>.yaml
"""
import argparse
import concurrent.futures
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ceco_core.data.loaders import get_loader_segment
from ceco_core.models.EMAT import EMAT
from ceco_core.utils.config import load_config, setup_logging
from ceco_core.utils.energy import compute_energy_batch
from ceco_core.utils.io import mkdir
from ceco_core.utils.metrics import evaluate
from ceco_lad_inference_pipeline.lad_bat_cloud import _load_thresholds, run as cloud_run
from ceco_lad_inference_pipeline.routing import compute_inv_cov, select_indices_by_distance


# ── Helpers ───────────────────────────────────────────────────────────────────

def _point_adjust(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = gt.astype(int)
    pred = pred.astype(int).copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def _run_edge_bat(
    combo: tuple,
    dataset: str,
    win_size: int,
    input_c: int,
    model_dir: Path,
    thresholds: dict,
    x: torch.Tensor,
    device: torch.device,
) -> "tuple | None":
    """Run one BAT checkpoint as an edge-scan model.

    Returns (energy_col [N,1], threshold) or None if checkpoint/threshold missing.
    Log format mirrors lad_qbat_edge.py so the frontend progress parser triggers.
    """
    ep, k, layers, bsz = combo
    name = f"{dataset}_e{ep}_k{k}_l{layers}_b{bsz}"
    ckpt = model_dir / f"{name}_checkpoint.pth"

    if not ckpt.exists():
        logging.warning("Edge checkpoint not found, skipping: %s", ckpt)
        return None
    if name not in thresholds:
        logging.warning("Edge threshold missing for '%s', skipping.", name)
        return None

    thresh = thresholds[name]
    model = EMAT(win_size=win_size, enc_in=input_c, c_out=input_c, e_layers=layers)
    model.load_state_dict(
        torch.load(str(ckpt), map_location=device, weights_only=True),
        strict=False,
    )
    model.to(device).eval()

    n = x.shape[0]
    parts = []
    with torch.no_grad():
        for start in range(0, n, 64):
            parts.append(compute_energy_batch(model, x[start:start + 64], win_size))
    energy = np.concatenate(parts)
    del model

    # Mirror lad_qbat_edge.py log format → frontend threshold display triggers
    logging.info("Edge agent: model '%s'  threshold=%.6f", name, thresh)
    return (energy.reshape(-1, 1), thresh)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CECO-LAD container demo inference.")
    parser.add_argument("--config", required=True, help="Inference YAML config path.")
    parser.add_argument("--skip-edge", action="store_true",
                        help="Skip Stage 1 (edge scan) and reuse existing energy_matrix.npy / edge_preds.npy.")
    args = parser.parse_args()

    os.chdir(str(ROOT))
    setup_logging("demo")
    cfg = load_config(args.config)

    dataset   = cfg["dataset"]
    win_size  = cfg.get("win_size", 100)
    input_c   = cfg.get("input_c", 10)
    batch_sz  = cfg.get("batch_size", 64)
    data_path = cfg.get("data_path", f"data/{dataset}")
    out_base  = cfg.get("output_dir", f"outputs/{dataset.lower()}")
    mkdir(out_base)

    cloud_cfg = cfg.get("cloud")
    if not cloud_cfg:
        logging.error("No 'cloud' section in config — cannot run demo inference for '%s'.", dataset)
        sys.exit(1)

    thresh_cloud_path = ROOT / cloud_cfg.get("thresholds_yaml", "")
    if not thresh_cloud_path.exists():
        logging.error(
            "Cloud threshold file not found: %s\nRun 'eval' first to generate thresholds.",
            thresh_cloud_path,
        )
        sys.exit(1)

    model_dir        = ROOT / cloud_cfg.get("model_save_path", "")
    cloud_thresholds = _load_thresholds(str(thresh_cloud_path))

    try:
        _cuda = torch.cuda.is_available()
    except Exception:
        _cuda = False
    device = torch.device("cuda:0" if _cuda else "cpu")

    # Select 3 lightweight BAT checkpoints as edge-proxy models.
    min_ep = min(cloud_cfg["num_epochs"])
    min_k  = min(cloud_cfg["k"])
    min_l  = min(cloud_cfg["e_layer_num"])
    edge_combos = [(min_ep, min_k, min_l, bsz) for bsz in sorted(cloud_cfg["batch_size"])[:3]]

    if args.skip_edge:
        # ── Stage 1: SKIPPED — load existing edge outputs ─────────────────────
        logging.info("=== Stage 1: Skipped (--skip-edge) — loading existing edge outputs ===")
        for fname in ("energy_matrix.npy", "edge_preds.npy", "ground_truth.npy"):
            if not os.path.exists(os.path.join(out_base, fname)):
                logging.error("--skip-edge requires %s but it was not found in %s", fname, out_base)
                sys.exit(1)
        energy_matrix = np.load(os.path.join(out_base, "energy_matrix.npy"))
        ground_truth  = np.load(os.path.join(out_base, "ground_truth.npy"))
        # Recover per-model thresholds from the edge threshold YAML
        import yaml as _yaml
        thresh_yaml = os.path.join(out_base, "thresholds_edge.yaml")
        if os.path.exists(thresh_yaml):
            with open(thresh_yaml) as _f:
                _tlist = _yaml.safe_load(_f).get("models", [])
            thresh_arr = np.array([t["threshold"] for t in _tlist])
        else:
            thresh_arr = np.array([
                cloud_thresholds.get(f"{dataset}_e{c[0]}_k{c[1]}_l{c[2]}_b{c[3]}", 0.0)
                for c in edge_combos
            ])
        # Still need scaled test windows to build routed windows for cloud.
        test_loader = get_loader_segment(
            [3, 1, 3, batch_sz], data_path,
            batch_size=batch_sz, win_size=win_size, step=win_size,
            mode="test", dataset=dataset,
        )
        test_windows = np.concatenate([x.numpy() for x, _ in test_loader], axis=0)
        logging.info("Loaded edge outputs: %d lines, %d edge models.", len(ground_truth), energy_matrix.shape[1])
    else:
        # ── Stage 1: Edge scan ────────────────────────────────────────────────
        logging.info("=== Stage 1: Edge Q-BAT Inference ===")

        test_loader = get_loader_segment(
            [3, 1, 3, batch_sz], data_path,
            batch_size=batch_sz, win_size=win_size, step=win_size,
            mode="test", dataset=dataset,
        )
        windows_list, labels_list = [], []
        for x_batch, lbl_batch in test_loader:
            windows_list.append(x_batch.numpy())
            labels_list.append(lbl_batch.numpy().reshape(-1))
        test_windows = np.concatenate(windows_list, axis=0)
        ground_truth = np.concatenate(labels_list).astype(int)

        demo_max = cfg.get("demo_max_windows", int(os.getenv("DEMO_MAX_WINDOWS", "0")))
        if demo_max > 0 and len(test_windows) > demo_max:
            idx = np.linspace(0, len(test_windows) - 1, demo_max, dtype=int)
            test_windows = test_windows[idx]
            ground_truth = ground_truth[idx]
            logging.info("Demo mode: using %d / %d test windows (evenly sampled).",
                         demo_max, len(idx) + (len(np.concatenate(labels_list)) - demo_max))

        x_tensor = torch.from_numpy(test_windows).float().to(device)
        n_edge   = len(edge_combos)
        logging.info("Edge agent: %d test windows, running %d Q-BAT models in parallel.",
                     len(test_windows), n_edge)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_edge) as executor:
            futures = [executor.submit(_run_edge_bat, combo, dataset, win_size, input_c,
                                       model_dir, cloud_thresholds, x_tensor, device)
                       for combo in edge_combos]
            raw_results = [f.result() for f in futures]

        valid = [r for r in raw_results if r is not None]
        if not valid:
            logging.error("No edge BAT models ran successfully. Check checkpoints in %s.", model_dir)
            sys.exit(1)
        logging.info("Edge agent: all %d/%d Q-BAT models complete.", len(valid), n_edge)

        energy_cols   = [r[0] for r in valid]
        thresh_arr    = np.array([r[1] for r in valid])
        energy_matrix = np.concatenate(energy_cols, axis=1)
        per_model     = (energy_matrix > thresh_arr).astype(int)
        predictions   = (per_model.sum(axis=1) > len(valid) / 2).astype(int)

        edge_adj = _point_adjust(ground_truth, predictions)
        np.save(os.path.join(out_base, "edge_preds.npy"),     edge_adj)
        np.save(os.path.join(out_base, "edge_preds_raw.npy"), predictions)
        np.save(os.path.join(out_base, "ground_truth.npy"),   ground_truth)
        np.save(os.path.join(out_base, "energy_matrix.npy"),  energy_matrix)
        evaluate(ground_truth, edge_adj, prefix="Edge")

    # Compute training energy for routing covariance (uses normal distribution only).
    # Try each edge combo as ensemble_param until one is accepted by the data loader.
    train_energy_matrix = None
    for combo in edge_combos:
        ep, k, layers, bsz = combo
        try:
            train_loader = get_loader_segment(
                [ep, k, layers, bsz], data_path,
                batch_size=batch_sz, win_size=win_size, step=win_size,
                mode="train", dataset=dataset,
            )
            train_windows_list = []
            for x_batch, _ in train_loader:
                train_windows_list.append(x_batch.numpy())
            train_windows  = np.concatenate(train_windows_list, axis=0)
            x_train_tensor = torch.from_numpy(train_windows).float().to(device)
            train_energy_cols = []
            for c in edge_combos:
                res = _run_edge_bat(c, dataset, win_size, input_c,
                                    model_dir, cloud_thresholds, x_train_tensor, device)
                if res is not None:
                    train_energy_cols.append(res[0])
            if train_energy_cols:
                train_energy_matrix = np.concatenate(train_energy_cols, axis=1)
            break
        except (ValueError, KeyError):
            continue
    if train_energy_matrix is None:
        logging.warning("Could not load training data for covariance — falling back to test energy.")
        train_energy_matrix = energy_matrix

    # ── Stage 2: Mahalanobis Routing ─────────────────────────────────────────
    logging.info("=== Stage 2: Mahalanobis Routing ===")

    tolerance     = cfg.get("routing_tolerance", 0.1)
    distance_type = cfg.get("routing_distance", "ma")

    routed_indices: list = []
    if energy_matrix.shape[1] >= 2:
        try:
            _, inv_cov = compute_inv_cov(train_energy_matrix)
            routed_indices = select_indices_by_distance(
                test_scores=energy_matrix,
                thresholds=thresh_arr,
                inv_covmat=inv_cov,
                distance_type=distance_type,
                tolerance=tolerance,
            )
        except np.linalg.LinAlgError:
            logging.warning("Singular covariance matrix — routing all predicted anomalies.")
            routed_indices = list(np.where(predictions == 1)[0])
    else:
        margin  = energy_matrix[:, 0] - thresh_arr[0]
        n_route = max(1, int(len(margin) * tolerance))
        routed_indices = sorted(np.argsort(margin)[-n_route:].tolist())

    logging.info(
        "Routing: %d / %d lines selected to cloud (tolerance=%.0f%%).",
        len(routed_indices), len(ground_truth), tolerance * 100,
    )
    routed_idx_arr = np.array(routed_indices, dtype=int)
    np.save(os.path.join(out_base, "routed_indices.npy"), routed_idx_arr)

    if not routed_indices:
        logging.info("No lines routed to cloud — edge results are final.")
        logging.info("=== Inference complete. Outputs in '%s' ===", out_base)
        return

    # Build windows from routed lines for cloud inference.
    test_lines   = test_windows.reshape(-1, input_c)
    routed_lines = test_lines[routed_idx_arr]
    n_full       = (len(routed_lines) // win_size) * win_size
    if n_full == 0:
        logging.info("Not enough routed lines to fill a window — edge results are final.")
        logging.info("=== Inference complete. Outputs in '%s' ===", out_base)
        return
    routed_windows = routed_lines[:n_full].reshape(-1, win_size, input_c)
    use_indices    = routed_idx_arr[:n_full]
    np.save(os.path.join(out_base, "routed_lines.npy"), routed_lines)

    # ── Stage 3: Cloud BAT Ensemble ───────────────────────────────────────────
    logging.info("=== Stage 3: Cloud BAT Ensemble ===")

    cloud_cfg.setdefault("dataset", dataset)
    cloud_cfg.setdefault("win_size", win_size)
    cloud_cfg.setdefault("input_c", input_c)
    cloud_cfg.setdefault("max_parallel_models", os.cpu_count() or 8)

    cloud_preds = cloud_run(routed_windows, cloud_cfg)
    np.save(os.path.join(out_base, "cloud_preds.npy"), cloud_preds)

    # ── Stage 4: Hybrid Evaluation ────────────────────────────────────────────
    logging.info("=== Stage 4: Hybrid Evaluation ===")
    logging.info(
        "Cloud preds: %d / %d routed lines anomalous.",
        int(cloud_preds.sum()), len(cloud_preds),
    )

    if 'predictions' not in dir():
        per_model  = (energy_matrix > thresh_arr).astype(int)
        predictions = (per_model.sum(axis=1) > energy_matrix.shape[1] / 2).astype(int)
    hybrid_raw = predictions.copy()
    hybrid_raw[use_indices] = cloud_preds

    hybrid_adj = _point_adjust(ground_truth, hybrid_raw)
    np.save(os.path.join(out_base, "hybrid_preds.npy"), hybrid_adj)
    evaluate(ground_truth, hybrid_adj, prefix="Hybrid")

    logging.info("=== Inference complete. Outputs in '%s' ===", out_base)


if __name__ == "__main__":
    main()
