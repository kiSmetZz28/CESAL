import argparse
import logging

import numpy as np

from ceco_core.utils.metrics import evaluate


def _load_arr(path: str, dtype=int) -> np.ndarray:
    """Load a 1-D array from either .npy (binary) or plain-text format."""
    if path.endswith('.npy'):
        return np.load(path).reshape(-1).astype(dtype)
    return np.loadtxt(path, dtype=dtype).reshape(-1)


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


def compute_edge_ensemble(
    edge_pred_files: list,
    label_file: str,
) -> tuple:
    """Majority-vote across Q-BAT model predictions and evaluate.

    Returns
    -------
    edge_raw : np.ndarray
        Ensemble binary predictions (majority vote).
    gt : np.ndarray
        Ground-truth labels.
    """
    preds_list = [_load_arr(p) for p in edge_pred_files]
    gt = _load_arr(label_file)

    lengths = [len(p) for p in preds_list]
    if not all(n == len(gt) for n in lengths):
        raise ValueError(
            f"Length mismatch: prediction lengths {lengths}, labels length {len(gt)}"
        )

    stacked = np.vstack(preds_list)
    votes = stacked.sum(axis=0)
    edge_raw = (votes >= (stacked.shape[0] // 2 + 1)).astype(int)

    logging.info("Edge ensemble predictions: %d samples, %d models", edge_raw.shape[0], stacked.shape[0])
    evaluate(gt, _point_adjust(gt, edge_raw), prefix="Edge")

    return edge_raw, gt


def compute_hybrid(
    edge_raw: np.ndarray,
    cloud_pred_file: str,
    indices_file: str,
    gt: np.ndarray,
) -> np.ndarray:
    """Merge edge and cloud predictions using routing indices, then evaluate.

    Parameters
    ----------
    edge_raw : np.ndarray
        Raw edge ensemble predictions.
    cloud_pred_file : str
        Path to cloud predictions for selected indices.
    indices_file : str
        Path to indices file produced by routing.py.
    gt : np.ndarray
        Ground-truth labels (full length).
    """
    cloud_pred = _load_arr(cloud_pred_file)
    indices    = _load_arr(indices_file)

    n_cloud = cloud_pred.shape[0]
    n_idx = indices.shape[0]

    if n_cloud > n_idx:
        raise ValueError(
            f"cloud_pred length {n_cloud} > indices length {n_idx}. "
            "Check selected data / windowing pipeline."
        )
    if n_cloud < n_idx:
        indices = indices[:n_cloud]

    if np.any(indices < 0) or np.any(indices >= edge_raw.shape[0]):
        raise ValueError("Some indices are out of range of edge prediction length.")

    hybrid_raw = edge_raw.copy()
    hybrid_raw[indices] = cloud_pred

    evaluate(gt, _point_adjust(gt, hybrid_raw), prefix="Hybrid")

    return hybrid_raw


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Compute edge ensemble and hybrid (edge+cloud) evaluation results."
    )
    parser.add_argument(
        "--edge_preds",
        type=str,
        nargs="+",
        required=True,
        help="Paths to edge prediction files to ensemble (e.g., three Q-BAT models).",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Path to ground-truth label file.",
    )
    parser.add_argument(
        "--cloud_pred",
        type=str,
        default=None,
        help="Optional: path to cloud predictions for selected indices.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Optional: path to indices file (for hybrid evaluation).",
    )

    args = parser.parse_args()

    edge_raw, gt = compute_edge_ensemble(args.edge_preds, args.label)

    if args.cloud_pred is not None and args.indices is not None:
        compute_hybrid(edge_raw, args.cloud_pred, args.indices, gt)


if __name__ == "__main__":
    main()
