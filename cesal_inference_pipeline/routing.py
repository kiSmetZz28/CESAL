import argparse
import logging
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.distance import euclidean, mahalanobis


def load_scores(score_files: List[str]) -> np.ndarray:
    """Load per-model scores and stack them into shape [n_samples, n_models].

    Each file is expected to contain one score per line. All files must have
    the same number of lines.
    """
    arrays = []
    for path in score_files:
        scores = np.loadtxt(path, dtype=float).reshape(-1)
        arrays.append(scores)
        logging.info("Loaded scores from %s", path)

    stacked = np.vstack(arrays).T
    logging.info("Stacked scores shape: %s", stacked.shape)
    return stacked


def compute_inv_cov(train_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute covariance and its inverse from training scores.

    train_scores shape: [n_samples, n_models]. This mirrors the logic in
    ensemble_result.get_threshold_indices_by_distance, but uses only the
    training set to estimate the covariance.
    """
    cov_matrix = np.cov(train_scores, rowvar=False)
    inv_covmat = np.linalg.inv(cov_matrix)
    logging.info("Covariance matrix computed from %d training samples", train_scores.shape[0])
    return cov_matrix, inv_covmat


def load_thresholds_from_yaml(
    yaml_path: str,
    dataset: str,
    energy_names: List[str],
) -> np.ndarray:
    """Load per-model thresholds from a YAML file.

    The YAML structure is expected to match what em_gmm_threshold.py writes:
        {dataset: {energy_name: threshold, ...}, ...}

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file with thresholds.
    dataset : str
        Dataset key under which thresholds are stored.
    energy_names : List[str]
        Ordered list of energy/model names; thresholds are returned in this order.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict) or dataset not in data:
        raise ValueError(f"Dataset '{dataset}' not found in YAML file: {yaml_path}")

    dataset_block = data.get(dataset, {})
    if not isinstance(dataset_block, dict):
        raise ValueError(f"Invalid structure under dataset '{dataset}' in YAML file: {yaml_path}")

    thresholds: List[float] = []
    for name in energy_names:
        if name not in dataset_block:
            raise ValueError(
                f"Energy/model name '{name}' not found under dataset '{dataset}' in YAML file: {yaml_path}",
            )
        thresholds.append(float(dataset_block[name]))

    logging.info(
        "Loaded thresholds from '%s': dataset='%s', energy_names=%s -> %s",
        yaml_path, dataset, energy_names, thresholds,
    )

    return np.asarray(thresholds, dtype=float)


def select_indices_by_distance(
    test_scores: np.ndarray,
    thresholds: np.ndarray,
    inv_covmat: Optional[np.ndarray],
    distance_type: str = "ma",
    tolerance: float = 0.1,
) -> List[int]:
    """Select indices of test samples closest to the threshold point (most uncertain).

    Parameters
    ----------
    test_scores : np.ndarray
        Test scores, shape [n_samples, n_models].
    thresholds : np.ndarray
        Threshold point in the same space, shape [n_models].
    inv_covmat : np.ndarray or None
        Inverse covariance matrix computed from training scores. Required
        when distance_type == 'ma'.
    distance_type : {'eu', 'ma'}
        'eu' for Euclidean, 'ma' for Mahalanobis.
    tolerance : float
        Fraction of samples to select (top by distance).

    Returns
    -------
    selected_indices : List[int]
        Sorted list of selected sample indices.
    """
    threshold_point = np.asarray(thresholds, dtype=float).reshape(-1)
    samples = np.asarray(test_scores, dtype=float)

    if samples.shape[1] != threshold_point.shape[0]:
        raise ValueError(
            f"Dimension mismatch: test_scores has {samples.shape[1]} dims, "
            f"thresholds has {threshold_point.shape[0]} dims",
        )

    if distance_type == "ma" and inv_covmat is None:
        raise ValueError("inv_covmat must be provided when distance_type is 'ma'")

    all_distances: List[Tuple[int, float]] = []
    for i, sample in enumerate(samples):
        if distance_type == "eu":
            dist = euclidean(sample, threshold_point)
        elif distance_type == "ma":
            dist = mahalanobis(sample, threshold_point, inv_covmat)
        else:
            raise ValueError(f"Invalid distance type '{distance_type}'. Use 'eu' or 'ma'.")
        all_distances.append((i, float(dist)))

    all_distances.sort(key=lambda x: x[1], reverse=True)

    num_select = max(int(len(all_distances) * tolerance), 0)
    selected_indices = sorted(idx for idx, _ in all_distances[:num_select])

    logging.info(
        "Total test samples: %d  |  tolerance: %.4f  |  selected: %d",
        len(all_distances), tolerance, len(selected_indices),
    )

    return selected_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute inverse covariance from training scores and select "
            "test indices for routing using Euclidean or Mahalanobis distance."
        ),
    )
    parser.add_argument(
        "--train_score_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to training score files (one per model, one value per line).",
    )
    parser.add_argument(
        "--test_score_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to test score files (one per model, one value per line).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Per-model thresholds forming the reference point in score space. "
            "Length must match the number of models (score files). "
            "If omitted, --thresholds_yaml / --dataset / --energy_names must be provided."
        ),
    )
    parser.add_argument(
        "--thresholds_yaml",
        type=str,
        default=None,
        help=(
            "Optional path to YAML file containing thresholds written by em_gmm_threshold.py. "
            "If provided, thresholds are loaded from YAML instead of --thresholds."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Dataset name used as key in the thresholds YAML file. "
            "Required when --thresholds_yaml is used."
        ),
    )
    parser.add_argument(
        "--energy_names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Ordered energy/model names whose thresholds should be read from the YAML file. "
            "Must match the order of score files when using --thresholds_yaml."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Fraction of test samples to route to cloud (default: 0.1).",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        choices=["eu", "ma"],
        default="ma",
        help="Distance type: 'eu' for Euclidean, 'ma' for Mahalanobis (default).",
    )
    parser.add_argument(
        "--output_indices",
        type=str,
        default=None,
        help="Optional path to save selected test indices (one index per line).",
    )
    parser.add_argument(
        "--output_inv_cov",
        type=str,
        default=None,
        help="Optional path to save the inverse covariance matrix as text.",
    )
    parser.add_argument(
        "--test_data_file",
        type=str,
        default=None,
        help=(
            "Optional path to the original test data file (one sample per line). "
            "If provided together with --output_selected_data, the script will "
            "save only the selected samples to that output file using the routed indices."
        ),
    )
    parser.add_argument(
        "--output_selected_data",
        type=str,
        default=None,
        help="Optional path to save selected test data samples (subset of lines from --test_data_file).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logging.info("Loading training scores...")
    train_scores = load_scores(args.train_score_files)

    logging.info("Loading test scores...")
    test_scores = load_scores(args.test_score_files)

    if train_scores.shape[1] != test_scores.shape[1]:
        raise ValueError(
            f"Dimension mismatch: train has {train_scores.shape[1]} models, "
            f"test has {test_scores.shape[1]} models",
        )

    cov_matrix, inv_covmat = compute_inv_cov(train_scores)

    if args.thresholds is not None:
        if len(args.thresholds) != len(args.train_score_files):
            raise ValueError("Number of thresholds must match number of models (score files).")
        thresholds = np.asarray(args.thresholds, dtype=float).reshape(-1)
    else:
        if args.thresholds_yaml is None or args.dataset is None or args.energy_names is None:
            raise ValueError(
                "When --thresholds is not provided, you must supply "
                "--thresholds_yaml, --dataset, and --energy_names.",
            )
        if len(args.energy_names) != len(args.train_score_files):
            raise ValueError("Number of energy_names must match number of models (score files).")
        thresholds = load_thresholds_from_yaml(
            yaml_path=args.thresholds_yaml,
            dataset=args.dataset,
            energy_names=args.energy_names,
        ).reshape(-1)

    selected_indices = select_indices_by_distance(
        test_scores=test_scores,
        thresholds=thresholds,
        inv_covmat=inv_covmat,
        distance_type=args.distance_type,
        tolerance=args.tolerance,
    )

    if args.output_indices is not None:
        np.savetxt(args.output_indices, np.array(selected_indices, dtype=int), fmt="%d")
        logging.info("Saved selected indices to %s", args.output_indices)

    if args.test_data_file is not None and args.output_selected_data is not None:
        logging.info("Extracting selected samples from %s", args.test_data_file)
        with open(args.test_data_file, "r") as f:
            all_lines = f.readlines()
        if len(all_lines) != test_scores.shape[0]:
            logging.warning(
                "Line count mismatch: test_data_file has %d lines but %d test samples.",
                len(all_lines), test_scores.shape[0],
            )
        selected_lines = [all_lines[i] for i in selected_indices if i < len(all_lines)]
        with open(args.output_selected_data, "w") as f:
            f.writelines(selected_lines)
        logging.info("Saved %d selected samples to %s", len(selected_lines), args.output_selected_data)

    if args.output_inv_cov is not None:
        np.savetxt(args.output_inv_cov, inv_covmat, fmt="%.10f")
        logging.info("Saved inverse covariance matrix to %s", args.output_inv_cov)


if __name__ == "__main__":
    main()
