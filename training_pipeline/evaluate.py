import argparse
import logging
from itertools import product

import numpy as np
from torch.backends import cudnn

from ceco_core.utils.config import load_config, setup_logging
from ceco_core.utils.io import mkdir
from ceco_core.utils.metrics import evaluate
from ceco_core.utils.voting import ensemble_method
from training_pipeline.solver import Solver


def _single_model_pred(config: argparse.Namespace) -> tuple:
    cudnn.benchmark = True
    mkdir(config.model_save_path)
    solver = Solver(vars(config))
    pred, gt = solver.singlemodelpred()
    f_score = evaluate(gt, pred)
    return pred, gt, f_score


def run_bat_ensemble(
    config_path: str,
    voting_method: str = 'majority',
    log_intermediate: bool = True,
) -> tuple:
    """Run the full BAT ensemble and return (predictions, ground_truth).

    Parameters
    ----------
    config_path : str
        Path to training YAML config.
    voting_method : {'majority', 'at least one', 'consensus', 'all'}
        Voting strategy. 'all' returns a dict mapping method -> predictions.
    log_intermediate : bool
        Log per-model and incremental ensemble metrics if True.
    """
    yaml_config = load_config(config_path)

    search_keys = ['num_epochs', 'k', 'e_layer_num', 'batch_size']
    search_space = [yaml_config[key] for key in search_keys]
    combinations = list(product(*search_space))
    base_config = {k: v for k, v in yaml_config.items() if k not in search_keys}

    ground_truth = None
    model_records = []  # (f_score, values, pred)

    for i, values in enumerate(combinations):
        config = argparse.Namespace(**{**base_config, **dict(zip(search_keys, values))})

        if log_intermediate:
            logging.info("\n=== Testing model %d/%d ===", i + 1, len(combinations))

        pred, gt, f_score = _single_model_pred(config)

        if ground_truth is None:
            ground_truth = gt
        elif not np.array_equal(ground_truth, gt):
            raise ValueError("Ground truth inconsistent across models.")

        model_records.append((f_score, values, pred.reshape(-1, 1)))

    # Sort all models by individual F1, low to high
    model_records.sort(key=lambda x: x[0])

    logging.info("\n=== Per-model F1 (sorted low → high) ===")
    for rank, (f_score, values, _) in enumerate(model_records, start=1):
        params = dict(zip(search_keys, values))
        logging.info(
            "  Rank %3d  e=%s k=%s l=%s b=%s  F1=%.4f",
            rank, params['num_epochs'], params['k'], params['e_layer_num'], params['batch_size'],
            f_score,
        )

    # Incremental ensemble in sorted order (1 → 81)
    methods = ['majority', 'at least one', 'consensus'] if voting_method == 'all' else [voting_method]
    logging.info("\n=== Incremental ensemble (sorted low → high F1) ===")
    for step in range(1, len(model_records) + 1):
        partial = np.concatenate([r[2] for r in model_records[:step]], axis=1)
        for method in methods:
            logging.info("-- %s voting (step %d/%d) --", method, step, len(model_records))
            evaluate(ground_truth, ensemble_method(method, partial))

    all_preds = np.concatenate([r[2] for r in model_records], axis=1)
    logging.info("\n=== Final ensemble (%d models) ===", len(model_records))

    if voting_method == 'all':
        results = {}
        for method in ['majority', 'at least one', 'consensus']:
            final = ensemble_method(method, all_preds)
            logging.info("-- %s voting --", method)
            evaluate(ground_truth, final)
            results[method] = final
        return results, ground_truth

    final = ensemble_method(voting_method, all_preds)
    evaluate(ground_truth, final)
    return final, ground_truth


if __name__ == '__main__':
    setup_logging('evaluate_bat')

    parser = argparse.ArgumentParser(description="BAT ensemble evaluation.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training/bgl.yaml',
        help='Path to the training YAML config.',
    )
    parser.add_argument(
        '--voting',
        type=str,
        default='all',
        choices=['all', 'majority', 'at least one', 'consensus'],
        help='Voting method for the ensemble.',
    )
    args, _ = parser.parse_known_args()

    run_bat_ensemble(args.config, voting_method=args.voting, log_intermediate=True)
