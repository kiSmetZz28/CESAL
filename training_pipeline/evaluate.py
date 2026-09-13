import argparse
import logging
from itertools import product

import numpy as np
from torch.backends import cudnn

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_core.utils.metrics import evaluate
from cesal_core.utils.steps import StepReporter
from cesal_core.utils.voting import ensemble_method
from training_pipeline.solver import Solver


_ABOUT = """
Measuring how well the trained models detect anomalies, alone and together.
Each model is scored on the test logs by itself, then the models are combined
one at a time — weakest first — so you can see how much the ensemble gains over
any single model, and where adding more stops helping.
"""


def _single_model_pred(config: argparse.Namespace) -> tuple:
    cudnn.benchmark = True
    mkdir(config.model_save_path)
    solver = Solver(vars(config))
    pred, gt = solver.singlemodelpred()
    # Per-model scores are intermediate: they are ranked below and summarised,
    # so they neither print nor enter the run summary.
    scores = evaluate(gt, pred, register=False, level=logging.DEBUG)
    return pred, gt, scores.f_score


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

    dataset = base_config.get('dataset', '')
    rep = StepReporter("eval", dataset=dataset, steps=steps.EVAL_STEPS, about=_ABOUT)

    ground_truth = None
    model_records = []  # (f_score, values, pred)

    # ── Step 1: score every model on its own ──────────────────────────────
    with rep.step("score") as st:
        st.detail("config", config_path)
        st.detail("models to score", len(combinations))
        st.detail("voting to test", voting_method)
        st.expect("models", len(combinations))

        for values in combinations:
            config = argparse.Namespace(**{**base_config, **dict(zip(search_keys, values))})
            name = (f"e{config.num_epochs}_k{config.k}"
                    f"_l{config.e_layer_num}_b{config.batch_size}")
            st.progress_note(name)

            pred, gt, f_score = _single_model_pred(config)

            if ground_truth is None:
                ground_truth = gt
            elif not np.array_equal(ground_truth, gt):
                raise ValueError("Ground truth inconsistent across models.")

            model_records.append((f_score, values, pred.reshape(-1, 1)))
            st.tick("models")

        # Rank low → high; the ensemble below is built in this order.
        model_records.sort(key=lambda x: x[0])
        for rank, (f_score, values, _) in enumerate(model_records, start=1):
            params = dict(zip(search_keys, values))
            logging.debug("  Rank %3d  e=%s k=%s l=%s b=%s  F1=%.4f", rank,
                          params['num_epochs'], params['k'], params['e_layer_num'],
                          params['batch_size'], f_score)

        f1s = [r[0] for r in model_records]   # already percentages
        best_params = dict(zip(search_keys, model_records[-1][1]))
        n_thresh = st._counters.get("thresholds written", {}).get("done", 0)
        st.outcome(**{
            "models scored": len(model_records),
            "weakest model F1": f"{f1s[0]:.2f}",
            "median model F1": f"{f1s[len(f1s) // 2]:.2f}",
            "best model F1": (f"{f1s[-1]:.2f} (e{best_params['num_epochs']}"
                              f"_k{best_params['k']}_l{best_params['e_layer_num']}"
                              f"_b{best_params['batch_size']})"),
            "thresholds rewritten": n_thresh,
        })
        if n_thresh:
            # Scoring recalibrates every model's threshold and overwrites the
            # bundled file in place, which changes what `run.py infer` will do.
            st.warn(f"{n_thresh} calibration thresholds were recomputed and written "
                    f"over the bundled ones — this changes subsequent inference runs.")

    # ── Step 2: combine them, adding one model at a time ──────────────────
    methods = ['majority', 'at least one', 'consensus'] if voting_method == 'all' else [voting_method]
    n = len(model_records)
    # A handful of sizes tells the story; scoring all 81 × 3 and printing each
    # would be 486 lines saying very little.
    milestones = sorted({1, 5, 10, 20, 40, n} & set(range(1, n + 1)))
    curve = {m: {} for m in methods}
    results = {}

    with rep.step("ensemble") as st:
        st.detail("ensemble order", "weakest model first")
        st.detail("voting methods", ", ".join(methods))
        st.expect("ensembles", n * len(methods))

        for step in range(1, n + 1):
            partial = np.concatenate([r[2] for r in model_records[:step]], axis=1)
            for method in methods:
                sc = evaluate(ground_truth, ensemble_method(method, partial),
                              register=False, level=logging.DEBUG)
                if step in milestones:
                    curve[method][step] = sc.f_score
                st.tick("ensembles")

        st.phase("how the ensemble grows")
        for method in methods:
            pts = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(curve[method].items()))
            logging.info("   %-14s F1 by ensemble size — %s", method, pts)

        st.phase(f"final vote across all {n} models")
        all_preds = np.concatenate([r[2] for r in model_records], axis=1)
        for method in methods:
            final = ensemble_method(method, all_preds)
            evaluate(ground_truth, final, prefix=method)
            results[method] = final

        best_method = max(methods, key=lambda m: curve[m].get(n, 0.0))
        st.outcome(**{
            "ensembles scored": n * len(methods),
            "best voting method": f"{best_method} (F1 {curve[best_method].get(n, 0.0):.2f})",
            "gain over best single model": f"{curve[best_method].get(n, 0.0) - f1s[-1]:+.2f} F1",
        })

    rep.finish(outputs=base_config.get('model_save_path', ''))

    if voting_method == 'all':
        return results, ground_truth
    return results[voting_method], ground_truth


if __name__ == '__main__':
    setup_logging('evaluate_bat')

    parser = argparse.ArgumentParser(description="BAT ensemble evaluation.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training/os.yaml',
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
