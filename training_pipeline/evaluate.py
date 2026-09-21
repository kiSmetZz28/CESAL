import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
from torch.backends import cudnn

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_core.utils.metrics import evaluate
from cesal_core.utils.steps import StepReporter
from cesal_core.utils.voting import ensemble_method
from training_pipeline.solver import Solver
from cesal_core.utils.reproducibility import environment, model_seed, seed_training, sha256, write_json


_ABOUT = """
Measure the detection performance of the trained BAT ensemble, per learner and
as a whole.

Every learner is scored on the test logs to calibrate its threshold, then all
of them vote. Only the ensemble result is reported: it is what the paper
claims, and per-learner figures are inflated by point adjustment.
"""


def _single_model_pred(config: argparse.Namespace) -> tuple:
    if getattr(config, 'seed', None) is not None:
        seed_training(model_seed(config.seed, config.dataset,
            (config.num_epochs, config.k, config.e_layer_num, config.batch_size)))
    else:
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
    model_f1: dict = None,
) -> tuple:
    """Run the full BAT ensemble and return (predictions, ground_truth).

    Parameters
    ----------
    config_path : str
        Path to training YAML config.
    voting_method : {'majority', 'at least one', 'consensus', 'all'}
        Voting strategy. Majority is the rule the paper reports and the
        default; the others remain available for comparison. 'all' evaluates
        every rule and returns a dict mapping method -> predictions.
    log_intermediate : bool
        Log per-model and incremental ensemble metrics if True.
    model_f1 : dict, optional
        Collect each full-precision learner's measured F1 by checkpoint model name.
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
        st.detail("learners to score", len(combinations))
        st.detail("voting", voting_method)
        st.expect("learners", len(combinations))

        for values in combinations:
            config = argparse.Namespace(**{**base_config, **dict(zip(search_keys, values))})
            name = (f"e{config.num_epochs}_k{config.k}"
                    f"_l{config.e_layer_num}_b{config.batch_size}")
            st.progress_note(name)

            pred, gt, f_score = _single_model_pred(config)
            if model_f1 is not None:
                model_f1[f'{dataset}_{name}'] = float(f_score)

            if ground_truth is None:
                ground_truth = gt
            elif not np.array_equal(ground_truth, gt):
                raise ValueError("Ground truth inconsistent across models.")

            model_records.append((f_score, values, pred.reshape(-1, 1)))
            st.tick("learners")

        # Per-learner F1 is recorded in bat_evaluation.json (the Q-BAT candidate
        # filter needs it) but not reported: after point adjustment a learner
        # that catches a single event in an anomaly segment scores like a
        # near-perfect detector, so only the ensemble figure is meaningful.
        n_thresh = st._counters.get("thresholds written", {}).get("done", 0)
        st.outcome(**{
            "learners scored": len(model_records),
            "thresholds rewritten": n_thresh,
        })
        if n_thresh:
            # Scoring recalibrates every model's threshold and overwrites the
            # bundled file in place, which changes what `run.py infer` will do.
            st.warn(f"{n_thresh} calibration thresholds were recomputed in "
                    f"{base_config.get('threshold_output', 'the default threshold file')}; "
                    "use them with the evaluated checkpoints.")

    # ── Step 2: combine them, adding one model at a time ──────────────────
    methods = ['majority', 'at least one', 'consensus'] if voting_method == 'all' else [voting_method]
    n = len(model_records)
    scores = {}
    results = {}

    with rep.step("ensemble") as st:
        st.detail("voting", ", ".join(methods))
        st.expect("ensembles", len(methods))

        st.phase(f"final vote across all {n} learners")
        all_preds = np.concatenate([r[2] for r in model_records], axis=1)
        for method in methods:
            final = ensemble_method(method, all_preds)
            scores[method] = evaluate(ground_truth, final, prefix=method)
            results[method] = final
            st.tick("ensembles")

        outcome = {"learners in the ensemble": n}
        if len(methods) > 1:
            best_method = max(methods, key=lambda m: scores[m].f_score)
            outcome["best voting method"] = f"{best_method} (F1 {scores[best_method].f_score:.2f})"
        else:
            outcome["BAT F1"] = f"{scores[methods[0]].f_score:.2f}"
        st.outcome(**outcome)

    rep.finish(outputs=base_config.get('model_save_path', ''))

    if voting_method == 'all':
        return results, ground_truth
    return results[voting_method], ground_truth


def report_path(config: dict) -> Path:
    return Path(config['threshold_output']).parent / 'bat_evaluation.json'


def signature(config_path: str) -> dict:
    """Hash everything the measured scores depend on, for later provenance checks."""
    config = load_config(config_path)
    files = [Path(config_path), Path(config['threshold_output'])]
    for e, k, depth, batch in product(*(config[key] for key in
                                        ('num_epochs', 'k', 'e_layer_num', 'batch_size'))):
        files.append(Path(config['model_save_path'])
                     / f'{config["dataset"]}_e{e}_k{k}_l{depth}_b{batch}_checkpoint.pth')
    files.extend(sorted(Path(config['data_path']).glob('*.txt')))
    root = Path(__file__).resolve().parent.parent
    files.extend([root / 'training_pipeline/solver.py', root / 'training_pipeline/evaluate.py',
                  *sorted((root / 'cesal_core').rglob('*.py'))])
    return {str(path.resolve()): sha256(path) for path in files}


def main() -> int:
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
        default='majority',
        choices=['majority', 'all', 'at least one', 'consensus'],
        help="Voting rule for the ensemble (default: majority, as in the paper; "
             "'all' also reports at-least-one and consensus).",
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Also record measured scores, per-learner F1 and artifact hashes in '
             'bat_evaluation.json beside the thresholds.',
    )
    args, _ = parser.parse_known_args()

    model_f1 = {} if args.report else None
    predictions, labels = run_bat_ensemble(args.config, voting_method=args.voting,
                                           log_intermediate=True, model_f1=model_f1)
    if not args.report:
        return 0
    if args.voting == 'all':
        predictions = predictions['majority']
    measured = evaluate(labels, predictions, register=False, level=logging.DEBUG)
    config = load_config(args.config)
    write_json(report_path(config), dict(
        seed=config.get('seed'), dataset=config['dataset'], voting=args.voting,
        scores=dict(accuracy=measured.accuracy, precision=measured.precision,
                    recall=measured.recall, f1=measured.f_score),
        model_f1=model_f1, evaluation_data='Full bundled test set',
        environment=environment(), artifact_sha256=signature(args.config)))
    print(f'BAT evaluation report: {report_path(config)}', flush=True)
    return 0


if __name__ == '__main__':
    setup_logging('evaluate_bat')
    raise SystemExit(main())
