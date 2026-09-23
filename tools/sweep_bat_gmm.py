"""Compare EM-GMM settings on a saved run's BAT energies, without retraining."""
import argparse
from datetime import datetime
from itertools import product
import json
import logging
from pathlib import Path
import time
import warnings

import numpy as np
import yaml
from joblib import Parallel, delayed, parallel_config

from cesal_core.utils.config import load_config
from cesal_core.utils.metrics import evaluate
from cesal_core.utils.reproducibility import environment, sha256, write_json
from cesal_inference_pipeline.cloud_runner import _point_adjust
from training_pipeline.solver import _fit_gmm, _log_cluster_percentages

KEYS = ('gmm_n_components', 'gmm_covariance_type', 'gmm_max_iter',
        'gmm_init_params', 'gmm_n_init')


def displays_as_99_99(value):
    """Whether a percentage would print as 99.99 or 100.00 to two decimals."""
    return 0 <= value <= 100 and f'{value:.2f}' in ('99.99', '100.00')

# Which energies the EM-GMM is fitted on, and which pool the percentile is taken
# from. 'train+thre' is the original CECO-LAD behaviour (Solver.COMPUTE_THRE_ENERGY
# = True); the thre loader reads the test split, so that pool contains the test
# energies. 'train' calibrates on the training energies alone.
CALIBRATION = 'calibration_energy'
CALIBRATIONS = ('train+thre', 'train')

GRIDS = ('single', 'components', 'refine', 'full')


def _one_at_a_time(baseline):
    """The original sweep: one EM-GMM parameter changed per trial."""
    for key, values in (
        ('gmm_n_components', (3, 5, 9, 11)),
        ('gmm_covariance_type', ('full',)),
        ('gmm_init_params', ('kmeans',)),
        ('gmm_n_init', (1, 20)),
        ('gmm_max_iter', (300,)),
    ):
        for value in values:
            if value != baseline[key]:
                yield f'{key}_{value}', dict(baseline, **{key: value})


def _component_scan(baseline):
    """n_components alone, which is what moves the largest-cluster percentile."""
    for value in range(3, 13):
        if value != baseline['gmm_n_components']:
            yield f'gmm_n_components_{value}', dict(baseline, gmm_n_components=value)


def _refine(baseline):
    """Everything except the component count, which a `components` scan already covered.

    gmm_max_iter is left out on purpose: the fits converge in a handful of
    iterations, so raising the cap cannot change the result.
    """
    for key, values in (
        ('gmm_covariance_type', ('tied', 'full')),
        ('gmm_init_params', ('k-means++', 'kmeans', 'random_from_data')),
        ('gmm_n_init', (1, 10, 20)),
    ):
        for value in values:
            if value != baseline[key]:
                yield f'{key}_{value}', dict(baseline, **{key: value})


def _full_product(baseline):
    """Every combination of the parameters that can change the fitted clustering."""
    axes = (('gmm_n_components', tuple(range(3, 12))),
            ('gmm_covariance_type', ('tied', 'full')),
            ('gmm_init_params', ('k-means++', 'kmeans', 'random_from_data')),
            ('gmm_n_init', (1, 10)))
    for values in product(*(values for _, values in axes)):
        settings = dict(baseline, **dict(zip((key for key, _ in axes), values)))
        if settings != baseline:
            yield '_'.join(f'{key.removeprefix("gmm_")}{value}' for key, value in zip(
                (key for key, _ in axes), values)), settings


def settings_grid(config, calibrations=CALIBRATIONS[:1], grid='single'):
    """Name every trial uniquely across both the parameter and calibration axes."""
    if grid not in GRIDS:
        raise ValueError(f'Unknown grid: {grid}')
    if not calibrations or any(item not in CALIBRATIONS for item in calibrations):
        raise ValueError(f'Unknown calibration energy: {calibrations}')
    expand = dict(single=_one_at_a_time, components=_component_scan,
                  refine=_refine, full=_full_product)[grid]
    candidates = []
    for calibration in dict.fromkeys(calibrations):
        baseline = {key: config[key] for key in KEYS}
        baseline[CALIBRATION] = calibration
        prefix = '' if calibration == CALIBRATIONS[0] else f'{calibration}__'
        candidates.append((f'{prefix}baseline', baseline))
        candidates.extend((prefix + name, settings) for name, settings in expand(baseline))
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError('Trial names must be unique')
    return candidates


def calibration_energy(arrays, calibration):
    if calibration == 'train+thre':
        return np.concatenate([arrays['train'], arrays['thre']])
    if calibration == 'train':
        return arrays['train']
    raise ValueError(f'Unknown calibration energy: {calibration}')


def load_arrays(cache, name):
    arrays = {kind: np.load(cache / f'{name}_{kind}.npy', allow_pickle=False)
              for kind in ('train', 'thre', 'test', 'labels')}
    for kind, values in arrays.items():
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError(f'{name}: invalid {kind} cache')
    if arrays['test'].shape != arrays['labels'].shape or not np.isin(arrays['labels'], (0, 1)).all():
        raise ValueError(f'{name}: invalid test labels')
    return arrays


def fit_model(cache, name, settings):
    arrays = load_arrays(cache, name)
    combined = calibration_energy(arrays, settings[CALIBRATION])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        clusters = _fit_gmm(combined.reshape(-1, 1), *(settings[key] for key in KEYS))
    ratio = _log_cluster_percentages(clusters)[0][1]
    threshold = float(np.percentile(combined, ratio))
    predictions = _point_adjust(arrays['labels'], (arrays['test'] > threshold).astype(int))
    return name, threshold, predictions.astype(np.int8), sorted({str(w.message) for w in caught})


def score_predictions(labels, predictions):
    result = evaluate(labels, predictions, register=False, level=logging.DEBUG)
    return dict(accuracy=result.accuracy, precision=result.precision,
                recall=result.recall, f1=result.f_score)


def summarize(labels, records):
    votes = np.zeros(len(labels), dtype=np.int16)
    individual = {}
    for name, threshold, pred, messages in records:
        votes += pred
        individual[name] = dict(threshold=threshold, f1=score_predictions(labels, pred)['f1'],
                                warnings=messages)
    final = (votes >= (len(records) // 2 + 1)).astype(int)
    scores = score_predictions(labels, final)
    return dict(scores=scores, accepted=displays_as_99_99(scores['f1']), models=individual,
                false_positives=int(np.sum((labels == 0) & (final == 1))),
                false_negatives=int(np.sum((labels == 1) & (final == 0))))


def verify_baseline(result, saved, thresholds):
    for key, value in saved['scores'].items():
        if not np.isclose(result['scores'][key], value, rtol=0, atol=1e-10):
            raise ValueError(f'Cached baseline does not reproduce saved BAT {key}')
    for name, model in result['models'].items():
        if not np.isclose(model['threshold'], thresholds[name], rtol=0, atol=1e-12):
            raise ValueError(f'Cached baseline threshold mismatch: {name}')
        if not np.isclose(model['f1'], saved['model_f1'][name], rtol=0, atol=1e-10):
            raise ValueError(f'Cached baseline F1 mismatch: {name}')


def run(config_path, output, workers, calibrations=CALIBRATIONS[:1], grid='single'):
    if workers < 1:
        raise ValueError('workers must be positive')
    config_path = config_path.resolve()
    config = load_config(config_path)
    dataset, seed = config['dataset'], config.get('seed')
    if seed is None:
        raise ValueError('The saved run must record its training seed.')
    names = [f'{dataset}_e{e}_k{k}_l{depth}_b{batch}' for e, k, depth, batch in
             product(*(config[key] for key in ('num_epochs', 'k', 'e_layer_num', 'batch_size')))]
    if len(names) != 81 or len(set(names)) != 81:
        raise ValueError('Expected all 81 distinct BAT learners')
    threshold_path = Path(config['threshold_output'])
    cache = threshold_path.parent / 'energies'
    saved_path = threshold_path.parent / 'bat_evaluation.json'
    saved = json.loads(saved_path.read_text())
    thresholds = {row['name']: row['threshold'] for row in load_config(threshold_path)['models']}
    # Check that the models, data, configuration and original thresholds still match the saved run.
    inputs = [config_path, threshold_path, *sorted(Path(config['data_path']).glob('*.txt')),
              *(Path(config['model_save_path']) / f'{name}_checkpoint.pth' for name in names)]
    hashes = {}
    for path in inputs:
        digest = sha256(path)
        if saved['artifact_sha256'].get(str(path.resolve())) != digest:
            raise ValueError(f'Saved BAT provenance mismatch: {path}')
        hashes[str(path.resolve())] = digest
    labels = None
    for name in names:
        arrays = load_arrays(cache, name)
        if labels is None:
            labels = arrays['labels']
        elif not np.array_equal(labels, arrays['labels']):
            raise ValueError(f'Ground truth inconsistent: {name}')
        for kind in arrays:
            path = cache / f'{name}_{kind}.npy'
            hashes[str(path.resolve())] = sha256(path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates = settings_grid(config, calibrations, grid)
    report = dict(status='running', source_config=str(config_path), seed=seed, dataset=dataset,
                  model_count=len(names), workers=workers, gmm_random_state=42, grid=grid,
                  calibrations=list(dict.fromkeys(calibrations)),
                  calibration_note='train+thre is the released behaviour; the thre loader reads the '
                                   'test split, so that pool contains the test energies. Reproducing '
                                   "'train' in the solver needs COMPUTE_THRE_ENERGY = False.",
                  evaluation_data='Full bundled test set used for parameter selection; not independent validation',
                  environment=environment(), artifact_sha256=hashes,
                  source_sha256={str(Path(__file__).resolve()): sha256(__file__),
                                 str(Path('training_pipeline/solver.py').resolve()): sha256('training_pipeline/solver.py')},
                  candidates=[dict(name=name, settings=settings) for name, settings in candidates], trials=[])
    write_json(output / 'sweep.json', report)
    try:
        with parallel_config(backend='loky', inner_max_num_threads=1):
            for index, (name, settings) in enumerate(candidates, 1):
                start = time.monotonic()
                print(f'[{index}/{len(candidates)}] {name}: fitting all 81 learners', flush=True)
                records = Parallel(n_jobs=workers)(delayed(fit_model)(cache, model, settings) for model in names)
                result = summarize(labels, records)
                result.update(name=name, settings=settings, seconds=time.monotonic() - start)
                if name == 'baseline':
                    verify_baseline(result, saved, thresholds)
                    report['baseline_verified'] = True
                trial = output / name
                trial.mkdir()
                # calibration_energy records the experiment condition; the solver reads
                # its module-level COMPUTE_THRE_ENERGY rather than this key.
                trial_config = dict(config, **settings, threshold_output=str(trial / 'thresholds_cloud.yaml'))
                (trial / 'training.yaml').write_text(yaml.safe_dump(trial_config, sort_keys=False))
                (trial / 'thresholds_cloud.yaml').write_text(yaml.safe_dump(dict(models=[
                    dict(name=model, threshold=data['threshold']) for model, data in result['models'].items()])))
                write_json(trial / 'scores.json', result)
                report['trials'].append(result)
                write_json(output / 'sweep.json', report)
                print(f"  BAT P={result['scores']['precision']:.4f}% R={result['scores']['recall']:.4f}% "
                      f"F1={result['scores']['f1']:.4f}% FP={result['false_positives']} "
                      f"FN={result['false_negatives']} ({result['seconds']:.1f}s)", flush=True)
        best = max(report['trials'], key=lambda trial: trial['scores']['f1'])
        report.update(status='complete', best_trial=best['name'],
                      best_per_calibration={calibration: max(
                          (trial for trial in report['trials']
                           if trial['settings'][CALIBRATION] == calibration),
                          key=lambda trial: trial['scores']['f1'])['name']
                          for calibration in report['calibrations']},
                      accepted_trials=[trial['name'] for trial in report['trials'] if trial['accepted']])
        print(f"Best trial: {best['name']}; BAT F1={best['scores']['f1']:.4f}%. "
              'Original defaults and thresholds are unchanged.', flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(output / 'sweep.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        default=None, help='Default: outputs/gmm_sweeps/<dataset>_seed<seed>_<timestamp>.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--calibration', dest='calibrations', action='append', choices=CALIBRATIONS,
                        help='Energy pool for EM-GMM fitting and the percentile; repeatable. '
                             f'Default: {CALIBRATIONS[0]}.')
    parser.add_argument('--grid', choices=GRIDS, default='single')
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        saved = load_config(args.config)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = Path('outputs/gmm_sweeps') / f"{saved['dataset'].lower()}_seed{saved.get('seed')}_{stamp}"
    run(args.config, output, args.workers, args.calibrations or CALIBRATIONS[:1], args.grid)


if __name__ == '__main__':
    main()
