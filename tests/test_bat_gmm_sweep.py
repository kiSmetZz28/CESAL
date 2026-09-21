"""Isolated GMM experiments must preserve scoring and reject inconsistent caches."""
import ast
import inspect
import textwrap

import numpy as np
import pytest

from tools import sweep_bat_gmm as sweep
from training_pipeline.solver import Solver


def test_cached_scoring_matches_solver_segment_logic():
    # Execute the original solver's postprocessing block as an independent reference.
    source = ast.parse(textwrap.dedent(inspect.getsource(Solver.singlemodelpred)))
    body = source.body[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'anomaly_state' for target in node.targets))
    code = compile(ast.Module(body=body[start:start + 2], type_ignores=[]), '<solver-scoring>', 'exec')
    rng = np.random.RandomState(62)
    for _ in range(30):
        labels = rng.randint(0, 2, 80)
        predictions = rng.randint(0, 2, 80)
        namespace = dict(gt=labels, pred=predictions.copy())
        exec(code, namespace)
        np.testing.assert_array_equal(sweep._point_adjust(labels, predictions), namespace['pred'])


def test_majority_keeps_zero_f1_learners():
    labels = np.array([0, 1, 1, 0])
    records = [('one', 1., labels, []), ('two', 1., np.zeros(4, dtype=int), []),
               ('three', 1., np.zeros(4, dtype=int), [])]
    report = sweep.summarize(labels, records)
    assert report['scores']['f1'] == 0
    assert report['false_negatives'] == 2
    assert len(report['models']) == 3


@pytest.mark.parametrize('calibration,energy,threshold,expected', [
    ('train+thre', [0., 1., 2., 3.], 1.5, [0, 1, 1, 0]),
    ('train', [0., 1.], 0.5, [1, 1, 1, 1]),
])
def test_fit_uses_the_requested_calibration_pool_and_strict_threshold(
        tmp_path, monkeypatch, calibration, energy, threshold, expected):
    arrays = dict(train=np.array([0., 1.]), thre=np.array([2., 3.]),
                  test=np.array([1.5, 2., 0.75, 0.75]), labels=np.array([0, 1, 1, 0]))
    for kind, values in arrays.items():
        np.save(tmp_path / f'model_{kind}.npy', values)
    settings = dict(zip(sweep.KEYS, (2, 'tied', 100, 'k-means++', 10)))
    settings[sweep.CALIBRATION] = calibration
    def fit(observed, *args):
        np.testing.assert_array_equal(observed.reshape(-1), energy)
        assert args == (2, 'tied', 100, 'k-means++', 10)
        return np.array([0] * (len(energy) // 2) + [1] * (len(energy) // 2))
    monkeypatch.setattr(sweep, '_fit_gmm', fit)
    name, measured, predictions, _ = sweep.fit_model(tmp_path, 'model', settings)
    assert measured == threshold
    np.testing.assert_array_equal(predictions, expected)
    np.save(tmp_path / 'model_test.npy', np.array([np.nan, 1, 1, 0]))
    with pytest.raises(ValueError, match='invalid test cache'):
        sweep.load_arrays(tmp_path, 'model')


def test_unknown_calibration_pool_is_rejected():
    arrays = dict(train=np.array([0.]), thre=np.array([1.]))
    with pytest.raises(ValueError, match='Unknown calibration energy'):
        sweep.calibration_energy(arrays, 'test')


def test_baseline_rejects_score_or_threshold_drift():
    result = dict(scores=dict(f1=99.94), models={'model': dict(threshold=1., f1=90.)})
    saved = dict(scores=dict(f1=99.94), model_f1={'model': 90.})
    sweep.verify_baseline(result, saved, {'model': 1.})
    with pytest.raises(ValueError, match='threshold mismatch'):
        sweep.verify_baseline(result, saved, {'model': 2.})
    with pytest.raises(ValueError, match='does not reproduce'):
        sweep.verify_baseline(result, dict(saved, scores=dict(f1=99.99)), {'model': 1.})


def test_grid_changes_one_parameter_at_a_time():
    config = dict(zip(sweep.KEYS, (7, 'tied', 100, 'k-means++', 10)))
    grid = sweep.settings_grid(config)
    assert grid[0] == ('baseline', dict(config, calibration_energy='train+thre'))
    for _, settings in grid[1:]:
        assert settings[sweep.CALIBRATION] == 'train+thre'
        assert sum(settings[key] != config[key] for key in sweep.KEYS) == 1
    assert config == dict(zip(sweep.KEYS, (7, 'tied', 100, 'k-means++', 10)))


@pytest.mark.parametrize('grid', sweep.GRIDS)
def test_each_calibration_repeats_the_same_settings_under_a_unique_name(grid):
    config = dict(zip(sweep.KEYS, (7, 'tied', 100, 'k-means++', 10)))
    both = sweep.settings_grid(config, sweep.CALIBRATIONS, grid)
    names = [name for name, _ in both]
    assert len(names) == len(set(names))
    halves = [both[:len(both) // 2], both[len(both) // 2:]]
    assert [name for name, _ in halves[0]] == [name.removeprefix('train__') for name, _ in halves[1]]
    for (_, first), (_, second) in zip(*halves):
        assert first[sweep.CALIBRATION] == 'train+thre'
        assert second[sweep.CALIBRATION] == 'train'
        assert {key: first[key] for key in sweep.KEYS} == {key: second[key] for key in sweep.KEYS}


def test_full_grid_covers_every_clustering_combination_once():
    config = dict(zip(sweep.KEYS, (7, 'tied', 100, 'k-means++', 10)))
    grid = sweep.settings_grid(config, ('train+thre',), 'full')
    combinations = {tuple(settings[key] for key in sweep.KEYS) for _, settings in grid}
    assert len(combinations) == len(grid) == 9 * 2 * 3 * 2
    assert all(settings['gmm_max_iter'] == 100 for _, settings in grid)


@pytest.mark.parametrize('kwargs', [dict(calibrations=()), dict(calibrations=('test',)), dict(grid='none')])
def test_unknown_grid_or_calibration_is_rejected(kwargs):
    config = dict(zip(sweep.KEYS, (7, 'tied', 100, 'k-means++', 10)))
    with pytest.raises(ValueError):
        sweep.settings_grid(config, **dict(dict(calibrations=('train',), grid='single'), **kwargs))


@pytest.mark.parametrize('score,expected', [
    (99.91, False), (99.98, False), (99.989, True), (99.99, True),
    (99.999, True), (100., True), (100.01, False), (float('nan'), False),
])
def test_two_decimal_display_threshold(score, expected):
    assert sweep.displays_as_99_99(score) == expected
