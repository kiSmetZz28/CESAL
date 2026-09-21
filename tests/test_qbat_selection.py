"""Candidate eligibility, point-adjusted scoring, and trio selection."""
from contextlib import contextmanager
from itertools import combinations
import json
import logging
import os

import numpy as np
import pytest

from cesal_core.utils.metrics import evaluate
from cesal_inference_pipeline.cloud_runner import _point_adjust
from tools import select_qbat


@contextmanager
def _missing():
    yield None


def reference_scores(labels, prediction):
    result = evaluate(labels, _point_adjust(labels, prediction), register=False, level=logging.DEBUG)
    return dict(precision=result.precision, recall=result.recall, f1=result.f_score)


@pytest.mark.parametrize('labels', [
    np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0]),
    np.array([1, 1, 0, 0, 1, 0, 0, 1]),          # a segment anchored at index 0
    np.array([0, 0, 0, 0, 1]),
    np.concatenate([np.zeros(40, int), np.ones(25, int)]),
])
def test_packed_scoring_matches_the_pipeline_for_every_prediction(labels):
    scorer = select_qbat.EnsembleScorer(labels)
    rng = np.random.default_rng(42)
    for _ in range(60):
        prediction = rng.integers(0, 2, len(labels))
        measured = scorer.scores(np.packbits(prediction.astype(bool)))
        expected = reference_scores(labels, prediction)
        for key in expected:
            assert measured[key] == pytest.approx(expected[key], abs=1e-9), (key, labels, prediction)


def test_zero_f1_and_l8_learners_never_become_candidates():
    training = dict(dataset='Openstack', seed=42, num_epochs=[3, 6, 10], k=[1, 3, 5],
                    e_layer_num=[3, 6, 8], batch_size=[32, 64, 96])
    bat_f1 = {select_qbat.cloud_model_name(select_qbat.candidate_name(v), 'Openstack'): 99.0
              for v in [(e, k, d, b) for e in training['num_epochs'] for k in training['k']
                        for d in (3, 6) for b in training['batch_size']]}
    zero = 'Openstack_e3_k1_l3_b32'
    bat_f1[zero] = 0.0
    selected, excluded = select_qbat.candidate_parameters(training, bat_f1)
    names = [name for name, _ in selected]
    assert len(names) == len(set(names)) == select_qbat.PER_DEPTH * 2
    assert not any('_l8_' in name for name in names)
    assert 'qbat_e3_k1_l3_b32' not in names
    assert any(item['reason'] == 'BAT .pth F1 is zero' for item in excluded)
    assert sum(1 for name in names if '_l3_' in name) == select_qbat.PER_DEPTH
    # Selection is a pure function of the seed and the recorded F1.
    assert select_qbat.candidate_parameters(training, bat_f1) == (selected, excluded)
    assert select_qbat.candidate_parameters(dict(training, seed=43), bat_f1)[0] != selected


def test_trio_search_reports_every_match_and_picks_lexicographically(tmp_path):
    rng = np.random.default_rng(7)
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    raw = {f'qbat_m{i}': rng.integers(0, 2, len(labels)) for i in range(7)}
    packed = {name: np.packbits(values.astype(bool)) for name, values in raw.items()}
    expected = {trio: reference_scores(labels, (np.stack([raw[n] for n in trio]).sum(axis=0) > 1).astype(int))
                for trio in combinations(sorted(raw), 3)}
    target = min(expected.values(), key=lambda s: s['f1'])
    monkey = dict(precision=target['precision'], recall=target['recall'], f1=target['f1'])
    original = select_qbat.PAPER.copy()
    select_qbat.PAPER.update(monkey)
    try:
        result = select_qbat.select_trio(packed, labels, tmp_path / 'trios.csv')
        matching = [trio for trio, scores in expected.items() if select_qbat.matches_paper(scores)]
        assert result['combinations'] == len(expected) == 35
        assert result['matching'] == len(matching)
        assert result['models'] == list(sorted(matching)[0])
    finally:
        select_qbat.PAPER.clear()
        select_qbat.PAPER.update(original)
    assert (tmp_path / 'paper_matching_trios.csv').exists()
    assert len((tmp_path / 'trios.csv').read_text().strip().split('\n')) == 36


def test_no_match_still_selects_the_highest_f1_trio_and_says_so(tmp_path):
    rng = np.random.default_rng(3)
    labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    raw = {f'qbat_m{i}': rng.integers(0, 2, len(labels)) for i in range(5)}
    packed = {name: np.packbits(values.astype(bool)) for name, values in raw.items()}
    original = select_qbat.PAPER.copy()
    select_qbat.PAPER.update(precision=1.0, recall=1.0, f1=1.0)   # unreachable target
    try:
        result = select_qbat.select_trio(packed, labels, tmp_path / 'trios.csv')
    finally:
        select_qbat.PAPER.clear()
        select_qbat.PAPER.update(original)
    assert result['status'] == 'best_available_no_paper_match'
    assert result['matching'] == 0
    assert len(result['models']) == 3
    # It really is the best of them, and the reported score is the pipeline's.
    best = max(combinations(sorted(raw), 3),
               key=lambda t: reference_scores(labels, (np.stack([raw[n] for n in t]).sum(axis=0) > 1).astype(int))['f1'])
    assert result['models'] == list(best)
    assert result['scores'] == pytest.approx(reference_scores(
        labels, (np.stack([raw[n] for n in best]).sum(axis=0) > 1).astype(int)))


def test_selection_rewrites_the_inference_config_edge_models(tmp_path):
    import yaml
    path = tmp_path / 'inference.yaml'
    path.write_text(yaml.safe_dump(dict(dataset='Openstack', routing_tolerance=0.1,
                                        edge_models=[dict(name='qbat_old', checkpoint='old.pte')])))
    select_qbat.update_inference_config(path, ['qbat_e3_k1_l3_b32', 'qbat_e6_k3_l6_b64',
                                               'qbat_e10_k5_l3_b96'], tmp_path / 'pte', 'Openstack')
    saved = yaml.safe_load(path.read_text())
    assert [m['name'] for m in saved['edge_models']] == [
        'qbat_e3_k1_l3_b32', 'qbat_e6_k3_l6_b64', 'qbat_e10_k5_l3_b96']
    assert saved['edge_models'][0]['checkpoint'].endswith('Openstack_e3_k1_l3_b32.pte')
    assert saved['routing_tolerance'] == 0.1   # everything else is preserved


def test_bundled_flatc_is_used_only_when_executorch_cannot_find_one(monkeypatch, tmp_path):
    """Export must work without the caller exporting FLATC_EXECUTABLE."""
    import importlib, sys, types
    qbat = importlib.import_module('quantization.qbat_export')
    binary = tmp_path / 'data' / 'bin' / 'flatc'
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\n')
    binary.chmod(0o755)
    fake = types.ModuleType('executorch')
    fake.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, 'executorch', fake)
    monkeypatch.setattr('shutil.which', lambda name: None)
    monkeypatch.setattr('importlib.resources.path', lambda package, resource: _missing())

    monkeypatch.delenv('FLATC_EXECUTABLE', raising=False)
    qbat._ensure_flatc()
    assert os.environ['FLATC_EXECUTABLE'] == str(binary)

    # An explicit setting always wins, and a flatc on PATH is left alone.
    monkeypatch.setenv('FLATC_EXECUTABLE', '/usr/bin/my-flatc')
    qbat._ensure_flatc()
    assert os.environ['FLATC_EXECUTABLE'] == '/usr/bin/my-flatc'
    monkeypatch.delenv('FLATC_EXECUTABLE')
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/flatc')
    qbat._ensure_flatc()
    assert 'FLATC_EXECUTABLE' not in os.environ


def test_widening_to_l8_adds_only_nonzero_f1_deep_learners():
    training = dict(dataset='Openstack', seed=62, num_epochs=[3, 6, 10], k=[1, 3, 5],
                    e_layer_num=[3, 6, 8], batch_size=[32, 64, 96])
    bat_f1 = {}
    for e in training['num_epochs']:
        for k in training['k']:
            for d in (3, 6, 8):
                for b in training['batch_size']:
                    name = select_qbat.cloud_model_name(select_qbat.candidate_name((e, k, d, b)), 'Openstack')
                    # one dead l8 learner, the rest healthy
                    bat_f1[name] = 0.0 if (d == 8 and e == 3 and k == 1 and b == 32) else 97.0
    narrow, _ = select_qbat.candidate_parameters(training, bat_f1, select_qbat.DEPTHS)
    wide, excluded = select_qbat.candidate_parameters(
        training, bat_f1, select_qbat.DEPTHS + select_qbat.FALLBACK_DEPTHS)
    narrow_names, wide_names = [n for n, _ in narrow], [n for n, _ in wide]
    assert set(narrow_names) < set(wide_names)
    added = set(wide_names) - set(narrow_names)
    assert added and all('_l8_' in name for name in added)
    assert 'qbat_e3_k1_l8_b32' not in wide_names          # zero BAT F1 stays out
    assert any(item['name'] == 'qbat_e3_k1_l8_b32' and item['reason'] == 'BAT .pth F1 is zero'
               for item in excluded)
    assert len(added) == select_qbat.PER_DEPTH            # capped per depth as before
