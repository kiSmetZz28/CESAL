"""Tests for the routing-ratio sweep (cesal_inference_pipeline/sweep.py)."""
import numpy as np
import pytest

from cesal_inference_pipeline import sweep as sw


def _write_run(tmp_path, name, gt, hybrid, routed):
    d = tmp_path / name
    d.mkdir()
    np.save(d / "ground_truth.npy", np.asarray(gt, dtype=np.int8))
    np.save(d / "hybrid_preds.npy", np.asarray(hybrid, dtype=np.int8))
    if routed is not None:
        np.save(d / "routed_indices.npy", np.asarray(routed, dtype=int))
    return str(d)


def test_scores_reads_a_finished_run(tmp_path):
    gt = [0, 1, 1, 0, 0, 1]
    hy = [0, 1, 1, 0, 0, 1]
    d = _write_run(tmp_path, "ratio_10", gt, hy, routed=[1, 2])
    p, r, f, n_routed, n_total = sw._scores(d)
    assert (p, r, f) == pytest.approx((100.0, 100.0, 100.0))
    assert n_routed == 2 and n_total == 6


def test_scores_counts_mistakes(tmp_path):
    gt = [0, 1, 1, 0]
    hy = [1, 1, 0, 0]      # one false positive, one miss
    d = _write_run(tmp_path, "ratio_20", gt, hy, routed=[0])
    p, r, f, _, _ = sw._scores(d)
    assert p == pytest.approx(50.0)
    assert r == pytest.approx(50.0)


def test_scores_returns_none_when_a_ratio_produced_nothing(tmp_path):
    """A ratio whose cloud stage failed must not break the whole sweep table."""
    d = tmp_path / "ratio_05"
    d.mkdir()
    assert sw._scores(str(d)) is None


def test_scores_tolerates_a_missing_routed_index_file(tmp_path):
    d = _write_run(tmp_path, "ratio_30", [0, 1], [0, 1], routed=None)
    assert sw._scores(d)[3] == 0


@pytest.mark.parametrize("bad", ["0", "-0.1", "1.5", "2"])
def test_ratios_outside_zero_to_one_are_rejected(monkeypatch, bad):
    monkeypatch.setattr("sys.argv", ["sweep", "--ratios", bad])
    with pytest.raises(SystemExit):
        sw.main()


def test_ratio_list_is_parsed(monkeypatch):
    seen = {}
    monkeypatch.setattr("sys.argv", ["sweep", "--ratios", "0.05, 0.1 ,0.25"])
    def finished(cfg, ratios):
        seen.update(ratios=ratios)
        return True
    monkeypatch.setattr(sw, "sweep", finished)
    sw.main()
    assert seen["ratios"] == [0.05, 0.1, 0.25]


def test_empty_ratio_list_is_rejected(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sweep", "--ratios", " , "])
    with pytest.raises(SystemExit) as exc:
        sw.main()
    assert exc.value.code == 2


def test_incomplete_sweep_exits_nonzero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sweep", "--ratios", "0.1"])
    monkeypatch.setattr(sw, "sweep", lambda *args: False)
    with pytest.raises(SystemExit) as exc:
        sw.main()
    assert exc.value.code == 1


@pytest.mark.parametrize('failure', ['exception', 'no_new_output'])
def test_failed_ratio_excludes_old_scores_and_continues(tmp_path, monkeypatch, failure):
    import csv
    from pathlib import Path

    # An earlier run left a complete result at the first ratio.
    first = _write_run(tmp_path, 'ratio_10', [0, 1, 1, 0], [0, 1, 1, 0], [1])
    stale = Path(first, 'hybrid_preds.npy').read_bytes()
    monkeypatch.setattr(sw, 'load_config', lambda _: {'dataset': 'Openstack', 'output_dir': str(tmp_path)})
    calls = []

    def run(config, ratio, output_dir, reuse_edge_from):
        calls.append(reuse_edge_from)
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        # Edge scan finishes even when the subsequent cloud verification fails.
        for name in ('edge_preds.npy', 'edge_preds_raw.npy', 'ground_truth.npy'):
            np.save(out / name, [0, 1, 1, 0])
        np.save(out / 'energy_matrix.npy', np.ones((4, 3)))
        np.save(out / 'routed_indices.npy', [1])
        if ratio == 0.1:
            if failure == 'exception':
                raise RuntimeError('cloud failed')
            return
        np.save(out / 'hybrid_preds.npy', [0, 1, 1, 0])

    monkeypatch.setattr(sw, 'run_inference', run)
    assert sw.sweep('config.yaml', [0.1, 0.2]) is False
    assert calls == [None, first]
    assert Path(first, 'hybrid_preds.npy').read_bytes() == stale
    with (tmp_path / 'routing_ratio_sweep.csv').open() as result:
        rows = list(csv.DictReader(result))
    assert rows[0]['precision'] == rows[0]['recall'] == rows[0]['f1'] == ''
    assert rows[1]['precision'] == rows[1]['recall'] == rows[1]['f1'] == '100.0000'


def test_failed_partial_edge_scan_is_not_reused(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(sw, 'load_config', lambda _: {'dataset': 'Openstack', 'output_dir': str(tmp_path)})
    calls = []

    def run(config, ratio, output_dir, reuse_edge_from):
        calls.append(reuse_edge_from)
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        np.save(out / 'edge_preds.npy', [0, 1, 1, 0])
        raise RuntimeError('edge scan incomplete')

    monkeypatch.setattr(sw, 'run_inference', run)
    assert sw.sweep('config.yaml', [0.1, 0.2]) is False
    assert calls == [None, None]
