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
    monkeypatch.setattr(sw, "sweep", lambda cfg, ratios: seen.update(ratios=ratios))
    sw.main()
    assert seen["ratios"] == [0.05, 0.1, 0.25]
