"""Exercise orchestration with fixed predictions, without model checkpoints."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from cesal_core.utils import steps
from cesal_inference_pipeline import cloud_runner, lad_bat_cloud, run


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    cfg = {
        'dataset': 'Openstack', 'output_dir': str(tmp_path), 'win_size': 2,
        'cloud': {'dataset': 'Openstack'},
    }
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    windows = np.arange(80, dtype=np.float32).reshape(4, 2, 10)
    result = run.lad_qbat_edge.EdgeResult(
        predictions=np.array([0, 0, 0, 1, 0, 0, 0, 0]),
        ground_truth=np.array([0, 0, 1, 1, 0, 1, 1, 0]),
        energy_matrix=np.arange(24, dtype=float).reshape(8, 3),
        train_energy_matrix=np.arange(24, dtype=float).reshape(8, 3),
        thresholds=np.ones(3), test_windows=windows,
    )
    monkeypatch.delenv('CESAL_STEP_HANDOFF', raising=False)
    monkeypatch.setenv('CESAL_EVENTS', '1')
    monkeypatch.setattr(run.lad_qbat_edge, 'run', lambda _: result)
    monkeypatch.setattr(run, 'compute_inv_cov', lambda _: (None, np.eye(3)))
    monkeypatch.setattr(run, 'select_indices_by_distance', lambda **kwargs: [0, 3, 5, 6])
    monkeypatch.setattr(run, '_detect_cloud_python', lambda: 'cloud-python')
    yield config, result
    steps._active_run = None
    steps._active_step = None


@pytest.mark.parametrize('failure', ['exit_code', 'missing_interpreter'])
def test_cloud_failure_propagates_and_marks_run_failed(pipeline, monkeypatch, capsys, failure):
    config, _ = pipeline
    error_type = subprocess.CalledProcessError if failure == 'exit_code' else FileNotFoundError

    def fail(command, **kwargs):
        assert kwargs['check'] is True
        if failure == 'exit_code':
            raise subprocess.CalledProcessError(17, command)
        raise FileNotFoundError('cloud interpreter not found')

    monkeypatch.setattr(run.subprocess, 'run', fail)
    with pytest.raises(error_type):
        run.run_inference(str(config))
    events = [json.loads(line[len('@@CESAL'):]) for line in capsys.readouterr().out.splitlines()
              if line.startswith('@@CESAL')]
    assert any(e['t'] == 'step_fail' and e['id'] == 'cloud' for e in events)
    assert any(e['t'] == 'step_skip' and e['id'] == 'hybrid' for e in events)
    assert not (config.parent / 'hybrid_preds.npy').exists()
    np.testing.assert_array_equal(np.load(config.parent / 'edge_preds.npy'), [0, 0, 1, 1, 0, 0, 0, 0])


def test_success_keeps_predictions_routing_and_point_adjustment(pipeline, monkeypatch):
    config, result = pipeline
    monkeypatch.setattr('cesal_core.utils.config.setup_logging', lambda _: None)

    def cloud_predict(windows, config):
        expected = result.test_windows.reshape(8, 10)[[0, 3, 5, 6]].reshape(2, 2, 10)
        np.testing.assert_array_equal(windows, expected)
        return np.array([1, 0, 1, 0])

    monkeypatch.setattr(lad_bat_cloud, 'run', cloud_predict)

    def cloud_process(command, **kwargs):
        assert kwargs['check'] is True
        with monkeypatch.context() as child:
            child.setenv('CESAL_STEP_HANDOFF', kwargs['env']['CESAL_STEP_HANDOFF'])
            child.setattr('sys.argv', command[1:])
            cloud_runner.main()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run.subprocess, 'run', cloud_process)
    run.run_inference(str(config))
    out = config.parent
    np.testing.assert_array_equal(np.load(out / 'routed_indices.npy'), [0, 3, 5, 6])
    np.testing.assert_array_equal(np.load(out / 'ground_truth.npy'), result.ground_truth)
    np.testing.assert_array_equal(np.load(out / 'energy_matrix.npy'), result.energy_matrix)
    np.testing.assert_array_equal(np.load(out / 'edge_preds_raw.npy'), result.predictions)
    np.testing.assert_array_equal(np.load(out / 'edge_preds.npy'), [0, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.load(out / 'cloud_preds.npy'), [1, 0, 1, 0])
    np.testing.assert_array_equal(np.load(out / 'hybrid_preds.npy'), [1, 0, 0, 0, 0, 1, 1, 0])
