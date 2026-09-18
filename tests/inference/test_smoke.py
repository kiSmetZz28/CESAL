"""Small-experiment inputs remain isolated from the published experiment."""
import json

import numpy as np
import pytest
import yaml

from tools import smoke
from cesal_inference_pipeline import lad_qbat_edge as edge
from cesal_inference_pipeline.lad_qbat_edge import select_test_windows


def test_sample_preserves_full_evaluation_windows_and_labels():
    indices = smoke.sample_indices(136913, 18434, 100)
    windows = np.arange(1553 * 100 * 10).reshape(1553, 100, 10)
    labels = np.r_[np.zeros(136913), np.ones(18387)]
    selected, gt = select_test_windows(windows, labels, indices)
    np.testing.assert_array_equal(selected, windows[indices])
    np.testing.assert_array_equal(gt, np.r_[np.zeros(500), np.ones(500)])
    unchanged, unchanged_labels = select_test_windows(windows, labels)
    assert unchanged is windows and unchanged_labels is labels


@pytest.mark.parametrize('indices', [[], [-1], [10], [1, 1], [3, 2], [1.5], [[1]]])
def test_invalid_window_selection_is_rejected(indices):
    with pytest.raises(ValueError, match='test_window_indices'):
        select_test_windows(np.zeros((10, 100, 10)), np.zeros(1000), indices)


def test_normal_run_keeps_every_window_and_bypasses_experiment_selection(monkeypatch):
    import torch

    windows = np.arange(120, dtype=np.float32).reshape(6, 2, 10)
    labels = np.tile([0, 1], (6, 1))
    model_inputs = []

    def unexpected_selection(*args):
        raise AssertionError('Normal inference must not use experiment sampling')

    def score(model, thresholds, inputs, path):
        model_inputs.append(inputs.copy())
        # Deterministic model outputs isolate the orchestration from model cost.
        return inputs[:, :, 0].reshape(-1, 1), thresholds[model['name']]

    monkeypatch.setattr(edge, 'select_test_windows', unexpected_selection)
    monkeypatch.setattr(edge, '_load_thresholds', lambda _: {'one': 25.0, 'two': 65.0})
    monkeypatch.setattr(edge, 'get_loader_segment', lambda *args, **kwargs:
                        [(torch.from_numpy(windows[:3]), torch.from_numpy(labels[:3])),
                         (torch.from_numpy(windows[3:]), torch.from_numpy(labels[3:]))])
    monkeypatch.setattr(edge, '_run_one_edge_model', score)
    result = edge.run({
        'dataset': 'Openstack', 'win_size': 2, 'batch_size': 3, 'data_path': 'unused',
        'edge_models': [{'name': 'one'}, {'name': 'two'}], 'threshold_output': 'unused',
    })
    assert len(model_inputs) == 2
    for inputs in model_inputs:
        np.testing.assert_array_equal(inputs, windows)
    np.testing.assert_array_equal(result.test_windows, windows)
    np.testing.assert_array_equal(result.ground_truth, labels.reshape(-1))
    np.testing.assert_array_equal(result.predictions, (windows[:, :, 0].reshape(-1) > 65).astype(int))


@pytest.mark.parametrize('dataset', ['os', 'hdfs'])
def test_normal_configs_keep_full_evaluation_settings(dataset):
    config = yaml.safe_load((smoke.ROOT / f'configs/inference/{dataset}.yaml').read_text())
    assert config.get('test_window_indices') is None
    assert len(config['edge_models']) == 3
    cloud = config['cloud']
    assert np.prod([len(cloud[key]) for key in ('num_epochs', 'k', 'e_layer_num', 'batch_size')]) == 81
    assert config['routing_tolerance'] == 0.1


def test_prepare_preserves_published_artifacts(tmp_path, monkeypatch):
    # Use the real config but fixture checkpoints/data: this test only checks
    # input preparation; run.py smoke performs actual model execution.
    cfg = yaml.safe_load((smoke.ROOT / 'configs/inference/os.yaml').read_text())
    monkeypatch.setattr(smoke, 'ROOT', tmp_path)
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write(tmp_path / 'configs/inference/os.yaml', yaml.safe_dump(cfg).encode())
    for name in ('train.txt', 'test_normal.txt', 'test_abnormal.txt'):
        write(tmp_path / cfg['data_path'] / name, ('1 2 3 4 5 6\n' * 100).encode())
    cloud_names = [f'Openstack_e3_k1_l3_b{b}' for b in (32, 64, 96)]
    for model in cfg['edge_models']:
        write(tmp_path / model['checkpoint'], b'fixture checkpoint')
    for name in cloud_names:
        write(tmp_path / cfg['cloud']['model_save_path'] / f'{name}_checkpoint.pth', b'fixture checkpoint')
    for path, names in [(cfg['threshold_output'], [m['name'] for m in cfg['edge_models']]),
                        (cfg['cloud']['thresholds_yaml'], cloud_names)]:
        write(tmp_path / path, yaml.safe_dump({'models': [{'name': n, 'threshold': 1.2} for n in names]}).encode())
    original = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path / 'experiment'
    output.mkdir()
    config, manifest = smoke.prepare(output)
    for path, data in original.items():
        assert path.read_bytes() == data
    for name in ('test_normal.txt', 'test_abnormal.txt'):
        assert (output / 'data' / name).read_bytes() == original[tmp_path / cfg['data_path'] / name]
    assert (output / 'data/train.txt').read_bytes() == original[tmp_path / cfg['data_path'] / 'train.txt']
    assert json.loads((output / 'experiment.json').read_text()) == manifest
    prepared = yaml.safe_load(config.read_text())
    assert prepared['routing_tolerance'] == cfg['routing_tolerance']
    assert len(prepared['edge_models']) == 3
    assert prepared['cloud']['batch_size'] == [32, 64, 96]
    assert prepared['output_dir'] == str(output)


def test_verification_rejects_incomplete_cloud_output(tmp_path):
    np.save(tmp_path / 'ground_truth.npy', np.r_[np.zeros(500), np.ones(500)])
    for name, values in {
        'energy_matrix.npy': np.ones((1000, 3)), 'edge_preds_raw.npy': np.zeros(1000),
        'edge_preds.npy': np.zeros(1000),
        'routed_indices.npy': np.arange(100), 'routed_lines.npy': np.zeros((100, 10)),
        'cloud_preds.npy': np.zeros(99), 'hybrid_preds.npy': np.zeros(1000),
    }.items():
        np.save(tmp_path / name, values)
    with pytest.raises(ValueError, match='cloud_preds'):
        smoke.verify(tmp_path)
