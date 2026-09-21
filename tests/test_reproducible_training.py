"""Seeded optimization, isolated artifacts, and threshold destinations."""
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from cesal_core.utils.reproducibility import model_seed, seed_training
from training_pipeline import workflow
from training_pipeline.workflow import prepare
from training_pipeline import train
from training_pipeline.solver import Solver


@pytest.fixture
def restore_rng():
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.are_deterministic_algorithms_enabled())
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
     torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = flags[:4]
    torch.use_deterministic_algorithms(flags[4])


def test_optimizer_and_shuffle_repeat_after_unrelated_learner(restore_rng):
    def train(parameters):
        seed_training(model_seed(42, 'Openstack', parameters))
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters())
        data = torch.randn(12, 3)
        order = torch.randperm(len(data))
        for x in data[order]:
            optimizer.zero_grad()
            model(x).square().sum().backward()
            optimizer.step()
        return {k: v.clone() for k, v in model.state_dict().items()}, order
    a, order_a = train((3, 1, 3, 32))
    other, _ = train((3, 1, 3, 64))
    b, order_b = train((3, 1, 3, 32))
    assert torch.equal(order_a, order_b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], other[k]) for k in a)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark


def test_prepared_baseline_splits_checkpoints_from_outputs(tmp_path):
    models, results = tmp_path / 'checkpoints', tmp_path / 'outputs'
    path = prepare('os', models, results, 42)
    cfg = yaml.safe_load(path.read_text())
    inference = yaml.safe_load((results / 'inference.yaml').read_text())
    assert cfg['seed'] == 42
    # Models under the checkpoint root, everything else under the results root.
    for artifact in (cfg['model_save_path'], *(m['checkpoint'] for m in inference['edge_models'])):
        assert str(artifact).startswith(str(models))
    for artifact in (cfg['threshold_output'], inference['output_dir'], inference['threshold_output']):
        assert str(artifact).startswith(str(results))
    assert not str(cfg['model_save_path']).startswith(str(results))
    with pytest.raises(FileExistsError):
        prepare('os', models, results, 43)


def test_calibration_and_energy_use_new_threshold_directory(tmp_path):
    solver = Solver.__new__(Solver)
    solver.dataset = 'Openstack'
    solver.num_epochs, solver.k, solver.e_layer_num, solver.batch_size = 3, 1, 3, 32
    target = tmp_path / 'results' / 'thresholds_cloud.yaml'
    solver.threshold_output = str(target)
    solver._update_threshold_config(0.25)
    solver._cache_energy('train', np.array([0.1, 0.2]))
    saved = yaml.safe_load(target.read_text())
    assert saved['models'] == [{'name': 'Openstack_e3_k1_l3_b32', 'threshold': 0.25}]
    assert (target.parent / 'energies' / 'Openstack_e3_k1_l3_b32_train.npy').exists()


def test_hdfs_edge_calibration_copies_corresponding_pth_thresholds(tmp_path, monkeypatch):
    from tools import calibrate_edge
    training_path = prepare('hdfs', tmp_path / 'ckpt', tmp_path / 'out', 42)
    cfg = yaml.safe_load(training_path.read_text())
    inference_path = training_path.parent / 'inference.yaml'
    inference = yaml.safe_load(inference_path.read_text())
    cloud_path = Path(cfg['threshold_output'])
    cloud_path.parent.mkdir()
    values = {item['name']: 0.123456789 + index for index, item in enumerate(inference['edge_models'])}
    cloud_path.write_text(yaml.safe_dump(dict(models=[dict(name=name.replace('qbat_', 'HDFS_'), threshold=value)
                                                     for name, value in values.items()])))
    monkeypatch.setattr('sys.argv', ['calibrate', '--training-config', str(training_path),
                                     '--inference-config', str(inference_path)])
    calibrate_edge.main()
    saved = yaml.safe_load(Path(inference['threshold_output']).read_text())
    assert {item['name']: item['threshold'] for item in saved['models']} == values


def test_default_dataset_and_preflight_failure_leave_no_stages(tmp_path, monkeypatch):
    target = tmp_path / 'run'
    monkeypatch.setattr('sys.argv', ['train', '--output-dir', str(target),
                                     '--models-dir', str(tmp_path / 'ckpt')])
    def fail(*args, **kwargs):
        raise RuntimeError('GPU unavailable')
    monkeypatch.setattr(workflow.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='GPU unavailable'):
        workflow.main()
    status = json.loads((target / 'status.json').read_text())
    assert status['datasets'] == ['os']
    assert status['status'] == 'failed'
    assert not status['stages']


def test_workflow_runs_every_stage_and_keeps_the_published_edge_models(tmp_path, monkeypatch):
    target = tmp_path / 'run'
    monkeypatch.setattr('sys.argv', ['train', 'os', '--output-dir', str(target),
                                     '--models-dir', str(tmp_path / 'ckpt')])
    monkeypatch.setattr(workflow.subprocess, 'run',
                        lambda command, **kwargs: SimpleNamespace(returncode=0))
    assert workflow.main() == 0
    status = json.loads((target / 'status.json').read_text())
    assert [stage['stage'] for stage in status['stages']] == [
        'train', 'evaluate_bat', 'convert', 'calibrate_edge', 'infer']
    assert status['status'] == 'complete'
    # The three edge architectures come from the released config, never from a search.
    published = yaml.safe_load(Path('configs/inference/os.yaml').read_text())
    inference = yaml.safe_load((target / 'os/inference.yaml').read_text())
    assert status['models_dir'] and not status['models_dir'].startswith(str(target))
    assert [m['name'] for m in inference['edge_models']] == [m['name'] for m in published['edge_models']]
    assert 'training_pipeline.evaluate' in status['stages'][1]['command']


def test_a_failing_stage_stops_the_run_and_is_recorded(tmp_path, monkeypatch):
    target = tmp_path / 'run'
    monkeypatch.setattr('sys.argv', ['train', 'os', '--output-dir', str(target),
                                     '--models-dir', str(tmp_path / 'ckpt')])
    def run(command, **kwargs):
        return SimpleNamespace(returncode=1 if 'quantization/qbat_export.py' in command else 0)
    monkeypatch.setattr(workflow.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='convert failed'):
        workflow.main()
    status = json.loads((target / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert [stage['stage'] for stage in status['stages']] == ['train', 'evaluate_bat', 'convert']
    assert status['stages'][-1]['status'] == 'failed'


@pytest.mark.parametrize('argv,message', [
    (['train', 'os', 'os'], 'only once'),
    (['train', 'bgl'], 'os or hdfs'),
    (['train', '--seed', '-1'], 'between 0'),
])
def test_invalid_arguments_are_rejected_before_creating_a_run(tmp_path, monkeypatch, argv, message):
    target = tmp_path / 'run'
    monkeypatch.setattr('sys.argv', argv + ['--output-dir', str(target),
                                            '--models-dir', str(tmp_path / 'ckpt')])
    with pytest.raises(SystemExit) as exc:
        workflow.main()
    assert exc.value.code == 2
    assert not target.exists()


def test_training_refuses_existing_checkpoints(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'original.pth'
    checkpoint.write_bytes(b'preserved published checkpoint')
    cfg = dict(dataset='Openstack', model_save_path=str(tmp_path),
               num_epochs=[3], k=[1], e_layer_num=[3], batch_size=[32])
    path = tmp_path / 'training.yaml'
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr('sys.argv', ['train', '--config', str(path)])
    with pytest.raises(FileExistsError, match='existing models are preserved'):
        train.main()
    assert checkpoint.read_bytes() == b'preserved published checkpoint'
    assert not (tmp_path / 'training_manifest.json').exists()


def test_bat_report_records_scores_hashes_and_per_learner_f1(tmp_path, monkeypatch):
    """--report is the provenance record the GMM sweep later verifies against."""
    from cesal_core.utils.metrics import Scores
    from training_pipeline import evaluate as bat
    config = tmp_path / 'training.yaml'
    config.write_text(yaml.safe_dump(dict(dataset='Openstack', seed=62,
                                          threshold_output=str(tmp_path / 'thresholds_cloud.yaml'))))
    def ensemble(path, voting_method, log_intermediate, model_f1):
        model_f1.update({'Openstack_e3_k1_l3_b32': 0.0, 'Openstack_e3_k1_l6_b32': 98.5})
        return np.array([0, 1]), np.array([0, 1])
    monkeypatch.setattr(bat, 'run_bat_ensemble', ensemble)
    monkeypatch.setattr(bat, 'evaluate', lambda *a, **k: Scores(99.9, 99.8, 100.0, 99.94))
    monkeypatch.setattr(bat, 'signature', lambda path: {'checkpoint': 'sha'})
    monkeypatch.setattr('sys.argv', ['evaluate', '--config', str(config), '--report'])
    assert bat.main() == 0
    report = json.loads((tmp_path / 'bat_evaluation.json').read_text())
    assert report['scores']['f1'] == 99.94
    assert report['seed'] == 62
    assert report['model_f1']['Openstack_e3_k1_l3_b32'] == 0.0
    assert report['artifact_sha256'] == {'checkpoint': 'sha'}
    assert 'accepted' not in report and 'criterion' not in report


def test_evaluation_without_report_writes_no_file(tmp_path, monkeypatch):
    from training_pipeline import evaluate as bat
    config = tmp_path / 'training.yaml'
    config.write_text(yaml.safe_dump(dict(dataset='Openstack', seed=62,
                                          threshold_output=str(tmp_path / 'thresholds_cloud.yaml'))))
    monkeypatch.setattr(bat, 'run_bat_ensemble',
                        lambda path, voting_method, log_intermediate, model_f1: (None, None))
    monkeypatch.setattr('sys.argv', ['evaluate', '--config', str(config)])
    assert bat.main() == 0
    assert not (tmp_path / 'bat_evaluation.json').exists()


def test_edge_thresholds_are_rederived_from_the_cloud_file(tmp_path, monkeypatch):
    """A .pte reuses its .pth threshold, so recalibration must refresh both."""
    from tools import calibrate_edge
    training_path = prepare('os', tmp_path / 'ckpt', tmp_path / 'out', 42)
    cfg = yaml.safe_load(training_path.read_text())
    inference_path = training_path.parent / 'inference.yaml'
    inference = yaml.safe_load(inference_path.read_text())
    cloud = Path(cfg['threshold_output'])
    cloud.parent.mkdir(parents=True, exist_ok=True)
    names = [m['name'] for m in inference['edge_models']]

    def write_cloud(values):
        cloud.write_text(yaml.safe_dump(dict(models=[
            dict(name=n.replace('qbat_', 'Openstack_'), threshold=v) for n, v in zip(names, values)])))

    def derive(force):
        argv = ['calibrate', '--training-config', str(training_path),
                '--inference-config', str(inference_path)] + (['--force'] if force else [])
        monkeypatch.setattr('sys.argv', argv)
        calibrate_edge.main()
        return [m['threshold'] for m in yaml.safe_load(Path(inference['threshold_output']).read_text())['models']]

    write_cloud([1.0, 2.0, 3.0])
    assert derive(force=False) == [1.0, 2.0, 3.0]
    # Recalibrating the cloud side must carry through to the edge file.
    write_cloud([4.5, 5.5, 6.5])
    with pytest.raises(FileExistsError, match='--force'):
        derive(force=False)
    assert derive(force=True) == [4.5, 5.5, 6.5]


def test_released_configs_resolve_their_cloud_threshold_path():
    """The shipped training configs omit threshold_output; the solver default applies."""
    from tools.calibrate_edge import cloud_threshold_path
    from cesal_core.utils.config import load_config
    for dataset, expected in (('os', 'outputs/os/thresholds_cloud.yaml'),
                              ('hdfs', 'outputs/hdfs/thresholds_cloud.yaml')):
        cfg = load_config(f'configs/training/{dataset}.yaml')
        assert 'threshold_output' not in cfg
        assert str(cloud_threshold_path(cfg)) == expected
