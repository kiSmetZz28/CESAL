"""Seeded optimization, isolated artifacts, and threshold destinations."""
import random

import numpy as np
import pytest
import torch
import yaml

from cesal_core.utils.reproducibility import model_seed, seed_training
from tools.retrain import prepare
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


def test_prepared_baseline_paths_do_not_use_published_outputs(tmp_path):
    target = tmp_path / 'new-baseline'
    path = prepare('os', target, 42)
    cfg = yaml.safe_load(path.read_text())
    inference = yaml.safe_load((target / 'inference.yaml').read_text())
    assert cfg['seed'] == 42
    for artifact in (cfg['model_save_path'], cfg['threshold_output'], inference['output_dir'],
                     inference['threshold_output'], *(m['checkpoint'] for m in inference['edge_models'])):
        assert str(artifact).startswith(str(target))
    with pytest.raises(FileExistsError):
        prepare('os', target, 43)


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
