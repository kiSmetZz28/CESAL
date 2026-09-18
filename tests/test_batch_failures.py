"""Partial batches keep their successful artifacts but report an unsuccessful run."""
from pathlib import Path

import pytest
import yaml

from cesal_core.utils import steps
from training_pipeline import train


@pytest.fixture
def config(tmp_path):
    cfg = {
        'dataset': 'Openstack', 'mode': 'train', 'win_size': 100, 'input_c': 10,
        'model_save_path': str(tmp_path / 'bat'),
        'num_epochs': [3, 6], 'k': [1], 'e_layer_num': [3], 'batch_size': [32],
    }
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    yield path, cfg
    steps._active_run = None
    steps._active_step = None


@pytest.mark.parametrize('failed', [set(), {3}, {3, 6}])
def test_training_exit_status_and_kept_checkpoints(config, monkeypatch, failed):
    path, cfg = config
    visited = []

    def train_one(config):
        visited.append(config.num_epochs)
        if config.num_epochs in failed:
            raise RuntimeError('training failed')
        out = Path(config.model_save_path)
        out.mkdir(exist_ok=True)
        (out / f'{config.num_epochs}.pth').write_bytes(b'successful checkpoint')

    monkeypatch.setattr(train, '_run_one', train_one)
    monkeypatch.setattr('sys.argv', ['train', '--config', str(path)])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == (1 if failed else 0)
    assert visited == [3, 6]
    for epoch in {3, 6} - failed:
        assert Path(cfg['model_save_path'], f'{epoch}.pth').read_bytes() == b'successful checkpoint'


@pytest.mark.parametrize('failed,missing', [(set(), set()), ({3}, set()), ({3, 6}, set()), (set(), {6})])
def test_conversion_exit_status_and_kept_exports(config, monkeypatch, failed, missing):
    pytest.importorskip('executorch.exir')
    pytest.importorskip('torchao.quantization')
    from quantization import qbat_export

    path, cfg = config
    source = Path(cfg['model_save_path'])
    source.mkdir()
    target = source.parent / 'qbat'
    for epoch in {3, 6} - missing:
        (source / f'Openstack_e{epoch}_k1_l3_b32_checkpoint.pth').write_bytes(b'source checkpoint')
    visited = []

    def convert(src, dst, *args):
        epoch = int(Path(src).name.split('_')[1][1:])
        visited.append(epoch)
        if epoch in failed:
            raise RuntimeError('conversion failed')
        Path(dst).parent.mkdir(exist_ok=True)
        Path(dst).write_bytes(b'successful export')

    monkeypatch.setattr(qbat_export, 'convert_one', convert)
    monkeypatch.setattr('sys.argv', ['convert', '--config', str(path), '--all', '--output_dir', str(target)])
    with pytest.raises(SystemExit) as exc:
        qbat_export.main()
    assert exc.value.code == (1 if failed or missing else 0)
    assert visited == sorted({3, 6} - missing)
    for epoch in {3, 6} - missing - failed:
        assert (target / f'Openstack_e{epoch}_k1_l3_b32.pte').read_bytes() == b'successful export'
