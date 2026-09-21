"""Archive installation preserves model bytes and fails before unsafe updates."""
import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from tools import download_checkpoints as download


def checksum(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


@pytest.fixture
def bat(tmp_path, monkeypatch):
    models = {f'Openstack_e3_k1_l3_b{b}_checkpoint.pth': f'model-{b}'.encode()
              for b in (32, 64, 96)}
    root = tmp_path / 'bat'
    root.mkdir()
    archive = root / 'ensemble_os.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, data in models.items():
            z.writestr('os/' + name, data)
    spec = dict(filename=archive.name, file_id='public-archive',
                models={name: checksum(data) for name, data in models.items()},
                **checksum(archive.read_bytes()))
    monkeypatch.setattr(download, '_BAT_ARCHIVES', {'os': spec})
    return archive, root / 'os', spec, models


def test_cached_archive_repairs_partial_install_without_replacing_existing(bat):
    archive, dest, _, models = bat
    dest.mkdir()
    name = next(iter(models))
    existing = dest / name
    existing.write_bytes(models[name])
    stat = existing.stat()
    download._download_bat('os', dest)
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == models
    assert existing.stat().st_ino == stat.st_ino
    assert existing.stat().st_mtime_ns == stat.st_mtime_ns
    assert not list(dest.parent.glob('.extract_*'))


def test_complete_install_needs_neither_archive_nor_network(bat, monkeypatch):
    archive, dest, _, models = bat
    dest.mkdir()
    for name, data in models.items():
        (dest / name).write_bytes(data)
    archive.unlink()
    monkeypatch.setitem(sys.modules, 'gdown', types.SimpleNamespace())
    download._download_bat('os', dest)
    assert not archive.exists()


def test_network_fetch_uses_one_archive_and_resume(bat, monkeypatch):
    archive, dest, spec, models = bat
    payload = archive.read_bytes()
    archive.unlink()
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        Path(kwargs['output']).write_bytes(payload)
        return kwargs['output']
    monkeypatch.setitem(sys.modules, 'gdown', types.SimpleNamespace(download=fetch))
    download._download_bat('os', dest)
    assert calls == [dict(id=spec['file_id'], output=str(archive), quiet=False, resume=True)]
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == models


def test_failed_download_does_not_create_models(bat, monkeypatch):
    archive, dest, _, _ = bat
    archive.unlink()
    monkeypatch.setitem(sys.modules, 'gdown', types.SimpleNamespace(download=lambda **kw: None))
    with pytest.raises(RuntimeError, match='did not complete'):
        download._download_bat('os', dest)
    assert not dest.exists()


def test_bad_archive_checksum_is_rejected_before_extraction(bat):
    archive, dest, _, _ = bat
    payload = bytearray(archive.read_bytes())
    payload[0] ^= 1
    archive.write_bytes(payload)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        download._download_bat('os', dest)
    assert not dest.exists()


def test_existing_different_model_is_not_overwritten(bat):
    _, dest, _, models = bat
    dest.mkdir()
    existing = dest / next(iter(models))
    existing.write_bytes(b'locally trained model')
    with pytest.raises(ValueError, match='left unchanged'):
        download._download_bat('os', dest)
    assert existing.read_bytes() == b'locally trained model'
    assert len(list(dest.iterdir())) == 1


@pytest.mark.parametrize('problem', ['traversal', 'duplicate', 'missing', 'wrong_weights'])
def test_invalid_members_do_not_install_partial_models(bat, problem):
    archive, dest, spec, models = bat
    with zipfile.ZipFile(archive, 'w') as z:
        for index, (name, data) in enumerate(models.items()):
            if problem == 'missing' and index == 2:
                continue
            z.writestr('os/' + name, b'x' * len(data) if problem == 'wrong_weights' and index == 2 else data)
        if problem == 'traversal':
            z.writestr('../outside.pth', b'outside')
        if problem == 'duplicate':
            name = next(iter(models))
            z.writestr('another/' + name, models[name])
    spec.update(checksum(archive.read_bytes()))
    with pytest.raises(ValueError):
        download._download_bat('os', dest)
    assert not dest.exists()
    assert not (archive.parent / 'outside.pth').exists()
    assert not list(archive.parent.glob('.extract_*'))


def test_bat_dispatch_uses_archives_qbat_dispatch_keeps_folders(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download, '_CHECKPOINTS_DIR', tmp_path)
    monkeypatch.setattr(download, '_download_bat', lambda ds, dest: calls.append(('bat', ds, dest)))
    monkeypatch.setattr(download, '_download_folder', lambda file_id, dest: calls.append(('qbat', file_id, dest)))
    download.download(None, 'os')
    assert calls == [('bat', 'os', tmp_path / 'bat/os'),
                     ('qbat', download._DRIVE_FOLDER_IDS['qbat']['os'], tmp_path / 'qbat/os')]


def test_published_manifests_have_all_81_configurations():
    from itertools import product
    manifest = json.loads(Path(download.__file__).with_name('bat_archives.json').read_text())
    for dataset, prefix in [('os', 'Openstack'), ('hdfs', 'HDFS')]:
        spec = manifest[dataset]
        k_values = (1, 3, 5) if dataset == 'os' else (3, 4, 5)
        assert set(spec['models']) == {
            f'{prefix}_e{e}_k{k}_l{l}_b{b}_checkpoint.pth'
            for e, k, l, b in product((3, 6, 10), k_values, (3, 6, 8), (32, 64, 96))}
        assert len(spec['sha256']) == 64


def test_generated_manifest_satisfies_the_downloader(tmp_path, monkeypatch):
    """The generator must produce entries the real download path accepts."""
    import json, zipfile, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import make_bat_manifest as gen

    monkeypatch.setattr(gen, 'EXPECTED_MODELS', 3)
    archive = tmp_path / 'ensemble_os.zip'
    payload = {f'Openstack_m{i}_checkpoint.pth': bytes([i]) * (100 + i) for i in range(3)}
    with zipfile.ZipFile(archive, 'w') as z:
        for name, blob in payload.items():
            z.writestr(f'os/{name}', blob)        # models sit under a dataset folder
        z.writestr('os/README.txt', b'ignored')   # non-.pth members are skipped

    out = tmp_path / 'manifest.json'
    monkeypatch.setattr('sys.argv', ['make', '--archive', f'os={archive}',
                                     '--file-id', 'os=DRIVEID', '--output', str(out)])
    assert gen.main() == 0
    spec = json.loads(out.read_text())['os']
    assert spec['file_id'] == 'DRIVEID' and spec['filename'] == 'ensemble_os.zip'
    assert set(spec['models']) == set(payload)            # keyed by basename, no 'os/' prefix
    assert spec['bytes'] == archive.stat().st_size
    for name, blob in payload.items():
        assert spec['models'][name]['bytes'] == len(blob)

    # The downloader's own verification must accept what was written.
    download._verify_file(archive, spec)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        download._verify_file(archive, dict(spec, sha256='0' * 64))


def test_manifest_generator_rejects_a_wrong_model_count(tmp_path, monkeypatch):
    import zipfile, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import make_bat_manifest as gen
    archive = tmp_path / 'ensemble_os.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('os/only_one_checkpoint.pth', b'x')
    with pytest.raises(SystemExit, match='expected 81 models'):
        gen.describe(archive, 'ID')
