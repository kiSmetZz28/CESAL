"""Regenerate tools/bat_archives.json from the published BAT archives.

Maintainer tool. Run it whenever the BAT checkpoints on the sharing service are
replaced: the downloader verifies the archive and all 81 models against this
file, so stale entries make every download fail.

    python tools/make_bat_manifest.py \\
        --archive os=/path/ensemble_os.zip --file-id os=<drive file id> \\
        --archive hdfs=/path/ensemble_hdfs.zip --file-id hdfs=<drive file id>

Datasets left unspecified keep their existing entry. The result is verified with
the downloader's own checks before it is written.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

# Run directly as a script, as the other tools do, without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cesal_core.utils.reproducibility import sha256

MANIFEST = Path(__file__).with_name('bat_archives.json')
EXPECTED_MODELS = 81


def _pairs(values, what):
    out = {}
    for item in values or ():
        dataset, _, value = item.partition('=')
        if not value or dataset not in ('os', 'hdfs'):
            raise SystemExit(f'--{what} expects os=<value> or hdfs=<value>, got {item!r}')
        out[dataset] = value
    return out


def describe(archive, file_id):
    """Hash the archive and every model inside it, exactly as the downloader will."""
    archive = Path(archive)
    with zipfile.ZipFile(archive) as bundle:
        # Select members exactly as tools/download_checkpoints.py does: .pth files
        # anywhere in the archive, keyed by basename, with the same safety checks.
        selected = {}
        for info in bundle.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or '..' in member.parts or '\\' in info.filename:
                raise SystemExit(f'{archive.name}: unsafe member {info.filename!r}')
            if info.is_dir() or member.suffix != '.pth':
                continue
            if member.name in selected:
                raise SystemExit(f'{archive.name}: duplicate model {member.name}')
            selected[member.name] = info
        if len(selected) != EXPECTED_MODELS:
            raise SystemExit(f'{archive.name}: expected {EXPECTED_MODELS} models, found {len(selected)}')
        models = {}
        for name in sorted(selected):
            info = selected[name]
            digest = hashlib.sha256()
            with bundle.open(info) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            models[name] = dict(bytes=info.file_size, sha256=digest.hexdigest())
    return dict(bytes=archive.stat().st_size, file_id=file_id, filename=archive.name,
                models=models, sha256=sha256(archive))


def verify(spec, archive):
    """Re-check the written entry the way tools/download_checkpoints.py does."""
    from tools import download_checkpoints as dl
    dl._verify_file(Path(archive), spec)
    with zipfile.ZipFile(archive) as bundle:
        members = {PurePosixPath(i.filename).name: i for i in bundle.infolist()
                   if not i.is_dir() and PurePosixPath(i.filename).suffix == '.pth'}
        if set(members) != set(spec['models']):
            raise SystemExit('archive members do not match the generated manifest')
        for name, info in members.items():
            if info.file_size != spec['models'][name]['bytes']:
                raise SystemExit(f'size mismatch for {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', action='append', metavar='DATASET=PATH', required=True)
    parser.add_argument('--file-id', action='append', metavar='DATASET=ID', required=True)
    parser.add_argument('--output', type=Path, default=MANIFEST)
    args = parser.parse_args()
    archives, ids = _pairs(args.archive, 'archive'), _pairs(args.file_id, 'file-id')
    if set(archives) != set(ids):
        raise SystemExit('every --archive needs a matching --file-id')
    manifest = json.loads(args.output.read_text()) if args.output.exists() else {}
    for dataset, archive in archives.items():
        print(f'{dataset}: hashing {archive}', flush=True)
        spec = describe(archive, ids[dataset])
        verify(spec, archive)
        manifest[dataset] = spec
        print(f'  {spec["filename"]}  {spec["bytes"]:,} bytes  sha256={spec["sha256"][:16]}…  '
              f'{len(spec["models"])} models verified', flush=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'Wrote {args.output}. Datasets: {", ".join(sorted(manifest))}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
