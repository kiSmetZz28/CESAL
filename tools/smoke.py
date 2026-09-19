"""A small real OpenStack detection experiment using published checkpoints."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def run_pipeline(config, output):
    """Show execution progress only; retain the unchanged pipeline output for diagnostics."""
    stages = {
        'edge': 'Data preprocessing and Q-BAT execution',
        'route': 'Event routing',
        'cloud': 'BAT cloud verification',
        'hybrid': 'Prediction merge and scoring',
    }
    command = [sys.executable, '-m', 'cesal_inference_pipeline.run', '--config', str(config)]
    # This environment belongs only to the smoke child and its cloud process.
    # Normal infer/eval output and all scoring calculations remain unchanged.
    env = dict(os.environ, CESAL_EVENTS='1', PYTHONUNBUFFERED='1')
    with (output / 'pipeline.log').open('w', encoding='utf-8') as log:
        log.write('SMOKE READINESS DIAGNOSTICS — subset metrics below are not paper-result estimates.\n'
                  'Use python run.py infer os for full detection evaluation.\n\n')
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if not line.startswith('@@CESAL '):
                    continue
                try:
                    event = json.loads(line[len('@@CESAL '):])
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get('id') not in stages:
                    continue
                title = stages[event['id']]
                kind = event.get('t')
                if kind == 'step_start':
                    print(f'RUNNING — {title}', flush=True)
                elif kind == 'step_done':
                    print(f'COMPLETED — {title}', flush=True)
                elif kind == 'step_progress':
                    print(f'  Execution progress — {title}: '
                          f'{event["done"]}/{event["total"]} {event["unit"]}', flush=True)
                elif kind == 'step_warn':
                    print(f'ATTENTION — {title}: {event["msg"]}', flush=True)
                elif kind == 'step_fail':
                    print(f'NEEDS ATTENTION — {title}: {event["error"]}', flush=True)
                # Metric events, prediction counts, and metric summary tables
                # stay in the diagnostic log, not the readiness display.
            returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def sample_indices(normal_events, abnormal_events, window_size):
    """Evenly sample five complete normal and five complete abnormal windows."""
    normal_end = normal_events // window_size
    abnormal_start = (normal_events + window_size - 1) // window_size
    test_end = (normal_events + abnormal_events) // window_size
    if normal_end < 5 or test_end - abnormal_start < 5:
        raise ValueError('The small experiment needs at least five full windows of each label.')
    return np.r_[np.linspace(0, normal_end - 1, 5, dtype=int),
                 np.linspace(abnormal_start, test_end - 1, 5, dtype=int)].tolist()


def prepare(output):
    """Create isolated inputs/config; published data, thresholds and models stay read-only."""
    cfg = yaml.safe_load((ROOT / 'configs/inference/os.yaml').read_text())
    cloud = cfg['cloud']
    # A small BAT ensemble keeps this a practical installation check on CPU.
    cloud.update(num_epochs=[3], k=[1], e_layer_num=[3], batch_size=[32, 64, 96])
    required = [ROOT / model['checkpoint'] for model in cfg['edge_models']]
    required += [ROOT / cloud['model_save_path'] / f'Openstack_e3_k1_l3_b{batch}_checkpoint.pth'
                 for batch in cloud['batch_size']]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path.relative_to(ROOT)}. Run python run.py download os first.')

    data = output / 'data'
    data.mkdir()
    original_data = ROOT / cfg['data_path']
    for name in ('train.txt', 'test_normal.txt', 'test_abnormal.txt'):
        (data / name).write_bytes((original_data / name).read_bytes())
    counts = []
    for name in ('test_normal.txt', 'test_abnormal.txt'):
        with (data / name).open() as source:
            counts.append(sum(len(line.split()) for line in source))
    cfg['test_window_indices'] = sample_indices(*counts, cfg['win_size'])

    references = {}
    for owner, key, filename, names in (
        (cfg, 'threshold_output', 'thresholds_edge.yaml', [m['name'] for m in cfg['edge_models']]),
        (cloud, 'thresholds_yaml', 'thresholds_cloud.yaml',
         [f'Openstack_e3_k1_l3_b{batch}' for batch in cloud['batch_size']]),
    ):
        source = ROOT / owner[key]
        content = source.read_bytes()
        stored = {row['name'] for row in yaml.safe_load(content)['models']}
        if not set(names).issubset(stored):
            raise ValueError(f'Missing model thresholds in {source}. Restore the published threshold file.')
        (output / filename).write_bytes(content)
        references[str(source.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
        owner[key] = str(output / filename)

    for model in cfg['edge_models']:
        model['checkpoint'] = str(ROOT / model['checkpoint'])
    cloud['model_save_path'] = str(ROOT / cloud['model_save_path'])
    cfg.update(data_path=str(data), output_dir=str(output), routing_tolerance=0.1)
    config = output / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = {
        'purpose': 'Small real detection experiment; not a reproduction of Table 3 scores.',
        'dataset': 'OpenStack', 'selection': 'Five evenly spaced full windows per label, selected after full preprocessing.',
        'source_window_indices': cfg['test_window_indices'],
        'training': 'Complete bundled training split; existing preprocessing and normalization.',
        'test_events': 1000, 'normal_events': 500, 'abnormal_events': 500,
        'window_size': 100, 'windows': 10, 'routing_ratio': 0.1,
        'edge_models': [m['name'] for m in cfg['edge_models']],
        'cloud_models': [f'Openstack_e3_k1_l3_b{b}' for b in cloud['batch_size']],
        'source_threshold_sha256': references,
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.txt')},
    }
    (output / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return config, manifest


def verify(output):
    """Require real edge, routing, cloud and hybrid outputs with aligned shapes."""
    from cesal_inference_pipeline.cloud_runner import _point_adjust

    gt = np.load(output / 'ground_truth.npy')
    np.testing.assert_array_equal(gt, np.r_[np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    for name, shape in (('energy_matrix.npy', (1000, 3)), ('edge_preds_raw.npy', (1000,)),
                        ('edge_preds.npy', (1000,)),
                        ('routed_indices.npy', (100,)), ('routed_lines.npy', (100, 10)),
                        ('cloud_preds.npy', (100,)), ('hybrid_preds.npy', (1000,))):
        values = np.load(output / name)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'{name}: expected finite values with shape {shape}, got {values.shape}.')
        if 'preds' in name and not np.isin(values, [0, 1]).all():
            raise ValueError(f'{name} contains non-binary predictions.')
    routed = np.load(output / 'routed_indices.npy')
    if len(np.unique(routed)) != 100 or not ((routed >= 0) & (routed < 1000)).all():
        raise ValueError('Routing indices are not 100 distinct positions in the test input.')
    edge = np.load(output / 'edge_preds_raw.npy')
    np.testing.assert_array_equal(np.load(output / 'edge_preds.npy'), _point_adjust(gt, edge))
    merged = edge.copy()
    merged[routed] = np.load(output / 'cloud_preds.npy')
    np.testing.assert_array_equal(np.load(output / 'hybrid_preds.npy'), _point_adjust(gt, merged))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['os'], default='os')
    parser.parse_args()
    base = ROOT / 'outputs' / 'smoke' / 'os'
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run_', dir=base))
    try:
        config, manifest = prepare(output)
        print('Small real OpenStack detection experiment', flush=True)
        print('1,000 events → 10 windows → 3 Q-BAT models → 10% routing → 3 BAT models → merge', flush=True)
        print('Readiness check only — detection scores are not displayed. Use run.py infer os for paper-result evaluation.', flush=True)
        print(f'Detailed diagnostics: {(output / "pipeline.log").relative_to(ROOT)}\n', flush=True)
        run_pipeline(config, output)
        verify(output)
        for name, digest in manifest['source_threshold_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'{name} changed during the experiment; check for another running evaluation.')
        (output / 'READY.txt').write_text(
            'Real OpenStack preprocessing, Q-BAT inference, routing, BAT verification and hybrid scoring completed.\n'
            'This validates the small detection configuration. Full-ensemble and LLM evaluation are separate.\n'
        )
        print('\nVERIFIED — real data preprocessing and edge model execution')
        print('VERIFIED — routing and cloud model execution')
        print('VERIFIED — prediction alignment, merge and scoring')
        print('Small detection experiment complete. Ready to proceed to the full detection evaluation.')
        print(f'Diagnostics and experiment record: {output.relative_to(ROOT)}')
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Experiment did not complete: {exc}', file=sys.stderr)
        print(f'Diagnostics and any completed outputs: {output}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
