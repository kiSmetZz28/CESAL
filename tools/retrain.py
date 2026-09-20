"""Run a new seeded baseline without replacing released models or thresholds."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from cesal_core.utils.reproducibility import sha256, write_json

ROOT = Path(__file__).resolve().parent.parent


def prepare(dataset, destination, seed):
    """Snapshot independent training/inference configs for the new baseline."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT / f'configs/training/{dataset}.yaml').read_text())
    cfg.update(seed=seed, data_path=str(ROOT / cfg['data_path']),
               model_save_path=str(destination / 'checkpoints/bat'),
               threshold_output=str(destination / 'results/thresholds_cloud.yaml'))
    train_path = destination / 'training.yaml'
    train_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    inference = yaml.safe_load((ROOT / f'configs/inference/{dataset}.yaml').read_text())
    inference.update(data_path=cfg['data_path'], output_dir=str(destination / 'results'),
                     threshold_output=str(destination / 'results/thresholds_edge.yaml'))
    for model in inference['edge_models']:
        model['checkpoint'] = str(destination / 'checkpoints/qbat' / Path(model['checkpoint']).name)
    inference['cloud'].update(model_save_path=cfg['model_save_path'],
                              thresholds_yaml=cfg['threshold_output'], seed=seed,
                              max_parallel_models=1)
    (destination / 'inference.yaml').write_text(yaml.safe_dump(inference, sort_keys=False))
    return train_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', choices=('os', 'hdfs'), default=['os', 'hdfs'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', help='New directory for the entire run.')
    parser.add_argument('--cloud-python', default=os.environ.get('CESAL_CLOUD_PYTHON', sys.executable))
    parser.add_argument('--edge-python', default=os.environ.get('EDGE_PYTHON', '/opt/conda/envs/cesal-edge/bin/python'))
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination = Path(args.output_dir or ROOT / f'outputs/retraining/seed_{args.seed}_{timestamp}').resolve()
    destination.mkdir(parents=True, exist_ok=False)
    status = dict(seed=args.seed, datasets=args.datasets, stages=[], status='running')
    write_json(destination / 'status.json', status)
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(args.seed), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               CESAL_CLOUD_PYTHON=args.cloud_python)
    print(f'New baseline directory: {destination}', flush=True)
    try:
        # Verify both interpreters before committing to the long training run.
        subprocess.run([args.cloud_python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA training GPU unavailable"'],
                       cwd=ROOT, env=env, check=True)
        subprocess.run([args.edge_python, '-c', 'import executorch.exir, torchao.quantization'],
                       cwd=ROOT, env=env, check=True)
        for dataset in args.datasets:
            out = destination / dataset
            config = prepare(dataset, out, args.seed)
            stages = [
                ('train', args.cloud_python, '-m', 'training_pipeline.train', '--config', str(config), '--seed', str(args.seed)),
                ('convert', args.edge_python, 'quantization/qbat_export.py', '--config', str(config), '--all',
                 '--seed', str(args.seed), '--output_dir', str(out / 'checkpoints/qbat')),
                ('calibrate_cloud', args.cloud_python, '-m', 'training_pipeline.evaluate', '--config', str(config)),
                ('calibrate_edge', args.edge_python, '-m', 'tools.calibrate_retrained_edge', '--training-config', str(config),
                 '--inference-config', str(out / 'inference.yaml')),
                ('infer', args.edge_python, '-m', 'cesal_inference_pipeline.run', '--config', str(out / 'inference.yaml')),
            ]
            for stage, *command in stages:
                stage_env = dict(env, PATH=str(Path(command[0]).resolve().parent) + os.pathsep + env.get('PATH', ''))
                log = out / f'{stage}.log'
                record = dict(dataset=dataset, stage=stage, command=command, log=str(log), status='running')
                status['stages'].append(record)
                write_json(destination / 'status.json', status)
                print(f'{dataset}: {stage} running; log: {log}', flush=True)
                with log.open('w') as stream:
                    result = subprocess.run(command, cwd=ROOT, env=stage_env, stdout=stream, stderr=subprocess.STDOUT)
                record.update(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode)
                write_json(destination / 'status.json', status)
                if result.returncode:
                    raise RuntimeError(f'{dataset} {stage} failed; inspect {log}')
            write_json(out / 'artifact_checksums.json', {
                str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                if p.suffix in ('.pth', '.pte', '.yaml')})
        status['status'] = 'complete'
    except BaseException as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(destination / 'status.json', status)


if __name__ == '__main__':
    main()
