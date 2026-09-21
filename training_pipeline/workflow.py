"""Train a seeded model set end to end, apart from the released artifacts.

Checkpoints land under ``checkpoints/baselines/<run>/`` and everything else
under ``outputs/baselines/<run>/``, so neither the published models nor the
published results are touched.
"""
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


def prepare(dataset, models_dir, results_dir, seed):
    """Snapshot independent training/inference configs for the new model set.

    Models go under ``models_dir`` and everything else under ``results_dir``,
    matching the repository's split between checkpoints and outputs.
    """
    models_dir, results_dir = Path(models_dir).resolve(), Path(results_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT / f'configs/training/{dataset}.yaml').read_text())
    cfg.update(seed=seed, data_path=str(ROOT / cfg['data_path']),
               model_save_path=str(models_dir / 'bat'),
               threshold_output=str(results_dir / 'results/thresholds_cloud.yaml'))
    train_path = results_dir / 'training.yaml'
    train_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    inference = yaml.safe_load((ROOT / f'configs/inference/{dataset}.yaml').read_text())
    inference.update(data_path=cfg['data_path'], output_dir=str(results_dir / 'results'),
                     threshold_output=str(results_dir / 'results/thresholds_edge.yaml'))
    # The edge architectures are kept from the released inference config, so a new
    # run quantizes the same three learners rather than searching for a trio.
    for model in inference['edge_models']:
        model['checkpoint'] = str(models_dir / 'qbat' / Path(model['checkpoint']).name)
    inference['cloud'].update(model_save_path=cfg['model_save_path'],
                              thresholds_yaml=cfg['threshold_output'], seed=seed,
                              max_parallel_models=1)
    (results_dir / 'inference.yaml').write_text(yaml.safe_dump(inference, sort_keys=False))
    return train_path


def stages_for(dataset, out, models, config, args):
    return [
        ('train', args.cloud_python, '-m', 'training_pipeline.train',
         '--config', str(config), '--seed', str(args.seed)),
        ('evaluate_bat', args.cloud_python, '-m', 'training_pipeline.evaluate',
         '--config', str(config), '--report'),
        ('convert', args.edge_python, 'quantization/qbat_export.py', '--config', str(config), '--all',
         '--seed', str(args.seed), '--output_dir', str(models / 'qbat')),
        ('calibrate_edge', args.edge_python, '-m', 'tools.calibrate_edge',
         '--training-config', str(config), '--inference-config', str(out / 'inference.yaml')),
        ('infer', args.edge_python, '-m', 'cesal_inference_pipeline.run',
         '--config', str(out / 'inference.yaml')),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', metavar='DATASET', help='os (default) or hdfs.')
    parser.add_argument('--seed', type=int, default=42, help='Master seed (default: 42).')
    parser.add_argument('--output-dir', help='Directory for configs, thresholds, results and logs.')
    parser.add_argument('--models-dir', help='Directory for the new checkpoints.')
    parser.add_argument('--cloud-python', default=os.environ.get('CESAL_CLOUD_PYTHON', sys.executable))
    parser.add_argument('--edge-python', default=os.environ.get('EDGE_PYTHON', '/opt/conda/envs/cesal-edge/bin/python'))
    args = parser.parse_args()
    args.datasets = args.datasets or ['os']
    if any(dataset not in ('os', 'hdfs') for dataset in args.datasets):
        parser.error('Datasets must be os or hdfs.')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Specify each dataset only once.')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 4294967295.')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run = f'seed_{args.seed}_{timestamp}'
    destination = Path(args.output_dir or ROOT / 'outputs/baselines' / run).resolve()
    models_root = Path(args.models_dir or ROOT / 'checkpoints/baselines' / run).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    status = dict(seed=args.seed, datasets=args.datasets, stages=[], status='running')
    write_json(destination / 'status.json', status)
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(args.seed), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               CESAL_CLOUD_PYTHON=args.cloud_python)
    status['models_dir'] = str(models_root)
    print(f'Checkpoints: {models_root}', flush=True)
    print(f'Results and logs: {destination}', flush=True)
    try:
        # Verify both interpreters before committing to the long training run.
        subprocess.run([args.cloud_python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA training GPU unavailable"'],
                       cwd=ROOT, env=env, check=True)
        subprocess.run([args.edge_python, '-c', 'import executorch.exir, torchao.quantization'],
                       cwd=ROOT, env=env, check=True)
        for dataset in args.datasets:
            out, models = destination / dataset, models_root / dataset
            config = prepare(dataset, models, out, args.seed)
            for stage, *command in stages_for(dataset, out, models, config, args):
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
                str(p.relative_to(root)): sha256(p) for root in (out, models)
                for p in root.rglob('*') if p.suffix in ('.pth', '.pte', '.yaml')})
        status['status'] = 'complete'
        scores = out / 'results/bat_evaluation.json'
        print(f'Training complete: {destination}', flush=True)
        if scores.exists():
            print(f"Measured BAT scores: {json.loads(scores.read_text())['scores']}", flush=True)
    except BaseException as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(destination / 'status.json', status)
    return 0


if __name__ == '__main__':
    sys.exit(main())
