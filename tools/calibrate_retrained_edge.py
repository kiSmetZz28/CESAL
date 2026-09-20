"""Calibrate new Q-BAT models using their bootstrapped training windows only.

This defines the new baseline's edge calibration protocol; released thresholds
are retained separately and are never copied to newly trained models.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from cesal_core.data.loaders import get_loader_segment
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.reproducibility import model_seed, seed_training, sha256, write_json
from cesal_inference_pipeline.lad_qbat_edge import _infer_model, compute_threshold_from_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', required=True)
    parser.add_argument('--inference-config', required=True)
    args = parser.parse_args()
    training = load_config(args.training_config)
    inference = load_config(args.inference_config)
    target = Path(inference['threshold_output'])
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing thresholds: {target}')
    records, models = [], []
    for item in inference['edge_models']:
        parameters = tuple(int(p[1:]) for p in Path(item['checkpoint']).stem.split('_')[1:])
        seed = model_seed(training['seed'], training['dataset'], parameters)
        seed_training(seed)
        loader = get_loader_segment(parameters, training['data_path'], batch_size=parameters[-1],
            win_size=training['win_size'], step=training['win_size'],
            data_seq_len=training['data_seq_len'], mode='train', dataset=training['dataset'])
        windows = np.concatenate([x.numpy() for x, _ in loader])
        energy = _infer_model(item['checkpoint'], windows, item['name'] + '_calibration')
        if not np.isfinite(energy).all():
            raise ValueError(f'Non-finite calibration energies for {item["name"]}')
        threshold, ratio, _ = compute_threshold_from_energy(energy)
        models.append(dict(name=item['name'], threshold=threshold))
        records.append(dict(name=item['name'], pte_sha256=sha256(item['checkpoint']),
                            seed=seed, windows=len(windows), normal_ratio=ratio))
        del windows, loader, energy
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(dict(models=models), sort_keys=False))
    write_json(target.with_suffix('.json'), dict(protocol='GMM on bootstrapped training-window energies only', models=records))


if __name__ == '__main__':
    setup_logging('calibrate_retrained_edge')
    main()
