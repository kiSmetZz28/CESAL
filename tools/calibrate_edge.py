"""Copy each Q-BAT model's corresponding BAT .pth threshold, unchanged."""
import argparse
import math
from pathlib import Path

import yaml

from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.reproducibility import sha256, write_json
from training_pipeline.solver import _THRESHOLD_OUTPUT


def cloud_model_name(edge_name, dataset):
    if not edge_name.startswith('qbat_'):
        raise ValueError(f'Unexpected Q-BAT model name: {edge_name}')
    return dataset + '_' + edge_name[len('qbat_'):]


def read_cloud_thresholds(path):
    entries = yaml.safe_load(Path(path).read_text())
    result = {}
    for item in (entries or {}).get('models', []):
        name, value = item['name'], float(item['threshold'])
        if name in result or not math.isfinite(value):
            raise ValueError(f'Duplicate or non-finite cloud threshold: {name}')
        result[name] = value
    if not result:
        raise ValueError(f'No cloud thresholds found: {path}')
    return result


def edge_thresholds(cloud, names, dataset):
    result = {}
    for name in names:
        source = cloud_model_name(name, dataset)
        if source not in cloud:
            raise ValueError(f'Missing .pth cloud threshold for {name}: {source}')
        result[name] = cloud[source]
    return result


def cloud_threshold_path(training):
    """Where this training config writes its cloud thresholds.

    Released configs leave ``threshold_output`` unset and rely on the solver's
    per-dataset default, so fall back to the same mapping the solver uses.
    """
    configured = training.get('threshold_output')
    if configured:
        return Path(configured)
    try:
        return Path(_THRESHOLD_OUTPUT[training['dataset']][0])
    except KeyError:
        raise ValueError(f"No cloud threshold path for dataset {training['dataset']!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', required=True)
    parser.add_argument('--inference-config', required=True)
    parser.add_argument('--force', action='store_true',
                        help='Re-derive the edge thresholds even if the file already exists.')
    args = parser.parse_args()
    training = load_config(args.training_config)
    inference = load_config(args.inference_config)
    target = Path(inference['threshold_output'])
    if target.exists() and not args.force:
        raise FileExistsError(f'Refusing to replace existing thresholds: {target}. Pass --force to re-derive them.')
    source = cloud_threshold_path(training)
    if source.resolve() != Path(inference['cloud']['thresholds_yaml']).resolve():
        raise ValueError('Training and inference must use the same cloud threshold file.')
    values = edge_thresholds(read_cloud_thresholds(source),
                             [item['name'] for item in inference['edge_models']], training['dataset'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(dict(models=[dict(name=name, threshold=value)
                                                 for name, value in values.items()]), sort_keys=False))
    write_json(target.with_suffix('.json'), dict(protocol='Reuse corresponding BAT .pth cloud threshold unchanged',
                                                source=str(source), source_sha256=sha256(source)))
    print(f'Copied {len(values)} corresponding BAT thresholds to {target}')


if __name__ == '__main__':
    setup_logging('calibrate_edge')
    main()
