import argparse
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

from torch.backends import cudnn

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_core.utils.steps import StepReporter
from training_pipeline.solver import Solver
from cesal_core.utils.reproducibility import model_seed, seed_training, environment, sha256, write_json

_ABOUT = """
Train the BAT ensemble: a set of EM-AT base learners whose votes decide whether
a window of log activity is normal.

Each learner sees the same training logs under a different hyper-parameter
setting, so their errors are not correlated and a vote across the ensemble is
steadier than any single member. Training is unsupervised — no attack is ever
labelled. Each learner models normal behaviour only, and whatever it cannot
reconstruct afterwards is treated as anomalous.
"""


def _run_one(config: argparse.Namespace) -> None:
    if getattr(config, 'seed', None) is not None:
        parameters = (config.num_epochs, config.k, config.e_layer_num, config.batch_size)
        seed = model_seed(config.seed, config.dataset, parameters)
        seed_training(seed)
        logging.info('   Reproducible training seed: %d', seed)
    else:
        cudnn.benchmark = True
    mkdir(config.model_save_path)
    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


def main() -> None:
    parser = argparse.ArgumentParser(description="BAT ensemble training via hyperparameter sweep.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training/os.yaml',
        help='Path to the training YAML config.',
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Master seed for deterministic per-learner training (default: 42).')
    parser.add_argument('--output-dir', help='Separate directory for newly trained checkpoints.')
    args, _ = parser.parse_known_args()

    yaml_config = load_config(args.config)
    yaml_config['seed'] = args.seed
    if args.output_dir:
        yaml_config['model_save_path'] = args.output_dir
    dataset = yaml_config.get('dataset', '')

    rep = StepReporter("train", dataset=dataset, steps=steps.TRAIN_STEPS, about=_ABOUT)

    search_keys = ['num_epochs', 'k', 'e_layer_num', 'batch_size']

    # ── Step 1: work out the sweep ────────────────────────────────────────
    with rep.step("plan") as st:
        st.detail("config", args.config)
        search_space = [yaml_config[key] for key in search_keys]
        combinations = list(product(*search_space))
        base_config = {k: v for k, v in yaml_config.items() if k not in search_keys}
        save_path = base_config.get('model_save_path', '')
        if list(Path(save_path).glob('*.pth')):
            raise FileExistsError(f'Checkpoints already exist in {save_path}. '
                                  'Choose a new --output-dir; existing models are preserved.')
        root = Path(__file__).resolve().parent.parent
        source_files = [Path(__file__), Path(__file__).with_name('solver.py')]
        source_files += list((root / 'cesal_core').rglob('*.py'))
        source_files += sorted((root / 'configs/training').glob('*.yaml'))
        splits = sorted(Path(base_config['data_path']).glob('*.txt')) if 'data_path' in base_config else []
        seed_training(args.seed)
        manifest = dict(master_seed=args.seed, config=yaml_config, environment=environment(),
                        bootstrap='Existing fixed per-combination sampling seeds from configs/training; unchanged',
                        inputs={str(p): sha256(p) for p in splits},
                        source_sha256={str(p.relative_to(root)): sha256(p) for p in source_files},
                        learners={}, status='running')
        write_json(Path(save_path) / 'training_manifest.json', manifest)

        for key in search_keys:
            st.detail(key, ", ".join(str(v) for v in yaml_config[key]))
        st.detail("saving to", save_path)
        st.outcome(**{
            "learners to train": len(combinations),
            "settings varied": " × ".join(
                f"{len(yaml_config[k])} {k}" for k in search_keys
            ),
        })

    # ── Step 2: train every base model ────────────────────────────────────
    with rep.step("sweep") as st:
        st.expect("learners", len(combinations))
        trained, failed = 0, 0
        slowest = ("", 0.0)

        for i, values in enumerate(combinations):
            config_dict = {**base_config, **dict(zip(search_keys, values))}
            config = argparse.Namespace(**config_dict)
            name = (f"e{config.num_epochs}_k{config.k}"
                    f"_l{config.e_layer_num}_b{config.batch_size}")

            # Name the model on the bar so it is clear which one is running.
            st.progress_note(name)
            logging.debug("Model %d/%d config: %s", i + 1, len(combinations),
                          sorted(vars(config).items()))

            t0 = time.time()
            try:
                _run_one(config)
                took = time.time() - t0
                trained += 1
                checkpoint = Path(save_path) / f'{dataset}_{name}_checkpoint.pth'
                manifest['learners'][name] = dict(seed=model_seed(args.seed, dataset, values),
                    status='complete', sha256=sha256(checkpoint) if checkpoint.exists() else None)
                if took > slowest[1]:
                    slowest = (name, took)
                logging.info("   %7s  %-22s trained in %s",
                             f"{i + 1}/{len(combinations)}", name,
                             steps.fmt_secs(took))
            except Exception as exc:  # one bad learner must not abort the sweep
                logging.debug('Learner training failure', exc_info=True)
                failed += 1
                manifest['learners'][name] = dict(status='failed', error=str(exc))
                st.warn(f"{name} ({i + 1}/{len(combinations)}) failed to "
                        f"train — {exc}")
            st.tick("learners")
            write_json(Path(save_path) / 'training_manifest.json', manifest)

        on_disk = 0
        if save_path and os.path.isdir(save_path):
            on_disk = len([f for f in os.listdir(save_path) if f.endswith('.pth')])

        st.outcome(**{
            "learners trained": f"{trained}/{len(combinations)}",
            "learners failed": failed,
            "checkpoints on disk": on_disk,
            "slowest learner": f"{slowest[0]} ({steps.fmt_secs(slowest[1])})" if slowest[0] else "—",
        })
        if trained == 0:
            st.fail(f"No model finished training ({failed} failed). "
                    f"The messages above name the cause for each.")
        elif failed:
            st.fail(f"Training is incomplete: {failed} of {len(combinations)} learners failed. "
                    "Successfully trained checkpoints have been kept.")

    manifest['status'] = 'complete' if trained and not failed else 'failed'
    write_json(Path(save_path) / 'training_manifest.json', manifest)
    rep.finish(outputs=save_path)
    sys.exit(0 if trained and not failed else 1)


if __name__ == '__main__':
    setup_logging('training_bat')
    main()
