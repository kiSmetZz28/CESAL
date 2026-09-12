import argparse
import logging
import os
import sys
import time
from itertools import product

from torch.backends import cudnn

from cesal_core.utils import steps
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.io import mkdir
from cesal_core.utils.steps import StepReporter
from training_pipeline.solver import Solver

_ABOUT = """
Training the BAT ensemble: a group of models that together decide whether a
stretch of log activity is normal or not.
Each model sees the training logs with a different set of learning settings, so
they make different mistakes — and a vote across all of them is steadier than
any single model. Nothing here is labelled as an attack: every model learns only
what normal looks like, and later treats whatever does not fit as suspicious.
"""


def _run_one(config: argparse.Namespace) -> None:
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
    args, _ = parser.parse_known_args()

    yaml_config = load_config(args.config)
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

        for key in search_keys:
            st.detail(key, ", ".join(str(v) for v in yaml_config[key]))
        st.detail("saving to", save_path)
        st.outcome(**{
            "models to train": len(combinations),
            "settings varied": " × ".join(
                f"{len(yaml_config[k])} {k}" for k in search_keys
            ),
        })

    # ── Step 2: train every base model ────────────────────────────────────
    with rep.step("sweep") as st:
        st.expect("models", len(combinations))
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
                if took > slowest[1]:
                    slowest = (name, took)
                logging.info(
                    "   model %d/%d · %-18s trained in %s",
                    i + 1, len(combinations), name, steps.fmt_secs(took),
                )
            except Exception as exc:  # one bad model must not abort the sweep
                failed += 1
                logging.warning(
                    "   model %d/%d · %-18s FAILED — %s",
                    i + 1, len(combinations), name, exc,
                )
            st.tick("models")

        on_disk = 0
        if save_path and os.path.isdir(save_path):
            on_disk = len([f for f in os.listdir(save_path) if f.endswith('.pth')])

        st.outcome(**{
            "models trained": f"{trained}/{len(combinations)}",
            "models failed": failed,
            "checkpoints on disk": on_disk,
            "slowest model": f"{slowest[0]} ({steps.fmt_secs(slowest[1])})" if slowest[0] else "—",
        })
        if trained == 0:
            st.fail(f"No model finished training ({failed} failed). "
                    f"The messages above name the cause for each.")

    rep.finish(outputs=save_path)
    sys.exit(0 if trained else 1)


if __name__ == '__main__':
    setup_logging('training_bat')
    main()
