import argparse
import logging
import time
from itertools import product

from torch.backends import cudnn

from ceco_core.utils.config import load_config, setup_logging
from ceco_core.utils.io import mkdir
from training_pipeline.solver import Solver


def _run_one(config: argparse.Namespace) -> None:
    cudnn.benchmark = True
    mkdir(config.model_save_path)
    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    setup_logging('training_bat')

    parser = argparse.ArgumentParser(description="BAT ensemble training via hyperparameter sweep.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training/os.yaml',
        help='Path to the training YAML config.',
    )
    args, _ = parser.parse_known_args()

    yaml_config = load_config(args.config)

    search_keys = ['num_epochs', 'k', 'e_layer_num', 'batch_size']
    search_space = [yaml_config[key] for key in search_keys]
    combinations = list(product(*search_space))

    base_config = {k: v for k, v in yaml_config.items() if k not in search_keys}

    logging.info("Starting BAT sweep: %d combinations from %s", len(combinations), args.config)

    for i, values in enumerate(combinations):
        config_dict = {**base_config, **dict(zip(search_keys, values))}
        config = argparse.Namespace(**config_dict)

        logging.info("\n=== Training model %d/%d ===", i + 1, len(combinations))
        for key, val in sorted(vars(config).items()):
            logging.info("  %s: %s", key, val)

        t0 = time.time()
        _run_one(config)
        logging.info("Model %d/%d done in %.2fs", i + 1, len(combinations), time.time() - t0)
