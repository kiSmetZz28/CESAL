import datetime
import logging
import os
from typing import Any, Dict

import yaml

_LOG_DIR = "logs"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(prefix: str) -> None:
    """Configure root logger to write to a timestamped file under logs/ and to stdout."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_filename = os.path.join(
        _LOG_DIR,
        f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
