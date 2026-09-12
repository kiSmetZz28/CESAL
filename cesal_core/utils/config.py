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


class _ConsoleFormatter(logging.Formatter):
    """Terminal formatter: bare messages, so the step banners in
    cesal_core.utils.steps stay aligned and readable.

    Only WARNING and above are tagged with their level — those must stand out.
    The file handler keeps the full timestamped format for the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def setup_logging(prefix: str) -> None:
    """Configure root logger to write to a timestamped file under logs/ and to stdout."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_filename = os.path.join(
        _LOG_DIR,
        f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    )

    # The file keeps the full record at DEBUG — including the per-model lines the
    # terminal summarises into milestone counters — while the console stays at
    # INFO so a run reads as a clean sequence of steps.
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ConsoleFormatter())

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    # Third-party DEBUG output would bury the pipeline's own record in the file.
    for noisy in ("matplotlib", "urllib3", "transformers", "filelock",
                  "huggingface_hub", "PIL", "asyncio", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
