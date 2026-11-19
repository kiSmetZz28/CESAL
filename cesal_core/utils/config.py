import datetime
import logging
import os
from typing import Any, Dict

import yaml

from cesal_core.utils import steps

_LOG_DIR = "logs"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class _ConsoleFormatter(logging.Formatter):
    """Terminal formatter: bare messages so step banners stay aligned. Pipeline warnings
    arrive already marked (``   ! ...``) and pass through; others (e.g. from libraries)
    get a level prefix. The log file keeps the full timestamped format."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING and not message.startswith(" "):
            return f"{record.levelname}: {message}"
        return message


class _ConsoleHandler(logging.StreamHandler):
    """Console handler that erases the progress bar, writes the line, then redraws the bar."""

    def emit(self, record: logging.LogRecord) -> None:
        steps.clear_bar()
        try:
            super().emit(record)
        finally:
            steps.redraw_bar()


def setup_logging(prefix: str) -> None:
    """Configure root logger to write to a timestamped file under logs/ and to stdout."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_filename = os.path.join(
        _LOG_DIR,
        f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    )

    # The file keeps everything at DEBUG (including per-model lines); the console
    # stays at INFO so a run reads as a clean sequence of steps.
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    console_handler = _ConsoleHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ConsoleFormatter())

    # force=True: importing torch/executorch/torchao may install a root handler first,
    # making basicConfig a silent no-op.
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[file_handler, console_handler], force=True)

    # Third-party DEBUG output would bury the pipeline's own record in the file.
    for noisy in ("matplotlib", "urllib3", "transformers", "filelock",
                  "huggingface_hub", "PIL", "asyncio", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
