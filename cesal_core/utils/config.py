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
    """Terminal formatter: bare messages, so the step banners in
    cesal_core.utils.steps stay aligned and readable.

    A warning or error raised by the pipeline itself arrives already indented
    and marked (``   ! ...``, ``   ✗ ...``) by the step reporter, so it is
    passed through untouched; prefixing it again would produce
    ``WARNING:    ! ...`` and break the column it was aligned to. Anything else
    — a warning from a library, say — is tagged with its level so it stands out
    against the pipeline's own output. The file handler keeps the full
    timestamped format for the record either way.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING and not message.startswith(" "):
            return f"{record.levelname}: {message}"
        return message


class _ConsoleHandler(logging.StreamHandler):
    """Console handler that co-operates with the in-place progress bar.

    The bar and the log share stderr, so a line written while the bar is on
    screen would other­wise be spliced into it. Erase the bar, write the line,
    then redraw it below.
    """

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

    # The file keeps the full record at DEBUG — including the per-model lines the
    # terminal summarises into milestone counters — while the console stays at
    # INFO so a run reads as a clean sequence of steps.
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    console_handler = _ConsoleHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ConsoleFormatter())

    # force=True: importing torch/executorch/torchao can install a root handler
    # before we get here, which would make basicConfig a silent no-op and leave
    # the default "WARNING:root:..." format spliced into the progress bar.
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[file_handler, console_handler], force=True)

    # Third-party DEBUG output would bury the pipeline's own record in the file.
    for noisy in ("matplotlib", "urllib3", "transformers", "filelock",
                  "huggingface_hub", "PIL", "asyncio", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
