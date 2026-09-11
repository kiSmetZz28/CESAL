"""Cross-platform entry point for CECO-LAD pipelines.

Usage
-----
  python run.py download [DATASET] [TYPE]
  python run.py train    [DATASET]
  python run.py eval     [DATASET] [VOTING]
  python run.py convert  [DATASET]
  python run.py infer    [DATASET]
  python run.py help

Examples
--------
  python run.py download                  # Download all pre-trained checkpoints
  python run.py download bgl              # Download BGL checkpoints only
  python run.py download bgl bat          # Download BGL full-precision only
  python run.py train bgl
  python run.py eval bgl all
  python run.py convert bgl
  python run.py infer bgl
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PY = sys.executable  # same interpreter that launched this script


def _run(*args: str) -> None:
    result = subprocess.run([_PY, *args], cwd=str(_ROOT))
    sys.exit(result.returncode)


def _run_keep_going(*args: str) -> int:
    """Like _run, but returns the exit code instead of exiting the process."""
    return subprocess.run([_PY, *args], cwd=str(_ROOT)).returncode


def _run_module(module: str, *args: str) -> None:
    _run("-m", module, *args)


def main() -> None:
    argv = sys.argv[1:]
    command = argv[0] if argv else "help"

    if command == "download":
        dataset = argv[1] if len(argv) > 1 else None
        ckpt_type = argv[2] if len(argv) > 2 else None
        extra = []
        if dataset:
            extra += ["--dataset", dataset]
        if ckpt_type:
            extra += ["--type", ckpt_type]
        print(f"[run] Downloading checkpoints — dataset: {dataset or 'all'}  type: {ckpt_type or 'both'}")
        rc = _run_keep_going(str(_ROOT / "tools" / "download_checkpoints.py"), *extra)
        if rc != 0:
            sys.exit(rc)

        # `.pte` files are useless without the ExecuTorch runtime, so when
        # qbat checkpoints are in the download set, also fetch and install
        # the runtime so that subsequent `run.py infer` / `run.py convert`
        # have everything they need.
        if ckpt_type != "bat":
            print("[run] Ensuring ExecuTorch runtime is installed (needed for infer / convert) …")
            sys.path.insert(0, str(_ROOT))
            from launch_dashboard import setup_executorch
            ok = setup_executorch()
            sys.exit(0 if ok else 1)
        sys.exit(0)

    elif command == "train":
        dataset = argv[1] if len(argv) > 1 else "bgl"
        print(f"[run] Training BAT ensemble — dataset: {dataset}")
        _run_module("training_pipeline.train", "--config", f"configs/training/{dataset}.yaml")

    elif command == "eval":
        dataset = argv[1] if len(argv) > 1 else "bgl"
        voting = argv[2] if len(argv) > 2 else "all"
        print(f"[run] Evaluating BAT ensemble — dataset: {dataset}  voting: {voting}")
        _run_module("training_pipeline.evaluate",
                    "--config", f"configs/training/{dataset}.yaml",
                    "--voting", voting)

    elif command == "convert":
        dataset = argv[1] if len(argv) > 1 else "bgl"
        print(f"[run] Converting BAT → Q-BAT — dataset: {dataset}")
        _run(str(_ROOT / "quantization" / "qbat_export.py"),
             "--config", f"configs/training/{dataset}.yaml", "--all")

    elif command == "infer":
        dataset = argv[1] if len(argv) > 1 else "bgl"
        print(f"[run] Inference pipeline — dataset: {dataset}")
        _run_module("ceco_lad_inference_pipeline.run",
                    "--config", f"configs/inference/{dataset}.yaml")

    elif command in ("help", "--help", "-h"):
        print(__doc__)

    else:
        print(f"[run] Unknown command: '{command}'", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
