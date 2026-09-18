"""Cross-platform entry point for CESAL pipelines.

Usage
-----
  python run.py check                      # software checks, grouped by component
  python run.py smoke    [os]              # small real edge-to-cloud experiment
  python run.py all      [DATASET]          # everything, start to finish
  python run.py download [DATASET] [TYPE]
  python run.py train    [DATASET]
  python run.py eval     [DATASET] [VOTING]
  python run.py convert  [DATASET]
  python run.py infer    [DATASET] [RATIO]
  python run.py sweep    [DATASET] [RATIOS]
  python run.py classify [MODELS]
  python run.py respond  [MODEL]
  python run.py help

Examples
--------
  python run.py all hdfs                  # checkpoints → detection → classification → response
  python run.py download                  # Download all pre-trained checkpoints
  python run.py download hdfs             # Download HDFS checkpoints only
  python run.py download hdfs bat         # Download HDFS full-precision only
  python run.py train hdfs
  python run.py eval hdfs all
  python run.py convert hdfs
  python run.py infer hdfs
  python run.py infer hdfs 0.2                       # escalate 20% to the cloud
  python run.py sweep hdfs 0.05,0.1,0.2,0.3         # routing-ratio table
  python run.py classify                            # all LLMs in configs/llm/hdfs.yaml
  python run.py classify qwen2.5-14b-instruct       # one LLM backbone
  python run.py respond                             # HDFS detections → anomaly queues → classification → workflows
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

    if command == "check":
        _run_module("tools.check_install")

    elif command == "smoke":
        _run_module("tools.smoke", "--dataset", argv[1] if len(argv) > 1 else "os")

    elif command == "download":
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
            from tools.setup_executorch import setup_executorch
            ok = setup_executorch()
            sys.exit(0 if ok else 1)
        sys.exit(0)

    elif command == "train":
        dataset = argv[1] if len(argv) > 1 else "os"
        print(f"[run] Training BAT ensemble — dataset: {dataset}")
        _run_module("training_pipeline.train", "--config", f"configs/training/{dataset}.yaml")

    elif command == "eval":
        dataset = argv[1] if len(argv) > 1 else "os"
        # Majority voting is what the paper reports; pass "all", "consensus" or
        # "at least one" as a third argument to evaluate the other rules too.
        voting = argv[2] if len(argv) > 2 else "majority"
        print(f"[run] Evaluating BAT ensemble — dataset: {dataset}  voting: {voting}")
        _run_module("training_pipeline.evaluate",
                    "--config", f"configs/training/{dataset}.yaml",
                    "--voting", voting)

    elif command == "convert":
        dataset = argv[1] if len(argv) > 1 else "os"
        print(f"[run] Quantizing EM-AT checkpoints → Q-BAT — dataset: {dataset}")
        _run(str(_ROOT / "quantization" / "qbat_export.py"),
             "--config", f"configs/training/{dataset}.yaml", "--all")

    elif command == "infer":
        dataset = argv[1] if len(argv) > 1 else "os"
        extra = ["--ratio", argv[2]] if len(argv) > 2 else []
        _run_module("cesal_inference_pipeline.run",
                    "--config", f"configs/inference/{dataset}.yaml", *extra)

    elif command == "sweep":
        dataset = argv[1] if len(argv) > 1 else "os"
        ratios = argv[2] if len(argv) > 2 else "0.05,0.1,0.2,0.3"
        _run_module("cesal_inference_pipeline.sweep",
                    "--config", f"configs/inference/{dataset}.yaml", "--ratios", ratios)

    elif command == "classify":
        models = argv[1] if len(argv) > 1 else None
        print(f"[run] LLM open-set incident classification — models: {models or 'all in configs/llm/hdfs.yaml'}")
        _run_module("incident_response.evaluate", "--config", "configs/llm/hdfs.yaml",
                    *(["--models", models] if models else []))

    elif command == "respond":
        model = argv[1] if len(argv) > 1 else None
        # queues.py runs step 1 and process_queues.py runs steps 2-4. The handoff
        # file carries the first process's step records to the second so the two
        # report as one numbered run (see cesal_core/utils/steps.py).
        import os
        os.environ.setdefault(
            "CESAL_STEP_HANDOFF",
            str(_ROOT / "outputs" / "hdfs" / "llm" / "queues" / ".run_steps.json"),
        )
        rc = _run_keep_going("-m", "incident_response.queues", "--config", "configs/llm/hdfs.yaml")
        if rc != 0:
            sys.exit(rc)
        _run_module("incident_response.process_queues", "--config", "configs/llm/hdfs.yaml",
                    *(["--model", model] if model else []))

    elif command == "all":
        # One command from a clean checkout to the paper's numbers: fetch the
        # published checkpoints, run detection, then classify each detected
        # incident and select its response workflow. Each stage is the same
        # command documented separately below, run in order.
        dataset = argv[1] if len(argv) > 1 else "hdfs"
        print(f"[run] Full pipeline — dataset: {dataset}\n")

        sys.path.insert(0, str(_ROOT))
        from tools.setup_executorch import executorch_present, setup_executorch

        bat  = _ROOT / "checkpoints" / "bat" / dataset
        qbat = _ROOT / "checkpoints" / "qbat" / dataset
        have = (len(list(bat.glob("*.pth"))) >= 81 if bat.is_dir() else False) and \
               (len(list(qbat.glob("*.pte"))) >= 3 if qbat.is_dir() else False) and \
               executorch_present()

        if have:
            # Checking here rather than inside the fetcher keeps the common case
            # fast: re-listing the remote folder takes minutes even when every
            # file is already on disk.
            print("[run] 1/3  Checkpoints and edge runtime — already present, skipping")
        else:
            print("[run] 1/3  Checkpoints and edge runtime")
            rc = _run_keep_going(str(_ROOT / "tools" / "download_checkpoints.py"),
                                 "--dataset", dataset)
            if rc != 0:
                sys.exit(rc)
            if not setup_executorch():
                sys.exit(1)

        print("\n[run] 2/3  Detection")
        rc = _run_keep_going("-m", "cesal_inference_pipeline.run",
                             "--config", f"configs/inference/{dataset}.yaml")
        if rc != 0:
            sys.exit(rc)

        if dataset != "hdfs":
            print(f"\n[run] 3/3  Classification and response — skipped: the open-set "
                  f"anomaly types and knowledge base are HDFS-only.")
            sys.exit(0)

        # respond needs torch + transformers, so it runs in the cloud environment.
        from cesal_inference_pipeline.run import _detect_cloud_python
        cloud_py = _detect_cloud_python()
        print(f"\n[run] 3/3  Classification and response  ({cloud_py})")
        rc = subprocess.run([cloud_py, str(_ROOT / "run.py"), "respond"],
                            cwd=str(_ROOT)).returncode
        sys.exit(rc)

    elif command in ("help", "--help", "-h"):
        print(__doc__)

    else:
        print(f"[run] Unknown command: '{command}'", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
