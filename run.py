"""Cross-platform entry point for CESAL pipelines.

Usage
-----
  python run.py check                      # software checks, grouped by component
  python run.py smoke    [os]              # small real edge-to-cloud experiment
  python run.py all      [DATASET]          # download → detect  (hdfs also classifies + responds)
  python run.py download [DATASET] [TYPE]
  python run.py train    [DATASET]
  python run.py baseline [os hdfs] [--edge-python PATH]
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

        # .pte files need the ExecuTorch runtime, so fetch it along with Q-BAT checkpoints.
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
        _run_module("training_pipeline.train", "--config", f"configs/training/{dataset}.yaml", *argv[2:])

    elif command == "baseline":
        # A complete seeded run — train, evaluate BAT, quantize, calibrate, detect —
        # kept apart from the standard checkpoint and output directories.
        _run_module("training_pipeline.workflow", *argv[1:])

    elif command == "eval":
        dataset = argv[1] if len(argv) > 1 else "os"
        # Majority voting is what the paper reports; pass "all", "consensus" or
        # "at least one" as a third argument to evaluate the other rules too.
        voting = argv[2] if len(argv) > 2 else "majority"
        print(f"[run] Evaluating BAT ensemble — dataset: {dataset}  voting: {voting}")
        _run_module("training_pipeline.evaluate",
                    "--config", f"configs/training/{dataset}.yaml",
                    "--voting", voting)
        # Eval rewrites the cloud thresholds and each .pte reuses its .pth threshold, so
        # re-derive the edge thresholds; they must never come from older weights.
        print(f"[run] Re-deriving Q-BAT edge thresholds from the cloud thresholds — dataset: {dataset}")
        _run_module("tools.calibrate_edge",
                    "--training-config", f"configs/training/{dataset}.yaml",
                    "--inference-config", f"configs/inference/{dataset}.yaml",
                    "--force")

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
        # queues.py runs step 1 and process_queues.py steps 2-4, sharing one handoff file.
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
        # Fetch checkpoints and detect (default OpenStack); 'hdfs' also classifies and
        # responds. Table 7 and Table 3 alone are `classify` and `eval`.
        dataset = argv[1] if len(argv) > 1 else "os"
        print(f"[run] Full pipeline — dataset: {dataset}\n")

        sys.path.insert(0, str(_ROOT))
        from tools.setup_executorch import executorch_present, setup_executorch

        bat  = _ROOT / "checkpoints" / "bat" / dataset
        qbat = _ROOT / "checkpoints" / "qbat" / dataset
        have = (len(list(bat.glob("*.pth"))) >= 81 if bat.is_dir() else False) and \
               (len(list(qbat.glob("*.pte"))) >= 3 if qbat.is_dir() else False) and \
               executorch_present()

        if have:
            # Checked here: re-listing the remote folder takes minutes even when complete.
            print("[run] 1/3  Checkpoints and edge runtime — already present, skipping")
        else:
            # Download vs. train is the evaluator's choice, so ask; without a terminal
            # (Docker, CI, nohup) print both routes and stop rather than hang.
            print(f"[run] 1/3  No detectors found for '{dataset}'. Choose how to obtain them:\n")
            print(f"  (a) Train them yourself — seeded, reproducible from the seed and config.")
            print(f"      81 BAT learners, their thresholds, then the quantized Q-BAT exports.")
            print(f"      OpenStack takes about an hour; HDFS considerably longer.\n")
            print(f"  (b) Download the published checkpoints the paper's numbers were measured")
            print(f"      on, with matching thresholds. About 3.2 GB per dataset.\n")
            choice = ""
            if sys.stdin.isatty():
                try:
                    choice = input("Choose [a/b, or Enter to stop]: ").strip().lower()
                except EOFError:
                    choice = ""
                print()
            if choice not in ("a", "b"):
                print(f"Nothing obtained. Run one of:")
                print(f"    conda activate cesal-cloud && python run.py train {dataset} && python run.py eval {dataset}")
                print(f"    conda activate cesal-edge  && python run.py convert {dataset}")
                print(f"  or")
                print(f"    python run.py download {dataset}")
                print(f"\nThen re-run: python run.py all {dataset}")
                sys.exit(2)
            if choice == "a":
                # Training and calibration need torch/sklearn from the cloud
                # environment; conversion and everything after run here.
                from cesal_inference_pipeline.run import _detect_cloud_python
                cloud_py = _detect_cloud_python()
                for stage in ("train", "eval"):
                    print(f"[run] 1/3  {stage} — dataset: {dataset}  ({cloud_py})")
                    rc = subprocess.run([cloud_py, str(_ROOT / "run.py"), stage, dataset],
                                        cwd=str(_ROOT)).returncode
                    if rc != 0:
                        sys.exit(rc)
                if not setup_executorch():
                    sys.exit(1)
                print(f"[run] 1/3  convert — dataset: {dataset}")
                rc = _run_keep_going(str(_ROOT / "quantization" / "qbat_export.py"),
                                     "--config", f"configs/training/{dataset}.yaml", "--all")
                if rc != 0:
                    sys.exit(rc)
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
