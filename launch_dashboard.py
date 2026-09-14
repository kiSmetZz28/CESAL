#!/usr/bin/env python3
"""Local startup script for CESAL.

Downloads any missing assets then launches the dashboard.
Safe to run multiple times — skips files that are already present.

Usage
-----
  python launch_dashboard.py                # download everything + launch dashboard
  python launch_dashboard.py --no-bat       # skip BAT checkpoints (3.5 GB each) + launch
  python launch_dashboard.py --setup-only   # download only, do not launch dashboard
  python launch_dashboard.py --status       # show what is present / missing, then exit

Assets downloaded
-----------------
  ExecuTorch build   ~1.4 GB   Google Drive
  Q-BAT checkpoints  ~220 MB   Google Drive
  HDFS raw logs      ~1.6 GB   Hugging Face assets
  BAT checkpoints    ~3.5 GB × dataset  Google Drive  (skippable with --no-bat)

Dashboard runs at http://localhost:8765
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # importable from any working directory

# ── Asset locations ───────────────────────────────────────────────────────────
# The ExecuTorch runtime is a pipeline asset, not a dashboard one: it is fetched
# and installed by tools/setup_executorch.py, which `run.py download` also calls.
from tools.setup_executorch import (            # noqa: E402
    executorch_present as _executorch_ok,
    extract_zip as _extract,
    setup_executorch as _setup_executorch,
)

QBAT_DIR     = ROOT / "checkpoints" / "qbat"
QBAT_DATASETS = ["hdfs", "os"]

LOG_ROOT      = Path(os.environ.get("CESAL_LOG_ROOT", Path.home() / "Desktop" / "Log Data"))
HDFS_SPLIT_DIR = LOG_ROOT
HDFS_FILES    = ["train.log", "test_normal.log", "test_abnormal.log"]

HF_ASSETS_REPO = "kiSmetZz/ceco-lad-assets"
BAT_DATASETS   = ["hdfs", "os"]
MIN_BAT_CKPTS  = 81


# ── Presence checks ───────────────────────────────────────────────────────────

def _qbat_ok() -> bool:
    return all(
        (QBAT_DIR / ds).exists() and len(list((QBAT_DIR / ds).glob("*.pte"))) >= 3
        for ds in QBAT_DATASETS
    )

def _hdfs_logs_ok() -> bool:
    return all((HDFS_SPLIT_DIR / f).is_file() for f in HDFS_FILES)

def _bat_ok(ds: str) -> bool:
    d = ROOT / "checkpoints" / "bat" / ds
    return d.exists() and len(list(d.glob("*.pth"))) >= MIN_BAT_CKPTS

def _outputs_ok(ds: str) -> bool:
    out = ROOT / "outputs" / ds
    required = [
        "ground_truth.npy", "edge_preds.npy", "hybrid_preds.npy",
        "cloud_preds.npy", "routed_indices.npy",
        "edge_preds_per_model.npy", "cloud_preds_per_model.npy",
    ]
    return all((out / f).exists() for f in required)


# ── Dependency helpers ────────────────────────────────────────────────────────

def _ensure_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)


# ── Download functions ────────────────────────────────────────────────────────

def _hf_download(filename: str, local_path: Path) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hf_hub_download(
            repo_id=HF_ASSETS_REPO, filename=filename,
            repo_type="dataset", local_dir=str(local_path.parent),
        )
        downloaded = Path(tmp)
        if downloaded != local_path:
            downloaded.rename(local_path)
        return local_path.exists()
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return False


# ── Per-asset setup ───────────────────────────────────────────────────────────

def setup_executorch() -> bool:
    return _setup_executorch(prefix="[1/4] ")


def setup_qbat() -> bool:
    if _qbat_ok():
        print("[2/4] Q-BAT checkpoints — already present, skipping.")
        return True
    print("[2/4] Downloading Q-BAT checkpoints (~220 MB) …")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "download_checkpoints.py"), "--type", "qbat"],
        cwd=str(ROOT),
    )
    if result.returncode == 0 and _qbat_ok():
        print("  Q-BAT checkpoints ready.")
        return True
    print("  WARNING: Q-BAT download failed — edge inference unavailable.")
    return False


def setup_hdfs_logs() -> bool:
    if _hdfs_logs_ok():
        print("[3/4] HDFS raw logs — already present, skipping.")
        return True
    print("[3/4] Downloading HDFS raw logs (~1.6 GB) from Hugging Face …")
    _ensure_hf_hub()
    zip_path = HDFS_SPLIT_DIR / "hdfs_split.zip"
    HDFS_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if not _hf_download("hdfs_split.zip", zip_path):
        print("  WARNING: HDFS log download failed — raw log panel will be empty.")
        return False
    _extract(zip_path, HDFS_SPLIT_DIR)
    if _hdfs_logs_ok():
        print("  HDFS raw logs ready.")
        return True
    print("  WARNING: some HDFS log files missing after extraction.")
    return False


def setup_outputs() -> bool:
    all_ok = True
    for ds in ("hdfs", "os"):
        if _outputs_ok(ds):
            print(f"  {ds.upper()} inference outputs — already present, skipping.")
            continue
        print(f"  {ds.upper()} inference outputs — downloading from Hugging Face …")
        _ensure_hf_hub()
        out_dir  = ROOT / "outputs" / ds
        zip_path = out_dir / f"{ds}_outputs.zip"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not _hf_download(f"outputs/{ds}.zip", zip_path):
            print(f"  WARNING: {ds.upper()} outputs download failed — prediction lookup will use live inference.")
            all_ok = False
            continue
        _extract(zip_path, out_dir)
        if _outputs_ok(ds):
            print(f"  {ds.upper()} inference outputs ready.")
        else:
            print(f"  WARNING: {ds.upper()} outputs incomplete after extraction.")
            all_ok = False
    return all_ok


def setup_bat(datasets: list) -> bool:
    all_ok = True
    for i, ds in enumerate(datasets, 1):
        label = f"[4/4] BAT {ds.upper()} checkpoints"
        if _bat_ok(ds):
            print(f"{label} — already present, skipping.")
            continue
        print(f"{label} — downloading (~3.5 GB) from Google Drive …")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "download_checkpoints.py"),
             "--type", "bat", "--dataset", ds],
            cwd=str(ROOT),
        )
        if result.returncode != 0 or not _bat_ok(ds):
            print(f"  WARNING: BAT {ds.upper()} download failed.")
            all_ok = False
        else:
            print(f"  BAT {ds.upper()} ready.")
    return all_ok


# ── Status display ────────────────────────────────────────────────────────────

def show_status() -> None:
    def _ok(cond: bool) -> str:
        return "OK" if cond else "MISSING"

    rows = [("ExecuTorch (executor_runner)", _executorch_ok()),
            ("Q-BAT checkpoints",            _qbat_ok()),
            ("HDFS raw logs",                _hdfs_logs_ok())]
    rows += [(f"BAT {ds.upper()} checkpoints", _bat_ok(ds)) for ds in BAT_DATASETS]
    rows += [(f"{ds.upper()} inference outputs", _outputs_ok(ds)) for ds in ("hdfs", "os")]
    width = max(len(name) for name, _ in rows)

    print("\n── Asset Status ─────────────────────────────────────────────")
    for name, ok in rows:
        print(f"  {name:<{width}} : {_ok(ok)}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up and launch CESAL locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--no-bat", action="store_true",
                        help="Skip BAT checkpoint download (3.5 GB per dataset).")
    parser.add_argument("--setup-only", action="store_true",
                        help="Download assets without launching the dashboard.")
    parser.add_argument("--status", action="store_true",
                        help="Show asset status and exit.")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    print("\nCESAL Local Setup")
    print("=" * 50)
    print("Skipped assets are already present. Downloads are resumable.\n")

    setup_executorch()
    setup_qbat()
    setup_hdfs_logs()
    setup_outputs()
    if not args.no_bat:
        setup_bat(BAT_DATASETS)
    else:
        print("[4/4] BAT checkpoints — skipped (--no-bat).")

    show_status()

    if args.setup_only:
        print("Setup complete. Run  python launch_dashboard.py  to launch the dashboard.")
        return

    print("Launching dashboard at http://localhost:8765 …\n")
    os.execv(sys.executable, [sys.executable, str(ROOT / "dashboard" / "app.py")])


if __name__ == "__main__":
    main()
