#!/usr/bin/env python3
"""Local startup script for CECO-LAD.

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
  BGL raw logs       ~700 MB   Hugging Face assets
  HDFS raw logs      ~1.6 GB   Hugging Face assets
  BAT checkpoints    ~3.5 GB × dataset  Google Drive  (skippable with --no-bat)

Dashboard runs at http://localhost:8765
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Asset locations ───────────────────────────────────────────────────────────
EXECUTORCH_DIR    = ROOT / "ceco_lad_inference_pipeline" / "executorch"
EXECUTOR_RUNNER   = EXECUTORCH_DIR / "cmake-out" / "executor_runner"
EXECUTORCH_GDRIVE   = "1IjZWI2mFijAs8VDapURfUVwxnYAevNtZ"  # ExecuTorch 0.5.0 pre-built

QBAT_DIR     = ROOT / "checkpoints" / "qbat"
QBAT_DATASETS = ["bgl", "hdfs", "os"]

LOG_ROOT      = Path(os.environ.get("CECO_LOG_ROOT", Path.home() / "Desktop" / "Log Data"))
BGL_SPLIT_DIR = LOG_ROOT / "BGL" / "split"
BGL_FILES     = ["bgl_train.log", "bgl_test_normal.log", "bgl_test_abnormal.log"]
HDFS_SPLIT_DIR = LOG_ROOT
HDFS_FILES    = ["train.log", "test_normal.log", "test_abnormal.log"]

HF_ASSETS_REPO = "kiSmetZz/ceco-lad-assets"
BAT_DATASETS   = ["bgl", "hdfs", "os"]
MIN_BAT_CKPTS  = 81


# ── Presence checks ───────────────────────────────────────────────────────────

def _executorch_ok() -> bool:
    return EXECUTOR_RUNNER.is_file()

def _qbat_ok() -> bool:
    return all(
        (QBAT_DIR / ds).exists() and len(list((QBAT_DIR / ds).glob("*.pte"))) >= 3
        for ds in QBAT_DATASETS
    )

def _bgl_logs_ok() -> bool:
    return all((BGL_SPLIT_DIR / f).is_file() for f in BGL_FILES)

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

def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown>=4.6"], check=True)


def _ensure_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)


# ── Download functions ────────────────────────────────────────────────────────

def _gdrive_download(file_id: str, dest: Path) -> bool:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(url=f"https://drive.google.com/uc?id={file_id}",
                       output=str(dest), quiet=False)
        return dest.exists()
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return False


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


def _extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(out_dir))
    zip_path.unlink(missing_ok=True)


# ── Per-asset setup ───────────────────────────────────────────────────────────

def _install_executorch_python_bindings() -> bool:
    install_script = EXECUTORCH_DIR / "install_requirements.py"
    if not install_script.exists():
        print("  WARNING: install_requirements.py not found — Python bindings skipped.")
        return False
    print("  Installing ExecuTorch Python bindings …")
    result = subprocess.run(
        [sys.executable, str(install_script)],
        cwd=str(EXECUTORCH_DIR),
    )
    if result.returncode == 0:
        edge_req = ROOT / "environment" / "edge" / "requirements.txt"
        if edge_req.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "-r", str(edge_req)], check=False)
        print("  ExecuTorch Python bindings installed.")
        return True
    print("  WARNING: Python binding install failed — C++ executor_runner fallback will be used.")
    return False


def setup_executorch() -> bool:
    bindings_ok = False
    try:
        from executorch.runtime import Runtime  # noqa: F401
        bindings_ok = True
    except ImportError:
        pass

    if _executorch_ok() and bindings_ok:
        print("[1/5] ExecuTorch — already present, skipping.")
        return True

    if not _executorch_ok():
        print("[1/5] Downloading ExecuTorch 0.5.0 (~1.4 GB) from Google Drive …")
        _ensure_gdown()
        zip_path = ROOT / "ceco_lad_inference_pipeline" / "executorch.zip"
        if not _gdrive_download(EXECUTORCH_GDRIVE, zip_path):
            print("  WARNING: ExecuTorch download failed — edge inference unavailable.")
            return False
        _extract(zip_path, ROOT / "ceco_lad_inference_pipeline")
        if not _executorch_ok():
            print("  WARNING: executor_runner not found after extraction.")
            return False

    if not bindings_ok:
        _install_executorch_python_bindings()

    print("  ExecuTorch ready.")
    return True


def setup_qbat() -> bool:
    if _qbat_ok():
        print("[2/5] Q-BAT checkpoints — already present, skipping.")
        return True
    print("[2/5] Downloading Q-BAT checkpoints (~220 MB) …")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "download_checkpoints.py"), "--type", "qbat"],
        cwd=str(ROOT),
    )
    if result.returncode == 0 and _qbat_ok():
        print("  Q-BAT checkpoints ready.")
        return True
    print("  WARNING: Q-BAT download failed — edge inference unavailable.")
    return False


def setup_bgl_logs() -> bool:
    if _bgl_logs_ok():
        print("[3/5] BGL raw logs — already present, skipping.")
        return True
    print("[3/5] Downloading BGL raw logs (~700 MB) from Hugging Face …")
    _ensure_hf_hub()
    zip_path = LOG_ROOT / "BGL" / "bgl_raw.zip"
    if not _hf_download("bgl_raw.zip", zip_path):
        print("  WARNING: BGL log download failed — raw log panel will be empty.")
        return False
    _extract(zip_path, BGL_SPLIT_DIR)
    if _bgl_logs_ok():
        print("  BGL raw logs ready.")
        return True
    print("  WARNING: some BGL log files missing after extraction.")
    return False


def setup_hdfs_logs() -> bool:
    if _hdfs_logs_ok():
        print("[4/5] HDFS raw logs — already present, skipping.")
        return True
    print("[4/5] Downloading HDFS raw logs (~1.6 GB) from Hugging Face …")
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
    for ds in ("bgl", "hdfs", "os"):
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
        label = f"[5/5] BAT {ds.upper()} checkpoints"
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

    print("\n── Asset Status ─────────────────────────────────────────────")
    print(f"  ExecuTorch (executor_runner) : {_ok(_executorch_ok())}")
    print(f"  Q-BAT checkpoints            : {_ok(_qbat_ok())}")
    print(f"  BGL raw logs                 : {_ok(_bgl_logs_ok())}")
    print(f"  HDFS raw logs                : {_ok(_hdfs_logs_ok())}")
    for ds in BAT_DATASETS:
        print(f"  BAT {ds.upper()} checkpoints          : {_ok(_bat_ok(ds))}")
    for ds in ("bgl", "hdfs", "os"):
        print(f"  {ds.upper()} inference outputs        : {_ok(_outputs_ok(ds))}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up and launch CECO-LAD locally.",
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

    print("\nCECO-LAD Local Setup")
    print("=" * 50)
    print("Skipped assets are already present. Downloads are resumable.\n")

    setup_executorch()
    setup_qbat()
    setup_bgl_logs()
    setup_hdfs_logs()
    setup_outputs()
    if not args.no_bat:
        setup_bat(BAT_DATASETS)
    else:
        print("[5/5] BAT checkpoints — skipped (--no-bat).")

    show_status()

    if args.setup_only:
        print("Setup complete. Run  python launch_dashboard.py  to launch the dashboard.")
        return

    print("Launching dashboard at http://localhost:8765 …\n")
    os.execv(sys.executable, [sys.executable, str(ROOT / "dashboard" / "app.py")])


if __name__ == "__main__":
    main()
