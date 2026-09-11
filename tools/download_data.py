"""Download assets needed to run CECO-LAD locally after cloning from GitHub.

Downloads:
  executorch.zip  → ceco_lad_inference_pipeline/executorch/   (~1.4 GB, Google Drive)
  bgl_raw.zip     → $CECO_LOG_ROOT/BGL/split/        (~700 MB, HF assets)
  hdfs_split.zip  → $CECO_LOG_ROOT/                  (~1.6 GB, HF assets)

OpenStack raw logs are already bundled in data/OpenStack/raw/ (tracked in git).
Log root defaults to ~/Desktop/Log Data; override with CECO_LOG_ROOT env var.

Usage
-----
  python tools/download_data.py                  # download everything
  python tools/download_data.py --skip executorch
  python tools/download_data.py --skip logs
  python tools/download_data.py --list
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
EXECUTORCH_DIR = _PROJECT_ROOT / "ceco_lad_inference_pipeline" / "executorch"

LOG_ROOT       = Path(os.environ.get("CECO_LOG_ROOT", Path.home() / "Desktop" / "Log Data"))
BGL_SPLIT_DIR  = LOG_ROOT / "BGL" / "split"
HDFS_SPLIT_DIR = LOG_ROOT

# ── Sources ───────────────────────────────────────────────────────────────────
HF_ASSETS_REPO    = "kiSmetZz/ceco-lad-assets"
EXECUTORCH_GDRIVE = "1IjZWI2mFijAs8VDapURfUVwxnYAevNtZ"

# ── Expected files ────────────────────────────────────────────────────────────
BGL_FILES  = ["bgl_train.log", "bgl_test_normal.log", "bgl_test_abnormal.log"]
HDFS_FILES = ["train.log", "test_normal.log", "test_abnormal.log"]
EXECUTORCH_MARKER = EXECUTORCH_DIR / "cmake-out" / "executor_runner"


# ── Dependency checks ─────────────────────────────────────────────────────────

def _check_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'gdown' is not installed.\n"
            "Install it with:  pip install 'gdown>=4.6'\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'huggingface_hub' is not installed.\n"
            "Install it with:  pip install huggingface_hub\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Download helpers ──────────────────────────────────────────────────────────

def _gdrive_download(file_id: str, dest: Path) -> bool:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url=url, output=str(dest), quiet=False)
        return dest.exists()
    except Exception as exc:
        print(f"ERROR: Google Drive download failed: {exc}", file=sys.stderr)
        return False


def _hf_download(filename: str, local_path: Path) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hf_hub_download(
            repo_id=HF_ASSETS_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_path.parent),
        )
        downloaded = Path(tmp)
        if downloaded != local_path:
            downloaded.rename(local_path)
        return local_path.exists()
    except Exception as exc:
        print(f"ERROR: HF download failed ({filename}): {exc}", file=sys.stderr)
        return False


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(out_dir))
    zip_path.unlink(missing_ok=True)


# ── Per-asset download functions ──────────────────────────────────────────────

def download_executorch() -> bool:
    if EXECUTORCH_MARKER.exists():
        print("ExecuTorch already present — skipping.")
        return True
    print("Downloading ExecuTorch (~1.4 GB) from Google Drive …")
    _check_gdown()
    zip_path = _PROJECT_ROOT / "ceco_lad_inference_pipeline" / "executorch.zip"
    if not _gdrive_download(EXECUTORCH_GDRIVE, zip_path):
        return False
    _extract_zip(zip_path, _PROJECT_ROOT / "ceco_lad_inference_pipeline")
    if EXECUTORCH_MARKER.exists():
        print("ExecuTorch ready.")
        return True
    print("WARNING: extraction finished but executor_runner not found.", file=sys.stderr)
    return False


def download_bgl() -> bool:
    if BGL_SPLIT_DIR.exists() and all((BGL_SPLIT_DIR / f).is_file() for f in BGL_FILES):
        print("BGL split logs already present — skipping.")
        return True
    print(f"Downloading BGL split logs (~700 MB) → {BGL_SPLIT_DIR} …")
    _check_huggingface_hub()
    zip_path = LOG_ROOT / "BGL" / "bgl_raw.zip"
    if not _hf_download("bgl_raw.zip", zip_path):
        return False
    _extract_zip(zip_path, BGL_SPLIT_DIR)
    if all((BGL_SPLIT_DIR / f).is_file() for f in BGL_FILES):
        print(f"BGL split logs ready.")
        return True
    print("WARNING: some BGL files are missing after extraction.", file=sys.stderr)
    return False


def download_hdfs() -> bool:
    if HDFS_SPLIT_DIR.exists() and all((HDFS_SPLIT_DIR / f).is_file() for f in HDFS_FILES):
        print("HDFS split logs already present — skipping.")
        return True
    print(f"Downloading HDFS split logs (~1.6 GB) → {HDFS_SPLIT_DIR} …")
    _check_huggingface_hub()
    zip_path = HDFS_SPLIT_DIR / "hdfs_split.zip"
    HDFS_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if not _hf_download("hdfs_split.zip", zip_path):
        return False
    _extract_zip(zip_path, HDFS_SPLIT_DIR)
    if all((HDFS_SPLIT_DIR / f).is_file() for f in HDFS_FILES):
        print("HDFS split logs ready.")
        return True
    print("WARNING: some HDFS files are missing after extraction.", file=sys.stderr)
    return False


# ── Status listing ────────────────────────────────────────────────────────────

def list_status() -> None:
    def _status(p: Path) -> str:
        if not p.exists():
            return "MISSING"
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            return f"{n} files"
        return f"{p.stat().st_size // 1024 // 1024} MB"

    print(f"ExecuTorch  ({EXECUTORCH_DIR}): {_status(EXECUTORCH_DIR)}")
    print(f"  executor_runner: {'OK' if EXECUTORCH_MARKER.exists() else 'MISSING'}")
    print(f"\nBGL split logs  ({BGL_SPLIT_DIR}):")
    for f in BGL_FILES:
        print(f"  {f}: {_status(BGL_SPLIT_DIR / f)}")
    print(f"\nHDFS split logs  ({HDFS_SPLIT_DIR}):")
    for f in HDFS_FILES:
        print(f"  {f}: {_status(HDFS_SPLIT_DIR / f)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download assets needed to run CECO-LAD locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip", choices=["executorch", "logs"], default=None,
        help="Skip a specific component.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print download status and exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_status()
        return

    ok = True
    if args.skip != "executorch":
        ok &= download_executorch()
    if args.skip != "logs":
        ok &= download_bgl()
        ok &= download_hdfs()

    print()
    if ok:
        print("All assets ready. You can now run CECO-LAD.")
    else:
        print("Some downloads failed — check errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
