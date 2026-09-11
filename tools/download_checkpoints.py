"""Download pre-trained BAT and Q-BAT checkpoints from Google Drive.

The Google Drive folder contains:
  bgl.zip            — BAT checkpoints for BGL   → checkpoints/bat/bgl/
  hdfs.zip           — BAT checkpoints for HDFS  → checkpoints/bat/hdfs/
  os.zip             — BAT checkpoints for OS    → checkpoints/bat/os/
  qbat/bgl/          — 3 Q-BAT .pte files        → checkpoints/qbat/bgl/
  qbat/hdfs/         — 3 Q-BAT .pte files        → checkpoints/qbat/hdfs/
  qbat/os/           — 3 Q-BAT .pte files        → checkpoints/qbat/os/

BAT zips are extracted automatically; Q-BAT .pte files are copied directly.

Usage
-----
  # Download everything (bat + qbat, all datasets):
  python tools/download_checkpoints.py

  # Download a specific checkpoint type only:
  python tools/download_checkpoints.py --type bat
  python tools/download_checkpoints.py --type qbat

  # Download a specific dataset only:
  python tools/download_checkpoints.py --dataset bgl

  # Combine both filters:
  python tools/download_checkpoints.py --type bat --dataset hdfs

  # List already-downloaded checkpoints:
  python tools/download_checkpoints.py --list

"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

# Per-file Google Drive IDs for BAT zip archives
_BAT_FILE_IDS = {
    "bgl":  "1vTqHLS7z8wbeYdibvEFrgI1CeD2Uox21",
    "hdfs": "1yk-jL0wra3qSe_Re1N2rw4g271BKC1NI",
    "os":   "10zi0AawRQ-8yPC7cPP_BshVv0-W0ESC7",
}

# Per-subfolder Google Drive IDs for Q-BAT folders (fill in if known)
_QBAT_FOLDER_IDS: dict = {
    # "bgl":  "<folder-id>",
    # "hdfs": "<folder-id>",
    # "os":   "<folder-id>",
}

# Fallback: full Drive folder URL used only when a specific ID is unavailable
_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1vZ_h_VVCsxIzzeov09i_uKTPjwyZTPmY"
)

# Project root is one level above this file  (tools/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_DIR = _PROJECT_ROOT / "checkpoints"

_VALID_TYPES = ("bat", "qbat")
_VALID_DATASETS = ("bgl", "hdfs", "os")


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(file_id: str, dest: Path) -> None:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=str(dest), quiet=False)


def _download_folder(folder_id: str, dest: Path) -> None:
    import gdown
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url=url, output=str(dest), quiet=False)


def _download_full_drive_folder(dest: Path) -> None:
    import gdown
    dest.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=_DRIVE_FOLDER_URL, output=str(dest), quiet=False)


def _extract_zip_flat(zip_path: Path, out_dir: Path) -> int:
    """Extract all files from *zip_path* into *out_dir*, ignoring internal directories.

    Returns the number of files extracted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = out_dir / Path(member.filename).name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return sum(1 for f in out_dir.iterdir() if f.is_file())


def _copy_folder(src: Path, dst: Path) -> int:
    """Recursively copy *src* into *dst*. Returns the number of files copied."""
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            count += 1
        elif item.is_dir():
            count += _copy_folder(item, dst / item.name)
    return count


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download(ckpt_type: Optional[str], dataset: Optional[str]) -> None:
    """Download and install checkpoints matching the given filters.

    Parameters
    ----------
    ckpt_type : {'bat', 'qbat'} or None
        Which checkpoint type to install. None means both.
    dataset : {'bgl', 'hdfs', 'os'} or None
        Which dataset to download. None means all three.
    """
    types = [ckpt_type] if ckpt_type else list(_VALID_TYPES)
    datasets = [dataset] if dataset else list(_VALID_DATASETS)

    print(f"Types    : {types}")
    print(f"Datasets : {datasets}")
    print(f"Output   : {_CHECKPOINTS_DIR}\n")

    step = 1
    total = ("bat" in types) + ("qbat" in types)

    if "bat" in types:
        print(f"Step {step}/{total}  Downloading BAT checkpoints...")
        step += 1
        for ds in datasets:
            if ds not in _BAT_FILE_IDS:
                print(f"  [bat/{ds}] WARNING: no file ID configured — skipping.")
                continue
            out_dir = _CHECKPOINTS_DIR / "bat" / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            zip_path = out_dir / f"{ds}.zip"
            print(f"  [bat/{ds}] Downloading {ds}.zip → {out_dir}")
            _download_file(_BAT_FILE_IDS[ds], zip_path)
            print(f"  [bat/{ds}] Extracting ...")
            n = _extract_zip_flat(zip_path, out_dir)
            zip_path.unlink()
            print(f"            {n} .pth file(s) installed.")

    if "qbat" in types:
        print(f"Step {step}/{total}  Downloading Q-BAT checkpoints...")
        for ds in datasets:
            out_dir = _CHECKPOINTS_DIR / "qbat" / ds
            if ds in _QBAT_FOLDER_IDS:
                print(f"  [qbat/{ds}] Downloading folder → {out_dir}")
                _download_folder(_QBAT_FOLDER_IDS[ds], out_dir)
                n = sum(1 for f in out_dir.glob("*.pte") if f.is_file())
                print(f"            {n} .pte file(s) installed.")
            else:
                # Fall back: download full Drive folder to a temp dir, copy the needed subset
                with tempfile.TemporaryDirectory(prefix="ceco_dl_") as tmp_str:
                    tmp_dir = Path(tmp_str)
                    print("  [qbat] No per-folder IDs configured — downloading full Drive folder...")
                    _download_full_drive_folder(tmp_dir)
                    src = tmp_dir / "qbat" / ds
                    if not src.exists():
                        print(f"  [qbat/{ds}] WARNING: folder not found — skipping.")
                        continue
                    print(f"  [qbat/{ds}] Copying → {out_dir}")
                    n = _copy_folder(src, out_dir)
                    print(f"            {n} .pte file(s) installed.")

    print()
    _verify(types, datasets)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def _verify(types: List[str], datasets: List[str]) -> None:
    print("--- Checkpoint inventory ---")
    for t in types:
        ext = ".pth" if t == "bat" else ".pte"
        for ds in datasets:
            folder = _CHECKPOINTS_DIR / t / ds
            if folder.exists():
                n = len(list(folder.glob(f"*{ext}")))
                print(f"  {t}/{ds}: {n} {ext} file(s)")
            else:
                print(f"  {t}/{ds}: (not downloaded)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pre-trained CECO-LAD checkpoints from Google Drive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--type",
        dest="ckpt_type",
        choices=_VALID_TYPES,
        default=None,
        help="Checkpoint type: 'bat' (full-precision .pth) or 'qbat' (quantized .pte)."
             " Omit to download both.",
    )
    parser.add_argument(
        "--dataset",
        choices=_VALID_DATASETS,
        default=None,
        help="Dataset: bgl, hdfs, or os. Omit to download all three.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print a summary of already-downloaded checkpoints and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    types = [args.ckpt_type] if args.ckpt_type else list(_VALID_TYPES)
    datasets = [args.dataset] if args.dataset else list(_VALID_DATASETS)

    if args.list:
        _verify(types, datasets)
        return

    _check_gdown()
    download(args.ckpt_type, args.dataset)


if __name__ == "__main__":
    main()
