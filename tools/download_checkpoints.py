"""Download pre-trained BAT and Q-BAT checkpoints from the CESAL Google Drive.

Drive folder: https://drive.google.com/drive/folders/16nz-irtjK2vURmSdlurYq7dd8jZUJnio
  pretrained_models/bat/ensemble_hdfs/  — 81 BAT .pth files   → checkpoints/bat/hdfs/
  pretrained_models/bat/ensemble_os/    — 81 BAT .pth files   → checkpoints/bat/os/
  pretrained_models/qbat/hdfs/          — 3 Q-BAT .pte files  → checkpoints/qbat/hdfs/
  pretrained_models/qbat/os/            — 3 Q-BAT .pte files  → checkpoints/qbat/os/

Each Drive subfolder is downloaded file by file into its checkpoints/ directory.

Usage
-----
  # Download everything (bat + qbat, all datasets):
  python tools/download_checkpoints.py

  # Download a specific checkpoint type only:
  python tools/download_checkpoints.py --type bat
  python tools/download_checkpoints.py --type qbat

  # Download a specific dataset only:
  python tools/download_checkpoints.py --dataset os

  # Combine both filters:
  python tools/download_checkpoints.py --type bat --dataset hdfs

  # List already-downloaded checkpoints:
  python tools/download_checkpoints.py --list

"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# CESAL Google Drive subfolder IDs, per checkpoint type and dataset
_DRIVE_FOLDER_IDS = {
    "bat": {
        "hdfs": "10w8XS4piWi_yExL1YxzBnDNCEYEIqz1H",   # pretrained_models/bat/ensemble_hdfs/
        "os":   "1yoyLstOOKJrDSeatPxZfTUHOJeG_l2fH",   # pretrained_models/bat/ensemble_os/
    },
    "qbat": {
        "hdfs": "1j9fvGrODijH5Ulx2E0sFgHEhr97Hnake",   # pretrained_models/qbat/hdfs/
        "os":   "1_PU133GGbeZrFLzCt0gxRbEVjSQcTlg6",   # pretrained_models/qbat/os/
    },
}

# Project root is one level above this file  (tools/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_DIR = _PROJECT_ROOT / "checkpoints"

_VALID_TYPES = ("bat", "qbat")
_VALID_DATASETS = ("hdfs", "os")


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

def _download_folder(folder_id: str, dest: Path) -> None:
    import gdown
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    # resume=True keeps files that finished downloading (and resumes partial ones),
    # so a re-run after an interruption doesn't fetch completed checkpoints again.
    gdown.download_folder(url=url, output=str(dest), quiet=False, resume=True)


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download(ckpt_type: Optional[str], dataset: Optional[str]) -> None:
    """Download and install checkpoints matching the given filters.

    Parameters
    ----------
    ckpt_type : {'bat', 'qbat'} or None
        Which checkpoint type to install. None means both.
    dataset : {'hdfs', 'os'} or None
        Which dataset to download. None means both.
    """
    types = [ckpt_type] if ckpt_type else list(_VALID_TYPES)
    datasets = [dataset] if dataset else list(_VALID_DATASETS)

    print(f"Types    : {types}")
    print(f"Datasets : {datasets}")
    print(f"Output   : {_CHECKPOINTS_DIR}\n")

    for step, t in enumerate(types, 1):
        label = "BAT" if t == "bat" else "Q-BAT"
        ext = ".pth" if t == "bat" else ".pte"
        print(f"Step {step}/{len(types)}  Downloading {label} checkpoints...")
        for ds in datasets:
            folder_id = _DRIVE_FOLDER_IDS[t].get(ds)
            if not folder_id:
                print(f"  [{t}/{ds}] WARNING: no Drive folder ID configured — skipping.")
                continue
            out_dir = _CHECKPOINTS_DIR / t / ds
            print(f"  [{t}/{ds}] Downloading folder → {out_dir}")
            _download_folder(folder_id, out_dir)
            n = sum(1 for f in out_dir.glob(f"*{ext}") if f.is_file())
            print(f"            {n} {ext} file(s) installed.")

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
        description="Download pre-trained CESAL checkpoints from Google Drive.",
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
        help="Dataset: hdfs or os. Omit to download both.",
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
