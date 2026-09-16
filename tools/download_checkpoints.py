"""Download pre-trained BAT and Q-BAT checkpoints from the CESAL Google Drive.

Drive folder: https://drive.google.com/drive/folders/16nz-irtjK2vURmSdlurYq7dd8jZUJnio
  pretrained_models/bat/ensemble_hdfs/  — 81 BAT .pth files   → checkpoints/bat/hdfs/
  pretrained_models/bat/ensemble_os/    — 81 BAT .pth files   → checkpoints/bat/os/
  pretrained_models/qbat/hdfs/          — 3 Q-BAT learners (.pte) → checkpoints/qbat/hdfs/
  pretrained_models/qbat/os/            — 3 Q-BAT learners (.pte) → checkpoints/qbat/os/

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
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Run directly as a script (as run.py does), so put the project root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cesal_core.utils import steps
from cesal_core.utils.config import setup_logging
from cesal_core.utils.steps import StepReporter

_ABOUT = """
Fetching the trained models CESAL needs before it can analyse anything.
These are the detectors produced by training — the full-size ones the cloud
uses, and the shrunk ones the edge device runs. They are downloaded once and
reused; anything already on disk is left alone.
"""

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

    rep = StepReporter("download", steps=steps.DOWNLOAD_STEPS, about=_ABOUT)

    # ── Step 1: see what is already on disk ───────────────────────────────
    targets = []
    with rep.step("check") as st:
        st.detail("checkpoint types", ", ".join(types))
        st.detail("datasets", ", ".join(datasets))
        st.detail("destination", str(_CHECKPOINTS_DIR))

        already = 0
        for t in types:
            ext = ".pth" if t == "bat" else ".pte"
            for ds in datasets:
                folder = _CHECKPOINTS_DIR / t / ds
                have = len(list(folder.glob(f"*{ext}"))) if folder.exists() else 0
                already += have
                if not _DRIVE_FOLDER_IDS[t].get(ds):
                    st.warn(f"No download location is configured for {t}/{ds} — skipping it.")
                    continue
                targets.append((t, ds, ext, folder))
                logging.info("   %-10s %s — %s already present", f"{t}/{ds}", ext,
                             f"{have:,}" if have else "none")

        st.outcome(**{
            "collections to fetch": len(targets),
            "files already present": already,
        })

    if not targets:
        rep.skip("fetch", "Nothing to download — no configured location for the "
                          "requested checkpoints.")
        rep.finish(outputs=str(_CHECKPOINTS_DIR))
        return

    # ── Step 2: fetch them ────────────────────────────────────────────────
    with rep.step("fetch") as st:
        st.expect("collections", len(targets))
        installed = 0
        for t, ds, ext, out_dir in targets:
            label = "BAT" if t == "bat" else "Q-BAT"
            st.progress_note(f"{label} · {ds}")
            st.phase(f"downloading {label} checkpoints for {ds}")
            _download_folder(_DRIVE_FOLDER_IDS[t][ds], out_dir)
            n = sum(1 for f in out_dir.glob(f"*{ext}") if f.is_file())
            installed += n
            logging.info("   %-10s %s %s file(s) now installed", f"{t}/{ds}",
                         f"{n:,}", ext)
            st.tick("collections")

        st.outcome(**{
            "collections fetched": len(targets),
            "checkpoint files on disk": installed,
            "total size": f"{_dir_size_mb(_CHECKPOINTS_DIR):,.0f} MB",
        })

    rep.finish(outputs=str(_CHECKPOINTS_DIR))


def _dir_size_mb(path: Path) -> float:
    """Total size of everything under `path`, in MB."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
    except OSError:
        return 0.0


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
    setup_logging("download")
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
