#!/usr/bin/env python3
"""Upload large CECO-LAD assets to a Hugging Face dataset repository.

This replaces Google Drive as the download source for HF Spaces startup.
Run this once locally whenever assets change.

Run from the project root:
    python tools/deploy/upload_assets.py
    python tools/deploy/upload_assets.py --repo kiSmetZz/ceco-lad-assets   # custom repo name
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "huggingface_hub>=0.20"], check=True)
    from huggingface_hub import HfApi, login

# This file lives in tools/deploy/, so the project root is two levels up.
ROOT = Path(__file__).resolve().parent.parent.parent

LOG_DATA_ROOT     = Path.home() / "Desktop" / "Log Data"
OPENSTACK_LOG_DIR = LOG_DATA_ROOT / "OpenStack"
OPENSTACK_LOGS = [
    OPENSTACK_LOG_DIR / "openstack_normal1.log",
    OPENSTACK_LOG_DIR / "openstack_normal2.log",
    OPENSTACK_LOG_DIR / "openstack_abnormal.log",
]
BGL_SPLIT_DIR  = LOG_DATA_ROOT / "BGL"     / "split"
BGL_SPLIT_FILES = ["bgl_train.log", "bgl_test_normal.log", "bgl_test_abnormal.log"]
HDFS_SPLIT_DIR   = LOG_DATA_ROOT
HDFS_SPLIT_FILES = ["train.log", "test_normal.log", "test_abnormal.log"]
BGL_DATA_DIR   = ROOT / "data" / "BGL"
BGL_DATA_FILES = ["bgl_train.txt", "bgl_test_normal.txt", "bgl_test_abnormal.txt"]
BAT_OS_DIR    = ROOT / "checkpoints" / "bat" / "os"
BAT_BGL_DIR   = ROOT / "checkpoints" / "bat" / "bgl"
BAT_HDFS_DIR  = ROOT / "checkpoints" / "bat" / "hdfs"
QBAT_DIR      = ROOT / "checkpoints" / "qbat"
RUNNER_PATH   = ROOT / "ceco_lad_inference_pipeline" / "executorch" / "cmake-out" / "executor_runner"
DEMO_VIDEO    = ROOT / "dashboard" / "static" / "demo.mp4"

# ── Pre-computed inference outputs (npy arrays) ───────────────────────────────
OUTPUT_NP_FILES = [
    "ground_truth.npy", "edge_preds.npy", "edge_preds_raw.npy",
    "hybrid_preds.npy", "cloud_preds.npy",
    "energy_matrix.npy", "routed_indices.npy",
    "edge_preds_per_model.npy", "cloud_preds_per_model.npy",
]


def upload(api: HfApi, repo: str, local: Path, remote: str) -> None:
    print(f"  Uploading {local.name} ({local.stat().st_size // 1024 // 1024} MB) → {remote} …")
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"  ✓ {remote}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload CECO-LAD assets to HF Hub.")
    parser.add_argument("--repo", default="kiSmetZz/ceco-lad-assets",
                        help="HF dataset repo (default: kiSmetZz/ceco-lad-assets)")
    args = parser.parse_args()
    repo = args.repo

    print(f"\nUploading CECO-LAD assets → {repo}")
    print("Log in to Hugging Face (token cached after first use):\n")
    login()

    api = HfApi()
    print(f"\nCreating dataset repo '{repo}' (public, exist_ok)…")
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    print("  Repo ready.\n")

    # ── 1. executor_runner binary ─────────────────────────────────────────────
    print("── Step 1/9  executor_runner binary ──")
    if RUNNER_PATH.exists():
        upload(api, repo, RUNNER_PATH, "executor_runner")
    else:
        print(f"  SKIP: {RUNNER_PATH} not found.")

    # ── 2. OpenStack raw log files ────────────────────────────────────────────
    print("\n── Step 2/9  OpenStack raw logs ──")
    missing = [f for f in OPENSTACK_LOGS if not f.exists()]
    if missing:
        print(f"  SKIP: log files not found: {[f.name for f in missing]}")
    else:
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        print(f"  Zipping {len(OPENSTACK_LOGS)} log files → {zip_path.name} …")
        subprocess.run(
            ["zip", "-j", str(zip_path)] + [str(f) for f in OPENSTACK_LOGS],
            check=True,
        )
        upload(api, repo, zip_path, "os_logs.zip")
        zip_path.unlink()

    # ── 3. BGL raw log split files ────────────────────────────────────────────
    print("\n── Step 3/9  BGL raw log (split files) ──")
    bgl_split_files = [BGL_SPLIT_DIR / f for f in BGL_SPLIT_FILES]
    missing = [f for f in bgl_split_files if not f.exists()]
    if missing:
        print(f"  SKIP: split files not found: {[f.name for f in missing]}")
    else:
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        total_mb = sum(f.stat().st_size for f in bgl_split_files) // 1024 // 1024
        print(f"  Zipping {len(bgl_split_files)} BGL split files (~{total_mb} MB) → {zip_path.name} …")
        subprocess.run(
            ["zip", "-j", str(zip_path)] + [str(f) for f in bgl_split_files],
            check=True,
        )
        upload(api, repo, zip_path, "bgl_raw.zip")
        zip_path.unlink()

    # ── 4. HDFS raw log split files ───────────────────────────────────────────
    print("\n── Step 4/9  HDFS raw log (split files) ──")
    hdfs_files = [HDFS_SPLIT_DIR / f for f in HDFS_SPLIT_FILES]
    missing = [f for f in hdfs_files if not f.exists()]
    if missing:
        print(f"  SKIP: HDFS split files not found: {[f.name for f in missing]}")
    else:
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        total_mb = sum(f.stat().st_size for f in hdfs_files) // 1024 // 1024
        print(f"  Zipping {len(hdfs_files)} HDFS split files (~{total_mb} MB) → {zip_path.name} …")
        subprocess.run(
            ["zip", "-j", str(zip_path)] + [str(f) for f in hdfs_files],
            check=True,
        )
        upload(api, repo, zip_path, "hdfs_split.zip")
        zip_path.unlink()

    # ── 5. BGL data files ─────────────────────────────────────────────────────
    print("\n── Step 5/9  BGL data files ──")
    bgl_files = [BGL_DATA_DIR / f for f in BGL_DATA_FILES]
    missing = [f for f in bgl_files if not f.exists()]
    if missing:
        print(f"  SKIP: BGL files not found: {[f.name for f in missing]}")
    else:
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        print(f"  Zipping {len(bgl_files)} BGL data files → {zip_path.name} …")
        subprocess.run(
            ["zip", "-j", str(zip_path)] + [str(f) for f in bgl_files],
            check=True,
        )
        upload(api, repo, zip_path, "bgl_logs.zip")
        zip_path.unlink()

    # ── 4. BAT OS checkpoints ─────────────────────────────────────────────────
    print("\n── Step 6/9  BAT OS checkpoints ──")
    pth_files = list(BAT_OS_DIR.glob("*.pth")) if BAT_OS_DIR.exists() else []
    if not pth_files:
        print(f"  SKIP: no .pth files found in {BAT_OS_DIR}")
    else:
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        print(f"  Zipping {len(pth_files)} OS checkpoints (~3.5 GB) → {zip_path.name} …")
        subprocess.run(
            ["zip", "-j", str(zip_path)] + [str(f) for f in pth_files],
            check=True,
        )
        upload(api, repo, zip_path, "bat/os.zip")
        zip_path.unlink()

    # ── 5. BAT BGL checkpoints ────────────────────────────────────────────────
    print("\n── Step 7/9  BAT BGL checkpoints ──")
    pth_files = list(BAT_BGL_DIR.glob("*.pth")) if BAT_BGL_DIR.exists() else []
    if not pth_files:
        print(f"  SKIP: no .pth files found in {BAT_BGL_DIR}")
    else:
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        print(f"  Zipping {len(pth_files)} BGL checkpoints (~3.5 GB) → {zip_path.name} …")
        subprocess.run(
            ["zip", "-j", str(zip_path)] + [str(f) for f in pth_files],
            check=True,
        )
        upload(api, repo, zip_path, "bat/bgl.zip")
        zip_path.unlink()

    # ── 6. BAT HDFS checkpoints ───────────────────────────────────────────────
    print("\n── Step 8/9  BAT HDFS checkpoints ──")
    pth_files = list(BAT_HDFS_DIR.glob("*.pth")) if BAT_HDFS_DIR.exists() else []
    if not pth_files:
        print(f"  SKIP: no .pth files found in {BAT_HDFS_DIR}")
    else:
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        print(f"  Zipping {len(pth_files)} HDFS checkpoints (~3.5 GB) → {zip_path.name} …")
        subprocess.run(
            ["zip", "-j", str(zip_path)] + [str(f) for f in pth_files],
            check=True,
        )
        upload(api, repo, zip_path, "bat/hdfs.zip")
        zip_path.unlink()

    # ── Q-BAT quantized checkpoints (.pte) ───────────────────────────────────
    print("\n── Step 9/10  Q-BAT quantized checkpoints ──")
    for ds in ("bgl", "hdfs", "os"):
        pte_dir   = QBAT_DIR / ds
        pte_files = sorted(pte_dir.glob("*.pte")) if pte_dir.exists() else []
        if not pte_files:
            print(f"  SKIP {ds}: no .pte files found in {pte_dir}"); continue
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        total_mb = sum(f.stat().st_size for f in pte_files) // 1024 // 1024
        print(f"  Zipping {len(pte_files)} Q-BAT {ds.upper()} models (~{total_mb} MB)…")
        # Store as ds/filename.pte so extraction into checkpoints/qbat/ works cleanly
        import zipfile as _zf
        with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED, compresslevel=6) as z:
            for f in pte_files:
                z.write(f, arcname=f"{ds}/{f.name}")
        upload(api, repo, zip_path, f"qbat/{ds}.zip")
        zip_path.unlink()

    # ── Dashboard demo video ──────────────────────────────────────────────────
    print("\n── Step 10/11  Dashboard demo video ──")
    if DEMO_VIDEO.is_file():
        upload(api, repo, DEMO_VIDEO, "demo.mp4")
    else:
        print(f"  SKIP: {DEMO_VIDEO} not found.")

    # ── Pre-computed npy outputs (prediction results + per-model arrays) ─────────
    print("\n── Step 11/11  Pre-computed inference outputs (npy) ──")
    for ds in ("bgl", "hdfs", "os"):
        out_dir = ROOT / "outputs" / ds
        files   = [out_dir / f for f in OUTPUT_NP_FILES if (out_dir / f).exists()]
        if not files:
            print(f"  SKIP {ds}: no npy files found in {out_dir}"); continue
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        total_mb = sum(f.stat().st_size for f in files) // 1024 // 1024
        print(f"  Zipping {len(files)} {ds.upper()} npy files (~{total_mb} MB)…")
        subprocess.run(["zip", "-j", str(zip_path)] + [str(f) for f in files], check=True)
        upload(api, repo, zip_path, f"outputs/{ds}.zip")
        zip_path.unlink()

    print("\n" + "=" * 60)
    print("  Upload complete!")
    print(f"\n  Repo: https://huggingface.co/datasets/{repo}")
    print("  HF Spaces will download from here on next cold start.")
    print("=" * 60)


if __name__ == "__main__":
    main()
