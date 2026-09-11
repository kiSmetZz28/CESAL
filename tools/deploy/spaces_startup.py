#!/usr/bin/env python3
"""First-launch startup script for HF Spaces / cloud deployments.

Downloads large assets from a Hugging Face dataset repository (no rate limits,
no quota issues unlike Google Drive).  Subsequent starts skip downloads.

Assets repo: kiSmetZz/ceco-lad-assets  (dataset, public)
  executor_runner      — ExecuTorch C++ binary     (~43 MB)
  bat/os.zip           — BAT OS checkpoints        (~3.5 GB)
  bat/bgl.zip          — BAT BGL checkpoints       (~3.5 GB)
  bat/hdfs.zip         — BAT HDFS checkpoints      (~3.5 GB)
  os_logs.zip          — OpenStack raw logs        (~59 MB)
  bgl_logs.zip         — BGL processed data files  (~200 MB)
  bgl_raw.zip          — BGL raw log (BGL.log)     (~709 MB)
  hdfs_raw.zip         — HDFS raw log (HDFS.log)   (~1.5 GB)
  demo.mp4             — Dashboard demo video      (~117 MB)
  (HDFS processed data files are bundled in the repo)

Usage (set automatically via Dockerfile CMD):
    python tools/deploy/spaces_startup.py
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# This file lives in tools/deploy/, so the project root is two levels up.
ROOT = Path(__file__).resolve().parent.parent.parent

# HF dataset repo that holds all large assets
HF_ASSETS_REPO = "kiSmetZz/ceco-lad-assets"

# ── BAT checkpoints ───────────────────────────────────────────────────────────
OS_CKPT_DIR   = ROOT / "checkpoints" / "bat" / "os"
BGL_CKPT_DIR  = ROOT / "checkpoints" / "bat" / "bgl"
HDFS_CKPT_DIR = ROOT / "checkpoints" / "bat" / "hdfs"
MIN_CKPTS     = 81

# ── Q-BAT quantized checkpoints ───────────────────────────────────────────────
QBAT_DIR      = ROOT / "checkpoints" / "qbat"
MIN_QBAT      = 3   # 3 .pte files per dataset

# ── executor_runner binary ────────────────────────────────────────────────────
RUNNER_PATH = ROOT / "ceco_lad_inference_pipeline" / "executorch" / "cmake-out" / "executor_runner"

# ── Dashboard demo video ──────────────────────────────────────────────────────
DEMO_VIDEO_PATH = ROOT / "dashboard" / "static" / "demo.mp4"
DEMO_VIDEO_MIN_BYTES = 1_000_000   # full file ~117 MB; LFS pointer is ~134 B

# ── OpenStack raw log files ───────────────────────────────────────────────────
OS_RAW_DIR   = ROOT / "data" / "OpenStack" / "raw"
OS_RAW_FILES = ["openstack_normal1.log", "openstack_normal2.log", "openstack_abnormal.log"]

# ── BGL data files ────────────────────────────────────────────────────────────
BGL_DATA_DIR   = ROOT / "data" / "BGL"
BGL_DATA_FILES = ["bgl_train.txt", "bgl_test_normal.txt", "bgl_test_abnormal.txt"]

# ── Raw structured log CSVs (for raw log display in the dashboard) ─────────────
LOG_DATA_ROOT    = Path.home() / "Desktop" / "Log Data"
BGL_LOG_DIR      = LOG_DATA_ROOT / "BGL"
BGL_SPLIT_DIR    = BGL_LOG_DIR / "split"
BGL_SPLIT_FILES  = ["bgl_train.log", "bgl_test_normal.log", "bgl_test_abnormal.log"]
HDFS_SPLIT_DIR   = LOG_DATA_ROOT          # train/test_normal/test_abnormal.log live here
HDFS_SPLIT_FILES = ["train.log", "test_normal.log", "test_abnormal.log"]


def _hf_download(filename: str, local_path: Path) -> bool:
    """Download one file from the HF assets repo to local_path."""
    try:
        from huggingface_hub import hf_hub_download
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hf_hub_download(
            repo_id=HF_ASSETS_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_path.parent),
        )
        # hf_hub_download may put file in a subdirectory — move it if needed
        downloaded = Path(tmp)
        if downloaded != local_path:
            downloaded.rename(local_path)
        return local_path.exists()
    except Exception as exc:
        print(f"[startup] HF download error ({filename}): {exc}", flush=True)
        return False


# ── executor_runner ───────────────────────────────────────────────────────────

def _runner_present() -> bool:
    return RUNNER_PATH.is_file() and os.access(str(RUNNER_PATH), os.X_OK)


def _download_runner() -> bool:
    print("[startup] executor_runner not found. Downloading from HF (~43 MB)…", flush=True)
    if _hf_download("executor_runner", RUNNER_PATH):
        os.chmod(str(RUNNER_PATH), 0o755)
        print("[startup] executor_runner ready.", flush=True)
        return True
    print("[startup] executor_runner download failed — edge inference unavailable.", flush=True)
    return False


# ── Dashboard demo video ──────────────────────────────────────────────────────

def _demo_video_present() -> bool:
    """The Docker COPY ships the LFS pointer (~134 B); reject anything that tiny."""
    return DEMO_VIDEO_PATH.is_file() and DEMO_VIDEO_PATH.stat().st_size >= DEMO_VIDEO_MIN_BYTES


def _download_demo_video() -> bool:
    print("[startup] Demo video not found (or only LFS pointer). Downloading from HF (~117 MB)…",
          flush=True)
    # Remove the stale pointer first so hf_hub_download writes the real blob.
    if DEMO_VIDEO_PATH.exists():
        DEMO_VIDEO_PATH.unlink(missing_ok=True)
    if _hf_download("demo.mp4", DEMO_VIDEO_PATH) and _demo_video_present():
        print("[startup] Demo video ready.", flush=True)
        return True
    print("[startup] Demo video download failed — Watch Demo button will be broken.", flush=True)
    return False


# ── OpenStack raw logs ────────────────────────────────────────────────────────

def _os_logs_present() -> bool:
    if not OS_RAW_DIR.exists():
        return False
    # Minimum sizes for the full log files (truncated/sample versions are too small)
    min_bytes = {
        "openstack_normal1.log":  10_000_000,   # full ~14 MB
        "openstack_normal2.log":  30_000_000,   # full ~38 MB
        "openstack_abnormal.log":  4_000_000,   # full ~5.2 MB
    }
    return all(
        (OS_RAW_DIR / f).is_file() and (OS_RAW_DIR / f).stat().st_size >= min_bytes.get(f, 0)
        for f in OS_RAW_FILES
    )


def _download_os_logs() -> bool:
    print("[startup] OpenStack raw logs not found. Downloading from HF (~59 MB)…", flush=True)
    zip_path = ROOT / "data" / "OpenStack" / "os_logs.zip"
    if not _hf_download("os_logs.zip", zip_path):
        print("[startup] OpenStack log download failed — synthetic messages will be used.", flush=True)
        return False
    try:
        OS_RAW_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(OS_RAW_DIR))
        zip_path.unlink(missing_ok=True)
        if _os_logs_present():
            print("[startup] OpenStack raw logs ready.", flush=True)
            return True
        print("[startup] Extraction finished but files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] OpenStack log extraction error: {exc}", flush=True)
        return False


# ── BGL split logs ────────────────────────────────────────────────────────────

def _bgl_split_present() -> bool:
    if not BGL_SPLIT_DIR.exists():
        return False
    min_bytes = {
        "bgl_train.log":        400_000_000,  # full ~489 MB
        "bgl_test_normal.log":  100_000_000,  # full ~172 MB
        "bgl_test_abnormal.log": 40_000_000,  # full ~55 MB
    }
    return all(
        (BGL_SPLIT_DIR / f).is_file() and (BGL_SPLIT_DIR / f).stat().st_size >= min_bytes.get(f, 0)
        for f in BGL_SPLIT_FILES
    )


def _download_bgl_raw() -> bool:
    print("[startup] BGL split log files not found. Downloading from HF (~200–400 MB)…",
          flush=True)
    zip_path = BGL_LOG_DIR / "bgl_raw.zip"
    BGL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _hf_download("bgl_raw.zip", zip_path):
        print("[startup] BGL raw log download failed — raw log panel will be empty.",
              flush=True)
        return False
    try:
        BGL_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(BGL_SPLIT_DIR))
        zip_path.unlink(missing_ok=True)
        if _bgl_split_present():
            print("[startup] BGL split log files ready.", flush=True)
            return True
        print("[startup] BGL extraction finished but split files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] BGL raw log extraction error: {exc}", flush=True)
        return False


# ── HDFS split logs ───────────────────────────────────────────────────────────

def _hdfs_split_present() -> bool:
    if not HDFS_SPLIT_DIR.exists():
        return False
    min_bytes = {
        "train.log":        10_000_000,   # full ~12 MB
        "test_normal.log": 1_000_000_000, # full ~1.4 GB
        "test_abnormal.log":  30_000_000, # full ~38 MB
    }
    return all(
        (HDFS_SPLIT_DIR / f).is_file() and (HDFS_SPLIT_DIR / f).stat().st_size >= min_bytes.get(f, 0)
        for f in HDFS_SPLIT_FILES
    )


def _download_hdfs_split() -> bool:
    print("[startup] HDFS split log files not found. Downloading from HF (~1.6 GB)…",
          flush=True)
    zip_path = HDFS_SPLIT_DIR / "hdfs_split.zip"
    HDFS_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if not _hf_download("hdfs_split.zip", zip_path):
        print("[startup] HDFS split log download failed — raw log panel will be empty.",
              flush=True)
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(HDFS_SPLIT_DIR))
        zip_path.unlink(missing_ok=True)
        if _hdfs_split_present():
            print("[startup] HDFS split log files ready.", flush=True)
            return True
        print("[startup] HDFS extraction finished but split files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] HDFS split log extraction error: {exc}", flush=True)
        return False


# ── BAT checkpoints ───────────────────────────────────────────────────────────

def _ckpts_present(ckpt_dir: Path) -> bool:
    return ckpt_dir.exists() and len(list(ckpt_dir.glob("*.pth"))) >= MIN_CKPTS


def _download_checkpoints(dataset: str, ckpt_dir: Path) -> bool:
    print(
        f"[startup] BAT {dataset.upper()} checkpoints not found. Downloading from HF (~3.5 GB) — "
        "this takes 5–15 min on first launch.",
        flush=True,
    )
    zip_path = ROOT / "checkpoints" / "bat" / f"{dataset}.zip"
    if not _hf_download(f"bat/{dataset}.zip", zip_path):
        print(f"[startup] {dataset.upper()} checkpoint download failed.", flush=True)
        return False
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(ckpt_dir))
        zip_path.unlink(missing_ok=True)
        n = len(list(ckpt_dir.glob("*.pth")))
        if n >= MIN_CKPTS:
            print(f"[startup] {n} BAT {dataset.upper()} checkpoints ready.", flush=True)
            return True
        print(f"[startup] Only {n}/{MIN_CKPTS} {dataset.upper()} checkpoints extracted.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] {dataset.upper()} checkpoint extraction error: {exc}", flush=True)
        return False


# ── Q-BAT quantized checkpoints ───────────────────────────────────────────────

def _qbat_present(dataset: str) -> bool:
    d = QBAT_DIR / dataset
    return d.exists() and len(list(d.glob("*.pte"))) >= MIN_QBAT


def _download_qbat(dataset: str) -> bool:
    print(f"[startup] Q-BAT {dataset.upper()} models not found. Downloading from HF (~50–100 MB)…",
          flush=True)
    zip_path = QBAT_DIR / f"{dataset}.zip"
    if not _hf_download(f"qbat/{dataset}.zip", zip_path):
        print(f"[startup] Q-BAT {dataset.upper()} download failed — edge inference unavailable.",
              flush=True)
        return False
    try:
        dest = QBAT_DIR / dataset
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Zip stores files as ds/filename.pte; extract flat into dest/
            for member in zf.namelist():
                fname = Path(member).name
                if fname.endswith(".pte"):
                    with zf.open(member) as src, open(dest / fname, "wb") as out:
                        out.write(src.read())
        zip_path.unlink(missing_ok=True)
        n = len(list(dest.glob("*.pte")))
        if n >= MIN_QBAT:
            print(f"[startup] {n} Q-BAT {dataset.upper()} models ready.", flush=True)
            return True
        print(f"[startup] Only {n}/{MIN_QBAT} Q-BAT {dataset.upper()} models extracted.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] Q-BAT {dataset.upper()} extraction error: {exc}", flush=True)
        return False


# ── BGL data files ────────────────────────────────────────────────────────────

def _bgl_data_present() -> bool:
    return BGL_DATA_DIR.exists() and all(
        (BGL_DATA_DIR / f).is_file() for f in BGL_DATA_FILES
    )


def _download_bgl_data() -> bool:
    print("[startup] BGL data files not found. Downloading from HF (~200 MB)…", flush=True)
    zip_path = BGL_DATA_DIR / "bgl_logs.zip"
    if not _hf_download("bgl_logs.zip", zip_path):
        print("[startup] BGL data download failed — BGL dataset unavailable.", flush=True)
        return False
    try:
        BGL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(BGL_DATA_DIR))
        zip_path.unlink(missing_ok=True)
        if _bgl_data_present():
            print("[startup] BGL data files ready.", flush=True)
            return True
        print("[startup] BGL extraction finished but files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] BGL data extraction error: {exc}", flush=True)
        return False


# ── Pre-computed npy outputs ──────────────────────────────────────────────────

OUTPUT_NP_FILES = [
    "ground_truth.npy", "edge_preds.npy", "edge_preds_raw.npy",
    "hybrid_preds.npy", "cloud_preds.npy",
    "energy_matrix.npy", "routed_indices.npy",
    "edge_preds_per_model.npy", "cloud_preds_per_model.npy",
]


def _outputs_present(ds: str) -> bool:
    out_dir = ROOT / "outputs" / ds
    required = [
        "ground_truth.npy", "edge_preds.npy", "hybrid_preds.npy",
        "cloud_preds.npy", "routed_indices.npy",
        "edge_preds_per_model.npy", "cloud_preds_per_model.npy",
    ]
    return all((out_dir / f).exists() for f in required)


def _download_outputs(ds: str) -> bool:
    print(f"[startup] {ds.upper()} inference outputs not found. Downloading from HF…",
          flush=True)
    out_dir  = ROOT / "outputs" / ds
    zip_path = out_dir / f"{ds}_outputs.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _hf_download(f"outputs/{ds}.zip", zip_path):
        print(f"[startup] {ds.upper()} outputs download failed — prediction lookup unavailable.",
              flush=True)
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(out_dir))
        zip_path.unlink(missing_ok=True)
        if _outputs_present(ds):
            present = [f for f in OUTPUT_NP_FILES if (out_dir / f).exists()]
            print(f"[startup] {ds.upper()} outputs ready ({len(present)} npy files).", flush=True)
            return True
        print(f"[startup] {ds.upper()} outputs extraction incomplete.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] {ds.upper()} outputs extraction error: {exc}", flush=True)
        return False


# ── Startup ───────────────────────────────────────────────────────────────────

def _clear_precomputed_results() -> None:
    pass


def _launch_app() -> None:
    port = int(os.getenv("PORT", "7860"))
    print(f"[startup] Starting CECO-LAD dashboard on port {port} …", flush=True)
    os.environ["PORT"] = str(port)
    os.execv(sys.executable, [sys.executable, str(ROOT / "dashboard" / "app.py")])


if __name__ == "__main__":
    _clear_precomputed_results()

    for _ds in ("bgl", "hdfs", "os"):
        if _outputs_present(_ds):
            print(f"[startup] {_ds.upper()} inference outputs present — skipping download.",
                  flush=True)
        else:
            _download_outputs(_ds)

    if _os_logs_present():
        print("[startup] OpenStack raw logs present — skipping download.", flush=True)
    else:
        _download_os_logs()

    if _bgl_data_present():
        print("[startup] BGL data files present — skipping download.", flush=True)
    else:
        _download_bgl_data()

    if _bgl_split_present():
        print("[startup] BGL split log files present — skipping download.", flush=True)
    else:
        _download_bgl_raw()

    if _hdfs_split_present():
        print("[startup] HDFS split log files present — skipping download.", flush=True)
    else:
        _download_hdfs_split()

    if _runner_present():
        print("[startup] executor_runner present — skipping download.", flush=True)
    else:
        _download_runner()

    if _demo_video_present():
        print("[startup] Demo video present — skipping download.", flush=True)
    else:
        _download_demo_video()

    for _ds in ("bgl", "hdfs", "os"):
        if _qbat_present(_ds):
            n = len(list((QBAT_DIR / _ds).glob("*.pte")))
            print(f"[startup] {n} Q-BAT {_ds.upper()} models present — skipping download.", flush=True)
        else:
            _download_qbat(_ds)

    if _ckpts_present(OS_CKPT_DIR):
        print(f"[startup] {len(list(OS_CKPT_DIR.glob('*.pth')))} BAT OS checkpoints present — skipping download.", flush=True)
    else:
        _download_checkpoints("os", OS_CKPT_DIR)

    if _ckpts_present(BGL_CKPT_DIR):
        print(f"[startup] {len(list(BGL_CKPT_DIR.glob('*.pth')))} BAT BGL checkpoints present — skipping download.", flush=True)
    else:
        _download_checkpoints("bgl", BGL_CKPT_DIR)

    if _ckpts_present(HDFS_CKPT_DIR):
        print(f"[startup] {len(list(HDFS_CKPT_DIR.glob('*.pth')))} BAT HDFS checkpoints present — skipping download.", flush=True)
    else:
        _download_checkpoints("hdfs", HDFS_CKPT_DIR)

    _launch_app()
