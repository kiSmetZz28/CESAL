"""Fetch and install the pre-built ExecuTorch 0.5.0 runtime.

The edge stage runs Q-BAT `.pte` models through ExecuTorch, so `run.py download`
and `run.py infer` need this whether or not the web dashboard is ever started.
It lives here, next to the other asset fetchers, so the terminal pipeline does
not import anything from the dashboard launcher; `launch_dashboard.py` imports
it from here too, and both callers share one implementation.

Usage (from project root):
  python tools/setup_executorch.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXECUTORCH_DIR    = ROOT / "cesal_inference_pipeline" / "executorch"
EXECUTOR_RUNNER   = EXECUTORCH_DIR / "cmake-out" / "executor_runner"
EXECUTORCH_GDRIVE = "1YyFOhLxOYOJJCxN6yxTEyMKSHhgaLWuh"  # ExecuTorch 0.5.0 pre-built


def executorch_present() -> bool:
    """True when the pre-built C++ executor_runner is installed."""
    return EXECUTOR_RUNNER.is_file()


def ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown>=4.6"], check=True)


def gdrive_download(file_id: str, dest: Path) -> bool:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(url=f"https://drive.google.com/uc?id={file_id}",
                       output=str(dest), quiet=False)
        return dest.exists()
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return False


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(out_dir))
    zip_path.unlink(missing_ok=True)


def install_python_bindings() -> bool:
    """Optional: the edge stage falls back to the C++ runner when these are absent."""
    install_script = EXECUTORCH_DIR / "install_requirements.py"
    if not install_script.exists():
        print("  WARNING: install_requirements.py not found — Python bindings skipped.")
        return False
    print("  Installing ExecuTorch Python bindings …")
    result = subprocess.run([sys.executable, str(install_script)], cwd=str(EXECUTORCH_DIR))
    if result.returncode == 0:
        edge_req = ROOT / "environment" / "edge" / "requirements.txt"
        if edge_req.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "-r", str(edge_req)], check=False)
        print("  ExecuTorch Python bindings installed.")
        return True
    print("  WARNING: Python binding install failed — C++ executor_runner fallback will be used.")
    return False


def setup_executorch(prefix: str = "") -> bool:
    """Install the runtime if it is missing. `prefix` labels the line for a caller
    that prints a numbered sequence (the dashboard launcher does)."""
    bindings_ok = False
    try:
        from executorch.runtime import Runtime  # noqa: F401
        bindings_ok = True
    except ImportError:
        pass

    if executorch_present() and bindings_ok:
        print(f"{prefix}ExecuTorch — already present, skipping.")
        return True

    if not executorch_present():
        print(f"{prefix}Downloading ExecuTorch 0.5.0 (~1.4 GB) from Google Drive …")
        ensure_gdown()
        zip_path = ROOT / "cesal_inference_pipeline" / "executorch.zip"
        if not gdrive_download(EXECUTORCH_GDRIVE, zip_path):
            print("  WARNING: ExecuTorch download failed — edge inference unavailable.")
            return False
        extract_zip(zip_path, ROOT / "cesal_inference_pipeline")
        if not executorch_present():
            print("  WARNING: executor_runner not found after extraction.")
            return False

    if not bindings_ok:
        install_python_bindings()

    print("  ExecuTorch ready.")
    return True


if __name__ == "__main__":
    sys.exit(0 if setup_executorch() else 1)
