#!/usr/bin/env python3
"""Deploy CECO-LAD to Hugging Face Spaces.

Uses HfApi.upload_folder() — no git, no git-lfs, no password needed.
Authentication is done with an access token (Write permission).

Run from the project root:
    python tools/deploy/deploy_hf.py kiSmetZz
    python tools/deploy/deploy_hf.py kiSmetZz ceco-lad   # custom space name
"""
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "huggingface_hub>=0.20"], check=True)
    from huggingface_hub import HfApi, login

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python tools/deploy/deploy_hf.py <hf-username> [space-name]")
    print("Example: python tools/deploy/deploy_hf.py kiSmetZz")
    sys.exit(1)

hf_user    = sys.argv[1]
space_name = sys.argv[2] if len(sys.argv) > 2 else "ceco-lad"
repo_id    = f"{hf_user}/{space_name}"
# This file lives in tools/deploy/, so the project root is two levels up.
root       = Path(__file__).resolve().parent.parent.parent

print(f"\nDeploying CECO-LAD  →  {repo_id}")
print(f"Public URL after build:  https://{hf_user}-{space_name}.hf.space\n")

# ── Step 1: Login ─────────────────────────────────────────────────────────────
print("Step 1/3  Log in to Hugging Face")
print("  Get your token at https://huggingface.co/settings/tokens")
print("  (click New Token → Write access → copy the token)\n")
login()          # prompts for token; token is cached after first use

# ── Step 2: Create Space ──────────────────────────────────────────────────────
api = HfApi()
print(f"\nStep 2/3  Creating Space '{repo_id}' (Docker SDK, public)…")
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
)
print("  Space ready.")

# ── Step 3: Upload ────────────────────────────────────────────────────────────
print("\nStep 3/3  Uploading files…")
print("  Skipped: checkpoints/bat/         (3.5 GB — downloaded at first launch)")
print("  Skipped: checkpoints/qbat/        (218 MB — downloaded at first launch)")
print("  Skipped: executorch build tree     (305 MB compiled libs — not needed in repo)")
print("  executor_runner: downloaded at container startup from HF assets repo")
print("  Uploading: code + data + outputs (~86 MB, well under 1 GB Space limit)")
print("  No git-lfs needed — large files are chunked automatically.\n")

# executor_runner is NOT uploaded here — it is downloaded at container startup
# by tools/deploy/spaces_startup.py using gdown (same mechanism as BAT checkpoints).
# This avoids the timing issue where upload_folder triggers HF's auto-build
# before a separately uploaded binary can arrive.
HF_README = f"""\
---
title: CECO-LAD
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

{(root / 'README.md').read_text()}"""

api.upload_file(
    path_or_fileobj=HF_README.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="space",
    commit_message="update README with HF Space config",
)

api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=str(root),
    # Delete any prior manual uploads of the demo video — the real blob lives
    # in the assets dataset repo and is fetched at container startup. Leaving a
    # copy here just bakes a stale LFS pointer (or stale file) into the build.
    # Also delete the old `inference_pipeline/` tree left over from before the
    # rename to `ceco_lad_inference_pipeline/`.
    delete_patterns=[
        "dashboard/static/demo.mp4",
        # Clean up the old inference_pipeline/ tree left over from before the
        # rename to ceco_lad_inference_pipeline/.
        "inference_pipeline/**",
        # Clean up any executorch download artefacts that may have been pushed
        # in prior deploys (under either the old or new folder name).
        "inference_pipeline/executorch.zip",
        "ceco_lad_inference_pipeline/executorch.zip",
        # Raw OpenStack log files inadvertently uploaded in prior deploys.
        "data/OpenStack/raw/**",
    ],
    ignore_patterns=[
        "README.md",             # already uploaded above with HF front matter
        # BAT checkpoints (3.5 GB) — downloaded at runtime from HF dataset repo
        "checkpoints/bat/**",
        # Q-BAT checkpoints (218 MB) — downloaded at runtime from HF dataset repo
        "checkpoints/qbat/**",
        # Entire executorch directory — compiled libs not needed in Space repo
        "ceco_lad_inference_pipeline/executorch/**",
        # The 696 MB executorch.zip is a download artefact left behind by
        # `run.py download` / `launch_dashboard.py`; it lives next to the
        # executorch/ folder so the pattern above does NOT match it.
        "ceco_lad_inference_pipeline/executorch.zip",
        # Raw OpenStack log files — fetched at runtime by spaces_startup.py
        # from the HF assets dataset repo; not needed in the Space repo.
        "data/OpenStack/raw/**",
        # Local database — rebuilt from scratch on startup
        "dashboard/ceco_lad.db",
        # Demo video (~117 MB) — downloaded at runtime from HF assets dataset repo.
        # If shipped via the Space repo it ends up as an LFS pointer (~134 B)
        # inside the Docker build context, which breaks playback.
        "dashboard/static/demo.mp4",
        # All output npy arrays are excluded from the Space repo to stay within
        # the 1 GB limit. They are all downloaded at container startup from the
        # HF assets dataset repo via tools/deploy/spaces_startup.py.
        "outputs/**/*.npy",
        # Dev / CI artefacts
        ".git/**",
        ".claude/**",
        "**/__pycache__/**",
        "**/*.pyc",
        ".pytest_cache/**",
        ".mypy_cache/**",
        "logs/**",
        ".vscode/**",
        ".idea/**",
        "environment/**",
        "pictures/**",
    ],
)

print("\n" + "=" * 60)
print("  Upload complete!")
print(f"\n  Space:      https://huggingface.co/spaces/{repo_id}")
print(f"  Public URL: https://{hf_user}-{space_name}.hf.space")
print("\n  HF is building the Docker image (~10 min first time).")
print("  On first visitor: BAT checkpoints download automatically (~10 min).")
print("  Every visit after that is instant.")
print("=" * 60)
