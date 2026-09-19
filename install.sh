#!/usr/bin/env bash
# CESAL — one-command installation (README Step 1 + the ExecuTorch runtime).
#
# Creates both Conda environments, installs their pinned requirements, installs
# CESAL into each, and fetches the pre-built ExecuTorch 0.5.0 runtime the edge
# tier executes. Safe to re-run: existing environments are reused.
#
#   ./install.sh
#
# Afterwards:  conda activate cesal-edge && python run.py download os
#
# Linux x86-64 only — the bundled executor_runner is an x86-64 binary
# (on Windows use WSL2). Docker users can skip this entirely; the image at
# Prepared images include the environments; build this revision's Dockerfile
# to include its current source and readiness commands.
set -euo pipefail

EDGE_ENV="${EDGE_ENV:-cesal-edge}"
CLOUD_ENV="${CLOUD_ENV:-cesal-cloud}"
PY_VER=3.10.0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found. Install Miniforge/Miniconda first:"
    echo "       https://github.com/conda-forge/miniforge"; exit 1; }
[ "$(uname -s)" = "Linux" ] || echo "WARNING: only Linux x86-64 is supported; on Windows use WSL2."
[ "$(uname -m)" = "x86_64" ] || echo "WARNING: the pre-built executor_runner is x86-64; $(uname -m) is unsupported."

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

make_env () {          # $1 env name   $2 requirements file   $3 pytorch index
    if conda env list | awk '{print $1}' | grep -qx "$1"; then
        echo ">>> $1 already exists — reusing"
    else
        echo ">>> creating $1 (python $PY_VER)"
        conda create -yn "$1" "python=$PY_VER"
    fi
    echo ">>> installing $2 into $1"
    conda run -n "$1" --no-capture-output pip install -r "$2" --extra-index-url "$3"
    conda run -n "$1" --no-capture-output pip install -e .
}

make_env "$EDGE_ENV"  environment/edge/requirements.txt  https://download.pytorch.org/whl/cpu
make_env "$CLOUD_ENV" environment/cloud/requirements.txt https://download.pytorch.org/whl/cu124

# ExecuTorch 0.5.0 + its bundled torchao: not on PyPI (PEP 440 local versions),
# so they are fetched and installed into the edge environment here.
echo ">>> installing the ExecuTorch 0.5.0 runtime into $EDGE_ENV"
conda run -n "$EDGE_ENV" --no-capture-output \
    env PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    python tools/setup_executorch.py

echo ">>> verifying"
conda run -n "$EDGE_ENV" --no-capture-output python -c "import executorch.exir, torchao; print('    executorch + torchao OK')"
test -x cesal_inference_pipeline/executorch/cmake-out/executor_runner \
    && echo "    executor_runner OK" \
    || { echo "ERROR: executor_runner missing — the ExecuTorch install did not complete."; exit 1; }
conda run -n "$EDGE_ENV" --no-capture-output python run.py check

cat <<MSG

Environment setup and core software checks completed:

    conda activate $EDGE_ENV     # edge tier: detection, conversion, dashboard
    conda activate $CLOUD_ENV    # cloud tier: BAT ensemble, LLM classification

Next:
    conda activate $EDGE_ENV
    python run.py download os    # fetch the published checkpoints
    python run.py smoke os       # verify real edge-to-cloud detection on a small input
    python run.py infer os       # Table 3 OpenStack Edge and CESAL rows (~3 h 20 m)
MSG
