#!/usr/bin/env bash
# CESAL installer — creates both conda environments and verifies the install.
#
#   ./install.sh                 # create both envs, install deps, run the test suite
#   ./install.sh --with-assets   # also download checkpoints + the ExecuTorch runtime (~5 GB)
#   ./install.sh --edge-only     # skip the cloud env (no GPU on this machine)
#   ./install.sh --help
#
# Environment names can be overridden:  EDGE_ENV=my-edge CLOUD_ENV=my-cloud ./install.sh
set -euo pipefail

EDGE_ENV="${EDGE_ENV:-cesal-edge}"
CLOUD_ENV="${CLOUD_ENV:-cesal-cloud}"
PYTHON_VERSION="3.10"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_ASSETS=0
EDGE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --with-assets) WITH_ASSETS=1 ;;
    --edge-only)   EDGE_ONLY=1 ;;
    --help|-h)     sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
}
# Make `conda activate` usable inside a non-interactive shell.
eval "$(conda shell.bash hook)"

create_env () {                       # $1 = env name, $2 = requirements file, $3 = torch index
  local env="$1" reqs="$2" torch_index="$3"
  if conda env list | awk '{print $1}' | grep -qx "$env"; then
    echo "==> Environment '$env' already exists — reusing it."
  else
    echo "==> Creating environment '$env' (python $PYTHON_VERSION)"
    conda create -y -n "$env" "python=$PYTHON_VERSION"
  fi
  echo "==> Installing $reqs into '$env'"
  conda activate "$env"
  python -m pip install --upgrade pip >/dev/null
  # --extra-index-url is required: the requirements pin local-version wheels
  # (torch==2.6.0+cpu / torch==2.4.0+cu124) that are not on PyPI.
  python -m pip install -r "$ROOT/$reqs" --extra-index-url "$torch_index"
  # --no-deps: the requirements files pin exact torch builds (CPU for edge, cu124 for
  # cloud). A plain editable install would re-resolve torch and replace those pins.
  python -m pip install -e "$ROOT" --no-deps
  conda deactivate
}

echo "=== CESAL install ==========================================="
create_env "$EDGE_ENV" "environment/edge/requirements.txt" "https://download.pytorch.org/whl/cpu"

if [ "$EDGE_ONLY" -eq 0 ]; then
  create_env "$CLOUD_ENV" "environment/cloud/requirements.txt" "https://download.pytorch.org/whl/cu124"
else
  echo "==> Skipping cloud environment (--edge-only)."
fi

if [ "$WITH_ASSETS" -eq 1 ]; then
  echo "==> Downloading checkpoints and the ExecuTorch runtime (this takes a while)"
  conda activate "$EDGE_ENV"
  python "$ROOT/run.py" download
  conda deactivate
else
  echo "==> Skipping asset download. Fetch it later with:"
  echo "      conda activate $EDGE_ENV && python run.py download"
fi

echo "==> Verifying the install (unit tests)"
conda activate "$EDGE_ENV"
python -m pytest "$ROOT/tests" -q
conda deactivate

cat <<EOF

=== Install complete ========================================
Environments: $EDGE_ENV (edge, CPU)$([ "$EDGE_ONLY" -eq 0 ] && echo ", $CLOUD_ENV (cloud, CUDA)")

Point the pipeline at these interpreters:
  export EDGE_PYTHON=\$(conda run -n $EDGE_ENV which python)
  export CLOUD_PYTHON=\$(conda run -n $CLOUD_ENV which python)

Next: a ~2-minute end-to-end check
  ./artifact/claims/claim4_scaled_down.sh

Full reproduction steps are in INSTALL.md; per-claim scripts in artifact/claims/.
EOF
