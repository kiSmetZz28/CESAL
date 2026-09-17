# CESAL artifact image — both Conda environments and the ExecuTorch runtime pre-installed,
# built by the same commands as README Step 1 and `run.py download`.
#
# Checkpoints and LLM weights are not baked in: `run.py download` and the first
# `run.py classify` fetch them into the volumes mounted in the README's docker run command.
#
# linux/amd64 only: the pre-built ExecuTorch executor_runner is an x86-64 binary.
# Ubuntu 24.04 because executor_runner and the ExecuTorch bindings need glibc >= 2.38.
FROM --platform=linux/amd64 ubuntu:24.04

LABEL org.opencontainers.image.source="https://github.com/kiSmetZz28/CESAL" \
      org.opencontainers.image.description="CESAL (ACSAC 2026) artifact: cesal-edge and cesal-cloud environments with ExecuTorch 0.5.0" \
      org.opencontainers.image.licenses="MIT"

ARG MINIFORGE_VERSION=26.7.2-0
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

# Compilers and git are needed by ExecuTorch's install_requirements.py (builds the pip package).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

# ── Environments (README Step 1) ──────────────────────────────────────────────
# Built from the requirement files alone, so source changes do not rebuild them.
COPY environment/ environment/

RUN conda create -yn cesal-cloud python=3.10.0 \
    && conda clean -afy \
    && /opt/conda/envs/cesal-cloud/bin/pip install -r environment/cloud/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124

RUN conda create -yn cesal-edge python=3.10.0 \
    && conda clean -afy \
    && /opt/conda/envs/cesal-edge/bin/pip install -r environment/edge/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# cesal-edge first on PATH, as `conda activate cesal-edge` would do: ExecuTorch's build calls
# the cmake that the edge requirements install, and one-off `docker run ... python` commands
# use this environment.
ENV PATH=/opt/conda/envs/cesal-edge/bin:$PATH

# ── ExecuTorch 0.5.0 runtime (what `run.py download` installs) ────────────────
# setup_executorch.py downloads the pre-built tree and runs its install_requirements.py.
# PIP_EXTRA_INDEX_URL lets its final re-install of the edge requirements find torch 2.6.0+cpu.
# The checks fail the build if the runner or the exir package (used by `run.py convert`) is missing.
# pip-out/ is build scratch, removed in the same layer to keep the image smaller.
COPY tools/__init__.py tools/setup_executorch.py tools/
RUN PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu python tools/setup_executorch.py \
    && test -x cesal_inference_pipeline/executorch/cmake-out/executor_runner \
    && python -c "import executorch.exir, torchao" \
    && rm -rf cesal_inference_pipeline/executorch/pip-out

# ── Source ────────────────────────────────────────────────────────────────────
COPY . .
RUN /opt/conda/envs/cesal-edge/bin/pip install -e . \
    && /opt/conda/envs/cesal-cloud/bin/pip install -e .

# The edge tier spawns the cloud tier with this interpreter; the dashboard reads the other two.
# expandable_segments avoids fragmentation OOMs when Qwen2.5-14B is partly offloaded on a 16 GB GPU.
ENV CESAL_CLOUD_PYTHON=/opt/conda/envs/cesal-cloud/bin/python \
    EDGE_PYTHON=/opt/conda/envs/cesal-edge/bin/python \
    CLOUD_PYTHON=/opt/conda/envs/cesal-cloud/bin/python \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# `conda activate cesal-edge` / `cesal-cloud` work as in the README; a new shell starts in cesal-edge.
RUN conda init bash && echo "conda activate cesal-edge" >> /root/.bashrc

EXPOSE 8765
CMD ["bash"]
