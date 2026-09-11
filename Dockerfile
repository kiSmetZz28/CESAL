FROM python:3.10-slim

WORKDIR /app

# libstdc++6 is required by the pre-compiled executor_runner (ExecuTorch C++ binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (saves ~1.3 GB vs the default CUDA build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Dashboard + inference dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.100" \
    "uvicorn[standard]>=0.20" \
    "aiofiles>=23.0" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "pyyaml>=6.0" \
    "pandas>=1.5" \
    "tqdm>=4.0" \
    "gdown>=4.6" \
    "huggingface_hub>=0.20"

# ── Core model code ───────────────────────────────────────────────────────────
COPY ceco_core/ /app/ceco_core/

# ── Inference pipeline ────────────────────────────────────────────────────────
COPY ceco_lad_inference_pipeline/__init__.py       /app/ceco_lad_inference_pipeline/__init__.py
COPY ceco_lad_inference_pipeline/lad_bat_cloud.py  /app/ceco_lad_inference_pipeline/lad_bat_cloud.py
COPY ceco_lad_inference_pipeline/routing.py        /app/ceco_lad_inference_pipeline/routing.py
COPY ceco_lad_inference_pipeline/lad_qbat_edge.py  /app/ceco_lad_inference_pipeline/lad_qbat_edge.py
COPY ceco_lad_inference_pipeline/run.py            /app/ceco_lad_inference_pipeline/run.py
COPY ceco_lad_inference_pipeline/evaluate.py       /app/ceco_lad_inference_pipeline/evaluate.py

# Create the directory for executor_runner (downloaded at first launch by tools/deploy/spaces_startup.py)
RUN mkdir -p /app/ceco_lad_inference_pipeline/executorch/cmake-out

# ── Dashboard ─────────────────────────────────────────────────────────────────
COPY dashboard/app.py          /app/dashboard/app.py
COPY dashboard/db.py           /app/dashboard/db.py
COPY dashboard/ingest.py       /app/dashboard/ingest.py
COPY dashboard/index.html      /app/dashboard/index.html
COPY dashboard/bat_predict.py  /app/dashboard/bat_predict.py
COPY dashboard/cloud_runner.py /app/dashboard/cloud_runner.py
COPY dashboard/demo_runner.py  /app/dashboard/demo_runner.py
COPY dashboard/static/         /app/dashboard/static/

# ── Static data ───────────────────────────────────────────────────────────────
COPY outputs/  /app/outputs/
COPY data/     /app/data/
COPY configs/  /app/configs/

# ── Q-BAT checkpoints downloaded at container startup by tools/deploy/spaces_startup.py ────
# (not bundled in the image — kept in HF assets repo to stay under 1 GB limit)

# ── Download helper + startup (fetches Q-BAT + BAT checkpoints at first launch) ─
COPY tools/download_checkpoints.py        /app/tools/download_checkpoints.py
COPY tools/deploy/spaces_startup.py       /app/tools/deploy/spaces_startup.py

# ── Environment ───────────────────────────────────────────────────────────────
ENV EDGE_PYTHON=/usr/local/bin/python
ENV CLOUD_PYTHON=/usr/local/bin/python
ENV DEMO_MODE=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "tools/deploy/spaces_startup.py"]
