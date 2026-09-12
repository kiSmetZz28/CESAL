#!/usr/bin/env bash
# Claim 2 — Open-set incident classification on HDFS (paper Section 4.6, Table 7)
#
#   ./claim2_classification.sh [MODEL]
#       no argument  -> all four backbones from configs/llm/hdfs.yaml
#       MODEL        -> one backbone, e.g. qwen2.5-14b-instruct
#
# Expected macro-average F1 (Table 7):
#   Llama-3.1-8B-Instruct  81.19      Qwen2.5-7B-Instruct   81.52
#   Gemma-2-9B-IT          81.71      Qwen2.5-14B-Instruct  83.03   <- CESAL default
#
# Requires a CUDA GPU. Meta-Llama-3.1-8B-Instruct and gemma-2-9b-it are gated on
# Hugging Face: accept their licenses and run `hf auth login` first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CLOUD_PYTHON="${CLOUD_PYTHON:-$HOME/miniconda3/envs/cesal-cloud/bin/python}"
[ -x "$CLOUD_PYTHON" ] || { echo "ERROR: CLOUD_PYTHON not executable. See INSTALL.md step 3." >&2; exit 1; }

echo "=== Claim 2: open-set incident classification (HDFS) ========"
echo "Cloud interpreter: $CLOUD_PYTHON"
"$CLOUD_PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'; \
print('GPU:', torch.cuda.get_device_name(0))"

echo "--- Test set and knowledge base ----------------------------"
echo "  test sequences : data/HDFS/open_set/open_set_test.csv"
echo "  knowledge base : data/HDFS/open_set/top100_split_template_sequences.csv"
echo "  (rebuild both from loghub with: python -m incident_response.data_prep)"

echo "--- Classifying --------------------------------------------"
"$CLOUD_PYTHON" run.py classify ${1:+"$1"}

echo
echo "--- Measured vs. Table 7 -----------------------------------"
"$CLOUD_PYTHON" - <<'PY'
import glob, os
import pandas as pd

ref_path = "outputs/hdfs/llm/table7_reference_metrics.csv"
paper = {"meta-llama/Meta-Llama-3.1-8B-Instruct": 81.19,
         "google/gemma-2-9b-it":                  81.71,
         "Qwen/Qwen2.5-7B-Instruct":              81.52,
         "Qwen/Qwen2.5-14B-Instruct":             83.03}

files = sorted(glob.glob("outputs/hdfs/llm/per_class_metrics_long*.csv")) or \
        sorted(glob.glob("outputs/hdfs/llm/results_*.csv"))
if not files:
    print("  No result files under outputs/hdfs/llm/ — did run.py classify finish?")
else:
    for f in files:
        df = pd.read_csv(f)
        if {"model", "f1"} <= set(df.columns):
            for model, g in df.groupby("model"):
                macro = 100 * g["f1"].mean()
                exp = paper.get(model)
                tag = f"  (Table 7: {exp})" if exp else ""
                print(f"  {model:<42} macro-F1 {macro:6.2f}{tag}")
    print(f"\n  Per-class reference values for direct comparison: {ref_path}")
PY
