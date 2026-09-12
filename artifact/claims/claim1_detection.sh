#!/usr/bin/env bash
# Claim 1 — Log-based incident detection (paper Section 4.2, Table 3)
#
#   ./claim1_detection.sh [hdfs|os]        default: os (much faster than hdfs)
#
# Expected (Table 3, point-adjusted):
#   HDFS       Q-BAT edge-only  P 99.06  R 100.00  F1 99.53
#              CESAL (10% routed) P 99.96  R 100.00  F1 99.98
#   OpenStack  Q-BAT edge-only  P 98.09  R 100.00  F1 99.03
#              CESAL (10% routed) P 99.90  R 100.00  F1 99.95
set -euo pipefail
DS="${1:-os}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EDGE_PYTHON="${EDGE_PYTHON:-$HOME/miniconda3/envs/cesal-edge/bin/python}"
CLOUD_PYTHON="${CLOUD_PYTHON:-$HOME/miniconda3/envs/cesal-cloud/bin/python}"
export EDGE_PYTHON CLOUD_PYTHON

echo "=== Claim 1: detection on ${DS^^} ==========================="
echo "Edge interpreter : $EDGE_PYTHON"
echo "Cloud interpreter: $CLOUD_PYTHON"
[ -x "$EDGE_PYTHON" ] || { echo "ERROR: EDGE_PYTHON not executable. See INSTALL.md step 3." >&2; exit 1; }

if [ "$DS" = "hdfs" ]; then
  cat <<'WARN'

WARNING: full HDFS is 221,540 windows. Without the ExecuTorch Python bindings the edge
stage invokes the C++ executor_runner once per window per model, which takes roughly a day.
Use ./claim4_scaled_down.sh for a fast end-to-end check, or run this on OpenStack.

WARN
fi

echo "--- Running the collaborative pipeline (edge -> routing -> cloud -> hybrid) ---"
"$EDGE_PYTHON" -m cesal_inference_pipeline.run --config "configs/inference/${DS}.yaml"

echo
echo "--- Measured vs. Table 3 -----------------------------------"
"$EDGE_PYTHON" - "$DS" <<'PY'
import sys, numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf

ds = sys.argv[1]
out = f"outputs/{ds}"
gt  = np.load(f"{out}/ground_truth.npy").reshape(-1).astype(int)

def point_adjust(gt, pred):
    """Protocol inherited from Anomaly Transformer; see training_pipeline/solver.py:394."""
    pred = pred.astype(int).copy(); state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not state:
            state = True
            for j in range(i, 0, -1):
                if gt[j] == 0: break
                pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0: break
                pred[j] = 1
        elif gt[i] == 0:
            state = False
        if state: pred[i] = 1
    return pred

def show(name, path):
    try:
        p = np.load(path).reshape(-1).astype(int)
    except FileNotFoundError:
        print(f"  {name:<22} (not produced)"); return
    n = min(len(p), len(gt))
    raw = prf(gt[:n], p[:n], average="binary", zero_division=0)
    adj = prf(gt[:n], point_adjust(gt[:n], p[:n]), average="binary", zero_division=0)
    print(f"  {name:<22} raw       P {100*raw[0]:6.2f}  R {100*raw[1]:6.2f}  F1 {100*raw[2]:6.2f}")
    print(f"  {'':<22} adjusted  P {100*adj[0]:6.2f}  R {100*adj[1]:6.2f}  F1 {100*adj[2]:6.2f}")

expected = {"hdfs": ("99.53", "99.98"), "os": ("99.03", "99.95")}[ds]
print(f"  ground truth: {len(gt):,} lines, {int(gt.sum()):,} anomalous")
segs = int(np.diff(np.concatenate(([0], gt, [0]))).clip(min=0).sum())
print(f"  ground-truth anomaly segments: {segs}")
show("Q-BAT (edge only)",  f"{out}/edge_preds_raw.npy")
show("CESAL (hybrid)",     f"{out}/hybrid_preds.npy")
print(f"\n  Table 3 expects (point-adjusted): edge F1 {expected[0]}, CESAL F1 {expected[1]}")
print("""
  NOTE ON THE PROTOCOL
  Table 3 reports point-adjusted scores. Point adjustment credits an entire ground-truth
  anomaly segment once any single line inside it is detected. Both test splits are built as
  all-normal lines followed by all-abnormal lines (cesal_core/data/loaders.py:45-53), so
  each has exactly ONE anomaly segment -- which is why 'segments' prints 1 above, and why
  the adjusted and raw numbers can differ substantially. Both are shown so the effect of
  the protocol is explicit.""")
PY
