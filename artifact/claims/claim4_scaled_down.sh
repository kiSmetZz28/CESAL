#!/usr/bin/env bash
# Claim 4 — Scaled-down end-to-end run (kick-the-tires, ~2 minutes)
#
#   ./claim4_scaled_down.sh [OUTPUT_DIR]
#
# Runs all four stages of the collaborative pipeline on a 300-window subsample of HDFS:
#   Stage 1  edge Q-BAT scan
#   Stage 2  Mahalanobis routing (10% of lines)
#   Stage 3  cloud BAT verification across all 81 checkpoints
#   Stage 4  hybrid evaluation
#
# Writes to a scratch directory (default: a temp dir) so the committed outputs/hdfs/
# arrays are never overwritten. Measured runtime on CPU: 1m55s.
#
# The subsample is far too small to reproduce Table 3 — use claim1_detection.sh for that.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-$(mktemp -d -t cesal-scaled-XXXXXX)}"
EDGE_PYTHON="${EDGE_PYTHON:-$HOME/miniconda3/envs/cesal-edge/bin/python}"
[ -x "$EDGE_PYTHON" ] || { echo "ERROR: EDGE_PYTHON not executable. See INSTALL.md step 3." >&2; exit 1; }

mkdir -p "$OUT"
echo "=== Claim 4: scaled-down end-to-end run ====================="
echo "Output directory: $OUT"

# Inference READS pre-computed EM-GMM thresholds (only `run.py train` writes them),
# so stage them into the scratch output directory.
cp outputs/hdfs/thresholds_edge.yaml outputs/hdfs/thresholds_cloud.yaml "$OUT/"

CFG="$OUT/hdfs_scaled.yaml"
"$EDGE_PYTHON" - "$OUT" "$CFG" <<'PY'
import sys, os, yaml
out, cfg_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/inference/hdfs_dashboard_full.yaml"))
cfg["output_dir"]               = out
cfg["threshold_output"]         = os.path.join(out, "thresholds_edge.yaml")
cfg["cloud"]["thresholds_yaml"] = os.path.join(out, "thresholds_cloud.yaml")
yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
print(f"  config written (demo_max_windows = {cfg.get('demo_max_windows')})")
PY

echo "--- Running all four stages --------------------------------"
time "$EDGE_PYTHON" dashboard/demo_runner.py --config "$CFG"

echo
echo "--- Artifacts produced -------------------------------------"
ls -1 "$OUT"/*.npy 2>/dev/null | while read -r f; do
  printf "  %-24s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done

echo
echo "Expected: edge_preds.npy, edge_preds_raw.npy, ground_truth.npy, energy_matrix.npy,"
echo "          routed_indices.npy, routed_lines.npy, cloud_preds.npy, hybrid_preds.npy"
echo "If all four stages logged and those files exist, the install is working end to end."
