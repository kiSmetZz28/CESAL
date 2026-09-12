#!/usr/bin/env bash
# Claim 3 — Controlled response workflows (paper Section 3.7, Table 1)
#
#   ./claim3_response.sh
#
# Shows that (a) each known anomaly type maps to its predefined Table 1 workflow with
# high-impact steps gated behind administrator approval, and (b) sequences classified as
# unknown map to "Unknown Anomaly Types", trigger no automated mitigation, and are
# preserved and flagged for human investigation.
#
# The workflow selection itself needs no GPU. Connecting detection to response
# (run.py respond) needs detection outputs and a GPU for the classifier.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EDGE_PYTHON="${EDGE_PYTHON:-$HOME/miniconda3/envs/cesal-edge/bin/python}"

echo "=== Claim 3: controlled response workflows =================="
echo "--- A known anomaly type: 4 steps, 1 requiring approval -----"
"$EDGE_PYTHON" -m incident_response.workflows --label "Replica immediately deleted"

echo
echo "--- An unknown type: no automated mitigation ----------------"
"$EDGE_PYTHON" -m incident_response.workflows --label "Other anomaly type"

echo
echo "--- All 11 workflows load and map correctly ------------------"
"$EDGE_PYTHON" - <<'PY'
from incident_response.workflows import WORKFLOWS, select_workflow
from incident_response.labels import KNOWN_LABELS, OTHER_LABEL
print(f"  workflows defined: {len(WORKFLOWS)}  (10 known types + Unknown Anomaly Types)")
n_appr = sum(any(s.requires_approval for s in w.steps) for w in WORKFLOWS.values())
n_esc  = sum(any(s.escalation      for s in w.steps) for w in WORKFLOWS.values())
print(f"  workflows with an approval-gated step: {n_appr}")
print(f"  workflows with an escalation step    : {n_esc}")
u = select_workflow(OTHER_LABEL)
print(f"  unknown -> {u.anomaly_type!r}, automated={u.automated}, steps={len(u.steps)}")
assert not u.automated, "unknown anomalies must not trigger automated mitigation"
for lab in KNOWN_LABELS:
    assert select_workflow(lab).automated, lab
print("  OK: every known type maps to an automated workflow; unknown does not.")
PY

echo
echo "--- Detection -> response (needs outputs/hdfs/*.npy + GPU) --"
if [ -f outputs/hdfs/edge_preds_raw.npy ]; then
  echo "  Detection outputs found. To build the anomaly queues and classify them:"
  echo "      conda activate cesal-cloud && python run.py respond"
  if [ -d outputs/hdfs/llm/queues ]; then
    "$EDGE_PYTHON" - <<'PY'
import csv, glob, os
f = sorted(glob.glob("outputs/hdfs/llm/queues/incidents_*.csv"))
if f:
    rows = list(csv.DictReader(open(f[-1], newline="", encoding="utf-8")))
    tot = len(rows)
    abn = sum(r["ground_truth"] == "1" for r in rows)
    hum = sum(r["automated_response"].strip().lower() != "true" for r in rows)
    print(f"\n  {os.path.basename(f[-1])}")
    print(f"    incidents: {tot:,}  ({abn:,} true anomalies, {tot-abn:,} detector false positives)")
    print(f"    routed to human investigation (unknown type): {hum:,}")
PY
  fi
else
  echo "  No detection outputs yet — run ./claim1_detection.sh hdfs first."
fi
