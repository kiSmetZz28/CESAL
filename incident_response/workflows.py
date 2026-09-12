"""Predefined controlled-response workflows for HDFS anomaly types (paper Table 1).

select_workflow() implements the workflow mapping w = Ω(â): each known anomaly type maps to its
predefined workflow; OTHER_LABEL ("Unknown Anomaly Types") maps to human investigation and never
triggers automated mitigation. Steps that submit administrator approval are marked
requires_approval; steps that escalate are marked escalation. This module selects workflows
only; it does not execute actions against a cluster.

Usage (from project root):
  python -m incident_response.workflows --label "Replica immediately deleted"
  python -m incident_response.workflows --results outputs/hdfs/llm/results_Qwen_Qwen2.5-14B-Instruct.csv
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from incident_response.labels import KNOWN_LABELS, OTHER_LABEL

UNKNOWN_WORKFLOW_NAME = "Unknown Anomaly Types"

# Table 1 text, one entry per anomaly type; steps are the ';'-separated clauses.
_TABLE_1: Dict[str, str] = {
    "Namenode not updated after deleting block": (
        "Collect NameNode edit logs, namespace snapshots, and DataNode block reports; run HDFS fsck to verify "
        "namespace and block consistency; refresh NameNode/DataNode metadata views through approved interfaces; "
        "if inconsistency persists, submit an approval-gated metadata synchronization task."
    ),
    "Write exception client give up": (
        "Collect client logs, DataNode status, and network diagnostics; identify the failed stage in the write pipeline; "
        "verify target DataNode availability and pipeline health; trigger a safe write retry after the pipeline is "
        "recovered; escalate repeated failures with the collected evidence."
    ),
    "Write failed at beginning": (
        "Check safe mode, permission, quota, and block allocation status; inspect client-side and NameNode "
        "initialization logs; correct safe configuration issues through approved management actions when permitted; "
        "trigger a controlled write retry after the initial write path is validated."
    ),
    "Replica immediately deleted": (
        "Inspect replication policy, block state, and replica placement records; determine whether the replica is "
        "invalid, corrupt, excessive, or prematurely removed; trigger re-replication when policy conditions are "
        "satisfied; submit administrator approval for destructive replica cleanup or metadata correction."
    ),
    "Received block that does not belong to any file": (
        "Compare DataNode block records with NameNode namespace metadata; run HDFS fsck to identify orphan, stale, or "
        "inconsistent blocks; quarantine suspicious block records through approved interfaces; submit administrator "
        "approval before permanent block cleanup or metadata modification."
    ),
    "Redundant addStoredBlock": (
        "Inspect duplicate block reports and repeated addStoredBlock events; verify whether the replica has already "
        "been registered in NameNode metadata; refresh block mappings through approved interfaces; suppress duplicate "
        "update handling when safe; escalate persistent metadata inconsistency."
    ),
    "Delete a block that no longer exists on data node": (
        "Compare the deletion request with the local DataNode block state; refresh DataNode block reports and NameNode "
        "metadata views; reconcile stale deletion requests through approved metadata update procedures; escalate "
        "repeated stale deletion events that indicate NameNode/DataNode state divergence."
    ),
    "Empty packet for block": (
        "Inspect block transfer logs, client status, and network conditions; determine whether the event is caused by "
        "timeout, transfer interruption, or client disconnect; restart or retry the block transfer after the connection "
        "and pipeline state are recovered; escalate repeated transfer failures."
    ),
    "Receive block exception": (
        "Collect receiver DataNode logs, disk I/O status, permissions, and network diagnostics; identify storage, "
        "permission, or communication causes; execute safe node-level recovery or retry actions when permitted; "
        "escalate failures requiring manual intervention."
    ),
    "Replication Monitor timeout": (
        "Check under-replicated block queues, NameNode workload, and live DataNode status; inspect delayed or blocked "
        "replication tasks; trigger approved rebalancing or restart delayed replication tasks when safe; submit "
        "administrator approval for service-level recovery operations."
    ),
    UNKNOWN_WORKFLOW_NAME: (
        "Flag the sequence as an unknown anomaly; preserve the full abnormal log sequence, retrieved evidence, and "
        "system context; notify engineers or domain experts for manual investigation."
    ),
}


@dataclass(frozen=True)
class WorkflowStep:
    action: str
    requires_approval: bool
    escalation: bool


@dataclass(frozen=True)
class Workflow:
    anomaly_type: str
    steps: Tuple[WorkflowStep, ...]
    automated: bool   # False for unknown anomalies: preserved and flagged for human investigation


def _parse_steps(text: str) -> Tuple[WorkflowStep, ...]:
    steps = []
    for clause in text.rstrip(".").split(";"):
        action = clause.strip()
        action = action[0].upper() + action[1:]
        steps.append(WorkflowStep(
            action=action,
            requires_approval="approval" in action.lower(),
            escalation=action.lower().startswith("escalate"),
        ))
    return tuple(steps)


WORKFLOWS: Dict[str, Workflow] = {
    name: Workflow(anomaly_type=name, steps=_parse_steps(text), automated=name != UNKNOWN_WORKFLOW_NAME)
    for name, text in _TABLE_1.items()
}
assert set(WORKFLOWS) == set(KNOWN_LABELS) | {UNKNOWN_WORKFLOW_NAME}


def select_workflow(predicted_label: str) -> Workflow:
    """Ω(â): map a predicted anomaly label to its predefined workflow (unknown → human investigation)."""
    if predicted_label == OTHER_LABEL or predicted_label not in WORKFLOWS:
        return WORKFLOWS[UNKNOWN_WORKFLOW_NAME]
    return WORKFLOWS[predicted_label]


def format_workflow(workflow: Workflow) -> str:
    lines = [f"Workflow: {workflow.anomaly_type}"
             + ("" if workflow.automated else "  (no automated mitigation — human investigation)")]
    for i, step in enumerate(workflow.steps, 1):
        tags = [t for t, on in (("requires administrator approval", step.requires_approval),
                                ("escalation", step.escalation)) if on]
        lines.append(f"  {i}. {step.action}" + (f"  [{', '.join(tags)}]" if tags else ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select predefined response workflows for predicted anomaly types.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", help="Predicted anomaly label.")
    group.add_argument("--results", help="results_<model>.csv from incident_response.evaluate.")
    args = parser.parse_args()

    if args.label:
        print(format_workflow(select_workflow(args.label)))
        return

    import pandas as pd
    counts = pd.read_csv(args.results)["pred_label"].value_counts()
    rows: List[Tuple[str, int, str]] = [(lab, int(n), select_workflow(lab).anomaly_type) for lab, n in counts.items()]
    width = max(len(r[0]) for r in rows)
    print(f"{'predicted label':<{width}}  {'sequences':>9}  workflow")
    for lab, n, wf in rows:
        print(f"{lab:<{width}}  {n:>9}  {wf}")


if __name__ == "__main__":
    main()
