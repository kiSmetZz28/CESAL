"""Tests for the Table 1 response-workflow mapping (incident_response/workflows.py).

The mapping w = Ω(â) is the paper's controlled-response step: a classified
anomaly type selects a predefined workflow, and an unknown type never triggers
automated mitigation.
"""
import pytest

from incident_response.labels import KNOWN_LABELS, OTHER_LABEL
from incident_response.workflows import (
    UNKNOWN_WORKFLOW_NAME, WORKFLOWS, select_workflow,
)


def test_every_known_anomaly_type_has_its_own_workflow():
    for label in KNOWN_LABELS:
        assert select_workflow(label).anomaly_type == label


def test_table_1_covers_the_known_types_plus_unknown():
    assert set(WORKFLOWS) == set(KNOWN_LABELS) | {UNKNOWN_WORKFLOW_NAME}
    assert len(WORKFLOWS) == 11


def test_unknown_type_gets_human_investigation_and_no_automation():
    w = select_workflow(OTHER_LABEL)
    assert w.anomaly_type == UNKNOWN_WORKFLOW_NAME
    assert w.automated is False
    assert not any(s.requires_approval for s in w.steps)


def test_a_label_the_model_invents_falls_back_to_unknown():
    """An LLM can emit text outside the label set; it must not select an action."""
    w = select_workflow("Catastrophic gremlin incursion")
    assert w.anomaly_type == UNKNOWN_WORKFLOW_NAME
    assert w.automated is False


def test_known_types_are_automated():
    for label in KNOWN_LABELS:
        assert select_workflow(label).automated is True


def test_every_workflow_has_steps():
    for name, w in WORKFLOWS.items():
        assert len(w.steps) >= 3, name
        assert all(s.action.strip() for s in w.steps), name


def test_steps_are_the_paper_wording_verbatim():
    """Splitting a Table 1 row into steps must not reword it — rejoining the
    steps has to reproduce the row character for character."""
    from incident_response.workflows import _TABLE_1
    for name, text in _TABLE_1.items():
        assert "; ".join(s.action for s in WORKFLOWS[name].steps) == text, name


def test_each_workflow_carries_its_table_1_row():
    from incident_response.workflows import _TABLE_1
    for name, w in WORKFLOWS.items():
        assert w.description == _TABLE_1[name]
        assert w.description.endswith(".")


def test_high_impact_steps_are_approval_gated():
    """Steps that submit administrator approval must be flagged; steps that merely
    act 'through approved interfaces' are low-impact and must not be."""
    gated, sanctioned = [], []
    for w in WORKFLOWS.values():
        for s in w.steps:
            low = s.action.lower()
            if "submit" in low and "approval" in low:
                gated.append(s)
            elif "through approved" in low or "trigger approved" in low:
                sanctioned.append(s)
    assert gated, "expected some approval-gated steps"
    assert sanctioned, "expected some steps using approved interfaces"
    assert all(s.requires_approval for s in gated)
    assert not any(s.requires_approval for s in sanctioned)


def test_escalation_steps_are_flagged():
    for w in WORKFLOWS.values():
        for s in w.steps:
            assert s.escalation == s.action.lower().startswith("escalate"), s.action


def test_at_least_one_workflow_of_each_control_kind():
    """Table 1 mixes automatic, approval-gated and escalating responses."""
    assert any(sum(s.requires_approval for s in w.steps) for w in WORKFLOWS.values())
    assert any(sum(s.escalation for s in w.steps) for w in WORKFLOWS.values())
    assert any(w.automated for w in WORKFLOWS.values())
    assert any(not w.automated for w in WORKFLOWS.values())


@pytest.mark.parametrize("label", KNOWN_LABELS + [OTHER_LABEL])
def test_selection_is_stable(label):
    assert select_workflow(label) is select_workflow(label)
