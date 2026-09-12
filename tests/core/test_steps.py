"""Tests for the shared step reporter (cesal_core/utils/steps.py).

These cover the behaviour both the CLI output and the dashboard depend on:
the numbered step lifecycle, the cross-process handoff that lets the cloud
process continue the edge process's run, and the structured event stream.
"""
import json

import pytest

from cesal_core.utils import steps
from cesal_core.utils.steps import StepReporter


@pytest.fixture(autouse=True)
def _clear_active_run():
    """No test may leak an active run into the next one."""
    yield
    steps._active_run = None
    steps._active_step = None


@pytest.fixture
def events_on(monkeypatch):
    """Turn on the structured event stream for one test."""
    monkeypatch.setenv("CESAL_EVENTS", "1")


def read_events(capsys):
    """Parse the events exactly as the dashboard does: @@CESAL-prefixed stdout."""
    out = capsys.readouterr().out
    return [
        json.loads(line[len(steps._EVENT_PREFIX):])
        for line in out.splitlines()
        if line.startswith(steps._EVENT_PREFIX)
    ]


# ── formatting ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("secs,expected", [
    (0.42, "0.4s"), (41.0, "41.0s"), (120.7, "2m 00.7s"), (3780, "1h 03m"),
])
def test_fmt_secs(secs, expected):
    assert steps.fmt_secs(secs) == expected


def test_fmt_count_groups_thousands():
    assert steps.fmt_count(221540) == "221,540"
    assert steps.fmt_count("3/3") == "3/3"


def test_leader_aligns_values_at_a_fixed_column():
    short = steps._leader("routed", "138")
    long_ = steps._leader("ground-truth anomalous", "500")
    assert short.index("138") == long_.index("500")


def test_leader_keeps_a_separator_when_the_label_overflows():
    line = steps._leader("a" * 80, "v")
    assert ".. v" in line


# ── step lifecycle ────────────────────────────────────────────────────────────

def test_step_records_state_and_outcome():
    rep = StepReporter("t", steps=[("a", "Step A")])
    with rep.step("a") as st:
        st.outcome(found=7)
    assert rep._records[0]["state"] == "done"
    assert rep._records[0]["outcome"] == {"found": "7"}


def test_failed_step_is_recorded_and_the_exception_propagates():
    rep = StepReporter("t", steps=[("a", "Step A")])
    with pytest.raises(ValueError):
        with rep.step("a"):
            raise ValueError("boom")
    assert rep._records[0]["state"] == "failed"
    assert "boom" in rep._records[0]["error"]


def test_current_exposes_the_running_step_and_clears_afterwards():
    rep = StepReporter("t", steps=[("a", "Step A")])
    assert steps.current().state == "inactive"
    with rep.step("a"):
        assert steps.current().id == "a"
    assert steps.current().state == "inactive"


def test_explicit_start_done_matches_the_context_manager():
    rep = StepReporter("t", steps=[("a", "Step A")])
    st = rep.step("a").start()
    st.outcome(n=1)
    st.done()
    assert rep._records[0]["state"] == "done"


def test_done_is_idempotent():
    rep = StepReporter("t", steps=[("a", "Step A")])
    st = rep.step("a").start()
    st.done()
    st.done()
    assert len(rep._records) == 1


def test_tick_is_thread_safe():
    import threading
    rep = StepReporter("t", steps=[("a", "Step A")])
    with rep.step("a") as st:
        st.expect("models", 200)
        threads = [threading.Thread(target=lambda: [st.tick("models") for _ in range(50)])
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert st._counters["models"]["done"] == 200


# ── no-op step outside a run ──────────────────────────────────────────────────

def test_stage_modules_work_without_an_active_run():
    """lad_qbat_edge / lad_bat_cloud call steps.current() unconditionally."""
    st = steps.current()
    st.detail("k", "v")
    st.expect("models", 3)
    st.tick("models")
    st.outcome(x=1)
    st.done()  # must not raise


# ── cross-process handoff ─────────────────────────────────────────────────────

def test_handoff_continues_numbering_and_summary(tmp_path):
    plan = [("a", "A"), ("b", "B"), ("c", "C")]
    parent = StepReporter("infer", dataset="OS", steps=plan)
    with parent.step("a"):
        pass
    parent.metric("Edge", 1.0, 2.0, 3.0, 4.0)
    handoff = tmp_path / "steps.json"
    parent.export(str(handoff))

    child = StepReporter("infer", steps=plan, adopt=str(handoff))
    # The child continues the parent's identity, records and metrics.
    assert child.dataset == "OS"
    assert [r["id"] for r in child._records] == ["a"]
    assert child._metrics[0]["label"] == "Edge"
    # And numbers its own steps by position in the shared plan.
    assert child.step("c").index == 3


def test_adopting_a_missing_handoff_falls_back_to_a_fresh_run(tmp_path):
    rep = StepReporter("infer", dataset="OS", steps=[("a", "A")],
                       adopt=str(tmp_path / "nope.json"))
    assert rep._records == []
    assert rep.total == 1


# ── structured events ─────────────────────────────────────────────────────────

def test_events_describe_the_whole_run(events_on, capsys):
    rep = StepReporter("infer", dataset="OS", steps=[("a", "A"), ("b", "B")])
    with rep.step("a") as st:
        st.detail("models", 3)
        st.expect("models", 2)
        st.tick("models")
        st.tick("models")
        st.outcome(flagged=5)
    rep.skip("b", "nothing to do")
    rep.finish(outputs="outputs/os")

    events = read_events(capsys)
    kinds = [e["t"] for e in events]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_done"
    for expected in ("step_start", "step_detail", "step_expect",
                     "step_progress", "step_done", "step_skip"):
        assert expected in kinds

    start = next(e for e in events if e["t"] == "step_start")
    assert (start["id"], start["n"], start["total"]) == ("a", 1, 2)

    done = next(e for e in events if e["t"] == "step_done")
    assert done["outcome"] == {"flagged": "5"}

    progress = [e for e in events if e["t"] == "step_progress"]
    assert progress[-1]["done"] == 2 and progress[-1]["total"] == 2


def test_no_events_emitted_unless_enabled(monkeypatch, capsys):
    monkeypatch.delenv("CESAL_EVENTS", raising=False)
    rep = StepReporter("infer", steps=[("a", "A")])
    with rep.step("a"):
        pass
    rep.finish()
    assert read_events(capsys) == []


def test_evaluate_registers_its_scores_with_the_active_run():
    import numpy as np
    from cesal_core.utils.metrics import evaluate

    rep = StepReporter("infer", steps=[("a", "A")])
    with rep.step("a"):
        scores = evaluate(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), prefix="Edge")
    assert rep._metrics[0]["label"] == "Edge"
    assert rep._metrics[0]["recall"] == pytest.approx(scores.recall)
    assert scores.recall == pytest.approx(50.0)
