"""Uniform step-by-step progress reporting for the CESAL pipelines.

Every runner — :mod:`cesal_inference_pipeline.run`, ``dashboard/cloud_runner.py``
and ``dashboard/demo_runner.py`` — reports through a single :class:`StepReporter`
so that:

* the terminal and the log file show the same numbered steps, each closed by an
  indented outcome block and an elapsed time, and
* the dashboard consumes *structured events* instead of regex-matching English
  prose out of the log stream.

Structured events are one-line JSON written to stdout with an ``@@CESAL``
prefix, and only when the environment variable ``CESAL_EVENTS=1`` is set — the
dashboard sets it when it spawns a runner, so interactive users never see them.

Typical use::

    rep = StepReporter("infer", dataset="HDFS", steps=INFER_STEPS)
    with rep.step("edge") as st:
        st.detail("models", "3 Q-BAT (.pte)")
        st.expect("models", 3)
        ...
        st.tick("models")
        st.outcome(**{"flagged anomalous": 5912})
    rep.finish(outputs="outputs/hdfs")

The inference pipeline spans two processes: the edge runner performs steps 1-2
and delegates steps 3-4 to ``cloud_runner.py`` in the cloud environment. The
parent calls :meth:`StepReporter.export` before spawning, and the child passes
``adopt=<path>`` so that step numbering continues correctly and the closing
summary covers all four steps rather than only the child's two.

Stage modules that do not own the reporter (``lad_qbat_edge``, ``lad_bat_cloud``)
reach the active step with :func:`current`; when no run is active that returns a
no-op step, so those modules still work standalone.
"""

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "StepReporter", "Step", "INFER_STEPS", "current", "current_run",
    "bars_suppressed", "fmt_count", "fmt_secs",
]

# The canonical four steps of the collaborative inference pipeline. Defined once
# here so the CLI runners, the demo runner and the dashboard all use the same
# ids, titles and ordering.
INFER_STEPS: List[Tuple[str, str]] = [
    ("edge",   "Edge Q-BAT scan"),
    ("route",  "Uncertainty routing"),
    ("cloud",  "Cloud BAT verification"),
    ("hybrid", "Hybrid merge & scoring"),
]

# Fixed width keeps banners aligned in both the terminal and the log file.
_WIDTH = 66
# Summary labels carry a step number and a status glyph, so they need a wider
# value column than the in-step detail lines.
_SUMMARY_W = 40
_RULE = "─" * _WIDTH
_RULE_HEAVY = "═" * _WIDTH
_EVENT_PREFIX = "@@CESAL "


def _events_enabled() -> bool:
    """Structured events are emitted only when the dashboard asks for them."""
    return os.environ.get("CESAL_EVENTS") == "1"


def bars_suppressed() -> bool:
    """True when tqdm progress bars should be turned off.

    The dashboard reads the runner's output as plain lines, so a bar's carriage
    returns arrive as unreadable fragments. Bars stay on for interactive CLI
    runs, where parsing millions of log lines genuinely benefits from one.
    """
    return _events_enabled() or not sys.stderr.isatty()


def fmt_count(n: Any) -> str:
    """Thousands-separated integers; anything else passes through unchanged."""
    return f"{n:,}" if isinstance(n, int) and not isinstance(n, bool) else str(n)


def fmt_secs(secs: float) -> str:
    """Human elapsed time: ``0.4s``, ``41.0s``, ``2m 00.7s``, ``1h 03m``."""
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{int(secs // 60)}m {secs % 60:04.1f}s"
    return f"{int(secs // 3600)}h {int((secs % 3600) // 60):02d}m"


def _leader(label: str, value: str, indent: int = 3, width: int = 28) -> str:
    """``   routed ............. 138 / 1,386`` — dot leader, aligned values.

    ``width`` is the column the value starts at; the summary uses a wider one
    because its labels carry a number and a status glyph.
    """
    pad = " " * indent
    dots = max(2, width - len(label))
    return f"{pad}{label} {'.' * dots} {value}"


# ── Active-run registry ───────────────────────────────────────────────────────
# Stage modules call current()/current_run() instead of taking a reporter
# argument, so their public signatures stay unchanged.

_active_lock = threading.Lock()
_active_run: Optional["StepReporter"] = None
_active_step: Optional["Step"] = None


def current() -> "Step":
    """The step currently executing, or a no-op step when no run is active."""
    with _active_lock:
        return _active_step or _NULL_STEP


def current_run() -> Optional["StepReporter"]:
    """The active reporter, or ``None`` outside a run."""
    with _active_lock:
        return _active_run


class Step:
    """One numbered pipeline step; created by :meth:`StepReporter.step`."""

    def __init__(self, reporter: "StepReporter", step_id: str, title: str, index: int):
        self.id = step_id
        self.title = title
        self.index = index
        self._rep = reporter
        self._lock = threading.Lock()
        self._started = 0.0
        self._outcome: Dict[str, Any] = {}
        self._counters: Dict[str, Dict[str, int]] = {}
        self.elapsed = 0.0
        self.state = "pending"

    # ── live reporting ────────────────────────────────────────────────────
    def detail(self, label: str, value: Any) -> None:
        """Print one aligned ``label ... value`` line as the step runs."""
        logging.info("%s", _leader(label, fmt_count(value)))
        self._rep._emit({"t": "step_detail", "id": self.id,
                         "k": label, "v": str(value)})

    def note(self, message: str) -> None:
        """Print an un-aligned informational line inside the step."""
        logging.info("   %s", message)
        self._rep._emit({"t": "step_note", "id": self.id, "msg": message})

    def warn(self, message: str) -> None:
        logging.warning("   ! %s", message)
        self._rep._emit({"t": "step_warn", "id": self.id, "msg": message})

    def expect(self, unit: str, total: int) -> None:
        """Declare how many items of ``unit`` this step will process."""
        with self._lock:
            self._counters[unit] = {"done": 0, "total": int(total),
                                    "mark": 0, "epct": -1}
        self._rep._emit({"t": "step_expect", "id": self.id,
                         "unit": unit, "total": int(total)})

    def tick(self, unit: str, n: int = 1) -> None:
        """Count ``n`` completed items.

        Thread-safe: the edge and cloud stages tick from worker threads. To keep
        the terminal readable a milestone line is printed at each 25% of the
        expected total rather than once per item — per-item records stay in the
        log file at DEBUG level. The dashboard receives every tick as an event.
        """
        with self._lock:
            c = self._counters.setdefault(unit, {"done": 0, "total": 0,
                                                 "mark": 0, "epct": -1})
            c.setdefault("epct", -1)  # counters created before this key existed
            c["done"] += n
            done, total = c["done"], c["total"]
            show = False   # print a milestone line to the terminal
            emit = False   # send a progress event to the dashboard
            if total > 0:
                pct = done * 100 // total
                if pct >= c["mark"] + 25 or done == total:
                    c["mark"] = (pct // 25) * 25
                    show = True
                # Events are throttled to whole-percent changes: a long stage can
                # tick thousands of times, and one event each would swamp the
                # dashboard's stream for no extra visible resolution.
                if pct != c["epct"] or done == total:
                    c["epct"] = pct
                    emit = True
            else:
                show = done % 25 == 0
                emit = done % 25 == 0
        if show:
            if total:
                logging.info("   %s/%s %s  (%d%%)", f"{done:,}", f"{total:,}",
                             unit, done * 100 // total)
            else:
                logging.info("   %s %s", f"{done:,}", unit)
        if emit:
            self._rep._emit({"t": "step_progress", "id": self.id, "unit": unit,
                             "done": done, "total": total})

    def outcome(self, **values: Any) -> None:
        """Record the step's result; rendered in its closing block."""
        self._outcome.update(values)

    # ── lifecycle ─────────────────────────────────────────────────────────
    # Usable either as a context manager (preferred) or explicitly via
    # start()/done(), which suits runners whose stage body is a long if/else
    # that would otherwise need re-indenting.
    def start(self) -> "Step":
        global _active_step
        self._started = time.perf_counter()
        self.state = "running"
        with _active_lock:
            _active_step = self
        logging.info("")
        logging.info("%s", _RULE)
        logging.info(" STEP %d/%d · %s", self.index, self._rep.total, self.title)
        logging.info("%s", _RULE)
        self._rep._emit({"t": "step_start", "id": self.id, "n": self.index,
                         "total": self._rep.total, "title": self.title})
        return self

    def __enter__(self) -> "Step":
        return self.start()

    def done(self) -> None:
        """Close a step opened with :meth:`start`."""
        self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, tb) -> bool:
        global _active_step
        if self.state != "running":  # already closed, or never started
            return False
        self.elapsed = time.perf_counter() - self._started
        with _active_lock:
            _active_step = None

        if exc_type is not None:
            self.state = "failed"
            logging.error("   ✗ failed after %s — %s", fmt_secs(self.elapsed), exc)
            self._rep._record(self, error=str(exc))
            self._rep._emit({"t": "step_fail", "id": self.id,
                             "secs": round(self.elapsed, 3), "error": str(exc)})
            return False  # never swallow the exception

        self.state = "done"
        for label, value in self._outcome.items():
            logging.info("%s", _leader(label.replace("_", " "), fmt_count(value)))
        logging.info("   ✓ done in %s", fmt_secs(self.elapsed))
        self._rep._record(self)
        self._rep._emit({"t": "step_done", "id": self.id,
                         "secs": round(self.elapsed, 3),
                         "outcome": {k: str(v) for k, v in self._outcome.items()}})
        return False


class _NullStep(Step):
    """No-op step returned by :func:`current` when no run is active."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips Step.__init__
        self.id = ""
        self.title = ""
        self.index = 0
        self.state = "inactive"
        self.elapsed = 0.0

    def detail(self, label: str, value: Any) -> None: pass
    def note(self, message: str) -> None: pass
    def warn(self, message: str) -> None: logging.warning("%s", message)
    def expect(self, unit: str, total: int) -> None: pass
    def tick(self, unit: str, n: int = 1) -> None: pass
    def outcome(self, **values: Any) -> None: pass
    def start(self) -> "Step": return self
    def done(self) -> None: pass
    def __enter__(self) -> "Step": return self
    def __exit__(self, exc_type, exc, tb) -> bool: return False


_NULL_STEP = _NullStep()


class StepReporter:
    """Drives a numbered sequence of steps for one pipeline run.

    Parameters
    ----------
    run : str
        Short name of the run, e.g. ``"infer"``.
    dataset : str
        Dataset label shown in the header.
    steps : sequence of (id, title)
        The full pipeline plan. Ignored when ``adopt`` supplies one.
    adopt : str, optional
        Path to a handoff file written by :meth:`export` in a parent process.
        Its plan, completed step records and metrics are absorbed, so this
        reporter continues the parent's numbering and its summary covers the
        whole pipeline.
    """

    def __init__(
        self,
        run: str,
        dataset: str = "",
        steps: Sequence[Tuple[str, str]] = (),
        adopt: str = "",
    ):
        global _active_run
        self.run = run
        self.dataset = dataset
        self.plan: List[Tuple[str, str]] = list(steps)
        self._records: List[Dict[str, Any]] = []
        self._skipped: Dict[str, str] = {}
        self._metrics: List[Dict[str, Any]] = []
        self._inherited_secs = 0.0
        self._started = time.perf_counter()

        inherited = self._adopt(adopt) if adopt else False

        self.total = len(self.plan)
        self._by_id = {sid: i + 1 for i, (sid, _) in enumerate(self.plan)}
        self._titles = dict(self.plan)

        with _active_lock:
            _active_run = self

        if not inherited:
            logging.info("")
            logging.info("%s", _RULE_HEAVY)
            logging.info(" CESAL · %s%s", run, f" · {dataset}" if dataset else "")
            logging.info("%s", _RULE_HEAVY)
            for i, (_, title) in enumerate(self.plan, start=1):
                logging.info("   %d. %s", i, title)
            self._emit({"t": "run_start", "run": run, "dataset": dataset,
                        "steps": [{"id": s, "title": t} for s, t in self.plan]})

    # ── cross-process handoff ─────────────────────────────────────────────
    def _adopt(self, path: str) -> bool:
        """Absorb a parent process's plan and completed steps. Returns success."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False  # a missing handoff must never break the run
        self.plan = [(s["id"], s["title"]) for s in data.get("plan", [])] or self.plan
        self._records = list(data.get("records", []))
        self._skipped = dict(data.get("skipped", {}))
        self._metrics = list(data.get("metrics", []))
        self._inherited_secs = float(data.get("secs", 0.0))
        self.run = data.get("run", self.run)
        self.dataset = data.get("dataset", self.dataset)
        return True

    def export(self, path: str) -> None:
        """Write this run's state so a child process can continue the summary."""
        payload = {
            "run": self.run,
            "dataset": self.dataset,
            "plan": [{"id": s, "title": t} for s, t in self.plan],
            "records": self._records,
            "skipped": self._skipped,
            "metrics": self._metrics,
            "secs": time.perf_counter() - self._started,
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except OSError as exc:
            logging.debug("Could not write step handoff to %s: %s", path, exc)

    # ── event channel ─────────────────────────────────────────────────────
    def _emit(self, payload: Dict[str, Any]) -> None:
        """Write one structured event for the dashboard (no-op when disabled)."""
        if not _events_enabled():
            return
        try:
            sys.stdout.write(_EVENT_PREFIX + json.dumps(payload, default=str) + "\n")
            sys.stdout.flush()
        except Exception:  # a broken pipe must never abort the pipeline
            pass

    def _record(self, step: Step, error: str = "") -> None:
        self._records.append({
            "id": step.id, "title": step.title, "index": step.index,
            "state": step.state, "secs": round(step.elapsed, 3),
            "outcome": {k: str(v) for k, v in step._outcome.items()},
            "error": error,
        })

    # ── step lifecycle ────────────────────────────────────────────────────
    def step(self, step_id: str, title: str = "") -> Step:
        """Open the step registered as ``step_id`` (use as a context manager)."""
        index = self._by_id.get(step_id, len(self._records) + 1)
        return Step(self, step_id, title or self._titles.get(step_id, step_id), index)

    def skip(self, step_id: str, reason: str) -> None:
        """Record a step that was deliberately not run, and say why."""
        title = self._titles.get(step_id, step_id)
        index = self._by_id.get(step_id, 0)
        self._skipped[step_id] = reason
        logging.info("")
        logging.info("%s", _RULE)
        logging.info(" STEP %d/%d · %s — SKIPPED", index, self.total, title)
        logging.info("%s", _RULE)
        logging.info("   %s", reason)
        self._emit({"t": "step_skip", "id": step_id, "n": index,
                    "title": title, "reason": reason})

    def metric(self, label: str, accuracy: float, precision: float,
               recall: float, f_score: float) -> None:
        """Register a scored result for the closing summary (percentages)."""
        row = {"label": label, "accuracy": accuracy, "precision": precision,
               "recall": recall, "f_score": f_score}
        self._metrics.append(row)
        self._emit({"t": "metric", **row})

    # ── closing summary ───────────────────────────────────────────────────
    def finish(self, outputs: str = "") -> None:
        """Print the end-of-run summary: every step, its state and its time."""
        global _active_run, _active_step
        total_secs = self._inherited_secs + (time.perf_counter() - self._started)
        by_id = {r["id"]: r for r in self._records}

        logging.info("")
        logging.info("%s", _RULE_HEAVY)
        logging.info(" RUN SUMMARY · %s%s", self.run,
                     f" · {self.dataset}" if self.dataset else "")
        logging.info("%s", _RULE_HEAVY)

        for index, (step_id, title) in enumerate(self.plan, start=1):
            rec = by_id.get(step_id)
            if rec and rec["state"] == "done":
                logging.info("%s", _leader(f"{index}. ✓ {title}",
                                           fmt_secs(rec["secs"]), width=_SUMMARY_W))
            elif rec and rec["state"] == "failed":
                logging.info("%s", _leader(f"{index}. ✗ {title}",
                                           f"failed after {fmt_secs(rec['secs'])}",
                                           width=_SUMMARY_W))
            elif step_id in self._skipped:
                logging.info("%s", _leader(f"{index}. – {title}", "skipped",
                                           width=_SUMMARY_W))
                logging.info("        %s", self._skipped[step_id])
            else:
                logging.info("%s", _leader(f"{index}. · {title}", "not run",
                                           width=_SUMMARY_W))

        if self._metrics:
            logging.info("   %s", "─" * (_WIDTH - 6))
            logging.info("   Detection quality")
            for m in self._metrics:
                logging.info("     %-9s P %6.2f   R %6.2f   F1 %6.2f",
                             m["label"], m["precision"], m["recall"], m["f_score"])

        logging.info("   %s", "─" * (_WIDTH - 6))
        if outputs:
            logging.info("%s", _leader("outputs", outputs, width=_SUMMARY_W))
        logging.info("%s", _leader("total time", fmt_secs(total_secs),
                                   width=_SUMMARY_W))
        logging.info("%s", _RULE_HEAVY)

        self._emit({"t": "run_done", "secs": round(total_secs, 3),
                    "outputs": outputs, "metrics": self._metrics})
        with _active_lock:
            _active_run = None
            _active_step = None
