"""Uniform step-by-step progress reporting for the CESAL pipelines.

Every runner — :mod:`cesal_inference_pipeline.run`, ``cesal_inference_pipeline/cloud_runner.py``
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
        st.detail("Q-BAT learners", "3 quantized EM-AT (.pte)")
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

# Progress milestone interval, in percent. Small enough that a slow stage
# (the edge scan runs for hours) visibly advances, so it is never mistaken
# for a hang; large enough not to flood the terminal.
_STEP_PCT = 5

__all__ = [
    "StepReporter", "Step", "INFER_STEPS", "TRAIN_STEPS", "CONVERT_STEPS",
    "EVAL_STEPS", "DOWNLOAD_STEPS", "CLASSIFY_STEPS", "RESPOND_STEPS",
    "current", "current_run",
    "bars_suppressed", "clear_bar", "redraw_bar", "fmt_count", "fmt_secs",
]

# Canonical step plans, defined once here so the CLI runners, the demo runner and
# the dashboard all use the same ids, titles and ordering. The third element of
# each entry is a plain-language line printed under the step's banner, so someone
# who does not know the system can still follow what it is doing.

INFER_STEPS: List[Tuple[str, str, str]] = [
    ("edge",   "Edge-side detection with Q-BAT",
     "The quantized Q-BAT ensemble scores every test window on the edge device and\n"
     "flags those whose energy exceeds its calibrated threshold."),
    ("route",  "Mahalanobis uncertainty routing",
     "A Mahalanobis distance policy selects the windows closest to the decision\n"
     "boundary — the least certain — for cloud verification."),
    ("cloud",  "Cloud-side verification with BAT",
     "The full-precision BAT ensemble re-evaluates only the escalated windows,\n"
     "at higher capacity than the edge tier can provide."),
    ("hybrid", "Collaborative merge and scoring",
     "Cloud verdicts supersede the edge verdicts for the escalated windows, and the\n"
     "merged prediction is scored against ground truth."),
]

TRAIN_STEPS: List[Tuple[str, str, str]] = [
    ("plan",  "Plan the BAT ensemble",
     "Enumerate the hyper-parameter grid and resolve the configuration of each\n"
     "base learner."),
    ("sweep", "Train the EM-AT base learners",
     "Each EM-AT learner is trained on normal log activity; deviation from that\n"
     "reconstruction is the anomaly signal."),
]

CONVERT_STEPS: List[Tuple[str, str, str]] = [
    ("locate",  "Locate the trained EM-AT checkpoints",
     "Resolve which trained checkpoints are present and eligible for conversion."),
    ("convert", "Quantize and export for the edge device",
     "Each learner is quantized to int8 activations and int4 weights, then exported\n"
     "as an ExecuTorch program for edge deployment."),
]

EVAL_STEPS: List[Tuple[str, str, str]] = [
    ("score",    "Score each EM-AT model on its own",
     "Each learner is evaluated independently to establish its standalone detection\n"
     "performance and calibrate its threshold."),
    ("ensemble", "Grow the BAT ensemble",
     "Learners are accumulated incrementally to quantify the ensemble gain over any\n"
     "individual member."),
]

# `run.py download` — fetch the checkpoints and runtime the pipeline needs.
DOWNLOAD_STEPS: List[Tuple[str, str, str]] = [
    ("check", "Check what is already here",
     "Inventory the local checkpoints and runtime so nothing is fetched twice."),
    ("fetch", "Download the missing pieces",
     "Retrieve the published checkpoints and the ExecuTorch runtime the edge tier\n"
     "requires."),
]

# `run.py classify` — benchmark the LLM incident classifier (paper Table 7).
CLASSIFY_STEPS: List[Tuple[str, str, str]] = [
    ("prepare",  "Build the RAG knowledge base",
     "Load the abnormal sequences and the reference corpus used for retrieval."),
    ("classify", "Open-set classification with the LLM",
     "For each sequence, similar incidents are retrieved and the language model\n"
     "assigns a known type or marks it as unknown."),
    ("score",    "Score against the ground-truth types",
     "Compare the assigned labels against ground truth and write per-class metrics."),
]

# `run.py respond` — detections → queues → classification → response workflows.
RESPOND_STEPS: List[Tuple[str, str, str]] = [
    ("queue",    "Fill the anomaly queues Q_E / Q_C",
     "Flagged sessions are queued by the tier that detected them — edge (Q_E) or\n"
     "cloud (Q_C)."),
    ("classify", "Open-set classification of each incident",
     "Each queued sequence is assigned a known incident type, or marked as an\n"
     "unknown type for human review."),
    ("workflow", "Select the response workflow",
     "Known types map to their predefined response workflow; unknown types are\n"
     "escalated for human investigation."),
    ("score",    "Score the classification",
     "Score the assigned types against ground truth for the labelled sessions."),
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
    """True when a subordinate tqdm progress bar should be turned off.

    Three reasons to suppress: the dashboard reads the runner's output as plain
    lines and a bar's carriage returns arrive there as unreadable fragments; the
    output is not a terminal at all; or a step-level bar is already on screen, in
    which case a second bar fighting for the same line only confuses. Bars stay
    on for interactive runs that have no step bar of their own — parsing millions
    of log lines genuinely benefits from one.
    """
    if _events_enabled() or not sys.stderr.isatty():
        return True
    with _bar_lock:
        return _active_bar is not None


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


# ── Terminal progress bar ─────────────────────────────────────────────────────
# Drawn in place on stderr for interactive runs. Log lines and the bar share the
# stream, so the console handler in cesal_core.utils.config clears the bar before
# writing a line and redraws it afterwards; without that the two interleave and
# leave fragments behind.

_BAR_WIDTH = 26
_bar_lock = threading.RLock()
_active_bar: Optional["_ProgressBar"] = None


class _ProgressBar:
    """An in-place ``████░░░░ 42%`` bar with a running time estimate."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(int(total), 1)
        self.done = 0
        self.note = ""
        self.started = time.perf_counter()
        self._drawn = False

    def _line(self) -> str:
        frac = min(self.done / self.total, 1.0)
        filled = int(round(frac * _BAR_WIDTH))
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        text = f"   {bar} {frac * 100:5.1f}%  {self.done:,}/{self.total:,} {self.label}"
        if 0 < self.done < self.total:
            elapsed = time.perf_counter() - self.started
            text += f" · ~{fmt_secs(elapsed / self.done * (self.total - self.done))} left"
        if self.note:
            text += f" · {self.note}"
        return text[:150]

    def draw(self) -> None:
        try:
            sys.stderr.write("\r\033[K" + self._line())
            sys.stderr.flush()
            self._drawn = True
        except Exception:
            pass

    def clear(self) -> None:
        if not self._drawn:
            return
        try:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        except Exception:
            pass
        self._drawn = False

    def close(self) -> None:
        """Leave the completed bar on screen and move to a fresh line."""
        if self._drawn:
            try:
                sys.stderr.write("\r\033[K" + self._line() + "\n")
                sys.stderr.flush()
            except Exception:
                pass
        self._drawn = False


def clear_bar() -> None:
    """Erase the active bar so a log line can be written cleanly."""
    with _bar_lock:
        if _active_bar is not None:
            _active_bar.clear()


def redraw_bar() -> None:
    """Redraw the active bar after a log line has been written."""
    with _bar_lock:
        if _active_bar is not None:
            _active_bar.draw()


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

    def __init__(self, reporter: "StepReporter", step_id: str, title: str,
                 index: int, explain: str = ""):
        self.id = step_id
        self.title = title
        self.explain = explain
        self.index = index
        self._rep = reporter
        self._lock = threading.Lock()
        self._started = 0.0
        self._outcome: Dict[str, Any] = {}
        self._failure = ""
        self._seen_details: set = set()
        self._phase_name = ""
        self._phase_started = 0.0
        self._counters: Dict[str, Dict[str, int]] = {}
        self._bar_unit: Optional[str] = None
        self.elapsed = 0.0
        self.state = "pending"

    # ── live reporting ────────────────────────────────────────────────────
    def detail(self, label: str, value: Any) -> None:
        """Print one aligned ``label ... value`` line as the step runs.

        Repeats of an identical label/value pair are dropped: a step that builds
        many models re-reports the same dataset facts each time, and printing
        them once is the useful behaviour.
        """
        key = (label, str(value))
        with self._lock:
            if key in self._seen_details:
                return
            self._seen_details.add(key)
        logging.info("%s", _leader(label, fmt_count(value)))
        self._rep._emit({"t": "step_detail", "id": self.id,
                         "k": label, "v": str(value)})

    def note(self, message: str) -> None:
        """Print an un-aligned informational line inside the step."""
        logging.info("   %s", message)
        self._rep._emit({"t": "step_note", "id": self.id, "msg": message})

    def phase(self, name: str) -> None:
        """Announce the sub-stage now running inside this step.

        A step like the edge scan is several distinct pieces of work — parsing
        the logs, scoring them, combining the votes — and a run that only says
        "Edge Q-BAT scan" gives no idea which of them is taking the time. The
        name is printed as the phase begins, so the detail lines it produces
        appear underneath it; its duration follows when it finishes, and is
        omitted for phases too brief to be worth a line.
        """
        self._close_phase()
        self._phase_name = name
        self._phase_started = time.perf_counter()
        logging.info("   ├─ %s", name)
        self._rep._emit({"t": "step_phase", "id": self.id, "name": name})

    # Phases quicker than this are not worth a line of their own.
    _PHASE_REPORT_SECS = 1.0

    def _close_phase(self, last: bool = False) -> None:
        if not self._phase_name:
            return
        secs = time.perf_counter() - self._phase_started
        if secs >= self._PHASE_REPORT_SECS:
            logging.info("   %s  took %s", "│" if not last else "╵", fmt_secs(secs))
        self._rep._emit({"t": "step_phase_done", "id": self.id,
                         "name": self._phase_name, "secs": round(secs, 3)})
        self._phase_name = ""

    def warn(self, message: str) -> None:
        logging.warning("   ! %s", message)
        self._rep._emit({"t": "step_warn", "id": self.id, "msg": message})

    def expect(self, unit: str, total: int, bar: bool = True) -> None:
        """Declare how many items of ``unit`` this step will process.

        On an interactive terminal this also starts a progress bar. Where a bar
        cannot be drawn (a redirected log, or the dashboard reading the stream)
        progress falls back to milestone lines at each 25%.
        """
        global _active_bar
        with self._lock:
            self._counters[unit] = {"done": 0, "total": int(total),
                                    "mark": 0, "epct": -1}
        if bar and total > 0 and not bars_suppressed():
            with _bar_lock:
                if _active_bar is not None:
                    _active_bar.close()
                _active_bar = _ProgressBar(unit, total)
                self._bar_unit = unit
                _active_bar.draw()
        self._rep._emit({"t": "step_expect", "id": self.id,
                         "unit": unit, "total": int(total)})

    def progress_note(self, text: str) -> None:
        """Set the trailing note on the progress bar (e.g. what is running now)."""
        with _bar_lock:
            if _active_bar is not None:
                _active_bar.note = text
                _active_bar.draw()

    def tick(self, unit: str, n: int = 1) -> None:
        """Count ``n`` completed items.

        Thread-safe: the edge and cloud stages tick from worker threads. To keep
        the terminal readable a milestone line is printed at each ``_STEP_PCT``
        percent of the expected total rather than once per item — per-item
        records stay in the log file at DEBUG level. The interval is small
        enough that a slow stage still shows movement, so a long scan is never
        mistaken for a hang. The dashboard receives every tick as an event.
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
                if pct >= c["mark"] + _STEP_PCT or done == total:
                    c["mark"] = (pct // _STEP_PCT) * _STEP_PCT
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

        with _bar_lock:
            if _active_bar is not None and unit == getattr(self, "_bar_unit", None):
                _active_bar.done = done
                _active_bar.draw()
                show = False  # the bar already conveys progress
        if show:
            if total:
                # Percentage only: the raw counts are an internal unit (for the
                # edge scan, windows x learners) and reading them as "windows"
                # is misleading.
                logging.info("   %s  %d%%", unit, done * 100 // total)
            else:
                logging.info("   %s %s", f"{done:,}", unit)
        if emit:
            self._rep._emit({"t": "step_progress", "id": self.id, "unit": unit,
                             "done": done, "total": total})

    def outcome(self, **values: Any) -> None:
        """Record the step's result; rendered in its closing block."""
        self._outcome.update(values)

    def fail(self, message: str) -> None:
        """Mark this step as failed without raising.

        For steps that process many items and finish the batch before deciding
        the whole thing did not succeed — reporting "done" there would be
        untrue. The closing block and the run summary both show the failure.
        """
        self._failure = message

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
        if self.explain:
            logging.info("   %s", self.explain)
            logging.info("")
        self._rep._emit({"t": "step_start", "id": self.id, "n": self.index,
                         "total": self._rep.total, "title": self.title,
                         "explain": self.explain})
        return self

    def __enter__(self) -> "Step":
        return self.start()

    def done(self) -> None:
        """Close a step opened with :meth:`start`."""
        self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, tb) -> bool:
        global _active_step, _active_bar
        if self.state != "running":  # already closed, or never started
            return False
        with _bar_lock:
            if _active_bar is not None:
                _active_bar.close()
                _active_bar = None
        self.elapsed = time.perf_counter() - self._started
        with _active_lock:
            _active_step = None

        if exc_type is not None:
            self.state = "failed"
            self._close_phase(last=True)
            logging.error("   ✗ failed after %s — %s", fmt_secs(self.elapsed), exc)
            self._rep._record(self, error=str(exc))
            self._rep._emit({"t": "step_fail", "id": self.id,
                             "secs": round(self.elapsed, 3), "error": str(exc)})
            return False  # never swallow the exception

        self._close_phase(last=True)
        for label, value in self._outcome.items():
            logging.info("%s", _leader(label.replace("_", " "), fmt_count(value)))

        if self._failure:
            self.state = "failed"
            logging.error("   ✗ %s", self._failure)
            self._rep._record(self, error=self._failure)
            self._rep._emit({"t": "step_fail", "id": self.id,
                             "secs": round(self.elapsed, 3),
                             "error": self._failure,
                             "outcome": {k: str(v) for k, v in self._outcome.items()}})
            return False

        self.state = "done"
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
    def phase(self, name: str) -> None: pass
    def warn(self, message: str) -> None: logging.warning("%s", message)
    def expect(self, unit: str, total: int, bar: bool = True) -> None: pass
    def progress_note(self, text: str) -> None: pass
    def tick(self, unit: str, n: int = 1) -> None: pass
    def outcome(self, **values: Any) -> None: pass
    def fail(self, message: str) -> None: pass
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
        steps: Sequence[Sequence[str]] = (),
        adopt: str = "",
        about: str = "",
    ):
        global _active_run
        self.run = run
        self.dataset = dataset
        self.about = about
        # Plan entries are (id, title) or (id, title, plain-language explanation).
        self.plan: List[Tuple[str, str, str]] = [
            (e[0], e[1], e[2] if len(e) > 2 else "") for e in steps
        ]
        self._records: List[Dict[str, Any]] = []
        self._skipped: Dict[str, str] = {}
        self._metrics: List[Dict[str, Any]] = []
        self._inherited_secs = 0.0
        self._started = time.perf_counter()

        inherited = self._adopt(adopt) if adopt else False

        self.total = len(self.plan)
        self._by_id = {e[0]: i + 1 for i, e in enumerate(self.plan)}
        self._titles = {e[0]: e[1] for e in self.plan}
        self._explains = {e[0]: e[2] for e in self.plan}

        with _active_lock:
            _active_run = self

        if not inherited:
            logging.info("")
            logging.info("%s", _RULE_HEAVY)
            logging.info(" CESAL · %s%s", run, f" · {dataset}" if dataset else "")
            logging.info("%s", _RULE_HEAVY)
            if about:
                for line in about.strip().splitlines():
                    logging.info("   %s", line.strip())
                logging.info("")
            for i, (_, title, _x) in enumerate(self.plan, start=1):
                logging.info("   %d. %s", i, title)
            self._emit({"t": "run_start", "run": run, "dataset": dataset,
                        "about": about,
                        "steps": [{"id": i, "title": t, "explain": x}
                                  for i, t, x in self.plan]})

    # ── cross-process handoff ─────────────────────────────────────────────
    def _adopt(self, path: str) -> bool:
        """Absorb a parent process's plan and completed steps. Returns success."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False  # a missing handoff must never break the run
        self.plan = [(s["id"], s["title"], s.get("explain", ""))
                     for s in data.get("plan", [])] or self.plan
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
            "plan": [{"id": i, "title": t, "explain": x} for i, t, x in self.plan],
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
        return Step(self, step_id, title or self._titles.get(step_id, step_id),
                    index, self._explains.get(step_id, ""))

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

        for index, (step_id, title, _x) in enumerate(self.plan, start=1):
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
            logging.info("   Scores")
            for m in self._metrics:
                logging.info("     %-22s P %6.2f   R %6.2f   F1 %6.2f",
                             m["label"][:22], m["precision"], m["recall"], m["f_score"])

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
