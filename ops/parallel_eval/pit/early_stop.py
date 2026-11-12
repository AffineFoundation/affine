"""RED-only early stop of the candidate probe.

OFF by default, in shadow mode and in live mode. It runs only when the PIT
config sets ``early_stop.enabled`` to true. No default config, install script
or shared test fixture sets it.

The rule (Jacob, 2026-11-11):

* The host reads the completion stream of the probe and nothing else. The
  stream is ``grades.jsonl`` in the run directory on the box: the worker
  appends one line when all rollouts of one task are graded.
* A check is due when ``min_tasks`` tasks are finished (default 100), and then
  after every ``check_every`` more (default 20). A check gives the finished
  candidate cells to the host scorer. That is the official
  ``baseline_v1.estimate`` with the full head baseline of the round.
* The probe stops when the estimate has ``z < bar_z - margin``. ``bar_z`` is
  the z bar of the gate (1.0) and the default ``margin`` is 2, so the default
  stop is at ``z < -1``.
* A stop ends the requests that still run. The decision is RED, with
  ``early_stop=true`` and the fraction of tasks that were finished.
* A stop never gives GREEN. Without a stop the probe runs to its end, and the
  decision is the normal one on the full sample.

The checks are a fixed function of the stream: check number k looks at the
first ``min_tasks + k * check_every`` lines. When the host reads the stream,
and how many lines one read returns, does not change the result.

Every doubt means no stop: a scorer that fails, a stream line that is not a
finished task, an estimate that is not RED. The probe then runs to its end.
"""
from __future__ import annotations

import json
import math
import shlex
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple

VERSION = 'pit-probe-early-stop-v1'
STREAM_NAME = 'grades.jsonl'
CELL_KEYS = ('n', 'c', 'w', 'u', 'b', 'x', 'tok', 'L')
# The live stall rule gives the worker this long to end its requests and write the partial result.
CANCEL_WAIT_S = 60


class EarlyStopError(ValueError):
    """Early-stop settings that are not valid."""


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class EarlyStop:
    """Settings of the early stop. ``EarlyStop()`` is off."""

    enabled: bool = False       # the flag; only the JSON value true turns the early stop on
    margin: float = 2.0         # m: stop when z < bar_z - m
    min_tasks: int = 100        # finished tasks before the first check
    check_every: int = 20       # finished tasks between two checks
    poll_s: float = 30.0        # seconds between two looks of the host at the stream

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EarlyStopError('early_stop.enabled must be true or false, not %r' % (self.enabled,))
        margin = _number(self.margin)
        if margin is None or margin < 0:
            # A negative margin would stop a candidate that is above the bar.
            raise EarlyStopError('early_stop.margin must be a number >= 0, not %r' % (self.margin,))
        for name in ('min_tasks', 'check_every'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EarlyStopError('early_stop.%s must be a whole number >= 1, not %r' % (name, value))
        poll = _number(self.poll_s)
        if poll is None or poll <= 0:
            raise EarlyStopError('early_stop.poll_s must be a number > 0, not %r' % (self.poll_s,))

    @classmethod
    def from_mapping(cls, raw: Any) -> 'EarlyStop':
        """Settings from the ``early_stop`` value of the PIT config. No value means off."""
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise EarlyStopError('early_stop must be a mapping such as {"enabled": true}, not %r' % (raw,))
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise EarlyStopError('early_stop has unknown settings: %s' % ', '.join(map(str, unknown)))
        return cls(**dict(raw))

    def is_due(self, tasks: int) -> bool:
        """True when a check is due after ``tasks`` finished tasks."""
        return tasks >= self.min_tasks and (tasks - self.min_tasks) % self.check_every == 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stop_verdict(scored: Any, settings: EarlyStop) -> Optional[Dict[str, Any]]:
    """What the official estimate of the finished cells says about a stop.

    ``scored`` is the answer of the host scorer. Returns the numbers of the stop, or None when there is no stop.
    An estimate that is not RED is never a stop, so a stop cannot carry a GREEN.
    """
    if not isinstance(scored, Mapping) or str(scored.get('decision') or '').lower() != 'red':
        return None
    evaluated = scored.get('evaluated')
    if not isinstance(evaluated, Mapping):
        return None
    z, tasks, bar = _number(evaluated.get('z')), _number(evaluated.get('n')), _number(scored.get('z_min'))
    if z is None or tasks is None or bar is None or tasks < settings.min_tasks:
        return None
    threshold = bar - float(settings.margin)
    if not z < threshold:
        return None
    return dict(z=z, bar_z=bar, margin=float(settings.margin), threshold=threshold, scored_tasks=int(tasks))


def parse_line(line: str) -> Optional[Tuple[str, int, Dict[str, Any]]]:
    """One line of the stream -> (environment id, task index, cell). None when it is not one finished task."""
    try:
        doc = json.loads(line)
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    env_id, index, rollouts = doc.get('env_id'), doc.get('index'), doc.get('n')
    if not isinstance(env_id, str) or isinstance(index, bool) or not isinstance(index, int):
        return None
    if isinstance(rollouts, bool) or not isinstance(rollouts, int) or rollouts < 1:
        return None
    return env_id, index, {key: doc[key] for key in CELL_KEYS if key in doc}


def cells_result(cells: Mapping[str, Mapping[str, Any]], base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A probe result that holds these finished cells and no other. ``cells`` is {environment id: {index: cell}}."""
    out = {key: value for key, value in (base or {}).items() if key not in ('envs', 'early_stop')}
    out['envs'] = {env_id: {'per_task': dict(per_task)} for env_id, per_task in cells.items()}
    out['partial'] = True
    return out


class Monitor:
    """Follows the completion stream of one probe run and applies the rule.

    ``read(lines_done)`` returns the text of the stream after its first ``lines_done`` lines.
    ``score(result)`` returns the official estimate of the cells in ``result``.
    ``cancel()`` ends the requests of the run that still run.
    """

    def __init__(self, settings: EarlyStop, *, read: Callable[[int], str], score: Callable[[Dict[str, Any]], Any],
                 cancel: Callable[[], None], tasks_total: int, clock: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self.tasks_total = int(tasks_total)
        self._read, self._score, self._cancel, self._clock = read, score, cancel, clock
        self._next_look = 0.0
        self._lines = 0                                     # lines of the stream that are taken
        self._seen: set = set()
        self._queue: Deque[Tuple[str, int, Dict[str, Any]]] = deque()
        self._order: List[Tuple[str, int]] = []             # finished tasks in stream order
        self._cells: Dict[str, Dict[str, Any]] = {}
        self._checked = 0                                   # tasks at the last check that gave an answer
        self.checks = 0
        self.failures = 0
        self.last_failure: Optional[str] = None
        self.broken: Optional[str] = None                   # why the stream cannot be used; then no stop, ever
        self.stop: Optional[Dict[str, Any]] = None
        self.cancelled = False

    # -------------------------------------------------------------- the stream
    def _pull(self) -> None:
        lines = (self._read(self._lines) or '').split('\n')
        tail = lines.pop()               # text after the last newline: a line that the worker has not ended yet
        for line in lines:
            self._take(line, complete=True)
        if tail.strip():
            # The link to the box can drop the last newline. A whole JSON object is then a whole line.
            self._take(tail, complete=False)

    def _take(self, line: str, *, complete: bool) -> None:
        if self.broken:
            return
        parsed = parse_line(line) if line.strip() else None
        if parsed is None:
            if not complete:
                return                   # read it again at the next look
            if line.strip():
                self.broken = 'line %d of the stream is not a finished task' % (self._lines + 1)
                return
        self._lines += 1
        if parsed is not None and parsed[:2] not in self._seen:
            self._seen.add(parsed[:2])
            self._queue.append(parsed)

    # -------------------------------------------------------------- the checks
    def _check(self, tasks: int) -> Optional[Dict[str, Any]]:
        """The estimate of the first ``tasks`` finished tasks. None: no answer yet. {}: no stop. Else the stop record."""
        scored = self._score(cells_result(self._cells))
        if isinstance(scored, Mapping) and str(scored.get('decision') or '').lower() == 'pending':
            return None                  # the head baseline is not ready; this check runs again at the next look
        self.checks += 1
        verdict = stop_verdict(scored, self.settings)
        if verdict is None:
            return {}
        total = max(self.tasks_total, tasks)
        return dict(
            verdict, version=VERSION, triggered=True, tasks=tasks, tasks_total=total, fraction=tasks / total,
            min_tasks=self.settings.min_tasks, check_every=self.settings.check_every, checks=self.checks,
            order=[[env_id, index] for env_id, index in self._order],
        )

    def _advance(self) -> None:
        if self.stop is not None or self.broken:
            return
        self._pull()
        while not self.broken:
            tasks = len(self._order)
            if self.settings.is_due(tasks) and self._checked < tasks:
                out = self._check(tasks)
                if out is None:
                    return
                self._checked = tasks
                if out:
                    self.stop = out
                    return
            if not self._queue:
                return
            env_id, index, cell = self._queue.popleft()
            self._order.append((env_id, index))
            self._cells.setdefault(env_id, {})[str(index)] = cell

    def _guarded(self, step: Callable[[], None]) -> None:
        try:
            step()
        except Exception as exc:  # a failure of the monitor must never fail the probe
            self.failures += 1
            self.last_failure = ('%s: %s' % (type(exc).__name__, exc))[:300]

    def poll(self) -> None:
        """One look at the stream, at most once per ``poll_s`` seconds. On a stop it ends the run. Never raises."""
        now = self._clock()
        if now < self._next_look:
            return
        self._next_look = now + float(self.settings.poll_s)
        self._guarded(self._advance)
        if self.stop is not None and not self.cancelled:
            self._guarded(self._cancel_run)

    def _cancel_run(self) -> None:
        self._cancel()
        self.cancelled = True

    def settle(self, probe_result: Dict[str, Any]) -> None:
        """Give the probe result of the stage its stop record. Only a partial result can get one.

        A partial result that the monitor did not stop itself (the host was down, or the probe stalled) is checked
        against the whole stream. The record is the same as when the host had watched: the checks depend on the
        stream alone.
        """
        if not probe_result.get('partial'):
            return
        if self.stop is None:
            self._guarded(self._advance)
        if self.stop is not None:
            probe_result['early_stop'] = dict(self.stop)

    def state(self) -> Dict[str, Any]:
        return dict(lines=self._lines, tasks=len(self._order), checks=self.checks, failures=self.failures,
                    last_failure=self.last_failure, broken=self.broken, stopped=self.stop is not None,
                    cancelled=self.cancelled)


def for_box(settings: EarlyStop, pod: Any, *, root: str, run_name: str, tasks_total: int,
            score: Callable[[Dict[str, Any]], Any], clock: Callable[[], float] = time.time) -> Monitor:
    """The monitor of one probe run on a box. ``pod.run`` runs a shell command on the box; ``root`` is the probe
    directory there. The stop uses the command of the live stall rule: ``podctl.py finalize-run``."""
    stream = shlex.quote('%s/runs/%s/%s' % (root.rstrip('/'), run_name, STREAM_NAME))
    finalize = 'python3 %s finalize-run %s %d' % (
        shlex.quote('%s/podctl.py' % root.rstrip('/')), shlex.quote(run_name), CANCEL_WAIT_S)

    def read(lines_done: int) -> str:
        return pod.run('tail -n +%d %s 2>/dev/null || true' % (lines_done + 1, stream), timeout=120, check=False) or ''

    def cancel() -> None:
        out = pod.run(finalize, timeout=CANCEL_WAIT_S + 120, check=False) or ''
        if '"finalized": true' not in out:
            raise RuntimeError('finalize-run did not end the run: %s' % out[-200:])

    return Monitor(settings, read=read, score=score, cancel=cancel, tasks_total=tasks_total, clock=clock)


# ------------------------------------------------------------------ the decision
@dataclass(frozen=True)
class Stopped:
    """The part of a partial probe result that a stop decision used."""

    cells: Dict[str, Dict[str, Any]]
    tasks: int
    tasks_total: int

    def result(self, probe_result: Mapping[str, Any]) -> Dict[str, Any]:
        return cells_result(self.cells, probe_result)


def stopped_cells(probe_result: Mapping[str, Any], settings: EarlyStop) -> Optional[Stopped]:
    """The cells of the stop record in a partial probe result. None when the result has no record that fits.

    The record is a claim. The caller scores these cells again and takes the stop only when ``stop_verdict`` of that
    score is a stop.
    """
    record = probe_result.get('early_stop')
    if not settings.enabled or not isinstance(record, Mapping) or record.get('triggered') is not True:
        return None
    order, total = record.get('order'), record.get('tasks_total')
    if not isinstance(order, list) or not settings.is_due(len(order)):
        return None
    if isinstance(total, bool) or not isinstance(total, int) or total < len(order):
        return None
    cells: Dict[str, Dict[str, Any]] = {}
    for entry in order:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return None
        env_id, index = entry
        if not isinstance(env_id, str) or isinstance(index, bool) or not isinstance(index, int):
            return None
        env = (probe_result.get('envs') or {}).get(env_id)
        cell = ((env or {}).get('per_task') or {}).get(str(index)) if isinstance(env, Mapping) else None
        if not isinstance(cell, Mapping) or not cell.get('n') or str(index) in cells.get(env_id, {}):
            return None
        cells.setdefault(env_id, {})[str(index)] = dict(cell)
    return Stopped(cells=cells, tasks=len(order), tasks_total=total)


def stamp(scored: Mapping[str, Any], stopped: Stopped, verdict: Mapping[str, Any], settings: EarlyStop) -> Dict[str, Any]:
    """The RED score of an early stop: the official estimate of the stopped cells, marked as an early stop."""
    fraction = stopped.tasks / stopped.tasks_total
    out = dict(scored)
    out['decision'] = 'red'
    out['early_stop'] = True
    out['early_stop_fraction'] = fraction
    out['early_stop_detail'] = dict(
        version=VERSION, tasks=stopped.tasks, tasks_total=stopped.tasks_total, fraction=fraction,
        z=verdict['z'], bar_z=verdict['bar_z'], margin=verdict['margin'], threshold=verdict['threshold'],
        min_tasks=settings.min_tasks, check_every=settings.check_every,
    )
    out['reasons'] = list(scored.get('reasons') or []) + [
        'early_stop: z %.3f < %.2f after %d of %d tasks' % (
            verdict['z'], verdict['threshold'], stopped.tasks, stopped.tasks_total)]
    return out


def log_extra(scored: Any) -> Optional[Dict[str, Any]]:
    """What the decision log keeps of an early stop. None for every other decision, so their entries do not change."""
    if not isinstance(scored, Mapping) or scored.get('early_stop') is not True:
        return None
    return dict(early_stop=True, early_stop_fraction=scored.get('early_stop_fraction'))
