"""Restart-safe box-side train+probe jobs (detached setsid/nohup).

Host threads die on restart; box-side work must not. A job under
``{job_root}/{item_id}/`` writes stage markers and a final ``RESULT.json``.
On re-adopt the host inspects markers, resumes polling, or fetches the
result — never re-runs a stage that already has a marker. Before launching
a missing stage, any stale box-side duplicate for that job is killed.
"""
from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Set


# 'eval' is the probe/serve stage (eval.done). Legacy 'probe.done' markers are
# accepted as aliases when reading so mid-flight boxes stay resume-safe.
STAGES: tuple[str, ...] = ('download', 'train', 'eval')
STAGE_ALIASES: Dict[str, str] = {'probe': 'eval'}
RESULT_NAME = 'RESULT.json'
PID_NAME = 'job.pid'
LOG_NAME = 'job.log'
# Per-attempt logs: job.log.0, job.log.1, … (never truncate prior attempts).
LOG_ATTEMPT_PREFIX = 'job.log.'
STATUS_RUNNING = 'running'
STATUS_DONE = 'done'
STATUS_NEED_LAUNCH = 'need_launch'

# After a detached stage exits with no marker yet, re-read markers/RESULT up to
# three times over ~10s (write-.done-then-exit race / NFS lag).
MARKER_REREAD_ATTEMPTS = 3
MARKER_REREAD_GAP_S = 3.0


class BoxJobError(RuntimeError):
    pass


def job_dir_for(item_id: str, *, job_root: str = '/workspace/pst/pit-jobs') -> str:
    safe = ''.join(c for c in str(item_id) if c.isalnum() or c in '-_')[:64] or 'item'
    return '%s/%s' % (job_root.rstrip('/'), safe)


def marker_path(job_dir: str, stage: str) -> str:
    return '%s/markers/%s.done' % (job_dir.rstrip('/'), stage)


def result_path(job_dir: str) -> str:
    return '%s/%s' % (job_dir.rstrip('/'), RESULT_NAME)


def pid_path(job_dir: str) -> str:
    return '%s/%s' % (job_dir.rstrip('/'), PID_NAME)


def log_path(job_dir: str) -> str:
    return '%s/%s' % (job_dir.rstrip('/'), LOG_NAME)


def log_attempt_path(job_dir: str, attempt: int) -> str:
    return '%s/%s%d' % (job_dir.rstrip('/'), LOG_ATTEMPT_PREFIX, int(attempt))


def next_log_attempt(store: 'BoxJobStore', job_dir: str) -> int:
    """Next unused ``job.log.<n>`` index (0-based); never reuse / truncate."""
    lister = getattr(store, 'list_log_attempts', None)
    if callable(lister):
        try:
            existing = sorted(int(x) for x in lister(job_dir))
            return (existing[-1] + 1) if existing else 0
        except Exception:
            pass
    # Fallback: probe job.log.0 .. job.log.63 for the first missing slot past max.
    found: list[int] = []
    for i in range(64):
        if store.read_text(log_attempt_path(job_dir, i)) is not None:
            found.append(i)
    return (found[-1] + 1) if found else 0


def read_log_tail(
    store: 'BoxJobStore',
    job_dir: str,
    *,
    n_lines: int = 200,
) -> Optional[str]:
    """Last ``n_lines`` of the box job log (stdout/stderr of the detached stage)."""
    path = log_path(job_dir)
    tail_fn = getattr(store, 'tail_text', None)
    if callable(tail_fn):
        text = tail_fn(path, int(n_lines))
    else:
        text = store.read_text(path)
        if text:
            lines = text.splitlines()
            text = '\n'.join(lines[-int(n_lines):])
    if not text:
        return None
    return text if text.endswith('\n') else text + '\n'


def next_missing_stage(markers: Set[str]) -> Optional[str]:
    for stage in STAGES:
        if stage not in markers:
            return stage
    return None


@dataclass
class BoxJobStatus:
    state: str
    markers: Set[str] = field(default_factory=set)
    result: Optional[Dict[str, Any]] = None
    pid: Optional[int] = None
    next_stage: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            'state': self.state,
            'markers': sorted(self.markers),
            'result': self.result,
            'pid': self.pid,
            'next_stage': self.next_stage,
            'error': self.error,
        }


class BoxJobStore(Protocol):
    def ensure_dir(self, job_dir: str) -> None: ...
    def read_text(self, path: str) -> Optional[str]: ...
    def write_text(self, path: str, text: str) -> None: ...
    def remove(self, path: str) -> None: ...
    def list_marker_stages(self, job_dir: str) -> Set[str]: ...
    def pid_alive(self, pid: int) -> bool: ...
    def kill_pids(self, pids: Sequence[int]) -> None: ...
    def pgrep(self, pattern: str) -> Sequence[int]: ...
    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int: ...


def _loads_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                doc = json.loads(line)
            except Exception:
                continue
            if isinstance(doc, dict):
                return doc
    try:
        doc = json.loads(raw)
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def read_markers(store: BoxJobStore, job_dir: str) -> Set[str]:
    return set(store.list_marker_stages(job_dir))


def read_pid(store: BoxJobStore, job_dir: str) -> Optional[int]:
    text = store.read_text(pid_path(job_dir))
    if not text:
        return None
    tok = text.strip().split()
    if not tok:
        return None
    try:
        return int(tok[0])
    except Exception:
        return None


def fetch_result(store: BoxJobStore, job_dir: str) -> Optional[Dict[str, Any]]:
    return _loads_json(store.read_text(result_path(job_dir)))


def canonicalize_stage(stage: str) -> str:
    s = str(stage or '')
    return STAGE_ALIASES.get(s, s)


def write_marker(
    store: BoxJobStore,
    job_dir: str,
    stage: str,
    payload: Optional[Mapping[str, Any]] = None,
) -> None:
    stage = canonicalize_stage(stage)
    if stage not in STAGES:
        raise BoxJobError('unknown stage %r' % stage)
    store.ensure_dir(job_dir)
    store.write_text(
        marker_path(job_dir, stage),
        json.dumps(dict(payload or {}), sort_keys=True, default=str) + '\n',
    )


def write_result(store: BoxJobStore, job_dir: str, result: Mapping[str, Any]) -> None:
    store.ensure_dir(job_dir)
    store.write_text(
        result_path(job_dir),
        json.dumps(dict(result), sort_keys=True, default=str) + '\n',
    )


def kill_stale_job(
    store: BoxJobStore,
    job_dir: str,
    *,
    extra_patterns: Sequence[str] = (),
) -> list[int]:
    killed: list[int] = []
    pid = read_pid(store, job_dir)
    if pid and store.pid_alive(pid):
        store.kill_pids([pid])
        killed.append(pid)
    patterns = ['pit-box-job', job_dir.rstrip('/')]
    patterns.extend(extra_patterns)
    for pat in patterns:
        for p in store.pgrep(pat):
            if p not in killed:
                store.kill_pids([p])
                killed.append(p)
    store.remove(pid_path(job_dir))
    return killed


def job_log_rc_ok(store: BoxJobStore, job_dir: str) -> bool:
    """True when ``job.log`` (or train.log) shows a clean ``rc=0`` exit."""
    texts: list[str] = []
    tail = read_log_tail(store, job_dir, n_lines=80)
    if tail:
        texts.append(tail)
    # Detached train also writes PST/train.log; Remote stores may surface it.
    for extra in (
        '%s/train.log' % job_dir.rstrip('/'),
        '/workspace/pst/train.log',
    ):
        raw = store.read_text(extra)
        if raw:
            texts.append(raw)
    for text in texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Prefer trailing lines (final exit) but accept anywhere for short logs.
        window = lines[-20:] if len(lines) > 20 else lines
        if any(ln == 'rc=0' or ln.endswith(' rc=0') for ln in window):
            return True
        if 'rc=0' in '\n'.join(window):
            return True
    return False


def train_artifacts_present(store: BoxJobStore, job_dir: str) -> bool:
    """True when on-box train shards / checkpoint artifacts exist.

    Prefer a store-specific checker when available (Remote / Mem); otherwise
    look for a recorded artifact hint under the job dir.
    """
    checker = getattr(store, 'train_artifacts_present', None)
    if callable(checker):
        try:
            return bool(checker(job_dir))
        except Exception:
            pass
    # Fallback hint file written by recovery tests / box helpers.
    hint = store.read_text('%s/train-artifacts.ok' % job_dir.rstrip('/'))
    return bool(hint and hint.strip())


def inspect_job(store: BoxJobStore, job_dir: str) -> BoxJobStatus:
    """Inspect box job state.

    Order matters: check pid-alive **before** reading markers. A stage that
    writes ``<stage>.done`` and exits between those two reads used to look like
    "not running + no marker" when markers were snapshotted first.
    """
    result = fetch_result(store, job_dir)
    if result is not None:
        return BoxJobStatus(state=STATUS_DONE, markers=set(STAGES), result=result)

    pid = read_pid(store, job_dir)
    alive = bool(pid and store.pid_alive(pid))
    # Pid-alive first — if still running, accept possibly-incomplete markers.
    if alive:
        markers = read_markers(store, job_dir)
        return BoxJobStatus(
            state=STATUS_RUNNING,
            markers=markers,
            pid=pid,
            next_stage=next_missing_stage(markers),
        )

    # Process dead: read markers *after* the pid check so a just-written
    # ``.done`` (and RESULT.json for eval) are visible.
    markers = read_markers(store, job_dir)
    result = fetch_result(store, job_dir)
    if result is not None:
        return BoxJobStatus(state=STATUS_DONE, markers=set(STAGES) | markers, result=result)
    nxt = next_missing_stage(markers)
    if nxt is None:
        return BoxJobStatus(state=STATUS_NEED_LAUNCH, markers=markers, next_stage='result')
    return BoxJobStatus(state=STATUS_NEED_LAUNCH, markers=markers, next_stage=nxt, pid=pid)


@dataclass
class BoxJobController:
    """Host-side controller for one item's box job."""

    store: BoxJobStore
    item_id: str
    job_root: str = '/workspace/pst/pit-jobs'
    sleep_fn: Callable[[float], None] = time.sleep
    now_fn: Callable[[], float] = time.time
    _stage_retries: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def job_dir(self) -> str:
        return job_dir_for(self.item_id, job_root=self.job_root)

    def status(self) -> BoxJobStatus:
        return inspect_job(self.store, self.job_dir)

    def ensure_dir(self) -> None:
        self.store.ensure_dir(self.job_dir)

    def mark_stage(self, stage: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        write_marker(self.store, self.job_dir, stage, payload)

    def publish_result(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        doc = dict(result)
        write_result(self.store, self.job_dir, doc)
        return doc

    def kill_stale(self, *, extra_patterns: Sequence[str] = ()) -> list[int]:
        return kill_stale_job(self.store, self.job_dir, extra_patterns=extra_patterns)

    def read_log_tail(self, *, n_lines: int = 200) -> Optional[str]:
        """Last lines of ``job.log`` — capture before release so host keeps the trail."""
        return read_log_tail(self.store, self.job_dir, n_lines=n_lines)

    def _read_stage_payload(self, stage: str) -> Dict[str, Any]:
        raw = self.store.read_text(marker_path(self.job_dir, stage))
        return _loads_json(raw) or {}

    def _await_late_marker(
        self,
        stage: str,
        *,
        attempts: int = MARKER_REREAD_ATTEMPTS,
        gap_s: float = MARKER_REREAD_GAP_S,
    ) -> Optional[Dict[str, Any]]:
        """Re-read markers (and RESULT for eval) after a short delay.

        Covers the write-.done-then-exit race and slow filesystem visibility.
        Up to ``attempts`` re-reads spaced by ``gap_s`` (~10s total by default).
        """
        for _ in range(max(1, int(attempts))):
            self.sleep_fn(float(gap_s))
            result = fetch_result(self.store, self.job_dir)
            if result is not None:
                return dict(result)
            markers = read_markers(self.store, self.job_dir)
            if stage in markers:
                return self._read_stage_payload(stage)
            if stage == 'eval':
                # Eval's durable completion is RESULT.json even without eval.done.
                result = fetch_result(self.store, self.job_dir)
                if result is not None:
                    return dict(result)
        return None

    def _recover_from_artifacts(self, stage: str) -> Optional[Dict[str, Any]]:
        """If on-box artifacts exist and job log rc==0, treat the stage as done.

        Never release on 'exited without marker' without this check first.
        """
        stage = canonicalize_stage(stage)
        if not job_log_rc_ok(self.store, self.job_dir):
            return None
        if stage == 'eval':
            result = fetch_result(self.store, self.job_dir)
            if result is not None:
                return dict(result)
            return None
        if stage == 'train':
            if not train_artifacts_present(self.store, self.job_dir):
                return None
            # Synthesize a minimal successful train marker from log / hint.
            payload: Dict[str, Any] = {'ok': True, 'recovered': 'artifacts+rc0'}
            hint = getattr(self.store, 'recovered_train_payload', None)
            if callable(hint):
                try:
                    extra = hint(self.job_dir)
                    if isinstance(extra, Mapping):
                        payload.update(dict(extra))
                except Exception:
                    pass
            ck = payload.get('checkpoint') or payload.get('candidate_id')
            if not ck:
                # Best-effort parse of checkpoint hex from job.log.
                tail = read_log_tail(self.store, self.job_dir, n_lines=200) or ''
                for line in reversed(tail.splitlines()):
                    line = line.strip()
                    if line.startswith('{'):
                        try:
                            doc = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(doc, dict):
                            ck = doc.get('checkpoint') or (doc.get('done') or {}).get('checkpoint')
                            if ck:
                                payload['checkpoint'] = ck
                                break
            write_marker(self.store, self.job_dir, 'train', payload)
            return payload
        return None

    def _handle_exited_without_marker(self, stage: str) -> Dict[str, Any]:
        """Recover or signal that the caller may retry once on the same box.

        Returns a payload when recovered. Raises ``BoxJobError`` with a
        ``retryable`` note when one same-box retry remains; raises a hard error
        after that retry is exhausted.
        """
        late = self._await_late_marker(stage)
        if late is not None:
            if stage not in read_markers(self.store, self.job_dir) and stage != 'eval':
                write_marker(self.store, self.job_dir, stage, late)
            elif stage in STAGES and stage not in read_markers(self.store, self.job_dir):
                # eval with RESULT only — stamp eval.done for resume.
                write_marker(self.store, self.job_dir, stage, {
                    'ok': True, 'recovered': 'late-result',
                    'candidate_id': late.get('candidate_id'),
                    'probe_result': late.get('probe_result'),
                })
            return late
        recovered = self._recover_from_artifacts(stage)
        if recovered is not None:
            return recovered
        used = int(self._stage_retries.get(stage, 0))
        if used < 1:
            self._stage_retries[stage] = used + 1
            raise BoxJobError(
                'detached stage %s exited without marker; retry once on same box' % stage
            )
        raise BoxJobError('detached stage %s exited without marker' % stage)

    def run_stage_once(
        self,
        stage: str,
        runner: Callable[[], Mapping[str, Any]],
        *,
        detached_command: Optional[str] = None,
        poll_s: float = 2.0,
        timeout_s: float = 7200.0,
        wait_for_marker: bool = False,
        extra_kill_patterns: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Skip if marker exists; else kill stale, run (or detach+poll), then mark.

        When ``detached_command`` is set the stage runs on the box under
        setsid/nohup. The box (or ``runner`` after poll) must produce the stage
        marker; if ``wait_for_marker`` the host only polls until the marker
        appears (runner optional). Otherwise ``runner()`` executes on the host
        after launch / instead of launch and its return value becomes the marker
        payload.

        Never releases the box on a bare 'exited without marker': re-reads
        markers/RESULT, checks artifacts + ``rc=0``, then retries the stage
        once on the same box before failing.
        """
        stage = canonicalize_stage(stage)
        if stage not in STAGES:
            raise BoxJobError('unknown stage %r' % stage)
        self.ensure_dir()
        deadline = float(self.now_fn()) + float(timeout_s)
        # Initial attempt + at most one same-box retry after recovery miss.
        for attempt in (0, 1):
            st = self.status()
            if st.result is not None:
                return dict(st.result)
            if stage in st.markers:
                return self._read_stage_payload(stage)

            # Stage missing: if a detached job is still running for this stage, poll it.
            # Sync stages (no detached_command) always kill stale leftovers and re-run.
            if (
                (wait_for_marker or detached_command)
                and st.state == STATUS_RUNNING
                and st.next_stage == stage
            ):
                while float(self.now_fn()) < deadline:
                    st = self.status()
                    if stage in st.markers:
                        return self._read_stage_payload(stage)
                    if st.result is not None:
                        return dict(st.result)
                    if st.state != STATUS_RUNNING:
                        break
                    self.sleep_fn(poll_s)
                st = self.status()
                if stage in st.markers:
                    return self._read_stage_payload(stage)
                if st.result is not None:
                    return dict(st.result)
                if st.state != STATUS_RUNNING and stage not in st.markers:
                    try:
                        return self._handle_exited_without_marker(stage)
                    except BoxJobError as exc:
                        if 'retry once' not in str(exc) or attempt >= 1:
                            raise
                        # Fall through: kill_stale + relaunch on same box (once).

            kill_patterns = list(extra_kill_patterns)
            if stage == 'train':
                kill_patterns.append('train_pit.py')
            elif stage == 'download':
                kill_patterns.extend(('gate_fetch.py', 'bundle_fetch.py'))
            elif stage == 'eval':
                # Re-run must clear leftover vLLM / podctl / harness from a prior host.
                kill_patterns.extend((
                    'vllm serve', 'podctl.py serve', 'podctl.py start-run',
                    'pit-box-job-eval', 'affine_gatepool',
                ))
            self.kill_stale(extra_patterns=kill_patterns)
            if detached_command:
                self.store.launch_detached(
                    self.job_dir, stage=stage, command=detached_command,
                )
                if wait_for_marker:
                    while float(self.now_fn()) < deadline:
                        st = self.status()
                        if stage in st.markers:
                            return self._read_stage_payload(stage)
                        if st.result is not None:
                            return dict(st.result)
                        if st.state != STATUS_RUNNING and stage not in st.markers:
                            try:
                                return self._handle_exited_without_marker(stage)
                            except BoxJobError as exc:
                                if 'retry once' not in str(exc) or attempt >= 1:
                                    raise
                                break  # for-loop: one same-box relaunch
                        self.sleep_fn(poll_s)
                    else:
                        raise BoxJobError('detached stage %s timed out' % stage)
                    continue  # retry launch once

                # Detached without wait_for_marker: runner runs on host after launch.
                payload = dict(runner() or {})
                write_marker(self.store, self.job_dir, stage, payload)
                return payload

            payload = dict(runner() or {})
            write_marker(self.store, self.job_dir, stage, payload)
            return payload

        raise BoxJobError('detached stage %s exited without marker' % stage)

    def resume_or_fetch(self) -> Optional[Dict[str, Any]]:
        """Return RESULT if present; None if host must continue missing stages."""
        st = self.status()
        if st.state == STATUS_DONE and st.result is not None:
            return dict(st.result)
        return None


# ---------------------------------------------------------------------------
# Remote store (SSH pod via P.run)
# ---------------------------------------------------------------------------
class PodLike(Protocol):
    def run(self, cmd: str, timeout: int = 60, check: bool = False, **kwargs: Any) -> str: ...


@dataclass
class RemoteBoxJobStore:
    pod: PodLike

    def ensure_dir(self, job_dir: str) -> None:
        self.pod.run(
            'mkdir -p %s/markers' % shlex.quote(job_dir.rstrip('/')),
            timeout=30, check=True,
        )

    def read_text(self, path: str) -> Optional[str]:
        out = self.pod.run(
            'test -f %s && cat %s || echo __MISSING__'
            % (shlex.quote(path), shlex.quote(path)),
            timeout=60, check=False,
        )
        text = (out or '').strip()
        if not text or text == '__MISSING__' or text.endswith('\n__MISSING__'):
            if text.endswith('\n__MISSING__'):
                body = text[: -len('\n__MISSING__')].rstrip()
                return body or None
            return None
        if '__MISSING__' in text and not text.startswith('{'):
            return None
        return text

    def write_text(self, path: str, text: str) -> None:
        parent = path.rsplit('/', 1)[0]
        self.pod.run(
            'mkdir -p %s && cat > %s <<\'EOF\'\n%sEOF'
            % (shlex.quote(parent), shlex.quote(path), text if text.endswith('\n') else text + '\n'),
            timeout=60, check=True,
        )

    def remove(self, path: str) -> None:
        self.pod.run('rm -f %s' % shlex.quote(path), timeout=15, check=False)

    def list_marker_stages(self, job_dir: str) -> Set[str]:
        out = self.pod.run(
            'for f in %s/markers/*.done; do [ -f "$f" ] && basename "$f" .done; done 2>/dev/null || true'
            % shlex.quote(job_dir.rstrip('/')),
            timeout=60, check=False,
        )
        found: Set[str] = set()
        for line in (out or '').splitlines():
            name = canonicalize_stage(line.strip())
            if name in STAGES:
                found.add(name)
        return found

    def pid_alive(self, pid: int) -> bool:
        out = self.pod.run(
            'kill -0 %d 2>/dev/null && echo ALIVE || echo DEAD' % int(pid),
            timeout=30, check=False,
        )
        return 'ALIVE' in (out or '')

    def train_artifacts_present(self, job_dir: str) -> bool:
        """True when safetensor shards exist under PST/out or KDIR (train succeeded)."""
        out = self.pod.run(
            "find /workspace/pst/out /opt/plain/probe/affine-env-probe/ckpt "
            "-name 'model-*-of-*.safetensors' 2>/dev/null | head -3",
            timeout=60, check=False,
        )
        text = (out or '').strip()
        return bool(text) and 'model-' in text

    def recovered_train_payload(self, job_dir: str) -> Optional[Dict[str, Any]]:
        """Best-effort checkpoint id from train.log / job.log done events."""
        for path in (
            '/workspace/pst/train.log',
            '%s/train.log' % job_dir.rstrip('/'),
            log_path(job_dir),
        ):
            raw = self.read_text(path)
            if not raw:
                continue
            for line in reversed(raw.splitlines()):
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    doc = json.loads(line)
                except Exception:
                    continue
                if not isinstance(doc, dict):
                    continue
                ck = doc.get('checkpoint')
                if not ck and isinstance(doc.get('done'), dict):
                    ck = doc['done'].get('checkpoint')
                if isinstance(ck, str) and len(ck) == 64:
                    return {'ok': True, 'checkpoint': ck, 'candidate_id': ck}
        return None

    def kill_pids(self, pids: Sequence[int]) -> None:
        for pid in pids:
            self.pod.run(
                'kill %d 2>/dev/null || true; sleep 1; kill -9 %d 2>/dev/null || true'
                % (int(pid), int(pid)),
                timeout=30, check=False,
            )

    def pgrep(self, pattern: str) -> Sequence[int]:
        raw = self.pod.run('pgrep -f %s || true' % shlex.quote(pattern), timeout=30, check=False)
        out: list[int] = []
        for tok in str(raw).split():
            if tok.isdigit():
                out.append(int(tok))
        return out

    def tail_text(self, path: str, n_lines: int = 200) -> Optional[str]:
        out = self.pod.run(
            'tail -n %d %s 2>/dev/null || true'
            % (int(n_lines), shlex.quote(path)),
            timeout=60, check=False,
        )
        text = (out or '').rstrip('\n')
        return text + '\n' if text else None

    def list_log_attempts(self, job_dir: str) -> Sequence[int]:
        out = self.pod.run(
            'ls -1 %s/job.log.[0-9]* 2>/dev/null | sed -n "s/.*job\\.log\\.\\([0-9][0-9]*\\)$/\\1/p" || true'
            % shlex.quote(job_dir.rstrip('/')),
            timeout=30, check=False,
        )
        found: list[int] = []
        for tok in str(out or '').split():
            if tok.isdigit():
                found.append(int(tok))
        return found

    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
        # Tag process with pit-box-job so kill_stale / pgrep can find duplicates.
        # Per-attempt log: never truncate prior job.log.N; point job.log → latest.
        attempt = next_log_attempt(self, job_dir)
        log_n = 'job.log.%d' % int(attempt)
        wrapped = (
            'mkdir -p {jd}/markers; cd {jd}; '
            'setsid nohup bash -c {cmd} > {log} 2>&1 < /dev/null & '
            'echo $! > {pid}; '
            'ln -sfn {log} {link} 2>/dev/null || cp -f {log} {link} 2>/dev/null || true; '
            'cat {pid}'
        ).format(
            jd=shlex.quote(job_dir.rstrip('/')),
            cmd=shlex.quote(
                'echo pit-box-job-%s; %s' % (stage, command)
            ),
            log=shlex.quote(log_n),
            link=shlex.quote(LOG_NAME),
            pid=shlex.quote(PID_NAME),
        )
        out = self.pod.run(wrapped, timeout=60, check=True)
        text = (out or '').strip().splitlines()
        if not text:
            raise BoxJobError('detached launch produced no pid')
        try:
            return int(text[-1].strip().split()[0])
        except Exception as exc:
            raise BoxJobError('bad pid from launch: %r' % out) from exc


# ---------------------------------------------------------------------------
# In-memory store for hermetic restart tests (no SSH / no host paths).
# ---------------------------------------------------------------------------
@dataclass
class MemBoxJobStore:
    files: Dict[str, str] = field(default_factory=dict)
    alive_pids: Set[int] = field(default_factory=set)
    next_pid: int = 1000
    pending: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # ticks before a launched stage auto-writes its marker
    default_ticks: int = 2
    crash_pids: Set[int] = field(default_factory=set)
    launch_log: list = field(default_factory=list)
    # Artifact-recovery hooks for hermetic tests (no real shards).
    has_train_artifacts: bool = False
    train_recovery_payload: Optional[Dict[str, Any]] = None

    def ensure_dir(self, job_dir: str) -> None:
        return None

    def train_artifacts_present(self, job_dir: str) -> bool:
        if self.has_train_artifacts:
            return True
        hint = self.files.get('%s/train-artifacts.ok' % job_dir.rstrip('/'))
        return bool(hint and str(hint).strip())

    def recovered_train_payload(self, job_dir: str) -> Optional[Dict[str, Any]]:
        if self.train_recovery_payload:
            return dict(self.train_recovery_payload)
        return None

    def read_text(self, path: str) -> Optional[str]:
        return self.files.get(path)

    def tail_text(self, path: str, n_lines: int = 200) -> Optional[str]:
        text = self.files.get(path)
        if not text:
            return None
        lines = text.splitlines()
        body = '\n'.join(lines[-int(n_lines):])
        return body + '\n' if body else None

    def write_text(self, path: str, text: str) -> None:
        self.files[path] = text

    def remove(self, path: str) -> None:
        self.files.pop(path, None)

    def list_marker_stages(self, job_dir: str) -> Set[str]:
        prefix = '%s/markers/' % job_dir.rstrip('/')
        found: Set[str] = set()
        for path in self.files:
            if path.startswith(prefix) and path.endswith('.done'):
                name = canonicalize_stage(path[len(prefix):-5])
                if name in STAGES:
                    found.add(name)
        return found

    def pid_alive(self, pid: int) -> bool:
        return int(pid) in self.alive_pids

    def advance(self) -> None:
        """Progress background jobs by one tick (wire to controller sleep_fn in tests)."""
        self._tick()

    def kill_pids(self, pids: Sequence[int]) -> None:
        for pid in pids:
            self.alive_pids.discard(int(pid))
            self.pending.pop(int(pid), None)

    def pgrep(self, pattern: str) -> Sequence[int]:
        # Only report alive job pids whose command/stage metadata matches.
        out: list[int] = []
        for pid, job in self.pending.items():
            blob = '%s %s %s' % (job.get('stage'), job.get('job_dir'), job.get('command'))
            if pattern in blob or pattern == 'pit-box-job':
                out.append(pid)
        return out

    def list_log_attempts(self, job_dir: str) -> Sequence[int]:
        prefix = '%s/%s' % (job_dir.rstrip('/'), LOG_ATTEMPT_PREFIX)
        found: list[int] = []
        for path in self.files:
            if path.startswith(prefix):
                tail = path[len(prefix):]
                if tail.isdigit():
                    found.append(int(tail))
        return found

    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.alive_pids.add(pid)
        self.files['%s/%s' % (job_dir.rstrip('/'), PID_NAME)] = str(pid)
        # Per-attempt log (no truncation of prior attempts); job.log → latest.
        attempt = next_log_attempt(self, job_dir)
        log_n = log_attempt_path(job_dir, attempt)
        self.files[log_n] = 'pit-box-job-%s pid=%d\n' % (stage, pid)
        self.files['%s/%s' % (job_dir.rstrip('/'), LOG_NAME)] = self.files[log_n]
        ticks = int(getattr(self, '_next_ticks', self.default_ticks))
        payload = dict(getattr(self, '_next_payload', {}) or {})
        auto_marker = bool(getattr(self, '_next_auto_marker', True))
        self.pending[pid] = {
            'stage': stage,
            'job_dir': job_dir,
            'command': command,
            'ticks_left': ticks,
            'payload': payload,
            'auto_marker': auto_marker,
            'log_attempt': attempt,
        }
        self.launch_log.append({
            'pid': pid, 'stage': stage, 'command': command, 'job_dir': job_dir,
            'log_attempt': attempt,
        })
        return pid

    def arm_launch(
        self,
        *,
        ticks: int = 2,
        payload: Optional[Mapping[str, Any]] = None,
        auto_marker: bool = True,
    ) -> None:
        self._next_ticks = ticks
        self._next_payload = dict(payload or {})
        self._next_auto_marker = auto_marker

    def _tick(self) -> None:
        done: list[int] = []
        for pid, job in list(self.pending.items()):
            if pid not in self.alive_pids:
                done.append(pid)
                continue
            job['ticks_left'] = int(job['ticks_left']) - 1
            if job['ticks_left'] > 0:
                continue
            jd = job['job_dir']
            stage = job['stage']
            if job.get('auto_marker') and stage in STAGES:
                path = marker_path(jd, stage)
                self.files[path] = json.dumps(job.get('payload') or {}, sort_keys=True) + '\n'
            self.alive_pids.discard(pid)
            self.files.pop('%s/%s' % (jd.rstrip('/'), PID_NAME), None)
            done.append(pid)
        for pid in done:
            self.pending.pop(pid, None)


def controller_for_pod(
    pod: PodLike,
    item_id: str,
    *,
    job_root: str = '/workspace/pst/pit-jobs',
) -> BoxJobController:
    return BoxJobController(store=RemoteBoxJobStore(pod), item_id=item_id, job_root=job_root)
