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
STATUS_RUNNING = 'running'
STATUS_DONE = 'done'
STATUS_NEED_LAUNCH = 'need_launch'


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


def inspect_job(store: BoxJobStore, job_dir: str) -> BoxJobStatus:
    result = fetch_result(store, job_dir)
    markers = read_markers(store, job_dir)
    if result is not None:
        return BoxJobStatus(state=STATUS_DONE, markers=set(STAGES), result=result)
    pid = read_pid(store, job_dir)
    if pid and store.pid_alive(pid):
        return BoxJobStatus(
            state=STATUS_RUNNING,
            markers=markers,
            pid=pid,
            next_stage=next_missing_stage(markers),
        )
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
        """
        stage = canonicalize_stage(stage)
        if stage not in STAGES:
            raise BoxJobError('unknown stage %r' % stage)
        self.ensure_dir()
        st = self.status()
        if st.result is not None:
            return dict(st.result)
        if stage in st.markers:
            raw = self.store.read_text(marker_path(self.job_dir, stage))
            return _loads_json(raw) or {}

        # Stage missing: if a detached job is still running for this stage, poll it.
        # Sync stages (no detached_command) always kill stale leftovers and re-run.
        deadline = float(self.now_fn()) + float(timeout_s)
        if (
            (wait_for_marker or detached_command)
            and st.state == STATUS_RUNNING
            and st.next_stage == stage
        ):
            while float(self.now_fn()) < deadline:
                st = self.status()
                if stage in st.markers:
                    raw = self.store.read_text(marker_path(self.job_dir, stage))
                    return _loads_json(raw) or {}
                if st.state != STATUS_RUNNING:
                    break
                self.sleep_fn(poll_s)
            st = self.status()
            if stage in st.markers:
                raw = self.store.read_text(marker_path(self.job_dir, stage))
                return _loads_json(raw) or {}

        stage = canonicalize_stage(stage)
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
            self.store.launch_detached(self.job_dir, stage=stage, command=detached_command)
            if wait_for_marker:
                while float(self.now_fn()) < deadline:
                    st = self.status()
                    if stage in st.markers:
                        raw = self.store.read_text(marker_path(self.job_dir, stage))
                        return _loads_json(raw) or {}
                    if st.state != STATUS_RUNNING and stage not in st.markers:
                        # Process died without marker — fail closed (caller may retry once).
                        raise BoxJobError('detached stage %s exited without marker' % stage)
                    self.sleep_fn(poll_s)
                raise BoxJobError('detached stage %s timed out' % stage)

        payload = dict(runner() or {})
        write_marker(self.store, self.job_dir, stage, payload)
        return payload

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

    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
        # Tag process with pit-box-job so kill_stale / pgrep can find duplicates.
        wrapped = (
            'mkdir -p {jd}/markers; cd {jd}; '
            'setsid nohup bash -c {cmd} > {log} 2>&1 < /dev/null & '
            'echo $! > {pid}; cat {pid}'
        ).format(
            jd=shlex.quote(job_dir.rstrip('/')),
            cmd=shlex.quote(
                'echo pit-box-job-%s; %s' % (stage, command)
            ),
            log=shlex.quote(LOG_NAME),
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

    def ensure_dir(self, job_dir: str) -> None:
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

    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.alive_pids.add(pid)
        self.files['%s/%s' % (job_dir.rstrip('/'), PID_NAME)] = str(pid)
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
        }
        self.launch_log.append({'pid': pid, 'stage': stage, 'command': command, 'job_dir': job_dir})
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
