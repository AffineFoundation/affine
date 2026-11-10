"""Detached on-box GREEN bench job (same shape as PIT ``box_job``).

Host ticks only **inspect** — never sleep/poll inside ``tick()``. Stages run
under ``setsid``/``nohup`` on the rented ``affine-pe-bench-*`` box:

  setup → fetch → run → RESULT.json

Box self-setup clones AffineFoundation/affine at ``BENCH_BOX_GIT_REF`` and runs
``bench/deploy/bootstrap.sh`` (venv-vllm + verifier envs). Idempotent markers
make re-adopt restart-safe without a second rent.

No ``bench.affine_bench`` import on the host. No standing-daemon state
(``daemon.json`` / ``NODE`` / ``C_POD``).
"""
from __future__ import annotations

import json
import os
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from .box_job import (
    LOG_ATTEMPT_PREFIX,
    LOG_NAME,
    PID_NAME,
    RESULT_NAME,
    STATUS_DONE,
    STATUS_NEED_LAUNCH,
    STATUS_RUNNING,
    BoxJobError,
    BoxJobStatus,
    RemoteBoxJobStore,
    next_log_attempt,
    read_log_tail,
)

# Pin the box-side bench/ + deploy/ tree. Production should set
# AFFINE_BENCH_GIT_REF to a commit SHA; default is the PIT PR branch that owns
# this on-demand path (no host paths).
BENCH_BOX_GIT_REF = os.environ.get('AFFINE_BENCH_GIT_REF') or 'cursor/pit-shadow-mode-0524'
BENCH_GIT_URL = os.environ.get(
    'AFFINE_BENCH_GIT_URL', 'https://github.com/AffineFoundation/affine.git',
)

# Box-local layout (on the rented pod — not a host path).
BENCH_BOX_ROOT = '/data/bench'
BENCH_JOB_ROOT = '/data/bench/jobs'

BENCH_STAGES: tuple[str, ...] = ('setup', 'fetch', 'run')
BENCH_PROFILE = 'qwen3.5-family-pp0'


def job_dir_for(head: str, *, job_root: str = BENCH_JOB_ROOT) -> str:
    safe = ''.join(c for c in str(head) if c.isalnum() or c in '-_')[:64] or 'head'
    return '%s/%s' % (job_root.rstrip('/'), safe)


def marker_path(job_dir: str, stage: str) -> str:
    return '%s/markers/%s.done' % (job_dir.rstrip('/'), stage)


def result_path(job_dir: str) -> str:
    return '%s/%s' % (job_dir.rstrip('/'), RESULT_NAME)


def pid_path(job_dir: str) -> str:
    return '%s/%s' % (job_dir.rstrip('/'), PID_NAME)


def log_path(job_dir: str) -> str:
    return '%s/%s' % (job_dir.rstrip('/'), LOG_NAME)


def next_missing_stage(markers: Set[str]) -> Optional[str]:
    for stage in BENCH_STAGES:
        if stage not in markers:
            return stage
    return None


def _loads_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None
    try:
        doc = json.loads(raw)
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def fetch_result(store: Any, job_dir: str) -> Optional[Dict[str, Any]]:
    return _loads_json(store.read_text(result_path(job_dir)))


def read_markers(store: Any, job_dir: str) -> Set[str]:
    return set(store.list_marker_stages(job_dir))


def read_pid(store: Any, job_dir: str) -> Optional[int]:
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


def inspect_bench_job(store: Any, job_dir: str) -> BoxJobStatus:
    """Non-blocking inspect (pid-alive before markers — same race rule as box_job)."""
    result = fetch_result(store, job_dir)
    if result is not None:
        return BoxJobStatus(state=STATUS_DONE, markers=set(BENCH_STAGES), result=result)

    pid = read_pid(store, job_dir)
    alive = bool(pid and store.pid_alive(pid))
    if alive:
        markers = read_markers(store, job_dir)
        return BoxJobStatus(
            state=STATUS_RUNNING,
            markers=markers,
            pid=pid,
            next_stage=next_missing_stage(markers),
        )

    markers = read_markers(store, job_dir)
    result = fetch_result(store, job_dir)
    if result is not None:
        return BoxJobStatus(state=STATUS_DONE, markers=set(BENCH_STAGES) | markers, result=result)
    nxt = next_missing_stage(markers)
    if nxt is None:
        return BoxJobStatus(state=STATUS_NEED_LAUNCH, markers=markers, next_stage='result')
    return BoxJobStatus(state=STATUS_NEED_LAUNCH, markers=markers, next_stage=nxt, pid=pid)


def setup_command(*, git_ref: str = BENCH_BOX_GIT_REF, git_url: str = BENCH_GIT_URL) -> str:
    """Idempotent clone + ``bench/deploy/bootstrap.sh``; writes ``markers/setup.done``."""
    root = BENCH_BOX_ROOT
    repo = '%s/repo' % root
    ref = shlex.quote(git_ref)
    url = shlex.quote(git_url)
    return (
        'set -euo pipefail; '
        'JD="$PWD"; mkdir -p {root}/{{models,hf,runs,logs,tmp,uvcache,jobs}} "$JD/markers"; '
        'if [ ! -d {repo}/.git ]; then '
        '  git clone --depth 1 --branch {ref} {url} {repo} '
        '    || (git clone {url} {repo} && git -C {repo} fetch --depth 1 origin {ref} '
        '        && git -C {repo} checkout {ref}); '
        'else '
        '  git -C {repo} fetch --depth 1 origin {ref} || git -C {repo} fetch origin {ref}; '
        '  git -C {repo} checkout {ref} || git -C {repo} checkout FETCH_HEAD; '
        'fi; '
        'bash {repo}/bench/deploy/bootstrap.sh; '
        'test -f {root}/venv-vllm/bin/python; '
        'test -f {root}/venv-vf/bin/python; '
        'printf \'%s\\n\' \'{{"ok":true,"ref":"{ref_raw}","bootstrap":"BOOTSTRAP_OK"}}\' '
        '> "$JD/markers/setup.done"'
    ).format(
        root=shlex.quote(root),
        repo=shlex.quote(repo),
        ref=ref,
        url=url,
        ref_raw=git_ref.replace('"', ''),
    )


def fetch_command(*, head: str) -> str:
    """Fetch king weights on the box (uses host-pushed descriptor/urls when present)."""
    head_q = shlex.quote(head)
    root = BENCH_BOX_ROOT
    return (
        'set -euo pipefail; '
        'JD="$PWD"; HEAD={head}; '
        'DESC="$JD/descriptor.json"; URLS="$JD/urls.private.json"; '
        'KINGS={root}/models/kings; mkdir -p "$KINGS" "$JD/markers"; '
        'if [ -f "$DESC" ] && [ -f "$URLS" ]; then '
        '  . {root}/env.sh; '
        '  python3 {root}/repo/bench/deploy/gate_fetch.py "$DESC" "$KINGS" "$URLS" --threads 32 '
        '    >> "$JD/fetch.log" 2>&1; '
        '  rm -f "$URLS"; '
        'elif [ -f "$KINGS/$HEAD/.verified" ]; then '
        '  echo "weights already verified"; '
        'else '
        '  echo "fetch inputs missing (descriptor/urls or .verified)" >&2; exit 2; '
        'fi; '
        'test -f "$KINGS/$HEAD/.verified" || test -d "$KINGS/$HEAD"; '
        'printf \'%s\\n\' "{{\\"ok\\":true,\\"head\\":\\"$HEAD\\"}}" > "$JD/markers/fetch.done"'
    ).format(head=head_q, root=shlex.quote(root))


def run_command(*, head: str, profile: str = BENCH_PROFILE) -> str:
    """Run ``affine-bench node-run`` for the king (box registry → pp0); write RESULT.json."""
    head16 = head[:16]
    mid = 'king-%s' % head16
    out = '%s/runs/%s' % (BENCH_BOX_ROOT, mid)
    model_dir = '%s/models/kings/%s' % (BENCH_BOX_ROOT, head)
    # Box-side only: registry import happens on the rented box after setup.
    py_resolve = (
        'import json,os,sys\n'
        'sys.path.insert(0, os.environ.get("BENCH_REPO","/data/bench/repo"))\n'
        'from bench.affine_bench import registry as R\n'
        'st=R.resolve_settings(%r, kind="king")\n'
        'assert st.get("profile")==%r, st\n'
        'assert float(st.get("presence_penalty",1))==0.0, st\n'
        'json.dump(st, open(os.environ["JD"]+"/sampling.json","w"), indent=1)\n'
        'bs=[b["id"] for b in R.benchmarks(enabled_only=True, runnable_only=True)]\n'
        'open(os.environ["JD"]+"/benchmarks.txt","w").write(",".join(bs))\n'
        'print("sampling_ok", st["profile"], st["presence_penalty"], len(bs))\n'
    ) % (mid, profile)
    py_result = (
        'import json,glob,os,time\n'
        'jd,out,mid,head=os.environ["JD"],os.environ["OUT"],os.environ["MID"],os.environ["HEAD"]\n'
        'st=json.load(open(jd+"/sampling.json"))\n'
        'rows=[]\n'
        'for f in sorted(glob.glob(out+"/*/summary.json")):\n'
        ' r=json.load(open(f)); r["model_id"]=mid; r["model_kind"]="king"\n'
        ' r["king"]=head; r["profile"]=st.get("profile"); r["presence_penalty"]=st.get("presence_penalty")\n'
        ' rows.append(r)\n'
        'doc=dict(ok=True,head=head,model_id=mid,profile=st.get("profile"),'
        'presence_penalty=float(st.get("presence_penalty") or 0),sampling=st,'
        'benchmarks=len(rows),results=rows,finished_at=time.time())\n'
        'open(jd+"/RESULT.json","w").write(json.dumps(doc,indent=1)+"\\n")\n'
        'open(jd+"/markers/run.done","w").write(json.dumps(dict(ok=True,benchmarks=len(rows)))+"\\n")\n'
        'print("RESULT_OK", len(rows))\n'
    )
    return (
        'set -euo pipefail; '
        'JD="$PWD"; . {root}/env.sh; '
        'export PATH={root}/venv-vf/bin:$PATH; '
        'export BENCH_REPO={root}/repo BENCH_ROOT={root}; '
        'HEAD={head}; MID={mid}; OUT={out}; MDIR={mdir}; '
        'mkdir -p "$OUT" "$JD/markers"; '
        'JD="$JD" python3 -c {py1}; '
        'BIDS=$(cat "$JD/benchmarks.txt"); test -n "$BIDS"; '
        'affine-bench node-run --model-id "$MID" --model-dir "$MDIR" '
        '--reasoning-parser qwen3 --benchmarks "$BIDS" --out "$OUT"; '
        'test -f "$OUT/DONE"; '
        'JD="$JD" OUT="$OUT" MID="$MID" HEAD="$HEAD" python3 -c {py2}'
    ).format(
        root=shlex.quote(BENCH_BOX_ROOT),
        head=shlex.quote(head),
        mid=shlex.quote(mid),
        out=shlex.quote(out),
        mdir=shlex.quote(model_dir),
        py1=shlex.quote(py_resolve),
        py2=shlex.quote(py_result),
    )


def stage_command(stage: str, *, head: str, profile: str = BENCH_PROFILE,
                  git_ref: str = BENCH_BOX_GIT_REF) -> str:
    if stage == 'setup':
        return setup_command(git_ref=git_ref)
    if stage == 'fetch':
        return fetch_command(head=head)
    if stage == 'run':
        return run_command(head=head, profile=profile)
    raise BoxJobError('unknown bench stage %r' % stage)


@dataclass
class BenchBoxController:
    """Non-blocking host controller for one head's on-box bench job."""

    store: Any
    head: str
    profile: str = BENCH_PROFILE
    git_ref: str = BENCH_BOX_GIT_REF
    job_root: str = BENCH_JOB_ROOT
    now_fn: Callable[[], float] = time.time

    @property
    def job_dir(self) -> str:
        return job_dir_for(self.head, job_root=self.job_root)

    def status(self) -> BoxJobStatus:
        return inspect_bench_job(self.store, self.job_dir)

    def ensure_dir(self) -> None:
        self.store.ensure_dir(self.job_dir)

    def read_log_tail(self, *, n_lines: int = 200) -> Optional[str]:
        return read_log_tail(self.store, self.job_dir, n_lines=n_lines)

    def tick_once(self) -> Dict[str, Any]:
        """Advance at most one launch; never sleep. Safe to call every host tick."""
        self.ensure_dir()
        st = self.status()
        if st.state == STATUS_DONE and st.result is not None:
            return {'op': 'done', 'result': st.result, 'state': st.state}
        if st.state == STATUS_RUNNING:
            return {
                'op': 'running',
                'pid': st.pid,
                'markers': sorted(st.markers),
                'next_stage': st.next_stage,
                'state': st.state,
            }
        # NEED_LAUNCH
        stage = st.next_stage
        if stage is None or stage == 'result':
            # All markers present but no RESULT — treat as failure for retry.
            return {
                'op': 'missing_result',
                'markers': sorted(st.markers),
                'state': st.state,
                'error': 'markers complete but RESULT.json missing',
            }
        if stage not in BENCH_STAGES:
            raise BoxJobError('unknown bench stage %r' % stage)
        # Kill stale before launch (same as box_job).
        kill = getattr(self.store, 'kill_pids', None)
        pid = read_pid(self.store, self.job_dir)
        if pid and self.store.pid_alive(pid) and callable(kill):
            kill([pid])
        cmd = stage_command(stage, head=self.head, profile=self.profile, git_ref=self.git_ref)
        launched = self.store.launch_detached(self.job_dir, stage=stage, command=cmd)
        return {
            'op': 'launched',
            'stage': stage,
            'pid': launched,
            'state': STATUS_RUNNING,
            'markers': sorted(st.markers),
        }


# ---------------------------------------------------------------------------
# In-memory store for hermetic tests (bench stages).
# ---------------------------------------------------------------------------
@dataclass
class RemoteBenchBoxStore:
    """SSH-backed store for bench stages (setup/fetch/run)."""

    pod: Any
    _inner: Optional[RemoteBoxJobStore] = field(default=None, init=False, repr=False)

    def _store(self) -> RemoteBoxJobStore:
        if self._inner is None:
            self._inner = RemoteBoxJobStore(self.pod)
        return self._inner

    def ensure_dir(self, job_dir: str) -> None:
        self._store().ensure_dir(job_dir)

    def read_text(self, path: str) -> Optional[str]:
        return self._store().read_text(path)

    def write_text(self, path: str, text: str) -> None:
        self._store().write_text(path, text)

    def remove(self, path: str) -> None:
        self._store().remove(path)

    def list_marker_stages(self, job_dir: str) -> Set[str]:
        out = self.pod.run(
            'for f in %s/markers/*.done; do [ -f "$f" ] && basename "$f" .done; done 2>/dev/null || true'
            % shlex.quote(job_dir.rstrip('/')),
            timeout=60, check=False,
        )
        found: Set[str] = set()
        for line in (out or '').splitlines():
            name = line.strip()
            if name in BENCH_STAGES:
                found.add(name)
        return found

    def pid_alive(self, pid: int) -> bool:
        return self._store().pid_alive(pid)

    def kill_pids(self, pids: Sequence[int]) -> None:
        self._store().kill_pids(pids)

    def pgrep(self, pattern: str) -> Sequence[int]:
        return self._store().pgrep(pattern)

    def tail_text(self, path: str, n_lines: int = 200) -> Optional[str]:
        return self._store().tail_text(path, n_lines)

    def list_log_attempts(self, job_dir: str) -> Sequence[int]:
        return self._store().list_log_attempts(job_dir)

    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
        return self._store().launch_detached(job_dir, stage=stage, command=command)


@dataclass
class MemBenchBoxStore:
    files: Dict[str, str] = field(default_factory=dict)
    alive_pids: Set[int] = field(default_factory=set)
    next_pid: int = 5000
    pending: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    default_ticks: int = 2
    launch_log: list = field(default_factory=list)
    fail_stage: Optional[str] = None  # when set, that stage exits without marker

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
                name = path[len(prefix):-5]
                if name in BENCH_STAGES:
                    found.add(name)
        return found

    def pid_alive(self, pid: int) -> bool:
        return int(pid) in self.alive_pids

    def kill_pids(self, pids: Sequence[int]) -> None:
        for pid in pids:
            self.alive_pids.discard(int(pid))
            self.pending.pop(int(pid), None)

    def pgrep(self, pattern: str) -> Sequence[int]:
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

    def arm_launch(
        self,
        *,
        ticks: int = 2,
        payload: Optional[Mapping[str, Any]] = None,
        auto_marker: bool = True,
        write_result: bool = False,
    ) -> None:
        self._next_ticks = ticks
        self._next_payload = dict(payload or {})
        self._next_auto_marker = auto_marker
        self._next_write_result = write_result

    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.alive_pids.add(pid)
        self.files['%s/%s' % (job_dir.rstrip('/'), PID_NAME)] = str(pid)
        attempt = next_log_attempt(self, job_dir)
        log_n = '%s/%s%d' % (job_dir.rstrip('/'), LOG_ATTEMPT_PREFIX, attempt)
        self.files[log_n] = 'pit-box-job-%s pid=%d\n' % (stage, pid)
        self.files['%s/%s' % (job_dir.rstrip('/'), LOG_NAME)] = self.files[log_n]
        ticks = int(getattr(self, '_next_ticks', self.default_ticks))
        payload = dict(getattr(self, '_next_payload', {}) or {})
        auto_marker = bool(getattr(self, '_next_auto_marker', True))
        write_res = bool(getattr(self, '_next_write_result', False))
        if self.fail_stage == stage:
            auto_marker = False
            write_res = False
        self.pending[pid] = {
            'stage': stage,
            'job_dir': job_dir,
            'command': command,
            'ticks_left': ticks,
            'payload': payload,
            'auto_marker': auto_marker,
            'write_result': write_res,
            'log_attempt': attempt,
        }
        self.launch_log.append({
            'pid': pid, 'stage': stage, 'command': command, 'job_dir': job_dir,
        })
        return pid

    def advance(self) -> None:
        self._tick()

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
            if job.get('auto_marker') and stage in BENCH_STAGES:
                path = marker_path(jd, stage)
                self.files[path] = json.dumps(job.get('payload') or {'ok': True}, sort_keys=True) + '\n'
            if job.get('write_result') or (stage == 'run' and job.get('auto_marker')):
                head = jd.rstrip('/').split('/')[-1]
                # Pad short test heads.
                if len(head) < 64:
                    head = (head + '0' * 64)[:64]
                doc = {
                    'ok': True,
                    'head': head if len(head) == 64 else 'a' * 64,
                    'model_id': 'king-%s' % (head[:16]),
                    'profile': BENCH_PROFILE,
                    'presence_penalty': 0.0,
                    'benchmarks': 1,
                    'results': [{
                        'model_id': 'king-%s' % head[:16],
                        'model_kind': 'king',
                        'benchmark': 'math500',
                        'benchmark_id': 'math500',
                        'score': 0.5,
                        'se': 0.01,
                        'profile': BENCH_PROFILE,
                        'presence_penalty': 0.0,
                        'env_version': 'math500/v1/bench-v2',
                    }],
                }
                self.files[result_path(jd)] = json.dumps(doc, indent=1) + '\n'
            self.alive_pids.discard(pid)
            self.files.pop('%s/%s' % (jd.rstrip('/'), PID_NAME), None)
            # Append failure note when stage fails.
            if not job.get('auto_marker'):
                logp = '%s/%s' % (jd.rstrip('/'), LOG_NAME)
                self.files[logp] = (self.files.get(logp) or '') + 'FATAL setup failed\nrc=1\n'
            done.append(pid)
        for pid in done:
            self.pending.pop(pid, None)
