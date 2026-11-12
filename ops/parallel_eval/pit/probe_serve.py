"""Box side of the probe layout: start vLLM as a layout says, prove it, run the probe.

The live gate starts vLLM with ``podctl.py serve`` and serve.sh: one replica per
GPU, the live flags. This file starts any ``probe_layout.ProbeLayout`` and keeps
the live pieces that decide scores as they are:

* the pinned ``podctl.py`` still owns ``status``, ``start-run``,
  ``finalize-run`` and ``stop-serve``, and the pinned ``probe_worker.py`` still
  sends every request and grades every answer. This file writes the same
  ``serve.*`` files podctl writes, so those commands see the replicas;
* a layout the live serve.sh can express (one replica per GPU, the live vLLM
  build, prefix caching on) is started by ``podctl.py serve`` itself;
* after the start, each replica is checked against the layout: its process
  arguments, its GPUs, the vLLM version it reports, the window it reports. A
  flag outside the layout stops the run.

Commands (all on the box):

  start   start the replicas of one layout and return
  stop    stop every replica
  eval    serve (with the plan's fallbacks) and run the probe of one PIT item;
          writes ``markers/eval.done`` and ``RESULT.json`` like the live eval
  side    serve one layout, run one probe plan, stop, write a side record for
          ``probe_parity``

The host copies this file and ``probe_layout.py`` into the job directory
(``ship``) and starts ``eval`` behind the live clean-up preamble
(``detached_command``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from . import probe_layout as PL
except ImportError:  # on the box the two files sit side by side, outside any package
    import probe_layout as PL

SIDE_VERSION = 'pit-probe-side-v1'
TOOL_FILES = ('probe_layout.py', 'probe_serve.py')
LAYOUTS_NAME = 'probe-layouts.json'
# RemoteBoxJobStore.write_text sends a file inside one shell argument (Linux limit: 131072 bytes).
MAX_SHIP_BYTES = 100_000

# The only vLLM arguments a replica may run with. A replica with another argument fails the proof. That keeps FP8
# out of the probe: --kv-cache-dtype (FP8 KV cache) and --quantization (FP8 weights) are not in this list.
ALLOWED_FLAGS = (
    '--host', '--port', '--served-model-name', '--dtype', '--max-model-len', '--gpu-memory-utilization',
    '--enable-prefix-caching', '--no-enable-prefix-caching', '--limit-mm-per-prompt', '--max-num-seqs',
    '--tensor-parallel-size',
)
SWITCHES = ('--enable-prefix-caching', '--no-enable-prefix-caching')

_VERSION_PY = 'import importlib.metadata as m; print(m.version("vllm"))'
_PACKAGES_PY = (
    'import importlib.metadata as m, re\n'
    'rows = {"%s==%s" % (re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(), d.version)\n'
    '        for d in m.distributions() if d.metadata["Name"]}\n'
    'print("\\n".join(sorted(rows)))\n'
)


class ServeError(RuntimeError):
    """The box does not serve the layout."""


@dataclass
class Box:
    """Where the probe lives on this machine, and how often to look."""

    root: str = PL.BOX_ROOT
    python: str = 'python3'
    port: int = PL.BASE_PORT
    poll_s: float = 5.0
    probe_poll_s: float = 10.0
    settle_s: float = 20.0
    drain_s: float = 60.0
    gpu_wait_s: float = PL.GPU_WAIT_S
    free_mib: int = 2000

    def p(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def args(self) -> List[str]:
        """These settings as options of this tool's command line."""
        return [
            '--root', self.root, '--python', self.python, '--port', str(self.port), '--poll-s', repr(self.poll_s),
            '--probe-poll-s', repr(self.probe_poll_s), '--settle-s', repr(self.settle_s),
            '--drain-s', repr(self.drain_s), '--gpu-wait-s', repr(self.gpu_wait_s), '--free-mib', str(self.free_mib),
        ]

    def podctl(self, *args: str, inp: Optional[bytes] = None, timeout: float = 120,
               check: bool = True, env: Optional[Mapping[str, str]] = None) -> str:
        proc = subprocess.run(
            [self.python, self.p('podctl.py'), *args], input=inp, capture_output=True, timeout=timeout,
            env=None if env is None else dict(env),
        )
        if check and proc.returncode != 0:
            raise ServeError('podctl %s failed (rc %d): %s' % (
                args[0], proc.returncode, proc.stderr.decode('utf-8', 'replace')[-300:]))
        return proc.stdout.decode('utf-8', 'replace')

    def status(self) -> Dict[str, Any]:
        return json.loads(self.podctl('status').strip().splitlines()[-1])


# ------------------------------------------------------------------ GPUs
def _smi(query: str) -> List[str]:
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=%s' % query, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ServeError('nvidia-smi did not run: %s' % exc) from exc
    if proc.returncode != 0:
        raise ServeError('nvidia-smi failed: %s' % (proc.stderr or proc.stdout)[-200:])
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def gpu_names() -> List[str]:
    names = _smi('name')
    if not names:
        raise ServeError('nvidia-smi lists no GPU')
    return names


def wait_gpus_free(box: Box) -> List[int]:
    """Wait until every GPU holds less than ``box.free_mib``; the trainer must be gone before vLLM starts."""
    t0 = time.time()
    while True:
        used = [int(float(row)) for row in _smi('memory.used')]
        if used and all(u < box.free_mib for u in used):
            return used
        if time.time() - t0 >= box.gpu_wait_s:
            raise ServeError('GPUs not free after %.0fs before serve: %s MiB' % (time.time() - t0, used))
        time.sleep(box.poll_s)


# ------------------------------------------------------------------ the vLLM environment
def venv_version(venv: str) -> Optional[str]:
    python = os.path.join(venv, 'bin', 'python')
    if not os.path.exists(python):
        return None
    proc = subprocess.run([python, '-c', _VERSION_PY], capture_output=True, text=True, timeout=300)
    text = proc.stdout.strip()
    return text if proc.returncode == 0 and text else None


def venv_packages(venv: str) -> List[str]:
    proc = subprocess.run(
        [os.path.join(venv, 'bin', 'python'), '-c', _PACKAGES_PY], capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ServeError('cannot list the packages of %s: %s' % (venv, proc.stderr[-200:]))
    return proc.stdout.strip().splitlines()


def packages_digest(packages: Sequence[str]) -> str:
    return hashlib.sha256(('\n'.join(packages) + '\n').encode()).hexdigest()


def _install(layout: PL.ProbeLayout, venv: str, box: Box) -> None:
    uv = shutil.which('uv') or next(
        (path for path in ('/usr/local/bin/uv', os.path.expanduser('~/.local/bin/uv')) if os.path.exists(path)), None)
    if uv is None:
        raise ServeError('uv is not on the box: install vLLM %s into %s first' % (layout.vllm_version, venv))
    if os.path.exists(venv):
        shutil.rmtree(venv)
    what = ['-r', layout.vllm_requirements] if layout.vllm_requirements else ['vllm==%s' % layout.vllm_version]
    env = dict(os.environ, UV_CACHE_DIR=box.p('uvcache'), VIRTUAL_ENV=venv)
    for step in ([uv, 'venv', '-q', '-p', '3.12', venv], [uv, 'pip', 'install', '-q'] + what):
        proc = subprocess.run(step, capture_output=True, text=True, timeout=PL.INSTALL_TIMEOUT_S, env=env)
        if proc.returncode != 0:
            raise ServeError('%s failed: %s' % (' '.join(step[1:3]), (proc.stderr or proc.stdout)[-300:]))


def runtime(layout: PL.ProbeLayout, box: Box) -> Dict[str, Any]:
    """Find the environment that holds the layout's vLLM build; install it when the layout allows.

    The package digest is checked against ``runtime_sha256`` when the layout pins one.
    """
    bundle, own = box.p('venv'), box.p('venv-vllm-%s' % layout.vllm_version)
    if layout.launcher() == 'podctl':
        places, target = [bundle], None     # serve.sh always runs the bundle's vLLM
    elif layout.vllm_venv:
        places, target = [layout.vllm_venv], None
    else:
        places, target = [bundle, own], own
    venv = next((place for place in places if venv_version(place) == layout.vllm_version), None)
    installed = False
    if venv is None:
        if target is None or not layout.install_vllm:
            raise ServeError('vLLM %s is not on the box (looked in %s)' % (layout.vllm_version, ', '.join(places)))
        _install(layout, target, box)
        if venv_version(target) != layout.vllm_version:
            raise ServeError('%s does not hold vLLM %s after the install' % (target, layout.vllm_version))
        venv, installed = target, True
    packages = venv_packages(venv)
    digest = packages_digest(packages)
    if layout.runtime_sha256 and digest != layout.runtime_sha256:
        raise ServeError('the packages of %s have digest %s; the layout pins %s' % (venv, digest, layout.runtime_sha256))
    return dict(venv=venv, vllm_version=layout.vllm_version, runtime_sha256=digest, packages=packages, installed=installed)


# ------------------------------------------------------------------ processes
def _argv_of(pid: int) -> List[str]:
    try:
        raw = Path('/proc/%d/cmdline' % pid).read_bytes()
    except OSError:
        return []
    return [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]


def _is_vllm_server(argv: Sequence[str]) -> bool:
    return any(os.path.basename(a) == 'vllm' and argv[i + 1:i + 2] == ['serve'] for i, a in enumerate(argv))


def vllm_servers() -> Dict[int, List[str]]:
    """Every ``vllm serve`` process on the machine: pid -> arguments."""
    found = {}
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            argv = _argv_of(int(entry))
            if _is_vllm_server(argv):
                found[int(entry)] = argv
    return found


def _stat(pid: int) -> List[str]:
    """Fields of /proc/<pid>/stat after the command name: state, parent, process group, session, ..."""
    try:
        return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    except (OSError, IndexError):
        return []


def running(pid: int) -> bool:
    """The process exists and has not exited. An exited process that nobody reaped does not run."""
    state = _stat(pid)[:1]
    return bool(state) and state[0] not in ('Z', 'X')


def _pid_from(path: str) -> Optional[int]:
    """The pid in a pid file when that process still runs."""
    try:
        pid = int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if running(pid) else None


def _session_of(pid: int) -> Optional[int]:
    fields = _stat(pid)
    return int(fields[3]) if len(fields) > 3 and fields[3].isdigit() else None


def _log_tail(box: Box, replica: int) -> str:
    try:
        return Path(box.p('serve.%d.log' % replica)).read_bytes()[-400:].decode('utf-8', 'replace')
    except OSError:
        return ''


def stop(box: Box) -> None:
    box.podctl('stop-serve', timeout=1200)


def _start_direct(layout: PL.ProbeLayout, cid: str, box: Box, venv: str, groups: Sequence[Sequence[int]]) -> None:
    """What ``podctl.py serve`` does, for any layout: one process per replica, in its own session, same state files."""
    Path(box.p('serve.ckpt')).write_text(cid)
    Path(box.p('serve.replicas')).write_text(str(len(groups)))
    vllm, ckpt = os.path.join(venv, 'bin', 'vllm'), box.p('ckpt', cid)
    for i, gpus in enumerate(groups):
        # Compile caches are per replica and per vLLM build: replicas race on a shared cache (serve.sh, 2026-10-29).
        cache = box.p('vcache', 'v%s-r%s' % (layout.vllm_version, '_'.join(str(g) for g in gpus)))
        for sub in ('vllm', 'triton', 'inductor'):
            os.makedirs(os.path.join(cache, sub), exist_ok=True)
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=','.join(str(g) for g in gpus),
            VLLM_CACHE_ROOT=os.path.join(cache, 'vllm'), TRITON_CACHE_DIR=os.path.join(cache, 'triton'),
            TORCHINDUCTOR_CACHE_DIR=os.path.join(cache, 'inductor'),
            VLLM_LOGGING_LEVEL='WARNING', HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
        )
        with open(box.p('serve.%d.log' % i), 'ab') as log:
            proc = subprocess.Popen(
                [vllm, 'serve', ckpt] + layout.vllm_args(box.port + i),
                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, env=env,
            )
        Path(box.p('serve.%d.pid' % i)).write_text(str(proc.pid))
    Path(box.p('serve.gen')).write_text('0')
    if os.path.exists(box.p('serve.swapcap')):      # podctl swap must not reload weights under this layout
        os.remove(box.p('serve.swapcap'))


def start(layout: PL.ProbeLayout, cid: str, box: Box, venv: str) -> List[List[int]]:
    """Stop what serves, then start the layout's replicas. Returns the GPU ids of each replica."""
    if not os.path.exists(box.p('ckpt', cid, 'config.json')):
        raise ServeError('checkpoint %s is not under %s' % (cid[:12], box.p('ckpt')))
    groups = [list(group) for group in layout.groups(len(gpu_names()))]
    stop(box)
    t0 = time.time()
    while True:     # podctl waits for each server's main process; its helpers may need a moment more
        left = vllm_servers()
        if not left:
            break
        if time.time() - t0 >= box.drain_s:
            raise ServeError('vLLM servers outside the probe state files are running: pids %s' % sorted(left))
        time.sleep(min(1.0, box.poll_s))
    if layout.launcher() == 'podctl':
        box.podctl('serve', cid, layout.gmu_text(), timeout=900,
                   env=dict(os.environ, MAX_SEQS=str(layout.max_num_seqs)))
    else:
        _start_direct(layout, cid, box, venv, groups)
    return groups


def wait_ready(layout: PL.ProbeLayout, cid: str, box: Box, need: int) -> float:
    """The live readiness rule: every replica answers, two polls in a row. A replica that exits ends the wait."""
    t0, ready = time.time(), 0
    while time.time() - t0 < layout.serve_timeout_s:
        st = box.status()
        if st.get('serve_ckpt') != cid or int(st.get('replicas') or 0) != need:
            raise ServeError('the box serves %s on %s replicas; expected %s on %d' % (
                str(st.get('serve_ckpt'))[:12], st.get('replicas'), cid[:12], need))
        dead = [i for i in range(need) if _pid_from(box.p('serve.%d.pid' % i)) is None]
        if dead:
            raise ServeError('replica %d exited while starting: %s' % (dead[0], _log_tail(box, dead[0])))
        if int(st.get('replicas_up') or 0) >= need:
            ready += 1
            if ready >= 2:
                return round(time.time() - t0, 1)
        else:
            ready = 0
        time.sleep(box.poll_s)
    raise ServeError('vLLM was not ready after %.0fs' % (time.time() - t0))


# ------------------------------------------------------------------ proof that the replicas are the layout
def parse_serve_argv(argv: Sequence[str]) -> Tuple[str, Dict[str, Optional[str]]]:
    """Arguments of one vLLM process -> (checkpoint dir, flags). Refuses a flag the layouts never set."""
    at = next((i for i, a in enumerate(argv) if os.path.basename(a) == 'vllm' and argv[i + 1:i + 2] == ['serve']), None)
    rest = list(argv[at + 2:]) if at is not None else []
    if not rest or rest[0].startswith('-'):
        raise ServeError('not a vLLM server command: %s' % ' '.join(argv)[:200])
    flags: Dict[str, Optional[str]] = {}
    i = 1
    while i < len(rest):
        flag = rest[i]
        if flag not in ALLOWED_FLAGS:
            raise ServeError('vLLM argument outside the layout: %r' % flag)
        if flag in flags:
            raise ServeError('vLLM argument given twice: %s' % flag)
        if flag in SWITCHES:
            flags[flag] = None
            i += 1
            continue
        if i + 1 >= len(rest):
            raise ServeError('vLLM argument without a value: %s' % flag)
        flags[flag] = rest[i + 1]
        i += 2
    return rest[0], flags


def argv_mismatch(layout: PL.ProbeLayout, argv: Sequence[str], *, ckpt: str, port: int) -> List[str]:
    """What the arguments of one replica say that the layout does not; empty when they agree."""
    model, flags = parse_serve_argv(argv)
    bad: List[str] = []

    def want(flag: str, value: Any, *, default: Optional[str] = None, number: bool = False) -> None:
        got = flags.get(flag, default)
        try:
            same = got is not None and (abs(float(got) - float(value)) < 1e-9 if number else got == str(value))
        except ValueError:
            same = False
        if not same:
            bad.append('%s is %s, the layout needs %s' % (flag, got, value))

    if os.path.realpath(model) != os.path.realpath(ckpt):
        bad.append('serves %s, not %s' % (model, ckpt))
    want('--host', '127.0.0.1')
    want('--port', port)
    want('--served-model-name', PL.SERVED_MODEL)
    want('--dtype', PL.GATE_DTYPE)
    want('--max-model-len', PL.GATE_MAX_MODEL_LEN)
    want('--max-num-seqs', layout.max_num_seqs)
    want('--gpu-memory-utilization', layout.gpu_memory_utilization, number=True)
    want('--tensor-parallel-size', layout.gpus_per_replica, default='1')
    try:
        if json.loads(flags.get('--limit-mm-per-prompt') or 'null') != json.loads(PL.LIMIT_MM):
            bad.append('--limit-mm-per-prompt is %s' % flags.get('--limit-mm-per-prompt'))
    except ValueError:
        bad.append('--limit-mm-per-prompt is not JSON')
    on, off = '--enable-prefix-caching' in flags, '--no-enable-prefix-caching' in flags
    if on == off or on != layout.prefix_caching:
        bad.append('prefix caching is %s, the layout needs %s' % (
            'on' if on and not off else 'off' if off and not on else 'unclear', 'on' if layout.prefix_caching else 'off'))
    return bad


def _get(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)
    except (OSError, ValueError) as exc:
        raise ServeError('%s did not answer: %s' % (url, exc)) from exc


def _visible_gpus(pid: int) -> Optional[str]:
    try:
        raw = Path('/proc/%d/environ' % pid).read_bytes()
    except OSError:
        return None
    for part in raw.split(b'\0'):
        if part.startswith(b'CUDA_VISIBLE_DEVICES='):
            return part.split(b'=', 1)[1].decode()
    return None


def verify(layout: PL.ProbeLayout, cid: str, box: Box, groups: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    """Every replica is this layout, and nothing else serves. Raises ``ServeError`` with every difference."""
    bad: List[str] = []
    replicas: List[Dict[str, Any]] = []
    for i, gpus in enumerate(groups):
        pid = _pid_from(box.p('serve.%d.pid' % i))
        if pid is None:
            bad.append('replica %d is not running' % i)
            continue
        want_gpus = ','.join(str(g) for g in gpus)
        if _visible_gpus(pid) != want_gpus:
            bad.append('replica %d sees GPUs %s, the layout gives it %s' % (i, _visible_gpus(pid), want_gpus))
        bad.extend('replica %d: %s' % (i, why) for why in argv_mismatch(
            layout, _argv_of(pid), ckpt=box.p('ckpt', cid), port=box.port + i))
        base = 'http://127.0.0.1:%d' % (box.port + i)
        version = _get(base + '/version').get('version')
        if version != layout.vllm_version:
            bad.append('replica %d runs vLLM %s, the layout needs %s' % (i, version, layout.vllm_version))
        cards = _get(base + '/v1/models').get('data') or []
        card = cards[0] if len(cards) == 1 else {}
        if card.get('id') != PL.SERVED_MODEL or card.get('max_model_len') != PL.GATE_MAX_MODEL_LEN:
            bad.append('replica %d reports model %s with a %s token window' % (i, card.get('id'), card.get('max_model_len')))
        replicas.append(dict(replica=i, pid=pid, gpus=list(gpus), port=box.port + i, vllm_version=version))
    # Each replica leads its own session, so a helper process that vLLM forks is in the session of its replica.
    own = {r['pid'] for r in replicas}
    extra = sorted(pid for pid in vllm_servers() if pid not in own and _session_of(pid) not in own)
    if extra:
        bad.append('vLLM servers outside the layout: pids %s' % extra)
    if bad:
        raise ServeError('the box does not serve layout %r: %s' % (layout.name, '; '.join(bad)))
    return replicas


def launch(layout: PL.ProbeLayout, cid: str, box: Box, venv: str) -> List[List[int]]:
    """``start`` in a short process of its own, as ``podctl.py serve`` is. Returns the GPU ids of each replica.

    The replicas are then no children of a long process. A replica that exits is reaped, and is not kept as an
    exited child that still looks alive to podctl.
    """
    layout_path = box.p('serve.layout.json')
    Path(layout_path).write_text(json.dumps(layout.as_dict(), sort_keys=True) + '\n')
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *box.args(),
         'start', '--checkpoint', cid, '--layout', layout_path, '--venv', venv],
        capture_output=True, text=True, timeout=1500,
    )
    if proc.returncode != 0:
        raise ServeError('start failed: %s' % (proc.stderr or proc.stdout)[-400:])
    return json.loads(proc.stdout.strip().splitlines()[-1])['gpu_groups']


def serve_one(layout: PL.ProbeLayout, cid: str, box: Box) -> Dict[str, Any]:
    """Start one layout, wait until it answers, verify it."""
    layout.check()
    rt = runtime(layout, box)
    groups = launch(layout, cid, box, rt['venv'])
    serve_s = wait_ready(layout, cid, box, len(groups))
    replicas = verify(layout, cid, box, groups)
    return dict(
        layout=layout.as_dict(), launcher=layout.launcher(), replicas=replicas, gpu_groups=groups, serve_s=serve_s,
        concurrency_per_replica=layout.concurrency(),
        runtime={k: v for k, v in rt.items() if k != 'packages'}, packages=rt['packages'], gpus=gpu_names(),
    )


def serve(plan: PL.ProbePlan, cid: str, box: Box) -> Dict[str, Any]:
    """Serve the first layout of the plan that starts and verifies. The record lists the layouts that did not."""
    failed: List[Dict[str, str]] = []
    for layout in plan.layouts:
        try:
            record = serve_one(layout, cid, box)
        except (ServeError, PL.LayoutError, subprocess.SubprocessError, OSError) as exc:
            failed.append(dict(layout=layout.name, error=str(exc)[:500]))
            try:
                stop(box)
            except (ServeError, subprocess.SubprocessError, OSError) as stop_exc:
                failed.append(dict(layout=layout.name, error='stop: %s' % str(stop_exc)[:300]))
            continue
        record['failed_layouts'] = failed
        return record
    raise ServeError('no layout of the plan serves: %s' % json.dumps(failed))


# ------------------------------------------------------------------ the probe
def run_probe(layout: PL.ProbeLayout, cid: str, plan: Mapping[str, Any], box: Box) -> Tuple[Dict[str, Any], str]:
    """``podctl.py start-run`` and the live stall rules, with the layout's limits. -> (result, 'ok' | 'finalized')."""
    plan = dict(plan, concurrency_per_replica=layout.concurrency())
    name = plan['run_name']
    if os.path.exists(box.p('live_math_manifest_subset.json')):
        raise ServeError('live_math_manifest_subset.json is on the box: the plan must carry the harness')
    box.podctl('start-run', cid, inp=json.dumps(plan).encode())
    t0 = last_t = time.time()
    last: Optional[Tuple[Any, ...]] = None
    status = 'ok'
    while True:
        run = (box.status().get('runs') or {}).get(name, {})
        if run.get('result'):
            break
        progress = run.get('progress') or {}
        seen = (progress.get('generated'), progress.get('samples'), progress.get('done'))
        if seen != last:
            last, last_t = seen, time.time()
        stalled = time.time() - last_t > layout.probe_stall_s
        if stalled or time.time() - t0 > layout.probe_wall_s:
            box.podctl('finalize-run', name, '60', check=False, timeout=180)
            if stalled:
                time.sleep(box.settle_s)
            status = 'finalized'
            break
        time.sleep(box.probe_poll_s)
    result = json.loads(Path(box.p('runs', name, 'result.json')).read_text())
    return result, status


def sampling_config(box: Box) -> Optional[Dict[str, Any]]:
    """The harness and the sampling contract of the probe bundle on this box (the frozen settings of a round).

    None when the bundle has no manifest. The host then has no fingerprint for this probe and gives no decision.
    """
    try:
        doc = json.loads(Path(box.p('live_math_manifest_subset.json')).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get('sampling_contract') is None:
        return None
    return dict(harness=doc.get('harness'), sampling_contract=doc.get('sampling_contract'))


def queue_use(result: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    """How the worker's one queue fed the replicas. An idle replica costs time, never a score, so this is a record."""
    need = len(record['replicas'])
    served = list(result.get('requests_per_replica') or [])
    want = record['concurrency_per_replica'] * need
    return dict(
        task_queue=record['layout']['task_queue'], replicas=need, replicas_fed=sum(1 for n in served if n),
        requests_per_replica=served, in_flight_limit=result.get('concurrency'), in_flight_limit_wanted=want,
        as_planned=(len(served) == need and all(served) and result.get('concurrency') == want),
    )


def _write_json(path: str, doc: Mapping[str, Any]) -> None:
    tmp = '%s.tmp.%d' % (path, os.getpid())
    Path(tmp).write_text(json.dumps(doc, sort_keys=True) + '\n')
    os.replace(tmp, path)


def eval_job(job_dir: str, cid: str, plan_path: str, layouts_path: str, box: Box) -> int:
    """The eval stage of one PIT item: serve, probe, write ``markers/eval.done`` and ``RESULT.json``.

    Same documents as the live detached eval, plus ``probe_result.serve_layout``: the layout that served.
    """
    os.makedirs(os.path.join(job_dir, 'markers'), exist_ok=True)

    def done(status: str, **extra: Any) -> None:
        doc = dict(ok=(status == 'ok'), status=status, candidate_id=cid, at=time.time(), **extra)
        _write_json(os.path.join(job_dir, 'markers', 'eval.done'), doc)
        _write_json(os.path.join(job_dir, 'RESULT.json'), doc)

    try:
        plan = PL.ProbePlan.from_doc(json.loads(Path(layouts_path).read_text()))
        probe_plan = json.loads(Path(plan_path).read_text())
        wait_gpus_free(box)
        record = serve(plan, cid, box)
        result, status = run_probe(PL.ProbeLayout.from_mapping(record['layout']), cid, probe_plan, box)
        result['serve_layout'] = dict({k: v for k, v in record.items() if k != 'packages'}, queue=queue_use(result, record),
                                      sampling=sampling_config(box))
        done(status, probe_result=result, run_name=probe_plan['run_name'])
        return 0
    except Exception as exc:  # the host must always get a marker, never a 2 h timeout
        done('failed', error=('%s: %s' % (type(exc).__name__, exc))[:500])
        return 1


def plan_identity(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """What two probe plans must share for a head-vs-head comparison: tasks, seed, rollouts, cap, harness."""
    core = {k: v for k, v in plan.items() if k not in ('run_name', 'concurrency', 'concurrency_per_replica', 'grade_workers')}
    envs = [dict(env_id=e['env_id'], k=e['k'], max_tokens=e['max_tokens'], tasks=len(e['indices']),
                 indices_sha256=hashlib.sha256(json.dumps(list(e['indices'])).encode()).hexdigest())
            for e in plan['envs']]
    return dict(seed=plan['seed'], checkpoint_id=plan['checkpoint_id'], envs=envs,
                plan_sha256=hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest())


def side(out_path: str, cid: str, plan_path: str, layout: PL.ProbeLayout, box: Box, *,
         tag: Optional[str] = None) -> Dict[str, Any]:
    """One side of a head-vs-head parity run: serve ``cid`` under one layout, run the plan, stop, write the record."""
    probe_plan = json.loads(Path(plan_path).read_text())
    if probe_plan.get('checkpoint_id') != cid:
        raise ServeError('the plan is for checkpoint %s' % str(probe_plan.get('checkpoint_id'))[:12])
    identity = plan_identity(probe_plan)
    probe_plan['run_name'] = '%s-%s' % (probe_plan['run_name'], tag or layout.name)
    if os.path.exists(box.p('runs', probe_plan['run_name'])):
        # The worker resumes from the grades of an earlier run with this name; they may be from another layout.
        raise ServeError('run %s already exists on this box; use another --tag' % probe_plan['run_name'])
    started = time.time()
    stop(box)
    wait_gpus_free(box)
    record = serve_one(layout, cid, box)
    try:
        result, status = run_probe(layout, cid, probe_plan, box)
    finally:
        stop(box)
    doc = dict(
        version=SIDE_VERSION, checkpoint_id=cid, tag=tag or layout.name, status=status, plan=identity,
        serve=dict(record, queue=queue_use(result, record)), result=result,
        started_at=started, finished_at=time.time(),
    )
    _write_json(out_path, doc)
    return doc


# ------------------------------------------------------------------ host side
def ship(write_text: Callable[[str, str], None], job_dir: str, plan: PL.ProbePlan, *,
         read_text: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, str]:
    """Copy this tool and the plan's layouts into the job directory on the box. -> remote paths.

    With ``read_text``, a file that is already on the box with this content is not written again. A host that
    restarts in the middle of an eval then does not rewrite a file that the running eval may start.
    """
    here = Path(__file__).resolve().parent
    job_dir = job_dir.rstrip('/')
    paths = {}
    for name, text in [(n, (here / n).read_text()) for n in TOOL_FILES] + [(LAYOUTS_NAME, PL.dumps(plan))]:
        if len(text.encode()) > MAX_SHIP_BYTES or 'EOF' in text.splitlines():
            raise ServeError('%s cannot go to the box through write_text' % name)
        paths[name] = '%s/%s' % (job_dir, name)
        if read_text is None or (read_text(paths[name]) or '').strip() != text.strip():
            write_text(paths[name], text)
    return paths


def live_preamble(live_command: str) -> str:
    """The clean-up shell of the live detached eval: its command up to the Python driver.

    It ends what a host that died left on the box (old job, GPU processes, old servers) and never ends itself.
    """
    head, found, _ = live_command.partition('python3 - ')
    if not found or 'pgrep' not in head or not head.endswith('\n'):
        raise ServeError('the live eval command has no clean-up preamble')
    return head


def detached_command(*, preamble: str, job_dir: str, candidate_id: str, plan_path: str,
                     root: str = PL.BOX_ROOT, box_args: Sequence[str] = ()) -> str:
    """Shell command of the detached eval: the live clean-up ``preamble``, then ``eval`` of this tool.

    ``box_args`` are more ``Box`` options for the tool (see ``Box.args``); a rented box needs none.
    """
    job_dir = job_dir.rstrip('/')
    words = ['python3', '%s/probe_serve.py' % job_dir, '--root', root, *box_args, 'eval', '--job-dir', job_dir,
             '--candidate', candidate_id, '--plan', plan_path, '--layouts', '%s/%s' % (job_dir, LAYOUTS_NAME)]
    return preamble + ' '.join(shlex.quote(word) for word in words) + '\n'


# ------------------------------------------------------------------ command line
def _box(a: argparse.Namespace) -> Box:
    return Box(root=a.root, python=a.python, port=a.port, poll_s=a.poll_s, probe_poll_s=a.probe_poll_s,
               settle_s=a.settle_s, drain_s=a.drain_s, gpu_wait_s=a.gpu_wait_s, free_mib=a.free_mib)


def _layout_file(path: str) -> PL.ProbeLayout:
    return PL.layout_from(json.loads(Path(path).read_text()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=PL.BOX_ROOT, help='probe directory on the box')
    ap.add_argument('--python', default='python3', help='interpreter that runs the pinned podctl.py')
    ap.add_argument('--port', type=int, default=PL.BASE_PORT, help='port of replica 0; must be the port in podctl.py')
    ap.add_argument('--poll-s', type=float, default=5.0, help='seconds between looks at the servers')
    ap.add_argument('--probe-poll-s', type=float, default=10.0, help='seconds between looks at the probe run')
    ap.add_argument('--settle-s', type=float, default=20.0, help='wait after a stalled run is closed')
    ap.add_argument('--drain-s', type=float, default=60.0, help='longest wait for old servers to exit')
    ap.add_argument('--gpu-wait-s', type=float, default=float(PL.GPU_WAIT_S), help='longest wait for free GPUs')
    ap.add_argument('--free-mib', type=int, default=2000, help='a GPU with less memory in use than this is free')
    sub = ap.add_subparsers(dest='cmd', required=True)
    one = sub.add_parser('start', help='start the replicas of one layout and return')
    one.add_argument('--checkpoint', required=True)
    one.add_argument('--layout', required=True, help='JSON file: a preset name or the settings of one layout')
    one.add_argument('--venv', required=True, help='Python environment that holds the vLLM build')
    sub.add_parser('stop', help='stop every replica')
    ev = sub.add_parser('eval', help='serve and probe one PIT item')
    ev.add_argument('--job-dir', required=True)
    ev.add_argument('--candidate', required=True)
    ev.add_argument('--plan', required=True, help='probe plan from pe_probe_plan.build_probe_plan')
    ev.add_argument('--layouts', required=True, help='probe-layouts.json written by ship()')
    sd = sub.add_parser('side', help='one side of a head-vs-head parity run')
    sd.add_argument('--checkpoint', required=True, help='id of a checkpoint under <root>/ckpt')
    sd.add_argument('--plan', required=True)
    sd.add_argument('--layout', required=True)
    sd.add_argument('--out', required=True, help='where the side record goes')
    sd.add_argument('--tag', default=None, help='name of this side; default is the layout name')
    a = ap.parse_args(argv)
    box = _box(a)
    if a.cmd == 'start':
        groups = start(_layout_file(a.layout), a.checkpoint, box, a.venv)
        print(json.dumps(dict(gpu_groups=groups)))
        return 0
    if a.cmd == 'stop':
        stop(box)
        return 0
    if a.cmd == 'eval':
        return eval_job(a.job_dir, a.candidate, a.plan, a.layouts, box)
    doc = side(a.out, a.checkpoint, a.plan, _layout_file(a.layout), box, tag=a.tag)
    print(json.dumps(dict(out=a.out, status=doc['status'], samples=doc['result'].get('samples'),
                          partial=doc['result'].get('partial'), layout=doc['serve']['layout']['name'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
