"""Fakepass sandbox isolation: /tmp roots, env deny-list, own R2 prefix.

Shared by SIMULATE and REAL fakepass. Inbound live values for deny-listed
keys are always dropped; REAL may then rebind a subset to sandbox paths
(``SANDBOX_REBIND``), each verified to resolve inside the sandbox root.
"""
from __future__ import annotations

import configparser
import enum
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .pe_rent import PREFIX_FAKEPASS


class FakepassError(RuntimeError):
    """Isolation or harness refusal."""


class FakepassPhase(enum.Enum):
    ISOLATE = 'isolate'
    RENT = 'rent'
    DOWNLOAD = 'download'
    TRAIN = 'train'
    EARLY_UPLOAD = 'early_upload'
    HOLD_VLLM = 'hold_vllm'
    FAKE_RESULT = 'fake_result'
    APPLY_TAIL = 'apply_tail'
    BENCH = 'bench'
    SECOND_STEP = 'second_step'
    RELEASE = 'release'
    CLEANUP = 'cleanup'
    DONE = 'done'


FAKE_Z = 9.99
BOX_NAME_PREFIX = PREFIX_FAKEPASS
R2_PREFIX_ROOT = 'pit-fakepass'
R2_PREFIX_TMPL = R2_PREFIX_ROOT + '/%s'

ENV_DENY_EXACT = frozenset({
    'PE_FLAG',
    'PE_HELDOUT',
    'HG_GATE_STATE',
    'PE_GATE_EP',
    'PE_GATE_STATE',
    'PE_LIFE_STATE',
})
ENV_DENY_PREFIXES = (
    'PE_SYNC_EXPECTED_HASH',
    'PIT_SYNC_EXPECTED_HASH',
)

# Deny-listed keys REAL rebinds to sandbox paths (never the inbound live value).
SANDBOX_REBIND = (
    'PE_FLAG',
    'PE_LIFE_STATE',
    'HG_GATE_STATE',
    'PE_SYNC_EXPECTED_HASH',
    'PE_SYNC_EXPECTED_HASH_LOG',
    'PE_SYNC_EXPECTED_HASH_ALERT',
)
# Interpreter path for the sandbox hash-sync hook; not a write target.
SANDBOX_REBIND_EXEMPT = ('PE_SYNC_EXPECTED_HASH_PYTHON',)

SANDBOX_PATH_KEYS = (
    'PE_STATE', 'PE_ST', 'PIT_STATE', 'LIFE_ROOT', 'PE_LIFE', 'PE_HERE', 'PE_JOBS',
    'PIT_UPLOAD_ROOT', 'PIT_FP32_PENDING_INDEX', 'PIT_FP32_MASTER_INDEX',
    'BENCH_RESULTS_DIR', 'AFFINE_BENCH_RESULTS',
) + SANDBOX_REBIND


def _die(msg: str) -> None:
    raise FakepassError(msg)


def _under(path: Path, root: Path) -> bool:
    p = Path(path).resolve()
    r = Path(root).resolve()
    return p == r or str(p).startswith(str(r) + os.sep)


def assert_isolated(*, state_root: Path, live_roots: Sequence[Path]) -> None:
    """Refuse unless state_root is under /tmp and outside every live root."""
    root = state_root.resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    if not str(root).startswith(str(tmp) + os.sep) and root != tmp:
        _die('fakepass state root must be under %s (got %s)' % (tmp, root))
    for live in live_roots:
        if live is None:
            continue
        try:
            live_r = Path(live).resolve()
        except Exception:
            continue
        if str(root).startswith(str(live_r) + os.sep) or root == live_r:
            _die('fakepass state root collides with live root %s' % live_r)
        if str(live_r).startswith(str(root) + os.sep):
            _die('live root %s is inside fakepass state — refuse' % live_r)


def live_roots_from_env(env: Optional[Mapping[str, str]] = None) -> List[Path]:
    src = env if env is not None else os.environ
    out: List[Path] = []
    for key in (
        'PE_STATE', 'PE_ST', 'PIT_STATE', 'LIFE_ROOT', 'PE_LIFE',
        'PE_LIFE_STATE', 'HG_GATE_STATE', 'PE_GATE_EP', 'PE_GATE_STATE',
        'PE_HERE', 'PE_JOBS', 'PIT_UPLOAD_ROOT',
        'PIT_FP32_PENDING_INDEX', 'PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX',
    ):
        v = src.get(key)
        if v:
            out.append(Path(v))
    out.extend([
        Path('/home/const/subnet120'),
        Path('/home/const/subnet120-rewrite'),
        Path.home() / 'subnet120-rewrite',
    ])
    return out


def assert_paths_outside_live(
    paths: Mapping[str, Path],
    live_roots: Sequence[Path],
) -> None:
    """Refuse if any resolved sandbox path sits under a live root."""
    lives: List[Path] = []
    for live in live_roots:
        try:
            lives.append(Path(live).resolve())
        except Exception:
            continue
    for name, path in paths.items():
        try:
            resolved = Path(path).resolve()
        except Exception as exc:
            _die('cannot resolve sandbox path %s: %s' % (name, exc))
        for live_r in lives:
            if str(resolved).startswith(str(live_r) + os.sep) or resolved == live_r:
                _die(
                    'sandbox path %s=%s resolves under live root %s'
                    % (name, resolved, live_r)
                )


def scrub_denied_env(env: Dict[str, str]) -> Dict[str, str]:
    """Drop live gate/flag/sync/heldout keys and refuse if any remain."""
    for key in list(env.keys()):
        if key in ENV_DENY_EXACT:
            env.pop(key, None)
            continue
        if any(key.startswith(p) for p in ENV_DENY_PREFIXES):
            env.pop(key, None)
    leftovers = sorted(
        k for k in env
        if k in ENV_DENY_EXACT or any(k.startswith(p) for p in ENV_DENY_PREFIXES)
    )
    if leftovers:
        _die('env deny-list leak after scrub: %s' % leftovers)
    return env


def assert_env_sandboxed(env: Mapping[str, str], root: Path) -> None:
    """Every deny-listed / state key present must point inside ``root``."""
    for key, value in env.items():
        denied = key in ENV_DENY_EXACT or any(key.startswith(p) for p in ENV_DENY_PREFIXES)
        if not denied and key not in SANDBOX_PATH_KEYS:
            continue
        if key in SANDBOX_REBIND_EXEMPT:
            continue
        if not value or not _under(Path(value), root):
            _die('env %s=%r is not inside the fakepass sandbox %s' % (key, value, root))


def refuse_live_artifacts(state_root: Path) -> None:
    """Document that ledger / PAYOUT / gate are never opened under live roots."""
    marker = state_root / 'FAKEPASS-ISOLATION.json'
    marker.write_text(json.dumps({
        'version': 'pit-fakepass-v2',
        'state_root': str(state_root),
        'r2_prefix': R2_PREFIX_TMPL % state_root.name,
        'box_name_prefix': BOX_NAME_PREFIX,
        'opens_live_ledger': False,
        'opens_live_payout': False,
        'opens_live_gate': False,
        'promote_live': False,
        'at': time.time(),
    }, indent=1, sort_keys=True) + '\n')


SANDBOX_DIRS = {
    'pit': 'pit',
    'pe': 'pe',
    'pe_here': 'pe-here',
    'pe_jobs': 'pe-jobs',
    'life': 'life',
    'life_state': 'life-state',
    'gate': 'gate',
    'uploads': 'uploads',
    'fp32_pending': 'fp32-pending',
    'fp32_public': 'fp32-master',
    'r2_mirror': 'r2-mirror',
    'bench': 'bench',
    'logs': 'logs',
    'box_local': 'box-local',
    'index': 'index',
    'apply': 'APPLY',
    'greens': 'GREENS',
    'head_chain': 'HEAD-CHAIN',
    'hash_sync': 'hash-sync',
    'workers': 'workers',
}


def make_sandbox(
    run_id: str,
    *,
    live_roots: Optional[Sequence[Path]] = None,
    root: Optional[Path] = None,
) -> Dict[str, Path]:
    lives = list(live_roots) if live_roots is not None else live_roots_from_env()
    created = root is None
    if root is None:
        root = Path(tempfile.mkdtemp(prefix='pit-fakepass-%s-' % run_id[:8]))
    else:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
    try:
        assert_isolated(state_root=root, live_roots=lives)
        paths: Dict[str, Path] = {'root': root}
        paths.update({k: root / v for k, v in SANDBOX_DIRS.items()})
        assert_paths_outside_live(paths, lives)
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        refuse_live_artifacts(root)
    except BaseException:
        if created:
            shutil.rmtree(root, ignore_errors=True)
        raise
    return paths


def cleanup_sandbox(paths: Mapping[str, Path]) -> None:
    root = paths.get('root')
    if root is None:
        return
    shutil.rmtree(root, ignore_errors=True)


def env_for_sandbox(paths: Mapping[str, Path], run_id: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.update({
        'PE_STATE': str(paths['pe']),
        'PE_ST': str(paths['pe']),
        'PIT_STATE': str(paths['pit']),
        'LIFE_ROOT': str(paths['life']),
        'PE_LIFE': str(paths['life']),
        'PIT_UPLOAD_ROOT': str(paths['uploads']),
        'PIT_FP32_PENDING_INDEX': str(paths['fp32_pending']),
        'PIT_FP32_MASTER_INDEX': str(paths['fp32_public']),
        'PIT_R2_LOCAL_MIRROR': str(paths['r2_mirror']),
        'PIT_R2_PENDING_PREFIX': R2_PREFIX_TMPL % run_id,
        'PIT_R2_CKPT_PREFIX': '%s/ckpts' % (R2_PREFIX_TMPL % run_id),
        'PIT_R2_FP32_PREFIX': '%s/fp32-master' % (R2_PREFIX_TMPL % run_id),
        'BENCH_RESULTS_DIR': str(paths['bench']),
        'AFFINE_BENCH_RESULTS': str(paths['bench']),
        'PIT_FAKEPASS': '1',
        'PIT_FAKEPASS_BOX_PREFIX': BOX_NAME_PREFIX,
    })
    return scrub_denied_env(env)


def append_timeline(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fh:
        fh.write(json.dumps(dict(row, at=time.time()), sort_keys=True, default=str) + '\n')


def tcp_ping(host: str, port: int, timeout_s: float = 1.0) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    err = None
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            ok = True
    except Exception as exc:
        err = str(exc)[:200]
    return {
        'ok': ok,
        'ms': round((time.time() - t0) * 1000.0, 1),
        'error': err,
        'host': host,
        'port': int(port),
    }


def fake_result_doc(*, candidate_id: str, z: float = FAKE_Z) -> Dict[str, Any]:
    return {
        'FAKE': True,
        'z': float(z),
        'candidate': candidate_id,
        'status': 'ok',
        'note': 'fakepass forced RESULT — must never promote live',
    }


def fake_green_fresh_stamp(*, candidate_id: str, z: float = FAKE_Z) -> Dict[str, Any]:
    """Sandbox-only green_fresh marker — can never promote live HEAD."""
    return {
        'FAKE': True,
        'decision': 'green_fresh',
        'z': float(z),
        'candidate': candidate_id,
        'promote_live': False,
        'sandbox_only': True,
        'note': 'fakepass green_fresh — refuse live promote',
    }


def phase_diagnostics(log_path: Path, phase: str) -> None:
    """Per-phase df/free/nvidia-smi/dmesg/ss snapshot → JSONL."""
    cmds = {
        'df': ['df', '-h'],
        'free': ['free', '-m'],
        'nvidia_smi': ['nvidia-smi'],
        'dmesg_tail': ['bash', '-c', 'dmesg | tail -n 40'],
        'ss': ['ss', '-s'],
        'ip': ['ip', '-s', 'link'],
    }
    snap: Dict[str, Any] = {'phase': phase, 'event': 'diagnostics'}
    for key, argv in cmds.items():
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
            snap[key] = {
                'rc': p.returncode,
                'stdout': (p.stdout or '')[-4000:],
                'stderr': (p.stderr or '')[-1000:],
            }
        except Exception as exc:
            snap[key] = {'error': str(exc)[:200]}
    append_timeline(log_path, snap)


def liveness_loop(
    *,
    log_path: Path,
    host: str,
    port: int,
    box_name: str,
    stop_at: float,
    interval_s: float = 15.0,
    lium_ps_fn: Optional[Callable[[], Any]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Ping the box's forwarded ssh port (never assume 22) until stopped."""
    while time.time() < stop_at:
        if stop_event is not None and stop_event.is_set():
            break
        ping = tcp_ping(host, port)
        lium_ps = None
        if lium_ps_fn is not None:
            try:
                lium_ps = lium_ps_fn()
            except Exception as exc:
                lium_ps = {'error': str(exc)[:200]}
        append_timeline(log_path, {
            'event': 'liveness',
            'box_name': box_name,
            'ping': ping,
            'lium_ps': lium_ps,
            'ssh': {'host': host, 'port': int(port), 'ok': ping.get('ok')},
        })
        remaining = stop_at - time.time()
        if remaining <= 0:
            break
        sleep_s = min(float(interval_s), remaining)
        if stop_event is not None:
            stop_event.wait(timeout=sleep_s)
        else:
            time.sleep(sleep_s)


LIUM_KEY_ENV = ('LIUM_API_API_KEY', 'LIUM_API_KEY')
LIUM_CONFIG_INI = Path('.lium') / 'config.ini'


def _lium_config_ini_source(path: Path, workspace: Optional[str]) -> Optional[str]:
    """Which ``config.ini`` entry holds the API key the Lium CLI would use, or None.

    Mirrors the CLI: a requested workspace means only that workspace's key;
    otherwise the stored default workspace, then ``[api] api_key``.
    """
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(path)
    except (configparser.Error, OSError, UnicodeDecodeError):
        return None

    def saved(name: str) -> Optional[str]:
        section = 'workspace.%s' % name.lower()
        if (parser.get(section, 'api_key', fallback='') or '').strip():
            return 'config.ini:[%s] api_key' % section
        return None

    if workspace:
        return saved(workspace)
    active = (parser.get('workspaces', 'active', fallback='') or '').strip()
    if active and saved(active):
        return saved(active)
    if (parser.get('api', 'api_key', fallback='') or '').strip():
        return 'config.ini:[api] api_key'
    return None


def lium_credential_source(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Non-secret label for the Lium credential in reach, or None. Never returns the key.

    Order is the Lium CLI's own: ``LIUM_WORKSPACE`` pins the key saved for that
    workspace; else ``LIUM_API_API_KEY`` / ``LIUM_API_KEY``; else the standard
    config file ``~/.lium/config.ini``. Older host layouts stay accepted last.
    """
    src = env if env is not None else os.environ
    home = Path(src.get('HOME') or Path.home())
    ini = home / LIUM_CONFIG_INI
    workspace = (src.get('LIUM_WORKSPACE') or '').strip() or None
    if workspace:
        return _lium_config_ini_source(ini, workspace) if ini.is_file() else None
    for key in LIUM_KEY_ENV:
        if src.get(key):
            return 'env:%s' % key
    if ini.is_file():
        found = _lium_config_ini_source(ini, None)
        if found:
            return found
    if src.get('LIUM_TOKEN'):
        return 'env:LIUM_TOKEN'
    for label, candidate in (
        ('legacy:config.json', home / '.lium' / 'config.json'),
        ('legacy:credentials', home / '.lium' / 'credentials'),
        ('env:LIUM_CONFIG', Path(src['LIUM_CONFIG']) if src.get('LIUM_CONFIG') else None),
    ):
        if candidate is not None and candidate.is_file():
            return label
    return None


def has_lium_credentials(env: Optional[Mapping[str, str]] = None) -> bool:
    return lium_credential_source(env) is not None


class SeveredPod:
    """Pod wrapper: after arm(), every Pod.run fails like ssh rc=255."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.severed = False
        self.ssh_attempts = 0

    def arm(self) -> None:
        self.severed = True

    def disarm(self) -> None:
        self.severed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def run(self, cmd: str, *args: Any, **kwargs: Any) -> str:
        if self.severed:
            self.ssh_attempts += 1
            if kwargs.get('check'):
                raise subprocess.CalledProcessError(
                    255, cmd, output='', stderr='ssh: connection refused (sever-after-green)',
                )
            raise FakepassError('ssh rc=255 (sever-after-green)')
        return self._inner.run(cmd, *args, **kwargs)
