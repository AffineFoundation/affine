#!/usr/bin/env python3
"""Forced-GREEN real-box harness — fully isolated from live PIT/PE/LIFE.

Jacob / aae897c1 review G: refuses to start unless isolated under /tmp with its
own R2 prefix, sandbox indexes, and ``affine-pe-pit-fakepass-*`` box names that
the live host ignores.

Never opens the live ledger, PAYOUT, or gate. Never promotes live HEAD.
"""
from __future__ import annotations

import argparse
import enum
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Never, Optional, Sequence, Tuple

from ops.parallel_eval.live.pe_artifact_upload import (
    gc_pending_artifacts,
    promote_pending_master_to_live,
    write_pending_master_index,
)
from ops.parallel_eval.live.pe_box import HostBoxIO

from . import PRICE_CEILING_USD_H
from .box import ScriptedBoxIO
from .gpu_policy import effective_gpu_order
from .pe_rent import (
    PREFIX_FAKEPASS,
    RentError,
    RentService,
    bind_host_rent_service,
    is_fakepass_name,
)


class FakepassError(RuntimeError):
    """Isolation or harness refusal."""


FAKE_Z = 9.99
BOX_NAME_PREFIX = PREFIX_FAKEPASS
R2_PREFIX_TMPL = 'pit-fakepass/%s'
CANDIDATE_FAKE = 'f' * 64
MASTER_FAKE = 'a' * 64
HEAD_FAKE = 'b' * 64

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
)


class FakepassMode(enum.Enum):
    DRY = 'dry'
    SIMULATE = 'simulate'
    REAL = 'real'


class FakepassPhase(enum.Enum):
    ISOLATE = 'isolate'
    RENT = 'rent'
    DOWNLOAD = 'download'
    TRAIN = 'train'
    EARLY_UPLOAD = 'early_upload'
    HOLD_VLLM = 'hold_vllm'
    FAKE_RESULT = 'fake_result'
    APPLY_TAIL = 'apply_tail'
    RELEASE = 'release'
    CLEANUP = 'cleanup'
    DONE = 'done'


def _die(msg: str) -> None:
    raise FakepassError(msg)


def _assert_never(phase: FakepassPhase) -> Never:
    raise FakepassError('unhandled fakepass phase: %s' % phase)


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
    ):
        v = src.get(key)
        if v:
            out.append(Path(v))
    out.extend([
        Path('/home/const/subnet120'),
        Path('/home/const/subnet120-rewrite'),
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


def refuse_live_artifacts(state_root: Path) -> None:
    """Document that ledger / PAYOUT / gate are never opened under live roots."""
    marker = state_root / 'FAKEPASS-ISOLATION.json'
    marker.write_text(json.dumps({
        'version': 'pit-fakepass-v1',
        'state_root': str(state_root),
        'r2_prefix': R2_PREFIX_TMPL % state_root.name,
        'box_name_prefix': BOX_NAME_PREFIX,
        'opens_live_ledger': False,
        'opens_live_payout': False,
        'opens_live_gate': False,
        'promote_live': False,
        'at': time.time(),
    }, indent=1, sort_keys=True) + '\n')


def make_sandbox(
    run_id: str,
    *,
    live_roots: Optional[Sequence[Path]] = None,
    root: Optional[Path] = None,
) -> Dict[str, Path]:
    lives = list(live_roots) if live_roots is not None else live_roots_from_env()
    if root is None:
        root = Path(tempfile.mkdtemp(prefix='pit-fakepass-%s-' % run_id[:8]))
    else:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
    assert_isolated(state_root=root, live_roots=lives)
    paths = {
        'root': root,
        'pit': root / 'pit',
        'pe': root / 'pe',
        'life': root / 'life',
        'uploads': root / 'uploads',
        'fp32_pending': root / 'fp32-pending',
        'fp32_public': root / 'fp32-master',
        'r2_mirror': root / 'r2-mirror',
        'bench': root / 'bench',
        'logs': root / 'logs',
        'box_local': root / 'box-local',
        'index': root / 'index',
        'apply': root / 'APPLY',
        'greens': root / 'GREENS',
        'head_chain': root / 'HEAD-CHAIN',
        'hash_sync': root / 'hash-sync',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    assert_paths_outside_live(paths, lives)
    refuse_live_artifacts(root)
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


def simulate_sever_ssh() -> Dict[str, Any]:
    """Model --sever-after-green: ssh fails rc=255; APPLY must use upload."""
    return {
        'ssh_rc': 255,
        'error': 'ssh: connection refused (sever-after-green)',
        'apply_from_upload': True,
    }


def simulate_restart_host_mid_eval(*, eval_pid: int, host_pid: int) -> Dict[str, Any]:
    """Model --restart-host-mid-eval: kill host by exact PID; eval pid unchanged."""
    if int(eval_pid) == int(host_pid):
        raise FakepassError('eval pid must differ from host pid')
    return {
        'killed_host_pid': int(host_pid),
        'eval_pid_unchanged': int(eval_pid),
        'no_free_gpus': True,
        'no_place_cand': True,
    }


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


def append_timeline(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fh:
        fh.write(json.dumps(dict(row, at=time.time()), sort_keys=True, default=str) + '\n')


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


def has_lium_credentials(env: Optional[Mapping[str, str]] = None) -> bool:
    src = env if env is not None else os.environ
    if src.get('LIUM_API_KEY') or src.get('LIUM_TOKEN'):
        return True
    home = Path(src.get('HOME') or Path.home())
    for candidate in (
        home / '.lium' / 'config.json',
        home / '.lium' / 'credentials',
        Path(src['LIUM_CONFIG']) if src.get('LIUM_CONFIG') else None,
    ):
        if candidate is not None and candidate.is_file():
            return True
    return False


class SeveredPod:
    """Pod wrapper: after arm(), Pod.run / ssh returns rc 255."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.severed = False
        self.ssh_attempts = 0

    def arm(self) -> None:
        self.severed = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def run(self, cmd: str, timeout: int = 60, check: bool = False, **kwargs: Any) -> str:
        if self.severed:
            self.ssh_attempts += 1
            err = subprocess.CalledProcessError(255, cmd, output='', stderr='ssh: connection refused (sever-after-green)')
            if check:
                raise err
            raise FakepassError('ssh rc=255 (sever-after-green)')
        return self._inner.run(cmd, timeout=timeout, check=check, **kwargs)


class ScriptedPod:
    """Minimal Pod stand-in for --simulate-local (no Lium)."""

    def __init__(self, *, box_id: str, name: str, ip: str = '127.0.0.1', box_root: Path):
        self.id = box_id
        self.name = name
        self.ip = ip
        self.box_root = Path(box_root)
        self.box_root.mkdir(parents=True, exist_ok=True)
        self.runs: List[str] = []
        self.free_gpus_calls = 0
        self.place_cand_calls = 0

    def run(self, cmd: str, timeout: int = 60, check: bool = False, **kwargs: Any) -> str:
        self.runs.append(cmd)
        low = cmd.lower()
        if 'free_gpus' in low or 'nvidia-smi' in low and 'kill' in low:
            self.free_gpus_calls += 1
        if 'place_cand' in low:
            self.place_cand_calls += 1
        if 'os.walk' in cmd or 'hashlib' in cmd:
            out: Dict[str, Any] = {}
            for p in self.box_root.rglob('*'):
                if p.is_file():
                    rel = p.relative_to(self.box_root).as_posix()
                    out[rel] = {
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'bytes': p.stat().st_size,
                    }
            return json.dumps(out)
        if 'PUT_OK' in cmd or 'urls' in cmd:
            return 'PUT_OK 1 BAD 0\n'
        if 'job.pid' in cmd:
            pid_file = self.box_root / 'job.pid'
            if pid_file.is_file():
                return pid_file.read_text()
            return ''
        return 'OK\n'


@dataclass
class FakepassRun:
    """Owns sandbox state and walks the forced-GREEN phase machine."""

    run_id: str
    mode: FakepassMode
    paths: Dict[str, Path]
    env: Dict[str, str]
    live_roots: List[Path]
    hold_vllm_s: int = 60
    sever_after_green: bool = False
    restart_host_mid_eval: bool = False
    candidate_id: str = CANDIDATE_FAKE
    master_digest: str = MASTER_FAKE
    head: str = HEAD_FAKE
    host_pid: Optional[int] = None
    eval_pid: Optional[int] = None
    liveness_interval_s: float = 15.0
    phase: FakepassPhase = FakepassPhase.ISOLATE
    rent: Optional[RentService] = None
    box_io: Any = None
    pod: Any = None
    box_id: Optional[str] = None
    box_name: Optional[str] = None
    train_receipt: Dict[str, Any] = field(default_factory=dict)
    early_upload: Dict[str, Any] = field(default_factory=dict)
    fake_result: Dict[str, Any] = field(default_factory=dict)
    green_stamp: Dict[str, Any] = field(default_factory=dict)
    apply_result: Dict[str, Any] = field(default_factory=dict)
    restart_result: Dict[str, Any] = field(default_factory=dict)
    called_free_gpus: bool = False
    called_place_cand: bool = False
    _liveness_stop: Optional[threading.Event] = field(default=None, repr=False)
    _liveness_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _host_proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _eval_proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    cleaned: bool = False

    @property
    def log_path(self) -> Path:
        return self.paths['logs'] / 'timeline.jsonl'

    def _note(self, event: str, **extra: Any) -> None:
        append_timeline(self.log_path, {'event': event, 'phase': self.phase.value, **extra})

    def _diag(self) -> None:
        phase_diagnostics(self.log_path, self.phase.value)

    def _advance(self, nxt: FakepassPhase) -> None:
        self.phase = nxt
        self._diag()

    def build_rent(self) -> RentService:
        order = effective_gpu_order(allow_sharded_gpus=False)
        svc = RentService(
            max_boxes=1,
            max_eval_nodes=99,
            gpu_order=order,
            price_ceiling_usd_h=float(PRICE_CEILING_USD_H),
            disable_auto_backup=True,
            require_host=self.mode is FakepassMode.REAL,
            shadow=True,
            name_prefix=BOX_NAME_PREFIX,
            share_pe_cap=False,
            ledger_path=str(self.paths['pe'] / 'fakepass-rent-ledger.json'),
        )
        if self.mode is FakepassMode.REAL:
            bind_host_rent_service(svc)
            if svc.host_rent is None:
                raise FakepassError('real mode requires host pe_rent credentials')
        return svc

    def build_box_io(self) -> Any:
        if self.mode is FakepassMode.SIMULATE:
            head_path = self.paths['box_local'] / 'head'
            master_path = self.paths['box_local'] / 'master'
            cand_path = self.paths['box_local'] / 'cand'
            cand_master = self.paths['box_local'] / 'cand-master'
            for p in (head_path, master_path, cand_path):
                p.mkdir(parents=True, exist_ok=True)
            cand_master.mkdir(parents=True, exist_ok=True)
            shard = cand_master / 'master-00001-of-00001.safetensors'
            if not shard.is_file():
                shard.write_bytes(b'FAKEPASS-WEIGHTS')
            (cand_path / 'model.safetensors').write_bytes(b'FAKE-BF16')
            return ScriptedBoxIO(
                head_path=str(head_path),
                master_path=str(master_path),
                train_result={
                    'candidate_id': self.candidate_id,
                    'candidate_path': str(cand_path),
                    'candidate_master_path': str(cand_master),
                    'candidate_master_digest': self.master_digest,
                    'candidate_sha256': self.candidate_id,
                    'candidate_files': {
                        'model.safetensors': hashlib.sha256(b'FAKE-BF16').hexdigest(),
                    },
                    'box_master_dir': str(cand_master),
                    'box_candidate_dir': str(cand_path),
                    'used_fp32_bootstrap_override': False,
                    'train_gpu': 'B300',
                    'training_report': {
                        'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
                        'batching': {'tasks': 125, 'rollouts': 1000},
                        'tokens': {
                            'prompt_tokens': 100,
                            'output_tokens': 200,
                            'loss_tokens': 150,
                        },
                        'update_size': 1.0,
                        'gradient_norm': 1.0,
                        'engine': 'packed',
                    },
                },
            )
        if self.mode is FakepassMode.REAL:
            return HostBoxIO()
        raise FakepassError('box_io not available in dry mode')

    def start_liveness(self, *, host: str, port: int, box_name: str) -> None:
        self._liveness_stop = threading.Event()
        stop_at = time.time() + max(1.0, float(self.hold_vllm_s) + 30.0)

        def _lium_ps() -> Any:
            if self.rent is not None and self.rent.list_pods is not None:
                return list(self.rent.list_pods() or [])
            return {'synthetic': True, 'box_name': box_name}

        self._liveness_thread = threading.Thread(
            target=liveness_loop,
            kwargs=dict(
                log_path=self.log_path,
                host=host,
                port=port,
                box_name=box_name,
                stop_at=stop_at,
                interval_s=self.liveness_interval_s,
                lium_ps_fn=_lium_ps,
                stop_event=self._liveness_stop,
            ),
            name='fakepass-liveness',
            daemon=True,
        )
        self._liveness_thread.start()

    def stop_liveness(self) -> None:
        if self._liveness_stop is not None:
            self._liveness_stop.set()
        if self._liveness_thread is not None:
            self._liveness_thread.join(timeout=2.0)

    def phase_isolate(self) -> None:
        self._advance(FakepassPhase.ISOLATE)
        assert_isolated(state_root=self.paths['root'], live_roots=self.live_roots)
        assert_paths_outside_live(self.paths, self.live_roots)
        scrub_denied_env(self.env)
        if not is_fakepass_name(BOX_NAME_PREFIX + 'x'):
            _die('BOX_NAME_PREFIX must be recognized by is_fakepass_name')
        self._note('sandbox_ready', run_id=self.run_id, root=str(self.paths['root']))

    def phase_rent(self) -> None:
        self._advance(FakepassPhase.RENT)
        self.rent = self.build_rent()
        try:
            out = self.rent.rent(item_id='fakepass-%s' % self.run_id, epoch=0, attempt=0)
        except RentError as exc:
            raise FakepassError('rent failed: %s' % exc) from exc
        self.box_id = str(out.get('box_id') or out.get('id'))
        self.box_name = str(out.get('name') or '')
        if not is_fakepass_name(self.box_name):
            raise FakepassError('rented name must be fakepass prefix: %s' % self.box_name)
        if self.mode is FakepassMode.SIMULATE:
            pod = ScriptedPod(
                box_id=self.box_id,
                name=self.box_name,
                box_root=self.paths['box_local'] / 'remote',
            )
            # Seed master tree for inventory/upload path.
            master = pod.box_root / 'master'
            master.mkdir(parents=True, exist_ok=True)
            (master / 'master-00001-of-00001.safetensors').write_bytes(b'FAKEPASS-WEIGHTS')
            self.pod = SeveredPod(pod) if self.sever_after_green else pod
        else:
            pod_obj = out.get('pod') if isinstance(out, Mapping) else None
            if pod_obj is None or not hasattr(pod_obj, 'run'):
                pod_obj = out
            self.pod = SeveredPod(pod_obj) if (
                self.sever_after_green and hasattr(pod_obj, 'run')
            ) else pod_obj
        self._note('rented', box_id=self.box_id, box_name=self.box_name, gpu=out.get('gpu'))

    def phase_download_train(self) -> None:
        self._advance(FakepassPhase.DOWNLOAD)
        self.box_io = self.build_box_io()
        head_handle = self.box_io.download_head(self.head)
        self._note('download_head', head=self.head, handle=head_handle)
        self._advance(FakepassPhase.TRAIN)
        receipt = dict(self.box_io.train_v834e(
            pod={'id': self.box_id, 'name': self.box_name} if self.mode is FakepassMode.SIMULATE else self.pod,
            expected_master_digest=self.master_digest,
            batch_handle='public-old-batch://fakepass',
            head_path=head_handle,
            item_id='fakepass-%s' % self.run_id,
            epoch=0,
            train_gpu='B300',
        ))
        self.candidate_id = str(receipt.get('candidate_id') or self.candidate_id)
        self.train_receipt = receipt
        self._note('train_done', candidate_id=self.candidate_id)

    def phase_early_upload(self) -> None:
        self._advance(FakepassPhase.EARLY_UPLOAD)
        os.environ.update({
            k: self.env[k] for k in (
                'PIT_R2_PENDING_PREFIX', 'PIT_R2_LOCAL_MIRROR',
                'PIT_FP32_PENDING_INDEX', 'PIT_FP32_MASTER_INDEX',
                'PIT_UPLOAD_ROOT', 'PIT_R2_FP32_PREFIX',
            ) if k in self.env
        })
        cand = self.candidate_id
        master_dir = Path(
            self.train_receipt.get('box_master_dir')
            or self.train_receipt.get('candidate_master_path')
            or (self.paths['box_local'] / 'cand-master')
        )
        master_dir.mkdir(parents=True, exist_ok=True)
        shard = master_dir / 'master-00001-of-00001.safetensors'
        if not shard.is_file():
            shard.write_bytes(b'FAKEPASS-WEIGHTS')
        inv = {
            'master-00001-of-00001.safetensors': {
                'sha256': hashlib.sha256(shard.read_bytes()).hexdigest(),
                'bytes': shard.stat().st_size,
            },
        }
        prefix = R2_PREFIX_TMPL % self.run_id
        mirror_dest = self.paths['r2_mirror'] / prefix / cand
        mirror_dest.mkdir(parents=True, exist_ok=True)
        for name in inv:
            dest = mirror_dest / name
            dest.write_bytes((master_dir / name).read_bytes())
        idx = write_pending_master_index(
            candidate_id=cand,
            bf16=cand,
            digest=self.master_digest,
            inv=inv,
        )
        up = self.paths['uploads'] / cand
        up.mkdir(parents=True, exist_ok=True)
        (up / 'RESULT.json').write_text(json.dumps({
            'FAKE': True,
            'candidate': cand,
            'pending_prefix': prefix,
        }, indent=1, sort_keys=True) + '\n')
        self.early_upload = {
            'ok': True,
            'pending': True,
            'index': str(idx),
            'pending_prefix': prefix,
            'candidate_id': cand,
            'PIT_R2_PENDING_PREFIX': self.env.get('PIT_R2_PENDING_PREFIX'),
        }
        self._note('early_upload', **self.early_upload)

    def phase_hold_vllm(self) -> None:
        self._advance(FakepassPhase.HOLD_VLLM)
        host = getattr(self.pod, 'ip', None) or '127.0.0.1'
        self.start_liveness(host=str(host), port=22, box_name=str(self.box_name or 'unknown'))
        if self.restart_host_mid_eval:
            self._do_restart_host_mid_eval()
        hold = max(0, int(self.hold_vllm_s))
        self._note('vllm_hold_start', seconds=hold)
        if hold > 0:
            time.sleep(hold)
        self._note('vllm_hold_end', seconds=hold)

    def _spawn_marker_proc(self, label: str) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(3600)'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _do_restart_host_mid_eval(self) -> None:
        # Eval process (job.pid) must survive; only the sandbox host PID is killed.
        if self.eval_pid is None:
            self._eval_proc = self._spawn_marker_proc('eval')
            self.eval_pid = int(self._eval_proc.pid)
        if self.host_pid is None:
            self._host_proc = self._spawn_marker_proc('host')
            self.host_pid = int(self._host_proc.pid)
        eval_before = int(self.eval_pid)
        host_before = int(self.host_pid)
        if eval_before == host_before:
            raise FakepassError('eval pid must differ from host pid')
        # Persist job.pid on the simulated box.
        if isinstance(getattr(self.pod, '_inner', self.pod), ScriptedPod):
            inner = getattr(self.pod, '_inner', self.pod)
            (inner.box_root / 'job.pid').write_text('%d\n' % eval_before)
        os.kill(host_before, signal.SIGKILL)
        try:
            os.waitpid(host_before, 0)
        except ChildProcessError:
            pass
        # Restart host marker; eval must be unchanged and still alive.
        self._host_proc = self._spawn_marker_proc('host-restarted')
        self.host_pid = int(self._host_proc.pid)
        try:
            os.kill(eval_before, 0)
            eval_alive = True
        except OSError:
            eval_alive = False
        if not eval_alive:
            raise FakepassError('eval job.pid died during host restart')
        # Invariant: harness must not free_gpus / place_cand on mid-eval restart.
        if self.called_free_gpus or self.called_place_cand:
            raise FakepassError('restart-host-mid-eval must not call free_gpus/place_cand')
        if isinstance(getattr(self.pod, '_inner', self.pod), ScriptedPod):
            inner = getattr(self.pod, '_inner', self.pod)
            if inner.free_gpus_calls or inner.place_cand_calls:
                raise FakepassError('pod recorded free_gpus/place_cand during restart')
        self.restart_result = simulate_restart_host_mid_eval(
            eval_pid=eval_before, host_pid=host_before,
        )
        self.restart_result['new_host_pid'] = self.host_pid
        self.restart_result['eval_alive'] = True
        self._note('restart_host_mid_eval', **self.restart_result)

    def phase_fake_result(self) -> None:
        self._advance(FakepassPhase.FAKE_RESULT)
        self.fake_result = fake_result_doc(candidate_id=self.candidate_id, z=FAKE_Z)
        self.green_stamp = fake_green_fresh_stamp(candidate_id=self.candidate_id, z=FAKE_Z)
        if self.green_stamp.get('promote_live') is not False:
            raise FakepassError('fake green_fresh must set promote_live=False')
        result_path = self.paths['pit'] / 'RESULT.json'
        result_path.write_text(json.dumps(self.fake_result, indent=1, sort_keys=True) + '\n')
        stamp_path = self.paths['greens'] / ('%s.json' % self.candidate_id)
        stamp_path.write_text(json.dumps(self.green_stamp, indent=1, sort_keys=True) + '\n')
        self._note('fake_result', result=self.fake_result, stamp=self.green_stamp)
        if self.sever_after_green:
            if isinstance(self.pod, SeveredPod):
                self.pod.arm()
            sever = simulate_sever_ssh()
            self._note('sever_after_green', **sever)

    def phase_apply_tail(self) -> None:
        self._advance(FakepassPhase.APPLY_TAIL)
        cand = self.candidate_id
        pending_idx = self.paths['fp32_pending'] / ('%s.json' % cand)
        if not pending_idx.is_file():
            raise FakepassError('APPLY refused: pending master index missing (upload required)')
        pending_doc = json.loads(pending_idx.read_text())
        if str(pending_doc.get('master_digest')) != self.master_digest:
            raise FakepassError('APPLY verify master digest mismatch')
        public_idx = promote_pending_master_to_live(candidate_id=cand, digest=self.master_digest)
        apply_doc = {
            'FAKE': True,
            'state': 'ready',
            'candidate': cand,
            'promote_live': False,
            'apply_from_upload': True,
            'sever_after_green': bool(self.sever_after_green),
            'ssh_required': False,
            'public_index': str(public_idx),
        }
        (self.paths['apply'] / ('%s.json' % cand)).write_text(
            json.dumps(apply_doc, indent=1, sort_keys=True) + '\n'
        )
        greens = {
            'FAKE': True,
            'promote_live': False,
            'entries': [self.green_stamp],
        }
        (self.paths['greens'] / 'GREENS.json').write_text(
            json.dumps(greens, indent=1, sort_keys=True) + '\n'
        )
        chain = {
            'FAKE': True,
            'promote_live': False,
            'applied': [{
                'new_head': cand,
                'prev_head': self.head,
                'sandbox_only': True,
                'at': time.time(),
            }],
        }
        (self.paths['head_chain'] / 'HEAD-CHAIN.json').write_text(
            json.dumps(chain, indent=1, sort_keys=True) + '\n'
        )
        (self.paths['pit'] / 'HEAD-CHAIN.json').write_text(
            json.dumps(chain, indent=1, sort_keys=True) + '\n'
        )
        hash_sync = {
            'FAKE': True,
            'ok': True,
            'candidate': cand,
            'expected': self.master_digest,
            'promote_live': False,
        }
        (self.paths['hash_sync'] / 'sync.json').write_text(
            json.dumps(hash_sync, indent=1, sort_keys=True) + '\n'
        )
        bench = {
            'FAKE': True,
            'minimal': True,
            'candidate': cand,
            'promote_live': False,
            'profile': 'fakepass-minimal',
        }
        (self.paths['bench'] / 'RESULT.json').write_text(
            json.dumps(bench, indent=1, sort_keys=True) + '\n'
        )
        if self.sever_after_green and isinstance(self.pod, SeveredPod):
            if self.pod.ssh_attempts:
                apply_doc['ssh_attempts_ignored'] = self.pod.ssh_attempts
            apply_doc['apply_from_upload'] = True
        self.apply_result = apply_doc
        self._note('apply_tail', **apply_doc)
        gc = gc_pending_artifacts(cand)
        self._note('r2_pending_gc', **gc)

    def phase_release(self) -> None:
        self._advance(FakepassPhase.RELEASE)
        if self.rent is not None and self.box_id:
            try:
                self.rent.release(self.box_id, reason='fakepass-done')
            except RentError as exc:
                self._note('release_error', error=str(exc)[:200])
        self._note('released', box_id=self.box_id)

    def phase_cleanup(self) -> None:
        self._advance(FakepassPhase.CLEANUP)
        self.stop_liveness()
        for proc in (self._host_proc, self._eval_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        cleanup_sandbox(self.paths)
        self.cleaned = True
        self.phase = FakepassPhase.DONE

    def _dispatch_phase(self, phase: FakepassPhase) -> None:
        if phase is FakepassPhase.ISOLATE:
            self.phase_isolate()
            return
        if phase is FakepassPhase.RENT:
            self.phase_rent()
            return
        if phase is FakepassPhase.DOWNLOAD:
            self.phase_download_train()
            return
        if phase is FakepassPhase.TRAIN:
            # download+train are paired; TRAIN alone is a no-op marker.
            return
        if phase is FakepassPhase.EARLY_UPLOAD:
            self.phase_early_upload()
            return
        if phase is FakepassPhase.HOLD_VLLM:
            self.phase_hold_vllm()
            return
        if phase is FakepassPhase.FAKE_RESULT:
            self.phase_fake_result()
            return
        if phase is FakepassPhase.APPLY_TAIL:
            self.phase_apply_tail()
            return
        if phase is FakepassPhase.RELEASE:
            self.phase_release()
            return
        if phase is FakepassPhase.CLEANUP:
            self.phase_cleanup()
            return
        if phase is FakepassPhase.DONE:
            return
        _assert_never(phase)

    def run(self) -> Dict[str, Any]:
        try:
            self._dispatch_phase(FakepassPhase.ISOLATE)
            if self.mode is FakepassMode.DRY:
                return {
                    'ok': True,
                    'dry_isolation_check': True,
                    'run_id': self.run_id,
                    'root': str(self.paths['root']),
                    'env_keys': sorted(
                        k for k in self.env if k.startswith(('PIT_', 'PE_', 'LIFE', 'BENCH'))
                    ),
                    'denied_absent': sorted(ENV_DENY_EXACT),
                }
            for phase in (
                FakepassPhase.RENT,
                FakepassPhase.DOWNLOAD,
                FakepassPhase.EARLY_UPLOAD,
                FakepassPhase.HOLD_VLLM,
                FakepassPhase.FAKE_RESULT,
                FakepassPhase.APPLY_TAIL,
                FakepassPhase.RELEASE,
            ):
                self._dispatch_phase(phase)
            out = {
                'ok': True,
                'run_id': self.run_id,
                'mode': self.mode.value,
                'box_id': self.box_id,
                'box_name': self.box_name,
                'candidate_id': self.candidate_id,
                'fake_result': self.fake_result,
                'green_stamp': self.green_stamp,
                'early_upload': self.early_upload,
                'apply_result': self.apply_result,
                'restart_result': self.restart_result or None,
                'sever_after_green': bool(self.sever_after_green),
                'promote_live': False,
                'root': str(self.paths['root']),
            }
            (self.paths['root'] / 'PLAN.json').write_text(
                json.dumps(out, indent=1, sort_keys=True, default=str) + '\n'
            )
            return out
        finally:
            # Always tear down /tmp sandbox (tests assert cleaned).
            if not self.cleaned:
                try:
                    self._dispatch_phase(FakepassPhase.CLEANUP)
                except Exception:
                    cleanup_sandbox(self.paths)
                    self.cleaned = True


def build_run_from_args(args: argparse.Namespace) -> Tuple[FakepassRun, List[Path]]:
    run_id = args.run_id or time.strftime('%Y%m%dT%H%M%S')
    # Capture live roots from the *pre-sandbox* environment.
    live_roots = live_roots_from_env()
    if args.dry_isolation_check:
        mode = FakepassMode.DRY
    elif args.real:
        mode = FakepassMode.REAL
    elif args.simulate_local:
        mode = FakepassMode.SIMULATE
    else:
        # Default: dry isolation when credentials missing; else require explicit mode.
        if has_lium_credentials():
            _die('refusing to start: pass --simulate-local, --real, or --dry-isolation-check')
        mode = FakepassMode.DRY

    if mode is FakepassMode.REAL and not has_lium_credentials():
        _die('real mode refused: Lium credentials missing (use --dry-isolation-check or --simulate-local)')

    paths = make_sandbox(run_id, live_roots=live_roots)
    env = env_for_sandbox(paths, run_id)
    return FakepassRun(
        run_id=run_id,
        mode=mode,
        paths=paths,
        env=env,
        live_roots=live_roots,
        hold_vllm_s=int(args.hold_vllm_s),
        sever_after_green=bool(args.sever_after_green),
        restart_host_mid_eval=bool(args.restart_host_mid_eval),
        candidate_id=str(args.candidate_id or CANDIDATE_FAKE),
        host_pid=int(args.host_pid) if args.host_pid else None,
        eval_pid=int(args.eval_pid) if args.eval_pid else None,
        liveness_interval_s=float(args.liveness_interval_s),
    ), live_roots


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ops.parallel_eval.pit.fakepass')
    p.add_argument('--run-id', default=None, help='sandbox id (default: timestamp)')
    p.add_argument('--dry-isolation-check', action='store_true')
    p.add_argument('--simulate-local', action='store_true',
                   help='ScriptedBoxIO + synthetic rent (no Lium)')
    p.add_argument('--real', action='store_true',
                   help='real RentService/HostBoxIO path (requires credentials)')
    p.add_argument('--sever-after-green', action='store_true',
                   help='ssh to box fails rc=255; APPLY must succeed from upload')
    p.add_argument('--restart-host-mid-eval', action='store_true',
                   help='kill sandbox host by exact PID and restart; eval pid unchanged')
    p.add_argument('--hold-vllm-s', type=int, default=60)
    p.add_argument('--liveness-interval-s', type=float, default=15.0)
    p.add_argument('--candidate-id', default=None)
    p.add_argument('--host-pid', type=int, default=None)
    p.add_argument('--eval-pid', type=int, default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        run, _lives = build_run_from_args(args)
    except FakepassError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}), file=sys.stderr)
        return 2

    try:
        out = run.run()
    except FakepassError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)[:400]}), file=sys.stderr)
        return 1

    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
