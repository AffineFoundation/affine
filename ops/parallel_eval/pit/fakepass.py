#!/usr/bin/env python3
"""Forced-GREEN harness — fully isolated from live PIT/PE/LIFE.

Jacob / aae897c1 review G: refuses to start unless isolated under /tmp with its
own R2 prefix, sandbox indexes, and ``affine-pe-pit-fakepass-*`` box names that
the live host ignores.

Modes:

* ``--dry-isolation-check``: build and verify the sandbox, nothing else.
* ``--simulate-local``: hermetic phase machine (ScriptedBoxIO, local R2 mirror)
  for unit tests.
* ``--real``: the real train / upload / serve / apply path on a rented box; only
  the eval verdict is fake (see ``fakepass_real``).

Never opens the live ledger, PAYOUT, or gate. Never promotes live HEAD.
"""
from __future__ import annotations

import argparse
import enum
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Never, Optional, Sequence, Tuple, Union

from ops.parallel_eval.live.pe_artifact_upload import (
    gc_pending_artifacts,
    live_fp32_r2_prefix,
    pending_r2_prefix,
    promote_pending_master_to_live,
    public_master_index_dir,
    write_pending_master_index,
)

from . import PRICE_CEILING_USD_H
from . import fakepass_real as REAL
from .box import ScriptedBoxIO
from .fakepass_sandbox import (
    BOX_NAME_PREFIX,
    ENV_DENY_EXACT,
    ENV_DENY_PREFIXES,
    FAKE_Z,
    R2_PREFIX_TMPL,
    FakepassError,
    FakepassPhase,
    SeveredPod,
    append_timeline,
    assert_isolated,
    assert_paths_outside_live,
    cleanup_sandbox,
    env_for_sandbox,
    fake_green_fresh_stamp,
    fake_result_doc,
    has_lium_credentials,
    live_roots_from_env,
    liveness_loop,
    make_sandbox,
    phase_diagnostics,
    refuse_live_artifacts,
    scrub_denied_env,
    tcp_ping,
)
from .gpu_policy import effective_gpu_order
from .pe_rent import RentError, RentService, is_fakepass_name

__all__ = [
    'BOX_NAME_PREFIX', 'ENV_DENY_EXACT', 'ENV_DENY_PREFIXES', 'FAKE_Z', 'R2_PREFIX_TMPL',
    'FakepassError', 'FakepassMode', 'FakepassPhase', 'FakepassRun', 'SeveredPod', 'ScriptedPod',
    'append_timeline', 'assert_isolated', 'assert_paths_outside_live', 'cleanup_sandbox',
    'env_for_sandbox', 'fake_green_fresh_stamp', 'fake_result_doc', 'has_lium_credentials',
    'live_roots_from_env', 'liveness_loop', 'make_sandbox', 'phase_diagnostics',
    'refuse_live_artifacts', 'scrub_denied_env', 'simulate_restart_host_mid_eval',
    'simulate_sever_ssh', 'tcp_ping', 'build_run_from_args', 'main',
]

CANDIDATE_FAKE = 'f' * 64
MASTER_FAKE = 'a' * 64
HEAD_FAKE = 'b' * 64
SIM_FP32_VERSION = 'v833f-fp32-head-master-v1'
RUN_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
SIM_ENV_KEYS = (
    'PIT_R2_PENDING_PREFIX', 'PIT_R2_LOCAL_MIRROR',
    'PIT_FP32_PENDING_INDEX', 'PIT_FP32_MASTER_INDEX',
    'PIT_UPLOAD_ROOT', 'PIT_R2_FP32_PREFIX',
)


class FakepassMode(enum.Enum):
    DRY = 'dry'
    SIMULATE = 'simulate'
    REAL = 'real'


def _die(msg: str) -> Never:
    raise FakepassError(msg)


def _assert_never(phase: FakepassPhase) -> Never:
    raise FakepassError('unhandled fakepass phase: %s' % phase)


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


class ScriptedPod:
    """Minimal Pod stand-in for --simulate-local (no Lium)."""

    def __init__(self, *, box_id: str, name: str, ip: str = '127.0.0.1', box_root: Path):
        self.id = box_id
        self.name = name
        self.ip = ip
        self.port = 22
        self.box_root = Path(box_root)
        self.box_root.mkdir(parents=True, exist_ok=True)
        self.runs: List[str] = []
        self.free_gpus_calls = 0
        self.place_cand_calls = 0

    def run(self, cmd: str, timeout: int = 60, check: bool = False, **kwargs: Any) -> str:
        self.runs.append(cmd)
        tag = REAL.classify_box_command(cmd)
        if tag == 'free_gpus':
            self.free_gpus_calls += 1
        if tag == 'place_cand':
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
    """SIMULATE / DRY phase machine. REAL runs use ``fakepass_real.RealFakepass``."""

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
    bench: Dict[str, Any] = field(default_factory=dict)
    second_step: Dict[str, Any] = field(default_factory=dict)
    restart_result: Dict[str, Any] = field(default_factory=dict)
    called_free_gpus: bool = False
    called_place_cand: bool = False
    _liveness_stop: Optional[threading.Event] = field(default=None, repr=False)
    _liveness_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _host_proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _eval_proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _env_saved: Dict[str, Optional[str]] = field(default_factory=dict, repr=False)
    cleaned: bool = False

    def __post_init__(self) -> None:
        if self.mode is FakepassMode.REAL:
            _die('FakepassRun is SIMULATE/DRY only; REAL uses fakepass_real.RealFakepass')

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
        return RentService(
            max_boxes=1,
            max_eval_nodes=99,
            gpu_order=effective_gpu_order(allow_sharded_gpus=False),
            price_ceiling_usd_h=float(PRICE_CEILING_USD_H),
            disable_auto_backup=True,
            require_host=False,
            shadow=True,
            name_prefix=BOX_NAME_PREFIX,
            share_pe_cap=False,
            ledger_path=str(self.paths['pe'] / 'fakepass-rent-ledger.json'),
        )

    def _sim_master_tree(self, master_dir: Path) -> Dict[str, Dict[str, Any]]:
        master_dir.mkdir(parents=True, exist_ok=True)
        shard = master_dir / 'master-00001-of-00001.safetensors'
        if not shard.is_file():
            shard.write_bytes(b'FAKEPASS-WEIGHTS')
        (master_dir / 'meta.json').write_text(json.dumps({
            'version': SIM_FP32_VERSION,
            'bf16_sha256': self.candidate_id,
            'master_digest': self.master_digest,
            'tensor_file': {'model.fake': shard.name},
        }, sort_keys=True))
        return {
            p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
            for p in (shard, master_dir / 'meta.json')
        }

    def build_box_io(self) -> Any:
        if self.mode is not FakepassMode.SIMULATE:
            _die('box_io not available in %s mode' % self.mode.value)
        head_path = self.paths['box_local'] / 'head'
        master_path = self.paths['box_local'] / 'master'
        cand_path = self.paths['box_local'] / 'cand'
        cand_master = self.paths['box_local'] / 'cand-master'
        for p in (head_path, master_path, cand_path):
            p.mkdir(parents=True, exist_ok=True)
        self._sim_master_tree(cand_master)
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

    def start_liveness(self, *, host: str, port: int, box_name: str) -> None:
        self._liveness_stop = threading.Event()
        stop_at = time.time() + max(1.0, float(self.hold_vllm_s) + 30.0)

        def _lium_ps() -> Any:
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
        pod = ScriptedPod(
            box_id=self.box_id,
            name=self.box_name,
            box_root=self.paths['box_local'] / 'remote',
        )
        self._sim_master_tree(pod.box_root / 'master')
        self.pod = SeveredPod(pod) if self.sever_after_green else pod
        self._note('rented', box_id=self.box_id, box_name=self.box_name, gpu=out.get('gpu'))

    def phase_download_train(self) -> None:
        self._advance(FakepassPhase.DOWNLOAD)
        self.box_io = self.build_box_io()
        head_handle = self.box_io.download_head(self.head)
        self._note('download_head', head=self.head, handle=head_handle)
        self._advance(FakepassPhase.TRAIN)
        receipt = dict(self.box_io.train_v834e(
            pod={'id': self.box_id, 'name': self.box_name},
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
        keys = [k for k in SIM_ENV_KEYS if k in self.env]
        self._env_saved.update({k: os.environ.get(k) for k in keys if k not in self._env_saved})
        os.environ.update({k: self.env[k] for k in keys})
        cand = self.candidate_id
        master_dir = Path(
            self.train_receipt.get('box_master_dir')
            or self.train_receipt.get('candidate_master_path')
            or (self.paths['box_local'] / 'cand-master')
        )
        inv = self._sim_master_tree(master_dir)
        mirror_dest = self.paths['r2_mirror'] / pending_r2_prefix(cand)
        mirror_dest.mkdir(parents=True, exist_ok=True)
        for name in inv:
            (mirror_dest / name).write_bytes((master_dir / name).read_bytes())
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
            'pending_prefix': pending_r2_prefix(cand),
        }, indent=1, sort_keys=True) + '\n')
        self.early_upload = {
            'ok': True,
            'pending': True,
            'index': str(idx),
            'pending_prefix': pending_r2_prefix(cand),
            'files': sorted(inv),
            'candidate_id': cand,
            'PIT_R2_PENDING_PREFIX': self.env.get('PIT_R2_PENDING_PREFIX'),
        }
        self._note('early_upload', **self.early_upload)

    def phase_hold_vllm(self) -> None:
        self._advance(FakepassPhase.HOLD_VLLM)
        host = getattr(self.pod, 'ip', None) or '127.0.0.1'
        port = int(getattr(self.pod, 'port', 0) or 22)
        self.start_liveness(host=str(host), port=port, box_name=str(self.box_name or 'unknown'))
        if self.restart_host_mid_eval:
            self._do_restart_host_mid_eval()
        hold = max(0, int(self.hold_vllm_s))
        self._note('vllm_hold_start', seconds=hold)
        if hold > 0:
            time.sleep(hold)
        self._note('vllm_hold_end', seconds=hold)

    def _spawn_marker_proc(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(3600)'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _do_restart_host_mid_eval(self) -> None:
        if self.eval_pid is None:
            self._eval_proc = self._spawn_marker_proc()
            self.eval_pid = int(self._eval_proc.pid)
        if self.host_pid is None:
            self._host_proc = self._spawn_marker_proc()
            self.host_pid = int(self._host_proc.pid)
        eval_before = int(self.eval_pid)
        host_before = int(self.host_pid)
        if eval_before == host_before:
            raise FakepassError('eval pid must differ from host pid')
        inner = getattr(self.pod, '_inner', self.pod)
        if isinstance(inner, ScriptedPod):
            (inner.box_root / 'job.pid').write_text('%d\n' % eval_before)
        os.kill(host_before, signal.SIGKILL)
        try:
            os.waitpid(host_before, 0)
        except ChildProcessError:
            pass
        self._host_proc = self._spawn_marker_proc()
        self.host_pid = int(self._host_proc.pid)
        try:
            os.kill(eval_before, 0)
        except OSError:
            raise FakepassError('eval job.pid died during host restart')
        if self.called_free_gpus or self.called_place_cand:
            raise FakepassError('restart-host-mid-eval must not call free_gpus/place_cand')
        if isinstance(inner, ScriptedPod) and (inner.free_gpus_calls or inner.place_cand_calls):
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
        (self.paths['pit'] / 'RESULT.json').write_text(
            json.dumps(self.fake_result, indent=1, sort_keys=True) + '\n'
        )
        (self.paths['greens'] / ('%s.json' % self.candidate_id)).write_text(
            json.dumps(self.green_stamp, indent=1, sort_keys=True) + '\n'
        )
        self._note('fake_result', result=self.fake_result, stamp=self.green_stamp)
        if self.sever_after_green:
            if isinstance(self.pod, SeveredPod):
                self.pod.arm()
            self._note('sever_after_green', **simulate_sever_ssh())

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
        apply_doc: Dict[str, Any] = {
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
        (self.paths['greens'] / 'GREENS.json').write_text(json.dumps({
            'FAKE': True,
            'promote_live': False,
            'entries': [self.green_stamp],
        }, indent=1, sort_keys=True) + '\n')
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
        for dest in (self.paths['head_chain'], self.paths['pit']):
            (dest / 'HEAD-CHAIN.json').write_text(json.dumps(chain, indent=1, sort_keys=True) + '\n')
        (self.paths['hash_sync'] / 'sync.json').write_text(json.dumps({
            'FAKE': True,
            'ok': True,
            'candidate': cand,
            'expected': self.master_digest,
            'promote_live': False,
        }, indent=1, sort_keys=True) + '\n')
        if isinstance(self.pod, SeveredPod) and self.pod.ssh_attempts:
            apply_doc['ssh_attempts_ignored'] = self.pod.ssh_attempts
        self.apply_result = apply_doc
        self._note('apply_tail', **apply_doc)

    def phase_bench(self) -> None:
        self._advance(FakepassPhase.BENCH)
        self.bench = {
            'FAKE': True,
            'minimal': True,
            'candidate': self.candidate_id,
            'promote_live': False,
            'profile': 'fakepass-minimal',
        }
        (self.paths['bench'] / 'RESULT.json').write_text(
            json.dumps(self.bench, indent=1, sort_keys=True) + '\n'
        )

    def phase_second_step(self) -> None:
        """Check the promoted index is what the next box's fetch_master_on_box reads."""
        self._advance(FakepassPhase.SECOND_STEP)
        cand = self.candidate_id
        doc = json.loads((public_master_index_dir() / ('%s.json' % cand)).read_text())
        REAL.validate_promoted_index(
            doc, candidate_id=cand, live_prefix=live_fp32_r2_prefix(cand),
            master_digest=self.master_digest,
        )
        meta_obj = self.paths['r2_mirror'] / doc['meta_json']['r2_key']
        meta = json.loads(meta_obj.read_text())
        if meta.get('bf16_sha256') != doc['bf16_sha256'] or meta.get('master_digest') != self.master_digest:
            raise FakepassError('promoted meta.json does not match the index')
        if meta.get('version') != SIM_FP32_VERSION or not meta.get('tensor_file'):
            raise FakepassError('promoted meta.json lacks trainer version/tensor_file')
        self.second_step = {'head': cand, 'meta_json': doc['meta_json'], 'meta_ok': True}
        self._note('second_step', **self.second_step)

    def phase_release(self) -> None:
        self._advance(FakepassPhase.RELEASE)
        if self.rent is not None and self.box_id:
            try:
                self.rent.release(self.box_id, reason='fakepass-done')
            except RentError as exc:
                self._note('release_error', error=str(exc)[:200])
        self._note('released', box_id=self.box_id)
        self._note('r2_pending_gc', **gc_pending_artifacts(self.candidate_id))

    def phase_cleanup(self) -> None:
        self._advance(FakepassPhase.CLEANUP)
        self.stop_liveness()
        for proc in (self._host_proc, self._eval_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception:
                    pass
        for key, prev in self._env_saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
        self._env_saved.clear()
        cleanup_sandbox(self.paths)
        self.cleaned = True
        self.phase = FakepassPhase.DONE

    def _dispatch_phase(self, phase: FakepassPhase) -> None:
        if phase is FakepassPhase.ISOLATE:
            self.phase_isolate()
        elif phase is FakepassPhase.RENT:
            self.phase_rent()
        elif phase is FakepassPhase.DOWNLOAD:
            self.phase_download_train()
        elif phase is FakepassPhase.TRAIN:
            return
        elif phase is FakepassPhase.EARLY_UPLOAD:
            self.phase_early_upload()
        elif phase is FakepassPhase.HOLD_VLLM:
            self.phase_hold_vllm()
        elif phase is FakepassPhase.FAKE_RESULT:
            self.phase_fake_result()
        elif phase is FakepassPhase.APPLY_TAIL:
            self.phase_apply_tail()
        elif phase is FakepassPhase.BENCH:
            self.phase_bench()
        elif phase is FakepassPhase.SECOND_STEP:
            self.phase_second_step()
        elif phase is FakepassPhase.RELEASE:
            self.phase_release()
        elif phase is FakepassPhase.CLEANUP:
            self.phase_cleanup()
        elif phase is FakepassPhase.DONE:
            return
        else:
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
                FakepassPhase.BENCH,
                FakepassPhase.SECOND_STEP,
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
                'bench': self.bench,
                'second_step': self.second_step,
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
            if not self.cleaned:
                try:
                    self._dispatch_phase(FakepassPhase.CLEANUP)
                except Exception:
                    cleanup_sandbox(self.paths)
                    self.cleaned = True


def _real_live_roots(base: List[Path], src: 'REAL.LiveSources') -> List[Path]:
    roots = list(base)
    roots.append(src.epoch_dir.parent.parent)
    roots.append(src.bundle_json.parent)
    if not src.master_index_origin.startswith('http'):
        roots.append(Path(src.master_index_origin).parent)
    return roots


def build_run_from_args(
    args: argparse.Namespace,
) -> Tuple[Union[FakepassRun, 'REAL.RealFakepass'], List[Path]]:
    run_id = args.run_id or time.strftime('%Y%m%dT%H%M%S')
    if not RUN_ID_RE.match(run_id):
        _die('run id must match %s (it names the R2 prefix)' % RUN_ID_RE.pattern)
    # Capture live roots from the *pre-sandbox* environment.
    live_roots = live_roots_from_env()
    if args.dry_isolation_check:
        mode = FakepassMode.DRY
    elif args.real:
        mode = FakepassMode.REAL
    elif args.simulate_local:
        mode = FakepassMode.SIMULATE
    else:
        if has_lium_credentials():
            _die('refusing to start: pass --simulate-local, --real, or --dry-isolation-check')
        mode = FakepassMode.DRY

    if mode is FakepassMode.REAL:
        host_check = REAL.preflight_real(os.environ)
        sources = REAL.resolve_live_sources(os.environ, epoch=args.epoch)
        live_roots = _real_live_roots(live_roots, sources)
        paths = make_sandbox(run_id, live_roots=live_roots)
        try:
            staged = REAL.stage_live_sources(sources, paths)
            env = REAL.real_env_for_sandbox(env_for_sandbox(paths, run_id), paths)
        except BaseException:
            cleanup_sandbox(paths)
            raise
        return REAL.RealFakepass(
            run_id=run_id,
            paths=paths,
            env=env,
            live_roots=live_roots,
            sources=sources,
            staged=staged,
            host_check=host_check,
            hold_vllm_s=int(args.hold_vllm_s),
            sever_after_green=bool(args.sever_after_green),
            restart_host_mid_eval=bool(args.restart_host_mid_eval),
            liveness_interval_s=float(args.liveness_interval_s),
            evidence_out=Path(args.evidence_out) if args.evidence_out else None,
        ), live_roots

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
                   help='real rent/train/upload/serve/apply on an affine-pe-pit-fakepass-* box')
    p.add_argument('--epoch', type=int, default=None,
                   help='REAL: finalized live epoch bound to HEAD (default: newest)')
    p.add_argument('--sever-after-green', action='store_true',
                   help='ssh to box fails rc=255; APPLY must succeed from upload')
    p.add_argument('--restart-host-mid-eval', action='store_true',
                   help='kill sandbox host by exact PID and restart; eval pid unchanged')
    p.add_argument('--hold-vllm-s', type=int, default=60,
                   help='eval hold; REAL raises it to %ds (%ds with restart)' % (
                       REAL.MIN_HOLD_S, REAL.MIN_HOLD_RESTART_S))
    p.add_argument('--liveness-interval-s', type=float, default=15.0)
    p.add_argument('--evidence-out', default=None,
                   help='REAL: copy sandbox logs here before the sandbox is removed')
    p.add_argument('--candidate-id', default=None, help='SIMULATE only')
    p.add_argument('--host-pid', type=int, default=None, help='SIMULATE only')
    p.add_argument('--eval-pid', type=int, default=None, help='SIMULATE only')
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
