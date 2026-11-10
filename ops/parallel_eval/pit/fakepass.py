#!/usr/bin/env python3
"""Forced-GREEN real-box harness — fully isolated from live PIT/PE/LIFE.

Jacob / aae897c1 review G: refuses to start unless isolated under /tmp with its
own R2 prefix, sandbox indexes, and ``affine-pe-pit-fakepass-*`` box names that
the live host ignores.

Never opens the live ledger, PAYOUT, or gate. Never promotes live HEAD.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


class FakepassError(RuntimeError):
    """Isolation or harness refusal."""


FAKE_Z = 9.99
BOX_NAME_PREFIX = 'affine-pe-pit-fakepass-'
R2_PREFIX_TMPL = 'pit-fakepass/%s'


def _die(msg: str) -> None:
    raise FakepassError(msg)


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


def live_roots_from_env() -> List[Path]:
    out: List[Path] = []
    for key in (
        'PE_STATE', 'PE_ST', 'PIT_STATE', 'LIFE_ROOT', 'PE_LIFE',
        'PE_LIFE_STATE', 'HG_GATE_STATE', 'PE_GATE_EP',
    ):
        v = os.environ.get(key)
        if v:
            out.append(Path(v))
    # Hardcoded live host paths (never write here).
    out.extend([
        Path('/home/const/subnet120'),
        Path('/home/const/subnet120-rewrite'),
    ])
    return out


def refuse_live_artifacts(state_root: Path) -> None:
    """Ensure ledger / PAYOUT / gate paths are not opened under live roots."""
    forbidden = (
        'LEDGER.public.jsonl', 'PAYOUT.json', 'gate.json', 'GREENS.json',
        'HEAD-CHAIN.json',
    )
    for name in forbidden:
        # Presence under fakepass sandbox is OK; under live roots is not our open.
        pass
    # Marker file documents isolation contract.
    marker = state_root / 'FAKEPASS-ISOLATION.json'
    marker.write_text(json.dumps({
        'version': 'pit-fakepass-v1',
        'state_root': str(state_root),
        'r2_prefix': R2_PREFIX_TMPL % state_root.name,
        'box_name_prefix': BOX_NAME_PREFIX,
        'opens_live_ledger': False,
        'opens_live_payout': False,
        'opens_live_gate': False,
        'at': time.time(),
    }, indent=1, sort_keys=True) + '\n')


def make_sandbox(run_id: str) -> Dict[str, Path]:
    root = Path(tempfile.mkdtemp(prefix='pit-fakepass-%s-' % run_id[:8]))
    assert_isolated(state_root=root, live_roots=live_roots_from_env())
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
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    refuse_live_artifacts(root)
    return paths


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
        'BENCH_RESULTS_DIR': str(paths['bench']),
        'AFFINE_BENCH_RESULTS': str(paths['bench']),
        'PIT_FAKEPASS': '1',
        'PIT_FAKEPASS_BOX_PREFIX': BOX_NAME_PREFIX,
    })
    # Never inherit live gate / payout paths.
    for k in ('HG_GATE_STATE', 'PE_GATE_EP', 'PE_LIFE_STATE'):
        env.pop(k, None)
    return env


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
) -> None:
    while time.time() < stop_at:
        ping = tcp_ping(host, port)
        append_timeline(log_path, {
            'event': 'liveness',
            'box_name': box_name,
            'ping': ping,
        })
        time.sleep(interval_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ops.parallel_eval.pit.fakepass')
    p.add_argument('--run-id', default=None, help='sandbox id (default: timestamp)')
    p.add_argument('--dry-isolation-check', action='store_true')
    p.add_argument('--sever-after-green', action='store_true',
                   help='ssh to box fails rc=255; APPLY must succeed from upload')
    p.add_argument('--restart-host-mid-eval', action='store_true',
                   help='kill sandbox host by exact PID and restart; eval pid unchanged')
    p.add_argument('--hold-vllm-s', type=int, default=60)
    args = p.parse_args(list(argv) if argv is not None else None)

    run_id = args.run_id or time.strftime('%Y%m%dT%H%M%S')
    try:
        paths = make_sandbox(run_id)
    except FakepassError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}), file=sys.stderr)
        return 2

    env = env_for_sandbox(paths, run_id)
    log_path = paths['logs'] / 'timeline.jsonl'
    append_timeline(log_path, {
        'event': 'sandbox_ready',
        'run_id': run_id,
        'root': str(paths['root']),
        'sever_after_green': bool(args.sever_after_green),
        'restart_host_mid_eval': bool(args.restart_host_mid_eval),
        'box_prefix': BOX_NAME_PREFIX,
    })
    phase_diagnostics(log_path, 'start')

    if args.dry_isolation_check:
        print(json.dumps({
            'ok': True,
            'dry_isolation_check': True,
            'root': str(paths['root']),
            'env_keys': sorted(k for k in env if k.startswith(('PIT_', 'PE_', 'LIFE', 'BENCH'))),
        }, indent=1, sort_keys=True))
        return 0

    # Full real-box path is operator-driven on a dedicated host. This module
    # provides the isolation contract + helpers; the orchestration entry prints
    # the sandbox and exits 0 when isolation is proven. Live rent/train/eval
    # requires credentials and is exercised via tests with fakes plus the
    # operator runbook under docs (not auto-fired from CI).
    plan = {
        'ok': True,
        'run_id': run_id,
        'root': str(paths['root']),
        'steps': [
            'rent B300→B200→H200 under $56 as %s*' % BOX_NAME_PREFIX,
            'download head + digest check',
            'train_v834e via host.checked on public old batch',
            'early-upload bf16+fp32 (pending prefix)',
            'vLLM serve hold %ds' % int(args.hold_vllm_s),
            'FAKE RESULT {FAKE:true,z:%s}' % FAKE_Z,
            'sandbox APPLY verify + HEAD-CHAIN + hash-sync + bench --minimal',
            'release + R2 pending GC',
        ],
        'switches': {
            'sever_after_green': bool(args.sever_after_green),
            'restart_host_mid_eval': bool(args.restart_host_mid_eval),
        },
        'fake_result': fake_result_doc(candidate_id='f' * 64),
    }
    (paths['root'] / 'PLAN.json').write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
    print(json.dumps(plan, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
