#!/usr/bin/env python3
"""Fail-closed cutover admission. No writes, service changes, rentals or secrets.
Pure audit() is replay-testable. Passing shadow decisions alone never authorizes
an incomplete live integration. Integration evidence must bind installed code.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REQUIRED = (
    'trainer_restore_consumer', 'candidate_keep_before_next_train',
    'miner_runtime_pins_verified', 'one_baseline_per_head_no_turn_refresh',
    'immediate_live_dispatch', 'current_head_round_binding',
    'trainer_fp32_master_merge_publish_keep', 'signed_merged_publication_readback',
    'durable_current_apply_reconciliation', 'gate_round_close_on_apply',
    'v829t_v829v_replay', 'infra_no_decision_k1_retry',
    'end_to_end_past_epoch_replay', 'surgical_revert_preserves_64k',
)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def audit(root):
    root = Path(root)
    errors = []
    for jid in ('shadow-e819c', 'shadow-e820'):
        jd = root / 'jobs' / jid
        ctl, status, result, score = [read(jd / n) or {} for n in ('CTL.json','STATUS.json','RESULT.json','SCORE.json')]
        if ctl.get('state') != 'scored' or status.get('status') != 'done':
            errors.append(jid + ': full shadow not scored')
        if not result or result.get('partial') is not False:
            errors.append(jid + ': result missing/partial')
        job = read(jd / 'JOB.json') or {}
        indices, k = job.get('indices') or [], job.get('k')
        expected = len(indices) * k if isinstance(k, int) else 0
        cells = (((result.get('envs') or {}).get(job.get('env_id')) or {}).get('per_task') or {})
        if (not expected or result.get('samples') != expected or result.get('planned_samples') != expected or
                set(cells) != {str(i) for i in indices} or
                any(c.get('n') != k for c in cells.values())):
            errors.append(jid + ': exact full plan/sample coverage unproven')
        if score.get('agree') is not True or score.get('pe_decision') != 'red' or score.get('pe_z') is None:
            errors.append(jid + ': RED parity unproven')
        if score.get('pe_n') != score.get('live_n') or not score.get('pe_n'):
            errors.append(jid + ': scored sample coverage differs from live')
        attempts = ctl.get('attempts') or []
        if jid == 'shadow-e820' and not any('gone from lium ps' in a.get('failed', '') for a in attempts):
            errors.append(jid + ': forced pod-loss recovery evidence missing')
        if not status.get('released'):
            errors.append(jid + ': pod release not confirmed')
    evidence = read(root / 'state' / 'LIVE-INTEGRATION-EVIDENCE.json') or {}
    checks = evidence.get('checks') or {}
    for name in REQUIRED:
        row = checks.get(name)
        if not isinstance(row, dict) or row.get('passed') is not True or not row.get('evidence'):
            errors.append('integration: ' + name + ' not proven')
            continue
        ev = (root / row['evidence']).resolve()
        if not ev.is_relative_to(root.resolve()) or not ev.is_file():
            errors.append('integration: ' + name + ' evidence not local/present')
    pins = evidence.get('code_sha256') or {}
    for name in ('pe_lifecycle.py', 'pe_controller.py', 'pe_gate_hook.py', 'pe_apply.py', 'pe_launcher.py', 'pe_master_merge.py'):
        path = root / 'code' / name
        if not path.is_file() or pins.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            errors.append('integration: missing/current-code pin ' + name)
    return errors


def main():
    errors = audit(HERE)
    print(json.dumps(dict(ready=not errors, blockers=errors), indent=1))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
