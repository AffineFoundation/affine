"""Gate probe error-rate guard (Jacob 2026-11-05 e757-rejudge).
If a head/candidate/baseline block has errored-sample rate > THRESHOLD, the gate must not judge:
hold the round, re-probe up to N times, keep holding + alert if errors persist.
Errors (cell 'x') are never treated as wrong answers.
Alert only — no auto-revert.
"""
from __future__ import annotations
import json, time, os
from pathlib import Path

THRESHOLD = 0.01          # >1% errored samples
REPROBE_MAX = 2
VERSION = 'sn120-probe-error-guard-v1'

def _cr(ts=None):
    import datetime
    t = ts if ts is not None else time.time()
    return datetime.datetime.fromtimestamp(t, datetime.timezone(datetime.timedelta(hours=-6))).strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)')

def cell_error_rate(per_task: dict) -> dict:
    """per_task values are probe cells with n/c/w/u/b/x. -> summary with rate."""
    xs = ns = 0
    if not isinstance(per_task, dict):
        return dict(n=0, x=0, rate=0.0, tasks=0)
    for v in per_task.values():
        if not isinstance(v, dict) or not v.get('n'):
            continue
        ns += int(v['n']); xs += int(v.get('x') or 0)
    rate = (xs / ns) if ns else 0.0
    return dict(n=ns, x=xs, rate=rate, tasks=sum(1 for v in per_task.values() if isinstance(v, dict) and v.get('n')))

def is_over_threshold(per_task: dict, threshold: float = THRESHOLD) -> bool:
    s = cell_error_rate(per_task)
    return s['n'] > 0 and s['rate'] > threshold

def alert(name: str, reason: str, **extra):
    """Write ALERT-only via sn120_alert if present; always append local guard log."""
    try:
        import sys
        sys.path.insert(0, '/home/const/subnet120-rewrite/ops/alerts')
        import sn120_alert
        sn120_alert.alert(name, reason, manual='probe-error-guard: hold+reprobe; NO auto-revert')
    except Exception as e:
        extra = dict(extra, alert_import_error=str(e)[:200])
    log_dir = Path('/home/const/subnet120-rewrite/state/alerts')
    log_dir.mkdir(parents=True, exist_ok=True)
    line = '%s %s ALERT (no action): %s %s\n' % (_cr(), name, reason[:300], json.dumps(extra)[:300] if extra else '')
    with open(log_dir / 'alerts.log', 'a') as f:
        f.write(line)
    # discord-watch log (same alerts.log is watched; also mailbox drop)
    mail = Path('/home/const/subnet120-rewrite/state/alerts/mailbox')
    mail.mkdir(parents=True, exist_ok=True)
    (mail / ('%s-%d.json' % (name, int(time.time())))).write_text(json.dumps(dict(
        name=name, reason=reason, at=_cr(), extra=extra, action='hold-reprobe-alert-only'), indent=2))

def hold_record(epoch_dir: Path, *, side: str, rate: float, attempt: int, detail: dict) -> dict:
    """Write/merge HELD.json in the epoch gate dir. Returns the record."""
    p = Path(epoch_dir) / 'HELD.json'
    rec = json.loads(p.read_text()) if p.exists() else dict(version=VERSION, holds=[])
    entry = dict(side=side, rate=rate, threshold=THRESHOLD, attempt=attempt, at=time.time(), at_cr=_cr(), detail=detail)
    rec['holds'] = [h for h in rec.get('holds') or [] if not (h.get('side') == side and h.get('attempt') == attempt)]
    rec['holds'].append(entry)
    rec['active'] = True
    rec['updated_at_cr'] = _cr()
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rec, indent=2)); os.replace(tmp, p)
    return rec

def should_judge(per_task: dict, *, side: str, epoch_dir: Path, attempt: int = 0) -> tuple[bool, dict]:
    """-> (ok_to_judge, summary). If not ok: writes HELD + alerts when attempts exhausted."""
    summary = cell_error_rate(per_task)
    if summary['n'] == 0:
        return False, dict(summary, reason='empty_probe')
    if summary['rate'] <= THRESHOLD:
        return True, summary
    hold_record(epoch_dir, side=side, rate=summary['rate'], attempt=attempt, detail=summary)
    if attempt >= REPROBE_MAX:
        alert('gate-probe-error-hold',
              'epoch probe %s error_rate=%.4f > %.2f after %d reprobes; holding judgment' % (side, summary['rate'], THRESHOLD, attempt),
              side=side, **summary)
    return False, summary
