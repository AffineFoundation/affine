#!/usr/bin/env python3
"""SN120 disk guard (runs on affinevalidator; systemd timer every 15 min).

Policy (Jacob 2026-10-30): keep ONLY the current king (gate head) and the next model queued for checking
(trainer CURRENT checkpoint / probe serve.ckpt); anything a process has open is never touched. Prunes:
  probe   ckpt/<id>[.*] (live plain-disk dir and the *.gocryptfs-orig copy)
  trainer private/promotion-gate/keep/<id>
Entries younger than MIN_AGE_MIN are skipped. Caps listed log files (copytruncate). Alerts (ALERT.json + journal)
when free space drops below the per-machine threshold (trainer needs >= 135 GB before a job).
Every deletion is appended to RA/20261030-disk-cleanup/DELETED.tsv.

  disk_guard.py            # dry run (default)
  disk_guard.py --apply    # prune + rotate
Disable: systemctl --user disable --now sn120-disk-guard.timer
"""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(os.environ.get('SN120_REPO', '/home/const/subnet120-rewrite'))
RA = REPO / 'state/root-audits'
GATE_STATUS = RA / '20261026-promotion-gate/state/status.json'
OUT = RA / '20261030-disk-cleanup'
LOG = OUT / 'DELETED.tsv'
POD_SIDE = Path(__file__).with_name('pod_side.sh')
KH = '/home/const/.lium/known_hosts/'
GB = 10 ** 9
# role: (host, port, known_hosts uuid, alert threshold bytes on the main data mount, log globs)
PODS = {
    'trainer': ('8.28.35.244', 40022, 'ef2b6085-adae-43e2-8a13-39182a3b3f7a', 150 * GB,
                '/workspace/affine-mimo-trainer-prod/workspace/*.log'),
    'probe': ('38.88.142.89', 40198, 'b4f3da9d-df66-436a-af38-829d1606966d', 100 * GB,
              '/root/affine-env-probe/serve*.log /root/affine-env-probe/*/serve*.log'),
    'eval': ('38.255.28.20', 20092, '6e469697-f3a7-4a6a-a1ae-40dde7120581', 40 * GB,
             '/root/affine-mimo-public-evaluator-20261013/*.log'),
}
HOST_MIN_FREE = 100 * GB
MIN_AGE_MIN = int(os.environ.get('MIN_AGE_MIN', '180'))
LOG_CAP_MB = int(os.environ.get('LOG_CAP_MB', '1024'))


def cfg_pods():
    f = OUT / 'pods.json'   # optional override when pods move: {"probe": [host, port, uuid, min_free, globs], ...}
    return {**PODS, **{k: tuple(v) for k, v in json.loads(f.read_text()).items()}} if f.exists() else PODS


def protected():
    ids = set()
    try: ids.add(json.loads(GATE_STATUS.read_text())['head'])
    except Exception: pass
    return ids


def ssh(role, script, env, timeout=900):
    h, p, k = cfg_pods()[role][:3]
    pre = ''.join('%s=%s\n' % (a, json.dumps(b)) for a, b in env.items())
    r = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
                        '-o', 'UserKnownHostsFile=' + KH + k, '-p', str(p), 'root@' + h, 'bash -s'],
                       input=pre + script, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


def pod_ids(role):
    """next-model ids reported by the pod itself (trainer CURRENT, probe serve.ckpt)."""
    cmd = {'trainer': "cat /workspace/affine-mimo-trainer-prod/trainer-state/*/CURRENT.json 2>/dev/null",
           'probe': "cat /root/affine-env-probe/serve.ckpt 2>/dev/null", 'eval': 'true'}[role]
    rc, out = ssh(role, cmd + '\n', {}, timeout=60)
    if rc != 0: raise RuntimeError('pod %s unreachable' % role)
    if role == 'trainer':
        d = json.loads(out); return {d.get('checkpoint'), d.get('input_checkpoint')} - {None}
    return {out.strip()} if len(out.strip()) == 64 else set()


def main():
    apply = '--apply' in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    if not LOG.exists(): LOG.write_text('ts_utc\thost\tpath\tbytes\tmtime\tstatus\n')
    head = protected()
    if not head: print('no gate head readable; refusing to prune'); apply = False
    try: nxt = pod_ids('trainer') | pod_ids('probe')
    except Exception as e: print('cannot read next-model ids (%s); refusing to prune' % e); nxt = set(); apply = False
    prot = head | nxt
    alerts, report = [], {'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'apply': apply, 'protected': sorted(prot)}
    for role, (_, _, _, min_free, globs) in cfg_pods().items():
        try:
            rc, out = ssh(role, POD_SIDE.read_text(), dict(ROLE=role, PROTECT=' '.join(prot), APPLY='1' if apply else '0',
                                                         MIN_AGE_MIN=str(MIN_AGE_MIN), LOG_CAP_MB=str(LOG_CAP_MB), LOG_GLOBS=globs))
        except Exception as e:
            alerts.append('%s: guard failed: %s' % (role, e)); continue
        rows = [l.split('\t') for l in out.splitlines() if l.count('\t') >= 3]
        report[role] = rows
        with LOG.open('a') as f:
            for r in rows:
                if r[0] in ('prune', 'rotate') and r[-1] in ('DELETED', 'ROTATED', 'RM-FAIL'):
                    f.write('\t'.join([report['at'], role] + r[1:]) + '\n')
        for r in rows:
            if r[0] == 'free' and r[1] in ('/workspace', '/') and int(r[3]) < min_free and not (role == 'trainer' and r[1] == '/' and any(x[1] == '/workspace' for x in rows if x[0] == 'free')):
                alerts.append('%s %s free %.0f GB < %.0f GB' % (role, r[1], int(r[3]) / GB, min_free / GB))
    st = os.statvfs('/')
    if st.f_bavail * st.f_frsize < HOST_MIN_FREE: alerts.append('affinevalidator / free %.0f GB' % (st.f_bavail * st.f_frsize / GB))
    report['alerts'] = alerts
    (OUT / 'LAST-RUN.json').write_text(json.dumps(report, indent=1))
    if alerts: (OUT / 'ALERT.json').write_text(json.dumps({'at': report['at'], 'alerts': alerts}))
    elif (OUT / 'ALERT.json').exists(): (OUT / 'ALERT.json').unlink()
    for a in alerts: print('ALERT', a, file=sys.stderr)
    for role in cfg_pods():
        for r in report.get(role, []): print(role, *r, sep='\t')


if __name__ == '__main__':
    main()
