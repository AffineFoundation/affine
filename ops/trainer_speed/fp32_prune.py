#!/usr/bin/env python3
"""fp32-master store pruner (RA/20261030-trainer-speed; Jacob standing rule via parent 14:16 CR: keep only the king and the next model
being checked). Runs on the AV host; read-only towards lifecycle/gate/trainer/publisher; only deletes trainer-pod fp32-master entries.
After each gate DECISION of epoch e (ledger applied red|green):
  candidate entry X(e) = the store entry written during e's training (mtime between TRAINING-JOB(e) and the decision).
  GREEN -> king = X(e).  RED -> king = X(e).meta.input_bf16_sha256 (the head the candidate was trained from).
  delete every entry that is not the king, was last modified before the decision (so an in-flight e+1 candidate is kept),
  and has no fresh PIN (publisher upload in progress, < 3 h). If the king cannot be determined: delete nothing, log.
Revert: systemctl --user stop ts-fp32-prune   (state in activation/fp32-prune.json, log activation/fp32-prune.log)"""
import json, os, subprocess, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); AV = RA / '20261030-trainer-speed/activation'
E = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'); GE = RA / '20261026-promotion-gate/state/epochs'
STORE = '/workspace/affine-mimo-trainer-prod/fp32-master'; ST = AV / 'fp32-prune.json'; FROM = int(os.environ.get('FROM', 276))
DRY = os.environ.get('DRY') == '1'
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '-o',
       'UserKnownHostsFile=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a', '-p', '40022', 'root@8.28.35.244']
cr = lambda: time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))
def log(**k):
    k['at_cr'] = cr(); open(AV / 'fp32-prune.log', 'a').write(json.dumps(k, sort_keys=True) + '\n')
def pod(cmd, t=120):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=t)
    if r.returncode: raise RuntimeError(f'pod rc {r.returncode}: {r.stderr[-200:]}')
    return r.stdout
LIST = r'''python3 - <<'PY'
import json, os, time
from pathlib import Path
out = []
for d in Path('%s').iterdir():
    if d.is_dir() and not d.name.startswith('.') and (d / 'meta.json').is_file():
        try: m = json.loads((d / 'meta.json').read_text())
        except Exception: continue
        pin = d / 'PIN'
        out.append(dict(name=d.name, mtime=(d / 'meta.json').stat().st_mtime, input=m.get('input_bf16_sha256'), bf16=m.get('bf16_sha256'),
                        pin=pin.exists() and time.time() - pin.stat().st_mtime < 3 * 3600))
print(json.dumps(out))
PY''' % STORE
def decided(e):
    d = GE / str(e) / 'DECISION.json'
    try: return json.loads(d.read_text()), d.stat().st_mtime
    except Exception: return None, None
def handle(e):
    dec, t_dec = decided(e); job = E / str(e) / 'TRAINING-JOB.json'
    if dec is None: return False
    verdict = dec.get('decision')
    if verdict not in ('red', 'green'): log(epoch=e, event='skip', decision=verdict); return True
    entries = json.loads(pod(LIST)); t_job = job.stat().st_mtime if job.exists() else None
    xs = [x for x in entries if t_job and t_job <= x['mtime'] <= t_dec]
    if len(xs) != 1:
        log(epoch=e, event='king-unknown', decision=verdict, window_entries=[x['name'][:12] for x in xs], entries=len(entries)); return True
    X = xs[0]; king = X['name'] if verdict == 'green' else X['input']
    victims = [x['name'] for x in entries if x['name'] != king and x['mtime'] <= t_dec and not x['pin']]
    kept = [x['name'][:12] for x in entries if x['name'] not in victims]
    if victims and not DRY:
        pod('cd %s && for v in %s; do case "$v" in *[!0-9a-f]*|"") exit 3;; esac; [ ${#v} -eq 64 ] && [ -d "$v" ] && rm -rf -- "$v"; done; true'
            % (STORE, ' '.join(victims)), t=600)
    free = pod("df -B1G --output=avail %s | tail -1" % STORE).strip()
    log(epoch=e, event='pruned' if not DRY else 'dry-run', decision=verdict, king=(king or '')[:12], candidate=X['name'][:12],
        deleted=[v[:12] for v in victims], kept=kept, pod_free_gb=free)
    return True
st = json.loads(ST.read_text()) if ST.exists() else {'next': FROM}
log(event='armed', next=st['next'], dry=DRY)
while True:
    try:
        if handle(st['next']): st['next'] += 1; ST.write_text(json.dumps(st))
    except Exception as x:
        log(event='error', epoch=st['next'], error=repr(x)[:300])
    if DRY and st['next'] > FROM: break
    time.sleep(15)
