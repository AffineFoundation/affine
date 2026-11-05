#!/usr/bin/env python3
"""hard-v1 32K re-filter orchestrator (host). Two pass pods (pod/, pod2/) share the work by task-id parity. Every 2 min: pull each pod's
grades (keep the last good copy, never shrink), merge, and when a pod's assigned set is (nearly) done, restart its worker on the next set:
tasks it owns that still need +4 rollouts under the adaptive rule (n<4 -> stage A; 4<=n<8 and c<n and b<0.75n -> B;
8<=n<12 and 1<=c<=0.75n and b<0.5n -> C; n>=12 or prompt error -> done). Per-task data stays in private/ (0600). Aggregates to stdout.
usage: orch.py [--once]"""
import collections, hashlib, json, os, subprocess, sys, time
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = D + '/private'
HEAD = '8389b90be4367d8eccf444f92508e38d87fe3736ae23c195e609f027991985b6'
CANDS = '/home/const/subnet120-rewrite/state/task-slices/hard-v1/private/candidates.jsonl'
PODS = {0: D + '/pod', 1: D + '/pod2'}
STATE = D + '/pod/orch_state.json'
os.umask(0o077)
def log(*a):
    m = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a); print(m, flush=True)
    open(D + '/pod/orch.log', 'a').write(m + '\n')
def pp(i, cmd, inp=None, timeout=300):
    r = subprocess.run([PODS[i] + '/pp.sh', cmd], input=inp, capture_output=True, timeout=timeout)
    if r.returncode: raise RuntimeError('pod%d rc=%d %s' % (i, r.returncode, r.stderr.decode(errors='replace')[-200:]))
    return r.stdout
def owner(t): return int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2
cand_lines = {}
for l in open(CANDS):
    if l.strip(): cand_lines[json.loads(l)['id']] = l if l.endswith('\n') else l + '\n'
def pull(i):
    f = P + '/grades.p%d.jsonl' % i
    if not os.path.exists(PODS[i] + '/pp.sh'): return
    try: raw = pp(i, 'cat /opt/prior/runs/hardv1/grades.jsonl 2>/dev/null || true', timeout=600)
    except Exception as e: log('pull pod%d failed: %s' % (i, str(e)[:120])); return
    old = os.path.getsize(f) if os.path.exists(f) else 0
    if len(raw) >= old:
        open(f + '.tmp', 'wb').write(raw); os.replace(f + '.tmp', f)
def merged():
    A = collections.defaultdict(lambda: [0, 0, 0]); err = set(); lines = []
    for i in PODS:
        f = P + '/grades.p%d.jsonl' % i
        if not os.path.exists(f): continue
        for l in open(f):
            try: g = json.loads(l)
            except Exception: continue
            if g.get('head') != HEAD: continue
            lines.append(l if l.endswith('\n') else l + '\n')
            if g.get('err'): err.add(g['task_id']); continue
            a = A[g['task_id']]; a[0] += g['n']; a[1] += g['c']; a[2] += g['b']
    open(P + '/grades.merged.jsonl.tmp', 'w').writelines(lines); os.replace(P + '/grades.merged.jsonl.tmp', P + '/grades.merged.jsonl')
    return A, err
def need(a):
    n, c, b = a
    if n < 4: return 'A'
    if n < 8: return 'B' if (c < n and b < 0.75 * n) else None
    if n < 12: return 'C' if (1 <= c <= 0.75 * n and b < 0.5 * n) else None
    return None
def restart(i, ids):
    body = ''.join(cand_lines[t] for t in sorted(ids))
    pp(i, 'cat > /opt/prior/data/assign.jsonl.tmp && mv /opt/prior/data/assign.jsonl.tmp /opt/prior/data/assign.jsonl', inp=body.encode())
    cmd = r'''set -e; R=/opt/prior/runs/hardv1; mkdir -p $R; touch $R/STOP; p=$(cat $R/worker.pid 2>/dev/null || true)
if [ -n "$p" ] && [ "$p" -gt 1 ]; then for k in $(seq 90); do kill -0 $p 2>/dev/null || break; sleep 2; done; kill -0 $p 2>/dev/null && kill $p || true; sleep 3; fi
for q in $(ps -eo pid=,args= | awk '$2=="/opt/prior/venv/bin/python" && $4=="priors/worker.py"{print $1}'); do kill -9 $q 2>/dev/null || true; done; sleep 2
python3 - <<'PY'
import json, os, time
R = '/opt/prior/runs/hardv1'; c = json.load(open('/opt/prior/code/sampler.json')); c['envs'] = {'dapo': {'path': '/opt/prior/data/assign.jsonl'}}
json.dump(c, open('/opt/prior/code/sampler.cur.json', 'w'), indent=1)
try: ps = json.load(open(R + '/PASS.json'))
except Exception: ps = None
head = json.load(open('/opt/prior/serve/SERVING.json'))['ckpt']
nxt = (ps['pass'] + 1) if ps else 0
json.dump(dict(**{'pass': nxt}, head=head, nonce=os.urandom(16).hex(), started=time.time()), open(R + '/PASS.json', 'w')); print('pass', nxt)
PY
rm -f $R/STOP; cd /opt/prior/code && (TOKENIZERS_PARALLELISM=false TMPDIR=/opt/prior/tmp HF_HUB_OFFLINE=1 HF_HOME=/opt/prior/hf setsid nohup /opt/prior/venv/bin/python -u priors/worker.py sampler.cur.json >> /opt/prior/logs/worker.log 2>&1 < /dev/null & echo $! > $R/worker.pid); sleep 1; cat $R/worker.pid'''
    out = pp(i, cmd, timeout=400).decode().strip().replace('\n', ' ')
    log('pod%d restarted on %d tasks: %s' % (i, len(ids), out))
def main():
    once = '--once' in sys.argv
    st = json.load(open(STATE)) if os.path.exists(STATE) else {'pods': {}}
    while True:
        for i in PODS: pull(i)
        A, err = merged()
        needs = {}
        for t in cand_lines:
            if t in err: continue
            nd = need(A.get(t, [0, 0, 0]))
            if nd: needs[t] = nd
        cnt = collections.Counter(needs.values())
        for i in PODS:
            if not os.path.exists(PODS[i] + '/pp.sh'): continue
            ps = st['pods'].get(str(i))
            if ps and ps.get('done'): continue
            ready = False
            if ps is None:
                ready = os.path.exists(PODS[i] + '/READY')
            else:
                rem = [t for t in ps['ids'] if t in needs and A.get(t, [0])[0] < ps['n0'].get(t, 0) + 4]
                prog = sum(A.get(t, [0])[0] for t in ps['ids'])
                if prog != ps.get('prog'): ps['prog'] = prog; ps['moved'] = time.time()
                ps['remaining'] = len(rem)
                ready = len(rem) == 0 or (len(rem) <= 300 and len(rem) <= 0.05 * len(ps['ids'])) or (len(rem) <= 20 and time.time() - ps.get('moved', 0) > 600)
            if not ready: continue
            mine = [t for t in needs if owner(t) == i]
            if not mine:
                st['pods'][str(i)] = dict(done=True, at=time.time()); log('pod%d: nothing left to assign' % i); continue
            try: restart(i, mine)
            except Exception as e: log('pod%d restart failed: %s' % (i, str(e)[:200])); continue
            st['pods'][str(i)] = dict(ids=mine, n0={t: A.get(t, [0])[0] for t in mine}, at=time.time(), moved=time.time(), prog=None,
                                      stages=dict(collections.Counter(needs[t] for t in mine)))
        json.dump(st, open(STATE + '.tmp', 'w')); os.replace(STATE + '.tmp', STATE)
        tot = sum(a[0] for a in A.values())
        log('scored %d tasks, %d rollouts; need A/B/C %s; pods %s' % (len(A), tot, dict(cnt),
            {k: ('done' if v.get('done') else '%d/%d left' % (v.get('remaining', len(v['ids'])), len(v['ids']))) for k, v in st['pods'].items()}))
        if all(v.get('done') for v in st['pods'].values()) and len(st['pods']) == len(PODS) and not needs:
            log('ALL DONE'); open(D + '/pod/ALL-DONE', 'w').write(time.ctime()); return
        if once: return
        time.sleep(120)
if __name__ == '__main__': main()
