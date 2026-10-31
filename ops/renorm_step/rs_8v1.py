#!/usr/bin/env python3
"""RA/20261031-renorm-step-live: Allan precondition (a). One 8-GPU v8.3.3i turn (epoch E, live) vs a 1-GPU replay of the same batch on the
SHADOW pod (off the live path, only on an idle GPU; never blocks epochs). Host: fetch the live candidate (R2 presigned GET, URLs only in a
0600 host tmp file), build the trained file (ff_format.cap of the epoch snapshot), push files over ssh (no URLs/creds on the pod), run the
replay + CPU compare on the pod, write tests/EIGHT-VS-ONE.json (the watchdog reverts the trainer on verdict "mismatch").
usage: rs_8v1.py EPOCH"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
E = int(sys.argv[1])
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); RS = RA / '20261031-renorm-step-live'; W = RS / '8v1' / ('e%d' % E)
SH = RA / '20261031-shadow-renorm-step'; TP = str(SH / 'tp.sh')
EP = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs') / str(E)
GC = RA / '20261026-promotion-gate/code'; FFR = RA / '20261030-free-file/release'
H = '87fa740811e75e4c221258ce6cf2d8cb1db1519d69aa0a65b42cb5fddea5a499'
W.mkdir(parents=True, exist_ok=True)
def log(m):
    line = '%s %s' % (time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600)), m)
    print(line, flush=True); open(W / 'log.txt', 'a').write(line + '\n')
def prog(m): open(RS / 'PROGRESS.md', 'a').write('- %s CR: 8v1 e%d: %s\n' % (time.strftime('%H:%M', time.gmtime(time.time() - 6 * 3600)), E, m))
def pod(cmd, inp=None, timeout=600):
    return subprocess.run([TP, cmd], input=inp, capture_output=True, timeout=timeout)
def result(verdict, **kw):
    d = dict(version='rs-8v1-v1', epoch=E, verdict=verdict, at=time.time(), **kw)
    (RS / 'tests' / 'EIGHT-VS-ONE.json').write_text(json.dumps(d, indent=1, default=str)); log('RESULT %s' % json.dumps(d, default=str)[:600])
while not (EP / 'TRAINING-RESULT.json').exists(): time.sleep(10)
r = json.loads((EP / 'TRAINING-RESULT.json').read_text()); r = r.get('payload', r); rep = r['report']; t = rep['training']
desc = rep['publication_receipt']['payload']['descriptor']; C = desc['id']; s = t['optimizer']['sgd']
if t['input_checkpoint'] != H: result('skipped', reason='head is not 87fa (shadow pod only has 87fa + its master)'); sys.exit(0)
(W / 'live_traj.json').write_text(json.dumps(dict(trajectories=t['trajectories'], objective_loss=t['objective_loss'], preclip_gradient_norm=t['preclip_gradient_norm'],
    raw_update_norm=s['raw_update_norm'], tensors_rel_capped=s['tensors_rel_capped'], sgd_direction=s.get('sgd_direction'), support_renorm=t.get('support_renorm'))))
turn = json.loads((EP / 'K1-TURN.json').read_text()); turn = turn.get('payload', turn)
raw = json.loads((EP / 'ff' / (turn['hotkey'] + '.json')).read_bytes())
sys.path.insert(0, str(FFR)); import ff_format
kept, dropped = ff_format.cap([(g, g['rollouts']) for g in raw['groups']], turn['caps']['max_tasks'], turn['caps']['max_rollouts'])
raw['groups'] = [g for g, _ in kept]
if dropped != turn['dropped_groups']: result('error', reason='trim mismatch'); sys.exit(1)
(W / 'ff.json').write_text(json.dumps(raw)); log('live e%d candidate %s, %d prompts, %d rollouts (dropped %d)' % (E, C[:12], len(raw['groups']), sum(len(g['rollouts']) for g in raw['groups']), dropped))
# fetch the live candidate onto host plain disk (same method as the gate / shadow fetch)
ck = RS / 'ckpt'; ck.mkdir(exist_ok=True)
if not (ck / C / '.verified').exists():
    sys.path.insert(0, str(GC)); import presign as PS
    fd, uf = tempfile.mkstemp(prefix='urls.', suffix='.private.json', dir='/tmp'); os.close(fd); os.chmod(uf, 0o600)
    try:
        json.dump(PS.presign(C, desc['files']), open(uf, 'w'))
        dp = ck / (C + '.descriptor.json'); dp.write_text(json.dumps(dict(id=C, files=desc['files'])))
        p = subprocess.run(['python3', str(GC / 'pod' / 'gate_fetch.py'), str(dp), str(ck), uf, '--threads', '16'], capture_output=True, text=True)
    finally:
        if os.path.exists(uf): os.remove(uf)
    log('fetch rc=%d %s' % (p.returncode, p.stdout.strip()[-200:]))
    if p.returncode != 0: result('error', reason='fetch failed'); sys.exit(1)
# push to the shadow pod (its own dir /data/rs; shadow files are only read)
pod('mkdir -p /data/rs/scripts /data/rs/e%d' % E)
for f in ('rs_replay_pod.py', 'rs_compare_pod.py'):
    pod('cat > /data/rs/scripts/%s' % f, (RS / 'pod' / f).read_bytes())
pod('cat > /data/rs/e%d/ff.json' % E, (W / 'ff.json').read_bytes()); pod('cat > /data/rs/e%d/live_traj.json' % E, (W / 'live_traj.json').read_bytes())
if pod('test -f /data/rs/live-%s/.pushed' % C).returncode != 0:
    t0 = time.time()
    tar = subprocess.Popen(['tar', 'cf', '-', '-C', str(ck), C], stdout=subprocess.PIPE)
    p = subprocess.run([TP, 'mkdir -p /data/rs/tmp && tar xf - -C /data/rs/tmp && rm -rf /data/rs/live-%s && mv /data/rs/tmp/%s /data/rs/live-%s && touch /data/rs/live-%s/.pushed' % (C, C, C, C)], stdin=tar.stdout, capture_output=True, timeout=3600)
    tar.wait(); log('push live candidate rc=%d %.0fs' % (p.returncode, time.time() - t0))
    if p.returncode != 0: result('error', reason='push failed'); sys.exit(1)
pod('cd /data/rs/live-%s && sha256sum --quiet -c <(python3 -c "import json;d=json.load(open(\'/dev/stdin\'));[print(v+\'  \'+k) for k,v in d[\'files\'].items()]" < /dev/stdin)' % C, json.dumps(dict(files=desc['files'])).encode())
# wait for a free GPU on the shadow pod (6 or 7, < 1 GB used); give up after 60 min without blocking anything
gpu = None; t0 = time.time()
while gpu is None and time.time() - t0 < 3600:
    q = pod('nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits').stdout.decode()
    free = [int(a) for a, b in (l.split(',') for l in q.strip().splitlines()) if int(b) < 1000 and int(a) in (7, 6)]
    if free: gpu = free[0]
    else: time.sleep(60)
if gpu is None: result('pending', reason='no idle GPU on the shadow pod within 60 min'); sys.exit(0)
log('replay on shadow GPU %d' % gpu); prog('1-GPU replay started on shadow pod GPU %d (live candidate %s)' % (gpu, C[:12]))
cmd = ('cd /tmp && export HOME=/data/home XDG_CACHE_HOME=/data/cache HF_HOME=/data/cache/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 '
       'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=%d TRITON_CACHE_DIR=/data/rs/triton%d; '
       'rm -rf /data/rs/e%d/out /data/rs/e%d/cand /data/rs/e%d/refcache; mkdir -p /data/rs/e%d/refcache; nohup /data/venvA/bin/python /data/rs/scripts/rs_replay_pod.py /data/rs/e%d/ff.json /data/rs/e%d/out /data/rs/e%d/cand '
       '> /data/rs/e%d/replay.log 2>&1 & echo $! > /data/rs/e%d/replay.pid' % ((gpu, gpu) + (E,) * 9))
pod(cmd); pid = pod('cat /data/rs/e%d/replay.pid' % E).stdout.decode().strip(); log('replay pid %s' % pid)
while pod('kill -0 %s 2>/dev/null' % pid).returncode == 0:
    time.sleep(30)
    if time.time() - t0 > 3 * 3600: result('error', reason='replay > 3 h'); sys.exit(1)
tail = pod('tail -c 1500 /data/rs/e%d/replay.log' % E).stdout.decode(); (W / 'replay.tail.txt').write_text(tail)
if 'STREAM_DONE' not in tail: result('error', reason='replay failed: ' + tail[-300:]); sys.exit(1)
p = pod('cd /tmp && /data/venvA/bin/python /data/rs/scripts/rs_compare_pod.py /data/ckpt/%s /data/rs/live-%s /data/rs/e%d/cand /data/rs/e%d/out /data/rs/e%d/live_traj.json /data/rs/e%d/compare.json' % (H, C, E, E, E, E), timeout=3600)
log('compare rc=%d %s %s' % (p.returncode, p.stdout.decode()[-400:], p.stderr.decode()[-300:]))
cj = pod('cat /data/rs/e%d/compare.json' % E).stdout.decode(); (W / 'compare.json').write_text(cj)
c = json.loads(cj)
result(c['verdict'], reason=c['reason'], candidate=C, gpu=gpu, weights=c['weights'], rollouts=c['rollouts'])
prog('verdict %s; max weight diff %s, bf16 update cos %s, mismatched %s elements; rollout loss max diff %s; outside-support live %s / replay %s'
     % (c['verdict'], c['weights']['candidate_max_abs_diff'], c['weights']['bf16_update_cos'], c['weights']['mismatched_elements'],
        c['rollouts'].get('rollout_loss_max_abs_diff'), c['rollouts'].get('support_outside_tokens_live'), c['rollouts'].get('support_outside_tokens_replay')))
