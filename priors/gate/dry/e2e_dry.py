#!/usr/bin/env python3
"""OFFLINE end-to-end dry run of the prior-v1 gate path (RA/20261101-baseline-switch). Copies only:
- gate code: patched COPIES in DRY/code (patch_gate.py apply), never the live code dir
- gate state / site / flag: fresh dirs under DRY (HG_GATE_STATE, HG_SITE, HG_FLAG), HG_NO_TRAINER=1
- probe: fakepod.py (HG_PROBE_SSH), synthetic grades from a prior-table SNAPSHOT (read-only sqlite backup)
- chain: the REAL finalized block hash (read-only JSON-RPC) for the commit-reveal seed
- lifecycle: past epochs' PLAN / TRAINING-RESULT / FINALIZED read from the live lifecycle dir (read-only)
Runs one round: open -> target block -> draw -> head baseline -> two turns (A/A candidate, +shift candidate) -> decisions ->
heldout publish (current.json / index.json / round reveal) -> gate_ledger gate.json -> privacy scan of every public file.
usage: e2e_dry.py DRY_DIR EPOCH_A EPOCH_B [SHIFT]"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
DRY = Path(sys.argv[1]); EA, EB = int(sys.argv[2]), int(sys.argv[3]); SHIFT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
LIVE_CODE = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
LIVE_FLAG = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json')
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
if DRY.exists(): shutil.rmtree(DRY)
for x in ('code', 'state', 'site', 'pod'): (DRY / x).mkdir(parents=True, mode=0o700)
for f in ('gate_service.py', 'heldout.py', 'gate_core.py', 'gate_ledger.py', 'training_params.py', 'presign.py'):
    shutil.copy2(LIVE_CODE / f, DRY / 'code' / f)
for f in ('prior_gate.py', 'baseline_v1.py'): shutil.copy2(HERE / f, DRY / 'code' / f)
subprocess.run([sys.executable, str(HERE / 'patch_gate.py'), 'apply', str(DRY / 'code')], check=True)
flag = json.load(open(LIVE_FLAG)); h = flag['heldout']
h.update(baseline='prior-v1', from_epoch=min(EA, EB) - 5,
         prior_v1=dict(N=500, k_cand=8, head_rollouts=12, head_blocks=3, scoring_mode='separate', block_delay=2,
                       table=os.environ.get('PG_TABLE', '/home/const/subnet120-rewrite/state/root-audits/20261101-prior-table/private/table/prior_table.private.sqlite')))
json.dump(flag, open(DRY / 'flag.json', 'w'), indent=1)
# synthetic truth for the fake pod: p_i = posterior mean of a read-only snapshot of the table (combined probe index)
sys.path.insert(0, str(DRY / 'code'))
import baseline_v1 as BV
snap = BV.snapshot_table(h['prior_v1']['table'], str(DRY / 'pod' / 'truth.private.sqlite'))
pool = [json.loads(l)['id'] for l in open('/home/const/subnet120-rewrite/state/root-audits/20261101-baseline-switch/public/gate-pool-v1.jsonl')]
rows = BV.load_rows(snap['path'], pool)
P = {str(BV.probe_index(r['task_id'])): (r['s'] + 1) / (r['n'] + 2) for r in rows}
Bf = {str(BV.probe_index(r['task_id'])): (min(1.0, r['b'] / max(1e-9, r['n'] - r['s'])) if r['n'] - r['s'] > 0 else 0.5) for r in rows}
json.dump(P, open(DRY / 'pod' / 'P.json', 'w')); json.dump(Bf, open(DRY / 'pod' / 'B.json', 'w'))
def cand(e):
    r = json.load(open(LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json')); rep = (r.get('payload') or r)['report']
    return rep['publication_receipt']['payload']['descriptor']['id'], rep.get('input_checkpoint')
CA, _ = cand(EA); CB, _ = cand(EB)
LG = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs')
HA = json.load(open(LG / str(EA) / 'SAMPLE.private.json'))['head']; HB = json.load(open(LG / str(EB) / 'SAMPLE.private.json'))['head']
assert HA == HB, 'pick two epochs with the same head'
os.environ.update(HG_GATE_STATE=str(DRY / 'state'), HG_SITE=str(DRY / 'site'), HG_FLAG=str(DRY / 'flag.json'), HG_NO_TRAINER='1',
                  HG_PROBE_SSH=json.dumps([sys.executable, str(HERE / 'fakepod.py')]), FAKEPOD_DIR=str(DRY / 'pod'),
                  FAKEPOD_P=str(DRY / 'pod' / 'P.json'), FAKEPOD_B=str(DRY / 'pod' / 'B.json'), FAKEPOD_SHIFT=json.dumps({CB[:16]: SHIFT}), FAKEPOD_CKPTS=json.dumps([HA, CA, CB]))
import gate_service as S
S.REPLAY_T0[EA] = time.time(); S.REPLAY_T0[EB] = time.time()
f = S.flag(); assert f['heldout']['baseline'] == 'prior-v1'
log = []
t0 = time.time()
while True:   # round open + target block + draw (real chain)
    s = S.ensure_sample(EA, HA, f)
    if s: break
    if time.time() - t0 > 300: raise SystemExit('draw did not complete in 300 s')
    time.sleep(6)
print('draw done in %.0f s: n=%d' % (time.time() - t0, len(s['indices'])))
(S.EP / str(EB)).mkdir(parents=True, exist_ok=True); s2 = S.ensure_sample(EB, HB, f)
for e in (EA, EB):
    for _ in range(60):
        st = S.process_epoch(e, f); log.append((e, st))
        if st == 'decided': break
    print('e%d -> %s' % (e, st))
S.heldout_publish()
import gate_ledger as GL
GL.EP = S.EP; GL.ST = S.ST; GL.LEDGER = S.ST / 'LEDGER.public.jsonl'; GL.REJECTED = S.ST / 'REJECTED.private.jsonl'; GL.SITE_OUT = DRY / 'site' / 'gate.json'
GL.SCORE_ALL_FLAG = DRY / 'none.json'
_pr = GL._prior_rule
GL._prior_rule = lambda: (lambda F: ('Promotion gate (held-out, ' + __import__('prior_gate').RULE + ')') if F else None)(True)
print('gate_ledger:', GL.tick())
# close the round to exercise the reveal path (as a head change would), then publish again
R = S.rounds(); cur = R.current()
if cur: R.close(cur['id'], 'dry run: closed to exercise the reveal', EB)
S.heldout_publish()
# training_params public block (mining.json heldout) from the patched module
import training_params as TP
TP.PG_FLAG = DRY / 'flag.json'; TP.HOME = DRY / 'fakehome'
(DRY / 'fakehome/subnet120/affine/website/gate/heldout').mkdir(parents=True)
shutil.copy2(DRY / 'site' / 'heldout' / 'index.json', DRY / 'fakehome/subnet120/affine/website/gate/heldout/index.json')
mj = TP.heldout_public(); json.dump(mj, open(DRY / 'site' / 'mining.heldout.json', 'w'), indent=1)
# privacy scan: no pi, no prior s/n, no table path in any public file
PI = None
for rd in (S.ST / 'heldout').glob('r2*'):
    PI = json.load(open(rd / 'PI.private.json'))
priv_vals = set()
for t, p in PI['pi'].items(): priv_vals.add(repr(p))
bad = []
pubs = sorted(p for p in (DRY / 'site').rglob('*.json'))
for p in pubs:
    s_ = p.read_text()
    for k in BV.PRIVATE_KEYS:
        if k in s_: bad.append((str(p), k))
    for tok in re.findall(r'\d\.\d{6,}', s_):
        if tok in priv_vals or any(v.startswith(tok) for v in list(priv_vals)[:0]): bad.append((str(p), 'pi value ' + tok))
    if 'prior_table' in s_ or '.sqlite' in s_: bad.append((str(p), 'table path'))
exact_hits = 0
for p in pubs:
    s_ = p.read_text()
    exact_hits += sum(1 for v in priv_vals if v in s_)
print('public files:', [str(p.relative_to(DRY)) for p in pubs])
print('privacy scan: %d key/path hits, %d exact pi-value hits' % (len(bad), exact_hits), bad[:5])
dec = {e: json.load(open(S.EP / str(e) / 'DECISION.json')) for e in (EA, EB)}
for e, d in dec.items():
    ev = d['evaluated']; print('e%d %s z %.3f gain %+.5f se %.5f (task %.5f run %.5f) n %d cap %.4f (pool %.4f) reasons %s' % (
        e, d['decision'], ev['z'], ev['gain'], ev['se'], ev['se_task'], ev['se_run'], ev['n'], ev['c_cap_rate'], ev['c_cap_rate_pool'], d['reasons']))
json.dump(dict(log=log, decisions=dec, privacy=dict(hits=bad, exact_pi_hits=exact_hits), public=[str(p) for p in pubs]),
          open(DRY / 'E2E-RESULT.private.json', 'w'), indent=1, default=str)
