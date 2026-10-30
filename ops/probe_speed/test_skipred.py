import sys, json, tempfile, time, os
from pathlib import Path
sys.path.insert(0, '/tmp/gsr_test')
import gate_service as GS, heldout as HO
tmp = Path(tempfile.mkdtemp()); GS.EP = tmp / 'epochs'; (GS.EP / '300').mkdir(parents=True); (GS.EP / '301').mkdir(parents=True)
LOG = []; GS.log = lambda **k: LOG.append(k); GS.clear_own_busy = lambda: None
GS.LIFE = tmp / 'life'; (GS.LIFE / 'epochs' / '301').mkdir(parents=True); (GS.LIFE / 'epochs' / '301' / 'TRAINING-RESULT.json').write_text('{}')
f = json.load(open('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/PROMOTION-GATE.json')) if os.path.exists('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/PROMOTION-GATE.json') else {}
f = dict(GS.flag() or {}, **f)
R = HO.Rounds(tmp / 'heldout', GS.wj, GS.rj, lambda **k: LOG.append(k), GS.cr); GS.ROUNDS = R
import gate_core as G
H = 'a' * 64; C = 'b' * 64
r = R.open(G, f, HO.cfg(f), H, 300, 'test'); rid = r['id']
(GS.EP / '300' / 'COMMIT.json').write_text(json.dumps(dict(commitment=r['commitment'])))
s = dict(head=H, round_id=rid, heldout=True)
# 1) closed round
R.close(rid, 'test close', 300)
GS.candidate_of = lambda e: dict(id=C, files={}, input=H)
out = GS.process_heldout_epoch(300, s, f); dec = json.load(open(GS.EP / '300' / 'DECISION.json'))
print('closed-round ->', out, dec['decision'], dec['reasons'], dec['heldout'])
assert out == 'decided' and dec['decision'] == 'red' and dec['head'] == H and dec['candidate'] == C and dec['commitment'] == r['commitment']
# lifecycle hook acceptance (v829i): head==H, candidate==C, decision in green/red
assert dec.get('head') == H and dec.get('candidate') == C and dec.get('decision') in ('green', 'red')
rs = ' '.join(dec['reasons']).lower(); assert not ('fail' in rs and 'closed' in rs), 'would trip fp32v-watchdog'
# 2) head mismatch in held-out path (open round, baseline ready)
r2 = R.open(G, f, HO.cfg(f), H, 301, 'test2'); s2 = dict(head=H, round_id=r2['id'], heldout=True)
GS.ensure_round_baseline = lambda e, r, f: {1: dict(n=4, c=2)}
GS.candidate_of = lambda e: dict(id=C, files={}, input='c' * 64)
out = GS.process_heldout_epoch(301, s2, f); dec = json.load(open(GS.EP / '301' / 'DECISION.json'))
print('head-mismatch ->', out, dec['decision'], dec['reasons'])
assert out == 'decided' and dec['decision'] == 'red' and dec['head'] == H
print('red schema keys', sorted(dec)); print('events', [k['event'] for k in LOG if 'event' in k]); print('ALL OK')
# 3) delta waits for TRAINING-RESULT when delta_after_result is on
started = []; GS.start_delta = lambda e, C, H, tl: (started.append(e), tl.__setitem__('delta_start', time.time()))
GS.opts = lambda: dict(delta=True, delta_after_result=True, delta_timeout=240)
tl = dict(head=H, early=True); st = dict(fetching=False, ckpts=[H])
(GS.LIFE / 'epochs' / '302').mkdir(parents=True)
r1 = GS.delta_stage(302, C, tl, st); assert r1 == 'delta-await-result' and not started, r1
(GS.LIFE / 'epochs' / '302' / 'TRAINING-RESULT.json').write_text('{}')
r2 = GS.delta_stage(302, C, tl, st); assert r2 == 'fetching-delta' and started == [302], r2
GS.opts = lambda: dict(delta=True, delta_after_result=False, delta_timeout=240)
tl2 = dict(head=H, early=True); (GS.LIFE / 'epochs' / '303').mkdir(parents=True)
assert GS.delta_stage(303, C, tl2, st) == 'fetching-delta'
print('delta OK')
