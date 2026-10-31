import json, sys, types, copy
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261030-free-file/release')
import ff_lifecycle as FL
G = json.load(open(FL.GRANTS)); G['enabled'] = True
FL.rj_orig = FL.rj
FL.rj = lambda p: G if str(p) == str(FL.GRANTS) else FL.rj_orig(p)
turns = [json.loads(x) for x in open('/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/state/K1-TURNS.jsonl') if x.strip()]
chain = json.load(open(FL.L_K1_CHAIN))
def served_hk(ts): return {t['hotkey'] for t in ts}
def served_regs(ts): return {(int(t['uid']), int(t.get('reg_block') or 0)) for t in ts if t.get('reg_block')}
FL.L = types.SimpleNamespace(k1_served_hotkeys=served_hk, k1_served_registrations=served_regs)
q = FL.queue(chain, turns)
print('first 12:', [(p, u, r['hotkey'][:8], r.get('grant')) for p, u, r in q[:12]])
assert [p for p, *_ in q[:10]] == list(range(-10, 0)), 'positions'
# a real later turn for the first granted hotkey consumes its grant
hk = q[0][2]['hotkey']
q2 = FL.queue(chain, turns + [dict(epoch_index=999, hotkey=hk, uid=q[0][1], reg_block=1)])
assert hk not in [r['hotkey'] for p, u, r in q2 if p < 0] and len([1 for p, *_ in q2 if p < 0]) == 9
# disabled -> exact old queue
G['enabled'] = False
q3 = FL.queue(chain, turns); assert all(p >= 0 for p, *_ in q3) and q3 == q[10:] or print('WARN normal part differs')
print('normal queue unchanged:', [x[:2] for x in q3[:3]] == [x[:2] for x in q[10:13]])
G['enabled'] = True
# ff_service eligible: grant rows' miner_hex present
print('grant identities:', sum(1 for _p, _u, r in FL.grant_rows(chain, turns) if r.get('miner_hex')))
print('OK')
