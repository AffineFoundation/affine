import json, os, sys, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); PE = RA / '20261106-parallel-cutover'
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'); EP = RA / '20261026-promotion-gate/state/epochs'
def rj(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}
out = {}
for e in [int(x) for x in sys.argv[1:]]:
    plan = rj(LIFE / str(e) / 'PLAN.json'); m = plan.get('manifest') or {}; m = m.get('payload', m)
    start = m.get('start'); tr = LIFE / str(e) / 'TRAINING-RESULT.json'
    td = tr.stat().st_mtime if tr.exists() else None
    dec = rj(EP / str(e) / 'DECISION.json'); da = dec.get('decided_at')
    st = rj(PE / 'jobs' / ('live-e%d' % e) / 'STATUS.json'); meta = rj(PE / 'jobs' / ('live-e%d' % e) / 'META.json'); ctl = rj(PE / 'jobs' / ('live-e%d' % e) / 'CTL.json')
    a = (ctl.get('attempts') or [{}])[-1]
    out[e] = dict(decision=dec.get('decision'), z=(dec.get('evaluated') or {}).get('z'), gpu=(st.get('pod') or {}).get('gpu'),
                  warm=(st.get("warm") if isinstance(st.get("warm"), dict) else {"name": st.get("warm")}).get("name"), warm_fetched=(st.get("warm") if isinstance(st.get("warm"), dict) else {}).get("fetched"), warm_served=(st.get("warm") if isinstance(st.get("warm"), dict) else {}).get("served"),
                  attempts=len(ctl.get('attempts') or []), fetch_s=st.get('fetch_s'), serve_s=st.get('serve_s'), sampling_s=meta.get('wall_s'),
                  last_attempt_s=round(st['done_at'] - a['launched'], 1) if st.get('done_at') and a.get('launched') else None,
                  reg_to_train_done_min=round((td - start) / 60, 2) if td and start else None,
                  train_done_to_decision_min=round((da - td) / 60, 2) if da and td else None,
                  reg_to_decision_min=round((da - start) / 60, 2) if da and start else None)
print(json.dumps(out, indent=1))
