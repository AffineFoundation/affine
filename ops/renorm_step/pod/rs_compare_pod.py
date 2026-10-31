#!/usr/bin/env python3
"""8-vs-1 compare (SHADOW pod, CPU only): live 8-GPU candidate vs 1-GPU replay candidate, both from head H; per-rollout details vs the live
trainer report. usage: rs_compare_pod.py HEAD_DIR LIVE_DIR REPLAY_DIR REPLAY_OUT LIVE_TRAJ_JSON OUT_JSON"""
import json, math, sys
from pathlib import Path
import torch
from safetensors import safe_open
head, live, rep, rout, ltraj, outp = map(Path, sys.argv[1:7])
idx = json.loads((head / 'model.safetensors.index.json').read_text())['weight_map']
files = sorted(set(idx.values()))
st = dict(tensors=0, elements=0, mismatched=0, max_abs=0.0, worst=None, changed_live=0, changed_rep=0, d2l=0.0, d2r=0.0, dot=0.0, diff2=0.0)
per = []
for f in files:
    with safe_open(str(head / f), 'pt') as H, safe_open(str(live / f), 'pt') as L, safe_open(str(rep / f), 'pt') as R:
        for k in H.keys():
            h, l, r = H.get_tensor(k), L.get_tensor(k), R.get_tensor(k)
            if l.shape != h.shape or r.shape != h.shape: raise SystemExit('shape ' + k)
            st['tensors'] += 1; st['elements'] += h.numel()
            if torch.equal(l, h) and torch.equal(r, h): continue
            ne = int((l != r).sum()); st['mismatched'] += ne
            dl = l.double() - h.double(); dr = r.double() - h.double()
            st['changed_live'] += int((dl != 0).sum()); st['changed_rep'] += int((dr != 0).sum())
            a = float((l.double() - r.double()).abs().max()) if ne else 0.0
            if a > st['max_abs']: st['max_abs'], st['worst'] = a, k
            nl, nr, dt = float(dl.pow(2).sum()), float(dr.pow(2).sum()), float((dl * dr).sum())
            st['d2l'] += nl; st['d2r'] += nr; st['dot'] += dt; st['diff2'] += float((dl - dr).pow(2).sum())
            per.append(dict(t=k, mismatched=ne, cos=dt / math.sqrt(nl * nr) if nl and nr else None, norm_live=math.sqrt(nl), norm_rep=math.sqrt(nr)))
cos = st['dot'] / math.sqrt(st['d2l'] * st['d2r']) if st['d2l'] and st['d2r'] else None
w = dict(candidate_max_abs_diff=st['max_abs'], worst_tensor=st['worst'], mismatched_elements=st['mismatched'], elements=st['elements'],
         changed_elements_live=st['changed_live'], changed_elements_replay=st['changed_rep'],
         mismatched_over_changed=st['mismatched'] / max(st['changed_live'], 1),
         bf16_update_norm_live=math.sqrt(st['d2l']), bf16_update_norm_replay=math.sqrt(st['d2r']), bf16_update_cos=cos,
         bf16_update_rel_diff=math.sqrt(st['diff2'] / st['d2l']) if st['d2l'] else None,
         worst_tensor_cos=sorted([p for p in per if p['cos'] is not None], key=lambda p: p['cos'])[:5])
D = json.loads((rout / 'details.json').read_text()); rows = D['rows']; L = json.loads(ltraj.read_text())
lt = L['trajectories']
r = dict(rows_live=len(lt), rows_replay=len(rows))
if len(lt) == len(rows):
    tokmis = sum(1 for a, b in zip(lt, rows) if a.get('loss_tokens') != b.get('loss_tokens') or a.get('tokens') != b.get('tokens'))
    dl = [abs(a['loss'] - b['loss']) for a, b in zip(lt, rows)]
    so = [abs((a.get('support_outside_tokens') or 0) - (b.get('support_outside_tokens') or 0)) for a, b in zip(lt, rows)]
    lq = [abs(a['support_logq_mean'] - b['support_logq_mean']) for a, b in zip(lt, rows) if a.get('support_logq_mean') is not None and b.get('support_logq_mean') is not None]
    r.update(token_mismatch_rows=tokmis, rollout_loss_max_abs_diff=max(dl), rollout_loss_mean_abs=sum(dl) / len(dl),
             support_outside_tokens_live=sum(a.get('support_outside_tokens') or 0 for a in lt), support_outside_tokens_replay=sum(b.get('support_outside_tokens') or 0 for b in rows),
             support_outside_rows_differing=sum(1 for x in so if x), support_outside_max_abs_row_diff=max(so),
             support_logq_mean_max_abs_diff=max(lq) if lq else None)
r.update(objective_live=L.get('objective_loss'), objective_replay=D['info'].get('objective'), preclip_live=L.get('preclip_gradient_norm'), preclip_replay=D['info'].get('preclip'))
steps = [json.loads(x) for x in (rout / 'steps.jsonl').read_text().splitlines() if x.strip()]
s = steps[-1] if steps else {}
r.update(raw_update_norm_live=L.get('raw_update_norm'), raw_update_norm_replay=s.get('raw_update_norm'),
         tensors_rel_capped_live=L.get('tensors_rel_capped'), tensors_rel_capped_replay=s.get('tensors_rel_capped'))
reasons = []
if r['rows_live'] != r['rows_replay']: reasons.append('row count')
else:
    if r['token_mismatch_rows']: reasons.append('%d rows with different tokens' % r['token_mismatch_rows'])
    if r['rollout_loss_max_abs_diff'] > 1e-3: reasons.append('rollout loss diff %.3g' % r['rollout_loss_max_abs_diff'])
    if abs(r['support_outside_tokens_live'] - r['support_outside_tokens_replay']) > max(2, 0.01 * max(r['support_outside_tokens_live'], 1)): reasons.append('outside-support count')
    if r['support_logq_mean_max_abs_diff'] is not None and r['support_logq_mean_max_abs_diff'] > 1e-3: reasons.append('support_logq_mean diff')
if r['preclip_live'] and abs(r['preclip_replay'] / r['preclip_live'] - 1) > 0.05: reasons.append('preclip norm off >5%')
if cos is None or cos < 0.95: reasons.append('bf16 update cos %s' % cos)
if w['mismatched_over_changed'] > 0.10: reasons.append('mismatched/changed %.3f' % w['mismatched_over_changed'])
res = dict(version='rs-8v1-v1', weights=w, rollouts=r, verdict='match' if not reasons else 'mismatch', reason='; '.join(reasons) or None)
outp.write_text(json.dumps(res, indent=1, default=str)); print(json.dumps(dict(verdict=res['verdict'], reason=res['reason'], max_abs=w['candidate_max_abs_diff'], cos=cos, mis=w['mismatched_elements'], loss=r.get('rollout_loss_max_abs_diff'))))
