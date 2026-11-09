#!/usr/bin/env python3
"""Score a PE probe RESULT with EXACTLY the live prior-v1 path (prior_gate.process_epoch math), host-only.
Same inputs as live: round r (ids, PI, pool, cfg), round head BASELINE per_task, candidate per_task via PG._cells_by_task,
tau by scoring_mode, z bar via PG.z_bar_for(PG.pv_of(flag)), cap_max from the live flag (PG.pv_of(flag)), min_scored/min_cover
from the round cfg; probe-error guard via PEG.cell_error_rate (read-only: never writes into live epoch dirs)."""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
sys.path.insert(0, str(GATE_CODE))
import baseline_v1 as BV
import prior_gate as PG
import probe_error_guard as PEG


class Pending(Exception):
    pass


def round_baseline(rid, epoch=None, per_turn=False):
    bd = HELDOUT / rid
    b = rj(bd / ('BASELINE%s.private.json' % (('-e%d' % epoch) if per_turn else ''))) or {}
    return b


def score(epoch, rid, result, flag=None):
    """-> decision dict (BV.estimate output + meta). Raises Pending if the round baseline is not ready."""
    f = flag if flag is not None else (rj(FLAG) or {})
    r = rj(HELDOUT / rid / 'ROUND.private.json') or {}
    if not r:
        raise RuntimeError('round %s missing' % rid)
    pv = r['cfg']
    per_turn = pv.get('scoring_mode') == 'same-run'
    b = round_baseline(rid, epoch, per_turn)
    bl = b.get('per_task')
    if not bl:
        raise Pending('baseline for round %s not ready' % rid)
    mt = PG._mt(pv)
    if int(b.get('max_tokens') or 0) != mt:
        raise RuntimeError('baseline cap %s != round cap %s' % (b.get('max_tokens'), mt))
    env = pv['probe_env']
    cpt_i = {int(k): v for k, v in (((result.get('envs') or {}).get(env) or {}).get('per_task') or {}).items()}
    cpt = PG._cells_by_task(cpt_i, pv)
    err = PEG.cell_error_rate(cpt_i)
    if err['n'] == 0 or err['rate'] > PEG.THRESHOLD:
        return dict(decision=None, guard='candidate-probe-errors', guard_summary=err)
    PI = rj(HELDOUT / rid / 'PI.private.json')
    drawn = {t: dict(env=PI['env'][t], pi=PI['pi'][t]) for t in r['ids']}
    tau = float(pv['tau_run_same'] if per_turn else pv['tau_run_separate'])
    zb = PG.z_bar_for(PG.pv_of(f)); zbar = float(zb['value'])
    dec = BV.estimate(drawn, bl, cpt, r['pool']['per_env'], shares=pv['shares'], tau=tau, sigma_run_floor=float(pv['sigma_run_floor']),
                      bar_z=zbar, z_inclusive=bool(pv['z_inclusive']), cap_max=float(PG.pv_of(f).get('cap_max', BV.CAP_HIT_MAX)),
                      min_scored=int(pv['min_scored']), min_cover=float(pv['min_cover']))
    dec.update(z_bar=zb, round_id=rid, baseline_done_at_cr=b.get('done_at_cr'), head_rollouts=b.get('rollouts_per_task'),
               settings_hash=b.get('settings_hash'), max_tokens=mt, k_cand=int(pv['k_cand']), set_size=r.get('n'),
               plan_seed=BV.cand_seed(r['seed'], epoch), tau_run=tau, scoring=BV.VERSION, cand_tasks=len(cpt), guard_summary=err)
    return dec


def score_epoch_shadow(epoch, result_path, out_path=None):
    e = int(epoch)
    d = EP / str(e)
    live = rj(d / 'DECISION.json') or {}
    sm = rj(d / 'SAMPLE.private.json') or {}
    rid = sm.get('round_id') or (live.get('heldout') or {}).get('round_id')
    res = rj(result_path) or {}
    dec = score(e, rid, res)
    ev_ = dec.get('evaluated') or {}
    lev = live.get('evaluated') or {}
    out = dict(epoch=e, round_id=rid, candidate=live.get('candidate'), head=live.get('head'),
               pe_decision=dec.get('decision'), pe_reasons=dec.get('reasons'), pe_evaluated=ev_,
               live_decision=live.get('decision'), live_reasons=live.get('reasons'),
               live_z=lev.get('z'), pe_z=ev_.get('z'), live_gain=lev.get('gain'), pe_gain=ev_.get('gain'),
               live_n=lev.get('n'), pe_n=ev_.get('n'), live_cap=lev.get('c_cap_rate'), pe_cap=ev_.get('c_cap_rate'),
               live_c_pass=lev.get('c_pass'), pe_c_pass=ev_.get('c_pass'),
               partial=bool(res.get('partial')), samples=res.get('samples'), planned_samples=res.get('planned_samples'),
               plan_seed=dec.get('plan_seed'), scored_at_cr=cr())
    out['z_delta'] = (out['pe_z'] - out['live_z']) if (out['pe_z'] is not None and out['live_z'] is not None) else None
    out['agree'] = out['pe_decision'] == out['live_decision']
    if out_path:
        wj(out_path, out)
    return out


if __name__ == '__main__':
    print(json.dumps(score_epoch_shadow(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), indent=1, default=str))
