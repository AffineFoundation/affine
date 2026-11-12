#!/usr/bin/env python3
"""Score a PE probe RESULT with EXACTLY the live prior-v1 path (prior_gate.process_epoch math), host-only.
Same inputs as live: round r (ids, PI, pool, cfg), round head BASELINE per_task, candidate per_task via PG._cells_by_task,
tau by scoring_mode, z bar via PG.z_bar_for(PG.pv_of(flag)), cap_max from the live flag (PG.pv_of(flag)), min_scored/min_cover
from the round cfg; probe-error guard via PEG.cell_error_rate (read-only: never writes into live epoch dirs)."""
import json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
sys.path.insert(0, str(GATE_CODE))
import baseline_v1 as BV
import prior_gate as PG
import probe_error_guard as PEG

_PIT_FILE = Path('ops') / 'parallel_eval' / 'pit' / 'shadow_compare.py'


def _repo_root():
    """The checkout that holds the PIT package (ops/parallel_eval/pit).

    This file runs from two places: ``<repo>/ops/parallel_eval/live`` (git) and the code directory of a root audit
    on the host (``<repo>/state/root-audits/<audit>/code``). The number of directories between this file and the
    repo is not the same in the two places. Thus the search does not count directories. The order is: the checkout
    that ``AFFINE_REPO`` names, each directory above this file, then ``REPO`` of pe_common.
    """
    here = Path(__file__).resolve()
    named = [Path(value) for value in (os.environ.get('AFFINE_REPO'),) if value]
    looked = named + list(here.parents) + [REPO]
    for base in looked:
        if (base / _PIT_FILE).is_file():
            return base
    raise ImportError('%s is not in AFFINE_REPO, not above %s and not in %s' % (_PIT_FILE, here.parent, REPO))


try:
    from ops.parallel_eval.pit.shadow_compare import evaluate_shadow_pass
except ImportError:  # script-mode / RA code dir without package layout
    _REPO = _repo_root()
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from ops.parallel_eval.pit.shadow_compare import evaluate_shadow_pass


class Pending(Exception):
    pass


def round_baseline(rid, epoch=None, per_turn=False):
    bd = HELDOUT / rid
    b = rj(bd / ('BASELINE%s.private.json' % (('-e%d' % epoch) if per_turn else ''))) or {}
    return b


def _per_turn(pv):
    return pv.get('scoring_mode') == 'same-run'


def scoring_baseline(epoch, rid):
    """The baseline that score() judges the candidate of this epoch against: the baseline of the round, or the
    baseline of the turn when the round scores each turn against its own head run. {} when there is no round.
    The PIT host reads the hardware and layout fingerprint from it before score() runs."""
    r = rj(HELDOUT / rid / 'ROUND.private.json') or {}
    if not r:
        return {}
    return round_baseline(rid, int(epoch), _per_turn(r['cfg']))


def score(epoch, rid, result, flag=None):
    """-> decision dict (BV.estimate output + meta). Raises Pending if the round baseline is not ready."""
    f = flag if flag is not None else (rj(FLAG) or {})
    r = rj(HELDOUT / rid / 'ROUND.private.json') or {}
    if not r:
        raise RuntimeError('round %s missing' % rid)
    pv = r['cfg']
    per_turn = _per_turn(pv)
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


def score_epoch_shadow(epoch, result_path, out_path=None, report_path=None, live_report_path=None):
    """SCORE-vs-live. Pass criterion is E1041 noise-tolerant (not equal hashes/z).

    Equal candidate hashes / equal z are NOT required. See
    ``ops.parallel_eval.pit.shadow_compare.evaluate_shadow_pass``.
    """
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
               plan_seed=dec.get('plan_seed'), scored_at_cr=cr(),
               train_gpu=res.get('train_gpu'), eval_gpu=res.get('eval_gpu'),
               live_train_gpu=live.get('train_gpu'), live_eval_gpu=live.get('eval_gpu'))
    out['z_delta'] = (out['pe_z'] - out['live_z']) if (out['pe_z'] is not None and out['live_z'] is not None) else None
    # Legacy label match (kept for charts); pass/fail uses shadow_pass below.
    out['decision_label_match'] = out['pe_decision'] == out['live_decision']

    box_report = rj(report_path) if report_path else (res.get('training_report') or {})
    live_report = rj(live_report_path) if live_report_path else (live.get('training_report') or {})
    box_pt = (((res.get('envs') or {}).get(dec.get('probe_env') or 'affine_gatepool') or {}).get('per_task')) or {}
    live_pt = live.get('per_task') or {}
    try:
        verdict = evaluate_shadow_pass(
            box_report=box_report or {},
            live_report=live_report or {},
            box_meta=dict(round_id=rid, plan_seed=dec.get('plan_seed'), heldout=dict(
                round_id=rid, settings_hash=dec.get('settings_hash'),
            )),
            live_meta=dict(
                round_id=rid,
                plan_seed=live.get('plan_seed') or (live.get('heldout') or {}).get('plan_seed'),
                heldout=live.get('heldout') or {},
            ),
            box_per_task=box_pt if isinstance(box_pt, dict) else {},
            live_per_task=live_pt if isinstance(live_pt, dict) else {},
            box_z=out['pe_z'],
            live_z=out['live_z'],
            train_gpu=out.get('train_gpu'),
            eval_gpu=out.get('eval_gpu'),
            live_train_gpu=out.get('live_train_gpu'),
            live_eval_gpu=out.get('live_eval_gpu'),
        )
    except Exception as exc:
        verdict = dict(passed=False, error=str(exc)[:200], pass_criterion='e1041-noise-tolerant-v1')
    out['shadow_pass'] = verdict
    out['agree'] = bool(verdict.get('passed'))  # noise-tolerant pass, not equal hashes/z
    if out_path:
        wj(out_path, out)
    return out


if __name__ == '__main__':
    print(json.dumps(score_epoch_shadow(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), indent=1, default=str))
