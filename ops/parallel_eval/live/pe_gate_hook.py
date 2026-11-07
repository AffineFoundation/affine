#!/usr/bin/env python3
"""Gate integration for parallel eval (wiring-v1, staged 2026-11-07; v1 = 2026-11-06 parallel cutover).
gate_service.process_epoch calls this for every epoch while PE is live:
  pe_owns(e)        -> True for every epoch >= parallel_eval.from_epoch (set at cutover) or with a CLAIM: the gate never
                       probes those candidates on its own probe pod.
  gate_side(S,e,f)  -> runs IN THE GATE PROCESS (single writer of gate/round state): for every undecided PE epoch
                       (a) computes the round's prior-v1 head baseline exactly as today (prior_gate.ensure_baseline on the gate's
                           own probe pod), (b) once the PE eval job has a full RESULT, writes DECISION.json with the SAME code
                           path/fields as prior_gate.process_epoch (+ a 'parallel_eval' block), turn_add, close round on GREEN,
                       (c) gave-up / never-judged -> the gate's own heldout_fail_closed (no_decision + K1 grant).
"""
import json, os, time
from pathlib import Path

FLAG = Path(os.environ.get("PE_FLAG") or "/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json")
_HERE = Path(os.environ.get("PE_HERE") or "/home/const/subnet120-rewrite/state/root-audits/20261106-parallel-cutover")
PE_ST = Path(os.environ.get("PE_ST") or (_HERE / "state"))
PE_JOBS = Path(os.environ.get("PE_JOBS") or (_HERE / "jobs"))
PE_LIFE = Path(os.environ.get("PE_LIFE_STATE") or (PE_ST / "life"))
REPROBE_MAX_PE = 1


def _cfg():
    try:
        return (json.loads(FLAG.read_text()).get("parallel_eval") or {})
    except Exception:
        return {}


def live():
    return _cfg().get("enabled") is True


def shadow():
    c = _cfg()
    return c.get("shadow") is True and not c.get("enabled")


def from_epoch():
    v = _cfg().get("from_epoch")
    return v if type(v) is int else None


def pe_owns(epoch):
    for name in ("live-e%d" % epoch,):
        st = PE_JOBS / name / "STATUS.json"
        if not st.exists():
            continue
        try:
            s = json.loads(st.read_text()).get("status")
        except Exception:
            continue
        if s in ("starting", "booting", "fetching", "serving", "probing", "done"):
            return True
    if not live():
        return False
    fe = from_epoch()
    return (PE_ST / ("CLAIM-e%d.json" % epoch)).exists() or (fe is not None and epoch >= fe)


def claim(epoch, candidate, head):
    PE_ST.mkdir(parents=True, exist_ok=True)
    p = PE_ST / ("CLAIM-e%d.json" % epoch)
    if p.exists():
        return json.loads(p.read_text())
    doc = dict(epoch=epoch, candidate=candidate, head=head, claimed_at=time.time())
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(p)
    return doc


def _rj(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return d


def _pending():
    out = {}
    for p in sorted((PE_LIFE / "pending").glob("e*.json")):
        d = _rj(p) or {}
        if type(d.get("epoch")) is int:
            out[d["epoch"]] = d
    return out


def gate_side(S, e, f):
    """Gate-process entry. Returns a status string; 'skip' when nothing occupies the gate probe pod (lets busy clear)."""
    pend = _pending()
    res = {}
    for pe_e in sorted(pend):
        if (S.EP / str(pe_e) / "DECISION.json").exists():
            continue
        try:
            res[pe_e] = decide_one(S, pe_e, pend[pe_e], f)
        except Exception as ex:
            res[pe_e] = "error: %s" % str(ex)[:200]
            S.log(event="pe_gate_side_error", epoch=pe_e, error=str(ex)[:300])
    if e in res:
        r = res[e]
    elif (S.EP / str(e) / "DECISION.json").exists():
        r = "decided"
    else:
        r = "pe-waiting"
    if any(str(v).startswith("prior-baseline") for v in res.values()):
        return r if str(r).startswith("prior-baseline") else "pe-baseline"   # probe pod busy with the head baseline
    return "decided" if r == "decided" else "skip"


def decide_one(S, e, pend, f):
    import prior_gate as PG
    import baseline_v1 as BV
    import probe_error_guard as PEG
    import heldout as HO
    d = S.EP / str(e); dp = d / "DECISION.json"
    s = S.rj(d / "SAMPLE.private.json")
    if not s:
        return "no-sample"
    C, H = pend["candidate"], pend["training_head"]
    cand = dict(id=C, files=(pend.get("checkpoint") or {}).get("files"), input=H)
    lt = S.rj(d / "LIFECYCLE-TIMEOUT.json")   # RA/20261107-timeout-promote guard kept for PE epochs: never decide (no late GREEN)
    if lt and s.get("heldout"):
        return S.heldout_fail_closed(e, s, dict(cand, id=lt.get("candidate") or C), f, "lifecycle decision timeout: %s" % lt.get("why"))
    if s.get("head") != H:
        return S.skip_red(e, s.get("head"), C, f, "PE candidate training head %s != sampled head %s" % (H[:12], str(s.get("head"))[:12]))
    R = S.rounds(); rid = s["round_id"]; r = R.get(rid)
    if not r or not PG.is_prior(r):
        return S.heldout_fail_closed(e, s, cand, f, "PE requires a prior-v1 held-out round (round %s)" % rid)
    pv = r["cfg"]; per_turn = pv["scoring_mode"] == "same-run"
    if per_turn:
        return S.heldout_fail_closed(e, s, cand, f, "PE does not support same-run scoring")
    # (a) head baseline: identical to today (one per round, gate probe pod; returns the cached per_task when ready).
    if R.closed(rid) and not R.baseline(rid):
        return S.heldout_fail_closed(e, s, cand, f, "round %s closed before its head baseline existed" % rid)
    bl = PG.ensure_baseline(S, e, r, f, per_turn)
    if not isinstance(bl, dict):
        age = time.time() - float(pend.get("at") or time.time())
        if age > float(HO.cfg(f).get("fail_closed_s") or 10 ** 9):
            return S.heldout_fail_closed(e, s, cand, f, "prior-v1 baseline not ready %ds after PE pending (%s)" % (int(age), bl))
        return "prior-" + str(bl)
    jd = PE_JOBS / ("live-e%d" % e)
    ctl = _rj(jd / "CTL.json") or {}
    if ctl.get("state") == "gave-up":
        return S.heldout_fail_closed(e, s, cand, f, "PE eval gave up after %d attempts (infra)" % len(ctl.get("attempts") or []))
    if ctl.get("state") not in ("done", "scored"):
        return "pe-eval-" + str(ctl.get("state") or "not-dispatched")
    job = _rj(jd / "JOB.json") or {}
    if job.get("candidate") != C or job.get("round_id") != rid:
        return S.heldout_fail_closed(e, s, cand, f, "PE job spec does not match pending candidate/round")
    res = _rj(jd / "RESULT.json") or {}
    pseed = BV.cand_seed(r["seed"], e)
    if str(job.get("plan_seed")) != str(pseed) or list(job.get("indices") or []) != list(r["indices"]) or int(job.get("k")) != int(pv["k_cand"]) \
            or int(job.get("max_tokens")) != int(PG._mt(pv)):
        return S.heldout_fail_closed(e, s, cand, f, "PE job is not the exact live replica (seed/indices/k/cap)")
    bfn = R.d(rid) / "BASELINE.private.json"
    b = S.rj(bfn) or {}
    if int(b.get("max_tokens") or 0) != PG._mt(pv):
        return S.heldout_fail_closed(e, s, cand, f, "round baseline cap %s != round cap %s" % (b.get("max_tokens"), PG._mt(pv)))
    _hok, _hsum = PEG.should_judge(b.get("per_task") or {}, side="head-baseline", epoch_dir=d, attempt=0)
    if not _hok:
        return "held-head-probe-errors"
    cpt_i = {int(k): v for k, v in (((res.get("envs") or {}).get(pv["probe_env"]) or {}).get("per_task") or {}).items()}
    cpt = PG._cells_by_task(cpt_i, pv)
    _cattempt = sum(1 for h in ((S.rj(d / "HELD.json") or {}).get("holds") or []) if h.get("side") == "candidate")
    _cok, _csum = PEG.should_judge(cpt_i, side="candidate", epoch_dir=d, attempt=_cattempt)
    if not _cok:
        if _cattempt < REPROBE_MAX_PE:
            S.wj(jd / "RERUN.json", dict(why="candidate probe errors", rate=_csum.get("rate"), at=time.time(), at_cr=S.cr()), 0o600)
            return "held-candidate-probe-errors(rerun)"
        return S.heldout_fail_closed(e, s, cand, f, "PE candidate probe error rate %.3f after rerun" % float(_csum.get("rate") or 0))
    PI = S.rj(R.d(rid) / "PI.private.json")
    drawn = {t: dict(env=PI["env"][t], pi=PI["pi"][t]) for t in r["ids"]}
    tau = float(pv["tau_run_separate"])
    zb = PG.z_bar_for(PG.pv_of(f)); zbar = float(zb["value"])
    dec = BV.estimate(drawn, bl, cpt, r["pool"]["per_env"], shares=pv["shares"], tau=tau, sigma_run_floor=float(pv["sigma_run_floor"]), bar_z=zbar,
                      z_inclusive=bool(pv["z_inclusive"]), cap_max=float(PG.pv_of(f).get("cap_max", BV.CAP_HIT_MAX)),
                      min_scored=int(pv["min_scored"]), min_cover=float(pv["min_cover"]))
    st = _rj(jd / "STATUS.json") or {}; meta = _rj(jd / "META.json") or {}
    S.wj(d / "C-PROBE.private.json", dict(candidate=C, per_task=cpt_i, wall=res.get("wall_seconds"), partial=res.get("partial"),
                                          plan_seed=pseed, k=int(pv["k_cand"]), heldout_round=rid, prior_v1=True, parallel_eval=True))
    S.wj(d / "H-PROBE.private.json", dict(head=H, per_task={str(BV.probe_index(t, pv)): v for t, v in bl.items()}, k=b.get("rollouts_per_task"),
                                          seeded=True, source="prior-v1-head-baseline", heldout_round=rid, plan_seeds=b.get("plan_seeds"),
                                          tasks=len(bl), baseline_done_at_cr=b.get("done_at_cr"), prior_v1=True))
    now = time.time(); t_res = float(pend.get("at") or now)
    dec.update(version=S.G.VERSION, epoch=e, head=H, candidate=C, mode=S.mode(f), decided_at=now, decided_at_cr=S.cr(),
               batch_tasks=None, early_staged=False, delta_rc=None,
               timing=dict(result_to_decision_s=round(now - t_res, 1), h_wait_s=0.0, fetch_s=st.get("fetch_s"), serve_s=st.get("serve_s"),
                           run_s=meta.get("wall_s")),
               red_credit=f.get("red_credit", "zero"), commitment=r["draw_commitment"], z_bar=zb,
               probe=dict(k=int(pv["k_cand"]), max_tokens=PG._mt(pv), seeded=True, common_random_numbers=False,
                          crn="none: candidate plan seed = sha256(seed:prior-baseline:cand:<epoch>), independent of the head blocks",
                          h_side="prior-v1 head baseline (%s rollouts/task on every drawn task, once per round)" % b.get("rollouts_per_task")),
               heldout=dict(round_id=rid, baseline="prior-v1", max_tokens=PG._mt(pv), set_size=r["n"], bar_z=zbar, z_inclusive=bool(pv["z_inclusive"]),
                            k_cand=int(pv["k_cand"]), head_rollouts=b.get("rollouts_per_task"), scoring_mode=pv["scoring_mode"], tau_run=tau,
                            sigma_run_floor=float(pv["sigma_run_floor"]), baseline_done_at_cr=b.get("done_at_cr"), scoring=BV.VERSION,
                            settings_hash=b.get("settings_hash")),
               parallel_eval=dict(job="live-e%d" % e, attempts=len(ctl.get("attempts") or []), gpu=(st.get("pod") or {}).get("gpu"),
                                  max_num_seqs=st.get("max_num_seqs"), samples=res.get("samples"), planned_samples=len(r["indices"]) * int(pv["k_cand"]),
                                  training_head=H, round_closed_before_decision=R.closed(rid)))
    BV.assert_public(dict(evaluated=dec["evaluated"], reasons=dec["reasons"], heldout=dec["heldout"]), True)
    S.wj(d / "EVALUATED.private.json", dict(evaluated=dec["evaluated"], drawn=len(drawn)), 0o600); S.wj(dp, dec, 0o600)
    R.turn_add(rid, dict(epoch=e, candidate=C, decision=dec["decision"], z=dec["evaluated"]["z"], plan_seed=pseed, batch=[],
                         evaluated=dec["evaluated"], reasons=dec["reasons"], decided_at=dec["decided_at"], prior_v1=True, parallel_eval=True))
    if dec["decision"] == "green" and not R.closed(rid):
        R.close(rid, "green e%d (candidate %s, parallel eval)" % (e, C[:12]), e)
    S.log(event="decision", epoch=e, decision=dec["decision"], mode=dec["mode"], z=round(dec["evaluated"]["z"], 3), heldout_round=rid,
          baseline="prior-v1", parallel_eval=True, n=dec["evaluated"]["n"], reasons=dec["reasons"])
    return "decided"

