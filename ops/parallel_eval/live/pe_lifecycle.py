#!/usr/bin/env python3
"""STAGED, NOT INSTALLED. PE lifecycle wrapper (non-blocking trainer) — see LIVE-INTEGRATION.md.
install_pe(Lifecycle) wraps the v829i-wrapped Lifecycle._training (outermost). Inert unless parallel_eval.enabled.
PE live per epoch e (head H = current checkpoint):
  1. restore guard: before a new TRAINING-JOB, the pod trainer must be on H (PE state TRAINER-AT.json == H); else request a
     restore (trainer_restore.sh H e, run by the PE controller) and wait up to restore_wait_s, then no-update.
  2. run the inner (un-held) training; on 'updated' C trained from H -> PENDING record (C, H, checkpoint, optimizer_step,
     authorization) for the PE controller to dispatch; head stays H (no-update 'pe-pending-e<e>-<C>-kept-<H>').
  3. a GREEN ready to apply on base H (APPLY-READY, green-time order, staleness<=2) replaces this epoch's result:
     'updated' -> that checkpoint (fresh: the candidate's own signed authorization, input_checkpoint == H).
     fresh-only phase 1 (Jacob 2026-11-07 07:27 CR): stale GREENs are merge_blocked by the controller (credited, never offered).
"""
import json, os, time
from pathlib import Path

PE_STATE = Path(os.environ.get("PE_LIFE_STATE", "/home/const/subnet120-rewrite/state/root-audits/20261106-parallel-cutover/state/life"))
FLAG = Path(os.environ.get("PE_FLAG", "/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json"))
RESTORE_WAIT_S = 1200
PIT_MAX_INFLIGHT = 5
PIT_DECISION_TIMEOUT_S = int(os.environ.get("PIT_DECISION_TIMEOUT_S", "21000"))


def _rj(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return d


def _wj(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name("." + p.name + ".%d.tmp" % os.getpid()); t.write_text(json.dumps(o, indent=1, sort_keys=True, default=str) + "\n"); os.chmod(t, 0o600); os.replace(t, p)


def pe_live():
    return ((_rj(FLAG, {}) or {}).get("parallel_eval") or {}).get("enabled") is True


def from_epoch():
    v = ((_rj(FLAG, {}) or {}).get("parallel_eval") or {}).get("from_epoch")
    return v if type(v) is int else None


def pit_cutover():
    """PIT live cutover: skip trainer dequeue; finalize via PIT DECISION / ready_apply.

    Confirmed against live ff_launcher_ntp + mimo_v829_train_gate_launcher: without a
    TRAINING-RESULT the serial gate path stalls. Under cutover we must not call the
    trainer ``inner``; instead wait for PIT DECISION.json / GREENS apply, then
    no-update (RED/no_decision) or offer the GREEN apply so CURRENT/weights advance.
    Nonces stay with the weight transaction path (untouched). Never all-zero weights:
    last-10-GREEN set remains non-empty as long as HEAD-CHAIN has GREENs.
    """
    pe = (_rj(FLAG, {}) or {}).get("parallel_eval") or {}
    pit = pe.get("pit") or {}
    if pit.get("revert") is True:
        return False
    return (
        pe.get("enabled") is True
        and pit.get("enabled") is True
        and pit.get("mode") == "live"
        and pit.get("cutover") is True
    )


def _gate_ep():
    return Path(os.environ.get(
        "PE_GATE_EP",
        "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs",
    ))


def _pit_decision(epoch):
    return _rj(_gate_ep() / str(epoch) / "DECISION.json") or {}


def _pit_roots():
    roots = []
    for env in ("PE_LIFE_STATE", "PE_STATE", "PIT_STATE"):
        v = os.environ.get(env)
        if v:
            roots.append(Path(v))
    return roots


def _pit_claim(epoch, directory=None):
    name = "PIT-CLAIM-e%d.json" % int(epoch)
    if directory is not None:
        p = Path(directory) / name
        if p.exists():
            return _rj(p)
    for root in _pit_roots():
        p = root / name
        if p.exists():
            return _rj(p)
    return None


def _pit_inflight_count():
    """Open PIT claims: written claim, no terminal DECISION yet."""
    open_epochs = []
    dedup = set()
    for root in _pit_roots():
        for p in root.glob("PIT-CLAIM-e*.json"):
            try:
                e = int(p.stem.split("-")[-1][1:])
            except Exception:
                continue
            if e in dedup:
                continue
            dedup.add(e)
            dec = _pit_decision(e)
            if dec.get("decision") or dec.get("no_decision") or dec.get("chart_status") == "no_decision":
                continue
            open_epochs.append(e)
    return len(open_epochs)


def _write_pit_claim(epoch, H, directory):
    doc = dict(epoch=int(epoch), training_head=H, source="pit", at=time.time())
    name = "PIT-CLAIM-e%d.json" % int(epoch)
    d = Path(directory)
    _wj(d / name, doc)
    life = Path(os.environ.get("PE_LIFE_STATE", str(PE_STATE)))
    life.mkdir(parents=True, exist_ok=True)
    _wj(life / name, doc)
    pe = os.environ.get("PE_STATE")
    if pe:
        _wj(Path(pe) / name, doc)
    pit = os.environ.get("PIT_STATE")
    if pit:
        _wj(Path(pit) / name, doc)
    pp = life / "pending" / ("e%d.json" % int(epoch))
    if not pp.exists():
        _wj(pp, dict(epoch=int(epoch), training_head=H, source="pit", pit_claimed=True, at=doc["at"]))


def _pit_promotion_marker(e, H, dec, *, reason):
    return dict(
        version="sn120-promotion-gate-pit", epoch_index=e, head=H,
        candidate=dec.get("candidate"), decision=dec.get("decision") or "no_decision",
        no_decision=bool(dec.get("no_decision")), red_credit=dec.get("red_credit", "zero"),
        merge_blocked=bool(dec.get("merge_blocked")), source="pit", reason=reason,
    )


def _pit_epoch_finalized(directory):
    env = _rj(Path(directory) / "FINALIZED.json")
    if not env:
        return False
    final = env.get("payload") or env
    res = final.get("result") or {}
    reason = str(res.get("reason") or "")
    return res.get("status") in ("no-update", "updated") and reason.startswith("pit-")


def closure_var(fn, name):
    for n, c in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if n == name:
            return c.cell_contents
    raise LookupError("no closure var %s in %s" % (name, fn))


def _no_update(current, reason):
    return dict(status="no-update", checkpoint=current["checkpoint"], optimizer_step=current["optimizer_step"],
                checkpoint_authorization=current.get("checkpoint_authorization"), reason=reason)


def ready_apply(H, epoch=None):
    """Oldest GREEN on H. An offered record is replayable ONLY in its original epoch.

    Offering is not consumption: lifecycle FINALIZED and CURRENT must both be
    durable before confirm_applied can acknowledge it.
    """
    rows = []
    for p in sorted((PE_STATE / "apply").glob("*.json")):
        a = _rj(p) or {}
        eligible = a.get("state") == "ready" or (
            a.get("state") == "offered" and a.get("offered_epoch") == epoch)
        if eligible and a.get("base_head") == H and a.get("checkpoint") and a.get("checkpoint_authorization"):
            rows.append((a.get("green_at", 0), p, a))
    # An epoch that already offered a particular GREEN must not select a
    # different, newly arriving GREEN after a process crash/retry.
    offered = [r for r in rows if r[2].get("state") == "offered"]
    if len(offered) > 1:
        raise RuntimeError("multiple PE applies offered in epoch %s" % epoch)
    return (offered or sorted(rows, key=lambda x: (x[0], str(x[1]))))[0][1:] if rows else (None, None)


def confirm_applied(root, current):
    """Acknowledge ONLY a finalized result that is also the durable CURRENT head.

    Crash before CURRENT: offered record remains replayable. Crash after CURRENT
    but before acknowledgment: a later tick reconciles it. No journal/head-chain
    append is made here; the apply consumer must idempotently reconcile this ACK.
    """
    confirmed = []
    for p in sorted((PE_STATE / "apply").glob("*.json")):
        a = _rj(p) or {}
        if a.get("state") != "offered":
            continue
        e = a["offered_epoch"]
        env = _rj(Path(root) / "epochs" / str(e) / "FINALIZED.json") or {}
        final = env.get("payload") or env
        result = final.get("result") or {}
        target = (a.get("checkpoint") or {}).get("id")
        if (current.get("epoch_index", -1) <= e or
                (current.get("checkpoint") or {}).get("id") != target or
                final.get("input_checkpoint") != a.get("base_head") or
                result.get("status") != "updated" or result.get("checkpoint") != a.get("checkpoint") or
                result.get("optimizer_step") != a.get("optimizer_step") or
                result.get("checkpoint_authorization") != a.get("checkpoint_authorization")):
            continue
        a.update(state="consumed", consumed_epoch=e, consumed_at=time.time(),
                 observed_current_epoch=current["epoch_index"], observed_current_head=target)
        _wj(p, a); confirmed.append(str(p))
    return confirmed


def install_pe(Lifecycle):
    if getattr(Lifecycle._training, "_pe", False):
        return dict(pe=True, already=True, flag=str(FLAG), live_now=pe_live())
    outer = Lifecycle._training
    try:
        inner = closure_var(outer, "prior")   # v829i's wrapped function (training without the gate hold)
    except LookupError:
        # RA/20261107-timeout-promote ff_launcher_ntp: outer = ntp wrapper (closure 'inner' = v829i wrapper). PE off -> outer
        # (ntp: timeout = no_decision, never promotes). PE on -> v829i's un-held prior; no gate hold, so no timeout path at all.
        inner = closure_var(closure_var(outer, "inner"), "prior")

    def _training(self, directory, opening, dispatch, current, now):
        if not pe_live():
            return outer(self, directory, opening, dispatch, current, now)
        d = Path(directory); e = int(d.name); H = current["checkpoint"]["id"]
        fe = from_epoch()
        if fe is not None and e < fe:
            # 2026-11-07 handover: an epoch opened before the cutover is finished by the serial gate path (ntp: never promotes on timeout)
            return outer(self, directory, opening, dispatch, current, now)

        # ---- PIT live cutover: never call the trainer; keep weights flowing ----
        if pit_cutover():
            finalized = _pit_epoch_finalized(d)
            ap, a = ready_apply(H, e)
            if a:
                a["state"] = "offered"; a["offered_epoch"] = e; a["offered_at"] = a.get("offered_at", time.time()); _wj(ap, a)
                return dict(status="updated", checkpoint=a["checkpoint"], optimizer_step=a["optimizer_step"],
                            checkpoint_authorization=a["checkpoint_authorization"],
                            **({"publication_receipt_sha256": a["publication_receipt_sha256"]} if a.get("publication_receipt_sha256") else {}))
            dec = _pit_decision(e)
            decision = dec.get("decision")
            claim = _pit_claim(e, d)

            if finalized and not a:
                # Claim→no-update already closed this epoch; async GREEN apply is a later tick.
                return None

            if decision == "green" and dec.get("merge_blocked"):
                marker = _pit_promotion_marker(e, H, dec, reason="pit-merge-blocked")
                _wj(d / "PROMOTION-GATE.json", marker)
                return _no_update(current, "pit-merge-blocked-e%d-kept-%s" % (e, H[:8]))

            if claim and not decision and not dec.get("no_decision"):
                age = time.time() - float(claim.get("at") or claim.get("claimed_at") or 0)
                if age > PIT_DECISION_TIMEOUT_S:
                    dec_timeout = dict(dec, no_decision=True, chart_status="no_decision", decision=None)
                    _wj(d / "PROMOTION-GATE.json", _pit_promotion_marker(e, H, dec_timeout, reason="pit-no-decision-timeout"))
                    return _no_update(current, "pit-no-decision-e%d-kept-%s" % (e, H[:8]))

            if decision == "green" and not dec.get("no_decision") and not dec.get("merge_blocked"):
                # GREEN credited; head advances only when ready_apply matches (possibly a later epoch).
                return _no_update(current, "pit-green-pending-apply-e%d-kept-%s" % (e, H[:8]))

            if decision in ("red",) or dec.get("no_decision") or dec.get("chart_status") == "no_decision":
                marker = _pit_promotion_marker(e, H, dec, reason="pit-%s" % (decision or "no_decision"))
                _wj(d / "PROMOTION-GATE.json", marker)
                return _no_update(current, "pit-%s-e%d-kept-%s" % (decision or "no_decision", e, H[:8]))

            if not claim:
                if _pit_inflight_count() >= PIT_MAX_INFLIGHT:
                    return None
                _write_pit_claim(e, H, d)
                return _no_update(current, "pit-claimed-e%d" % e)

            # Claimed, decision not ready: do not stall the serial gate (next epoch may open).
            return _no_update(current, "pit-claimed-e%d" % e)

        if not (d / "TRAINING-JOB.json").exists():
            at = (_rj(PE_STATE / "TRAINER-AT.json") or {}).get("head")
            if at != H:
                req = PE_STATE / ("RESTORE-REQUEST-e%d.json" % e)
                r = _rj(req) or dict(epoch=e, head=H, requested_at=time.time())
                _wj(req, r)
                if time.time() - r["requested_at"] < RESTORE_WAIT_S:
                    return None
                return _no_update(current, "pe-restore-pending-e%d" % e)
        res = inner(self, directory, opening, dispatch, current, now)
        if res is None:
            return None
        if res.get("status") == "updated":
            pp = PE_STATE / "pending" / ("e%d.json" % e)
            if not pp.exists():
                _wj(pp, dict(epoch=e, candidate=res["checkpoint"]["id"], training_head=H, checkpoint=res["checkpoint"],
                             optimizer_step=res["optimizer_step"], checkpoint_authorization=res["checkpoint_authorization"],
                             publication_receipt_sha256=res.get("publication_receipt_sha256"), at=time.time()))
            _wj(PE_STATE / "TRAINER-AT.json", dict(head=res["checkpoint"]["id"], epoch=e, why="trained", at=time.time()))
        ap, a = ready_apply(H, e)
        if a:
            a["state"] = "offered"; a["offered_epoch"] = e; a["offered_at"] = a.get("offered_at", time.time()); _wj(ap, a)
            return dict(status="updated", checkpoint=a["checkpoint"], optimizer_step=a["optimizer_step"],
                        checkpoint_authorization=a["checkpoint_authorization"],
                        **({"publication_receipt_sha256": a["publication_receipt_sha256"]} if a.get("publication_receipt_sha256") else {}))
        if res.get("status") == "updated":
            return _no_update(current, "pe-pending-e%d-%s-kept-%s" % (e, res["checkpoint"]["id"][:8], H[:8]))
        return res

    _training.__qualname__, _training.__module__ = outer.__qualname__, outer.__module__
    _training._pe = True
    Lifecycle._training = _training
    if hasattr(Lifecycle, "tick"):
        prior_tick = Lifecycle.tick

        def tick(self, *, now):
            result = prior_tick(self, now=now)
            # The prior tick has released lifecycle.lock. Re-read the actual
            # durable CURRENT, never infer persistence from its return value.
            if pe_live():
                confirm_applied(self.root, self._current())
            return result

        tick.__qualname__, tick.__module__ = prior_tick.__qualname__, prior_tick.__module__
        tick._pe = True
        Lifecycle.tick = tick
    return dict(pe=True, flag=str(FLAG), live_now=pe_live())
