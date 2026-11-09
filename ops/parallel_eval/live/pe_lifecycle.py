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

PIT cutover (Jacob 2026-11-09 / Gate B): **decoupled from the one-open-epoch model**.
Lifecycle never holds dispatch waiting for a PIT decision and never hands the job
to the standing trainer (WARM-OFF). Cutover STILL runs ff/v829 selection so
K1-TURN + TRAIN-DISPATCH-FF + signed TRAINING-JOB are written, then marks the
epoch ``PIT-DISPATCHED.json`` and returns no-update. queue_ingest picks up
PIT-DISPATCHED epochs even after FINALIZED. Lifecycle also (a) offers a ready
PIT GREEN apply when base head matches. k1/weights read PIT DECISION.json /
HEAD-CHAIN / PIT-RESULT for miner credit.
"""
import json, os, time
from pathlib import Path

# Defaults only when env is unset (operator must set PE_FLAG / PE_LIFE_STATE in
# real deploys). Read via flag_path()/pe_state_path() on every call so suite
# order cannot freeze a prior test's temp paths into module globals.
_FLAG_DEFAULT = os.environ.get(
    "PE_FLAG_DEFAULT",
    "",
)
_PE_STATE_DEFAULT = os.environ.get(
    "PE_LIFE_STATE_DEFAULT",
    "",
)
RESTORE_WAIT_S = 1200


def flag_path() -> Path:
    """Current promotion-gate flag path (env PE_FLAG; re-read every call)."""
    env = os.environ.get("PE_FLAG")
    if env:
        return Path(env)
    if _FLAG_DEFAULT:
        return Path(_FLAG_DEFAULT)
    raise RuntimeError("PE_FLAG is unset (no import-time host default)")


def pe_state_path() -> Path:
    """Current PE life-state root (env PE_LIFE_STATE; re-read every call)."""
    env = os.environ.get("PE_LIFE_STATE")
    if env:
        return Path(env)
    if _PE_STATE_DEFAULT:
        return Path(_PE_STATE_DEFAULT)
    raise RuntimeError("PE_LIFE_STATE is unset (no import-time host default)")


# Back-compat aliases for callers that still read module attributes. Prefer
# flag_path()/pe_state_path() — these aliases are refreshed by _sync_paths().
FLAG = Path(_FLAG_DEFAULT) if _FLAG_DEFAULT else Path("/tmp/pe-flag-unset")
PE_STATE = Path(_PE_STATE_DEFAULT) if _PE_STATE_DEFAULT else Path("/tmp/pe-life-unset")


def _sync_paths() -> None:
    """Refresh FLAG/PE_STATE aliases from the current process environment."""
    global FLAG, PE_STATE
    FLAG = flag_path()
    PE_STATE = pe_state_path()


def _rj(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return d


def _wj(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name("." + p.name + ".%d.tmp" % os.getpid()); t.write_text(json.dumps(o, indent=1, sort_keys=True, default=str) + "\n"); os.chmod(t, 0o600); os.replace(t, p)


def pe_live():
    _sync_paths()
    return ((_rj(flag_path(), {}) or {}).get("parallel_eval") or {}).get("enabled") is True


def from_epoch():
    _sync_paths()
    v = ((_rj(flag_path(), {}) or {}).get("parallel_eval") or {}).get("from_epoch")
    return v if type(v) is int else None


def pit_cutover():
    """PIT live cutover: trainer off; lifecycle never waits on PIT decisions.

    Jacob 2026-11-09: PIT owns the batch queue (cap N boxes) independent of the
    one-open-epoch lifecycle. This wrapper only (a) offers a ready PIT GREEN apply
    when base head matches, or (b) returns no-update so weight nonces keep moving.
    Never returns None waiting for a per-epoch PIT DECISION.
    """
    _sync_paths()
    pe = (_rj(flag_path(), {}) or {}).get("parallel_eval") or {}
    pit = pe.get("pit") or {}
    if pit.get("revert") is True:
        return False
    return (
        pe.get("enabled") is True
        and pit.get("enabled") is True
        and pit.get("mode") == "live"
        and pit.get("cutover") is True
    )


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
    _sync_paths()
    rows = []
    for p in sorted((pe_state_path() / "apply").glob("*.json")):
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
    _sync_paths()
    confirmed = []
    for p in sorted((pe_state_path() / "apply").glob("*.json")):
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
    _sync_paths()
    if getattr(Lifecycle._training, "_pe", False):
        return dict(pe=True, already=True, flag=str(flag_path()), live_now=pe_live())
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
        state = pe_state_path()

        # ---- PIT live cutover: decoupled from epoch dispatch (Jacob 2026-11-09) ----
        # Never hold returning None for a PIT decision. Never hand the job to the
        # standing trainer (WARM-OFF at cutover). Still run ff/v829 so K1-TURN +
        # TRAIN-DISPATCH-FF + signed TRAINING-JOB land for queue_ingest. Mark
        # PIT-DISPATCHED, then finalize no-update so weight nonces keep moving.
        # Head advances only via ready_apply when a verified fresh PIT GREEN is ready.
        if pit_cutover():
            ap, a = ready_apply(H, e)
            if a:
                a["state"] = "offered"; a["offered_epoch"] = e; a["offered_at"] = a.get("offered_at", time.time()); _wj(ap, a)
                return dict(status="updated", checkpoint=a["checkpoint"], optimizer_step=a["optimizer_step"],
                            checkpoint_authorization=a["checkpoint_authorization"],
                            **({"publication_receipt_sha256": a["publication_receipt_sha256"]} if a.get("publication_receipt_sha256") else {}))
            marker = d / "PIT-DISPATCHED.json"
            if not marker.exists():
                # Drive ff selection / job write. Standing trainer is WARM-OFF, so
                # a returned None (waiting for TRAINING-RESULT) is expected — we
                # do NOT wait and do NOT apply an "updated" standing-trainer result.
                res = inner(self, directory, opening, dispatch, current, now)
                if isinstance(res, dict) and res.get("status") == "no-update" and not (d / "TRAINING-JOB.json").exists():
                    # k1-skip / no eligible tasks — no PIT work for this epoch.
                    return res
                if not (d / "TRAINING-JOB.json").exists():
                    # Inner did not write a job yet (signing / restore). Retry next tick;
                    # do not finalize an empty epoch forever.
                    return None
                _wj(marker, dict(
                    epoch=e,
                    head=H,
                    at=time.time(),
                    status="pit-dispatched",
                    has_dispatch_ff=(d / "TRAIN-DISPATCH-FF.json").exists(),
                    has_dispatch_k1=(d / "TRAIN-DISPATCH-K1.json").exists(),
                    has_k1_turn=(d / "K1-TURN.json").exists(),
                    note="standing trainer not invoked; PIT queue owns the batch",
                ))
            # Finalize so the serial lifecycle can open the next epoch. PIT
            # queue_ingest still accepts this epoch via PIT-DISPATCHED.
            return _no_update(current, "pit-decoupled-e%d-kept-%s" % (e, H[:8]))

        if not (d / "TRAINING-JOB.json").exists():
            at = (_rj(state / "TRAINER-AT.json") or {}).get("head")
            if at != H:
                req = state / ("RESTORE-REQUEST-e%d.json" % e)
                r = _rj(req) or dict(epoch=e, head=H, requested_at=time.time())
                _wj(req, r)
                if time.time() - r["requested_at"] < RESTORE_WAIT_S:
                    return None
                return _no_update(current, "pe-restore-pending-e%d" % e)
        res = inner(self, directory, opening, dispatch, current, now)
        if res is None:
            return None
        if res.get("status") == "updated":
            pp = state / "pending" / ("e%d.json" % e)
            if not pp.exists():
                _wj(pp, dict(epoch=e, candidate=res["checkpoint"]["id"], training_head=H, checkpoint=res["checkpoint"],
                             optimizer_step=res["optimizer_step"], checkpoint_authorization=res["checkpoint_authorization"],
                             publication_receipt_sha256=res.get("publication_receipt_sha256"), at=time.time()))
            _wj(state / "TRAINER-AT.json", dict(head=res["checkpoint"]["id"], epoch=e, why="trained", at=time.time()))
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
    return dict(pe=True, flag=str(flag_path()), live_now=pe_live())
