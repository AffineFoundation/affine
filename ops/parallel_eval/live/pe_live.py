#!/usr/bin/env python3
"""PE LIVE WIRING (staged wiring-v1, overnight 2026-11-07, NOT INSTALLED until activate_live).
Connects the installed lifecycle wrapper (pe_lifecycle) to the controller. Called once per controller tick, live mode only.

  restore_tick   : state/life/RESTORE-REQUEST-e<N>.json (written by pe_lifecycle before TRAINING-JOB) ->
                   (a) if the pod trainer currently holds an unresolved PENDING candidate C, copy C's generation into the
                       pod's pe-keep/<C> (pe_trainer_keep.py keep C) so a later fresh GREEN apply of C can be restored;
                   (b) stage pe-keep/<H> into the gate keep dir if the gate keep for H is missing (pe_trainer_keep.py stage H);
                   (c) run the LIVE gate's own trainer_restore.sh H N (same script/protocol as every live RED restore today);
                   (d) rc 0 -> state/life/TRAINER-AT.json {head: H}. Background process, exact-PID tracked, bounded retries.
  dispatch_tick  : state/life/pending/e<N>.json -> CLAIM-e<N> (gate stays off this epoch) -> job live-e<N> with the EXACT live
                   gate replica spec: the epoch's own committed round draw (SAMPLE.private.json == ROUND indices), plan seed
                   cand_seed(round seed, N), k_cand, max_tokens = round cap (64K), env, MAX_SEQS 512. Any non-5090 GPU (pe_rent).
  apply_tick     : gate DECISION green for a PE epoch -> GREENS queue (green-time order) -> pe_apply.decide_apply:
                   fresh (staleness 0)   -> state/life/apply/e<N>.json state=ready (candidate's own signed authorization)
                   merge (staleness 1-2) -> MERGE-BLOCKED: needs a trainer-signed merged publication (not built) -> credited only
                   overflow / unknown    -> stale_overflow (credited only)
  reconcile_tick : apply record consumed by pe_lifecycle.confirm_applied (FINALIZED + durable CURRENT) -> HEAD-CHAIN append.
Writes only PE state (+ gate CLAIM files). Never chain, weights, CURRENT, FINALIZED. Test hooks: PE_FAKE_TRAINER=<json path>.
"""
import json, os, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
# Single pe_apply module object under both import names. Package loaders
# (ops.parallel_eval.live.pe_apply) and bare `import pe_apply` must share
# identity — otherwise cutover patches / marks land on a different object
# than apply_tick reads via PA.next_green.
_PE_APPLY_PKG = "ops.parallel_eval.live.pe_apply"
if _PE_APPLY_PKG in sys.modules:
    PA = sys.modules[_PE_APPLY_PKG]
    sys.modules["pe_apply"] = PA
else:
    import pe_apply as PA  # noqa: E402
    sys.modules[_PE_APPLY_PKG] = PA
import pe_gate_hook as HOOK

def _life_state_root() -> Path:
    """PE_LIFE_STATE when set (required for PIT cutover); else legacy ST/life."""
    env = os.environ.get("PE_LIFE_STATE")
    if env:
        return Path(env)
    return ST / "life"


def _pending_dir() -> Path:
    return _life_state_root() / "pending"


def _apply_dir() -> Path:
    return _life_state_root() / "apply"


# Back-compat aliases (refreshed by _sync_life_paths). Prefer _pending_dir/_apply_dir.
LST = Path(os.environ.get("PE_LIFE_STATE") or (ST / "life"))
PENDING = LST / "pending"
APPLY = LST / "apply"
RESTORE_SH = Path(os.environ.get("PE_RESTORE_SH") or (GATE_CODE / "trainer_restore.sh"))
RSSH = os.environ.get("PE_RSSH") or "rssh"
POD_KEEP = os.environ.get("PE_POD_KEEP") or (
    "/workspace/affine-mimo-trainer-prod/private/promotion-gate/pe_trainer_keep.py"
)
FAKE = os.environ.get("PE_FAKE_TRAINER")
RESTORE_MAX_ATTEMPTS = 5
RESTORE_BACKOFF_S = 60
RESTORE_TIMEOUT_S = 2400


def _sync_life_paths() -> None:
    global LST, PENDING, APPLY
    LST = _life_state_root()
    PENDING = _pending_dir()
    APPLY = _apply_dir()


def _cur():
    c = rj(LIFE / "CURRENT.json") or {}
    return c, int(c.get("epoch_index", -1)), (c.get("checkpoint") or {}).get("id")


def _pid_alive(pid, needle):
    try:
        os.kill(int(pid), 0)
        return needle in Path("/proc/%d/cmdline" % int(pid)).read_text(errors="ignore")
    except Exception:
        return False


# ---------------------------------------------------------------- trainer pod ops (real or fake)
def pod_keep(cmd, ck):
    """pe_trainer_keep.py keep|stage <ck> on the trainer pod -> dict."""
    if FAKE:
        f = Path(FAKE); s = rj(f, {}) or {}
        if s.get("fail_keep") and cmd == "keep":
            return dict(ok=False, error="fake keep failure")
        if cmd == "keep":
            if s.get("pod_current") != ck:
                return dict(ok=False, error="pod CURRENT is %s, not %s" % (str(s.get("pod_current"))[:8], ck[:8]))
            s.setdefault("pe_keep", []).append(ck)
        elif cmd == "prune":
            need = set(json.loads(ck)); s["pe_keep"] = [x for x in (s.get("pe_keep") or []) if x in need or x == s.get("pod_current")]
        elif cmd == "stage":
            if ck not in (s.get("pe_keep") or []) and ck not in (s.get("gate_keep") or []):
                return dict(ok=False, error="no pe-keep for %s" % ck[:8])
            s.setdefault("gate_keep", []).append(ck)
        wj(f, s); return dict(ok=True, fake=True)
    try:
        p = subprocess.run([RSSH, "python3 %s %s %s" % (POD_KEEP, cmd, _q(ck))], capture_output=True, text=True, timeout=1800,
                           stdin=subprocess.DEVNULL)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as ex:
        return dict(ok=False, error=scrub(str(ex))[:300])


def pod_current():
    """Pod trainer CURRENT checkpoint id (None if unknown)."""
    if FAKE:
        return (rj(Path(FAKE), {}) or {}).get("pod_current")
    try:
        p = subprocess.run([RSSH, "python3 -c 'import json;print(json.load(open(\"/workspace/affine-mimo-trainer-prod/trainer-state/affine-mimo9b-grpo-20261013/CURRENT.json\"))[\"checkpoint\"])'"],
                           capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        v = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else None
        return v if v and len(v) == 64 else None
    except Exception:
        return None


def restore_cmd(H, e):
    if FAKE:
        return [HOST_PY, "-B", str(Path(__file__).resolve().parent / "pe_fake_restore.py"), FAKE, H, str(e)]
    return ["bash", str(RESTORE_SH), H, str(e)]


# ---------------------------------------------------------------- restore
def restore_tick(now=None):
    now = now or time.time()
    cur, N, H = _cur()
    out = {}
    for rq in sorted(LST.glob("RESTORE-REQUEST-e*.json")):
        r = rj(rq) or {}
        e = int(r.get("epoch", -1)); sp = LST / ("RESTORE-STATE-e%d.json" % e)
        s = rj(sp) or dict(epoch=e, head=r.get("head"), attempts=[], state="new")
        if s["state"] in ("done", "obsolete", "failed"):
            continue
        ed = LIFE / "epochs" / str(e)
        if e != N or r.get("head") != H or (ed / "TRAINING-JOB.json").exists():
            if s.get("bg") and _pid_alive(s["bg"]["pid"], "restore"):
                out[e] = "restoring(obsolete-wait)"; continue
            s.update(state="obsolete", why="current moved (N=%s head=%s) or job exists" % (N, str(H)[:8]), at_cr=cr()); wj(sp, s)
            ev(event="restore_obsolete", epoch=e); out[e] = "obsolete"; continue
        at = rj(LST / "TRAINER-AT.json") or {}
        if at.get("head") == H:
            s.update(state="done", note="trainer already on H", at_cr=cr()); wj(sp, s); out[e] = "done(already)"; continue
        bg = s.get("bg")
        if bg:
            if _pid_alive(bg["pid"], "restore") and now - bg["at"] < RESTORE_TIMEOUT_S:
                out[e] = "restoring"; continue
            if _pid_alive(bg["pid"], "restore"):
                try: os.kill(int(bg["pid"]), 15)
                except Exception: pass
            rc = None
            try: rc = int(Path(bg["rcf"]).read_text().strip())
            except Exception: pass
            s["attempts"].append(dict(n=bg["n"], rc=rc, seconds=round(now - bg["at"], 1), at_cr=cr()))
            s.pop("bg")
            if rc == 0:
                wj(LST / "TRAINER-AT.json", dict(head=H, epoch=e, why="restored", at=now, at_cr=cr()))
                s.update(state="done", at_cr=cr()); wj(sp, s); ev(event="restore_done", epoch=e, head=H[:12]); out[e] = "done"; continue
            ev(event="restore_failed", epoch=e, rc=rc, attempt=bg["n"])
            if len(s["attempts"]) >= RESTORE_MAX_ATTEMPTS:
                s.update(state="failed", at_cr=cr()); wj(sp, s); out[e] = "failed"; continue
            s["retry_at"] = now + RESTORE_BACKOFF_S; wj(sp, s); out[e] = "backoff"; continue
        if now < float(s.get("retry_at") or 0):
            out[e] = "backoff"; continue
        # (0) handover: the previous epoch was decided by the serial gate -> its own restore runs; never run a second one
        pdec = rj(EP / str(e - 1) / "DECISION.json") or {}
        if pdec.get("decision") == "red" and pdec.get("head") == H and not (EP / str(e - 1) / "TRAINER-RESTORED.json").exists() \
                and not HOOK.pe_owns(e - 1) and now - float(pdec.get("decided_at") or now) < 1800:
            out[e] = "gate-restore-wait"; continue
        pc = pod_current()
        if pc == H:
            wj(LST / "TRAINER-AT.json", dict(head=H, epoch=e, why="pod CURRENT already H", at=now, at_cr=cr()))
            s.update(state="done", note="pod CURRENT already H (no restore needed)", at_cr=cr()); wj(sp, s)
            ev(event="restore_skip_pod_on_head", epoch=e, head=H[:12]); out[e] = "done(pod-on-H)"; continue
        # (a) keep the pod's freshly trained pending candidate before leaving it
        if at.get("why") == "trained" and at.get("head") and at["head"] != H:
            C = at["head"]; pk = _pending_by_candidate(C)
            if pk and pk[1].get("pe_keep") is None:
                k = pod_keep("keep", C)
                pk[1]["pe_keep"] = bool(k.get("ok")); pk[1]["pe_keep_info"] = {kk: k.get(kk) for kk in ("ok", "error", "gen", "already")}
                wj(pk[0], pk[1]); ev(event="pe_keep", candidate=C[:12], ok=bool(k.get("ok")), err=k.get("error"))
                if k.get("ok"):
                    need = needed_keeps(H)
                    pr = pod_keep("prune", json.dumps(sorted(need)))
                    ev(event="pe_keep_prune", ok=bool(pr.get("ok")), kept=len(need), pruned=pr.get("pruned"))
        # (b) H restorable from the gate keep dir (stage our pe-keep copy there if needed)
        st = pod_keep("stage", H)
        if not st.get("ok"):
            s.setdefault("stage_errors", []).append(dict(at_cr=cr(), err=st.get("error")))
            if len(s["stage_errors"]) >= RESTORE_MAX_ATTEMPTS:
                s.update(state="failed", why="no restorable keep for H"); wj(sp, s); out[e] = "failed(no-keep)"; continue
            s["retry_at"] = now + RESTORE_BACKOFF_S; wj(sp, s); out[e] = "stage-failed"; continue
        # (c) the live gate's restore script, background, exact PID
        n = len(s["attempts"]); rcf = LST / ("restore-e%d-a%d.rc" % (e, n)); lg = LST / ("restore-e%d-a%d.out" % (e, n))
        shell = "%s; echo $? > %s" % (" ".join(map(_q, restore_cmd(H, e))), _q(str(rcf)))
        p = subprocess.Popen(["bash", "-c", "exec -a pe-restore bash -c " + _q(shell)], stdout=open(lg, "ab"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        s["bg"] = dict(pid=p.pid, n=n, at=now, rcf=str(rcf), log=str(lg)); s["state"] = "restoring"; wj(sp, s)
        ev(event="restore_start", epoch=e, head=H[:12], attempt=n, pid=p.pid); out[e] = "started"
    return out


def _q(x):
    import shlex
    return shlex.quote(x)


def needed_keeps(H):
    """pe-keep copies still needed: current head, the last 3 applied heads, every pending candidate not finally resolved."""
    need = {H}
    need.update(a["new_head"] for a in PA.chain().get("applied", [])[-3:])
    gs = {g["candidate"]: g for g in PA.greens()}
    for p in PENDING.glob("e*.json"):
        d = rj(p) or {}; C = d.get("candidate"); e = d.get("epoch")
        if not C:
            continue
        dec = rj(EP / str(e) / "DECISION.json") or {}
        if not dec.get("decision"):
            need.add(C); continue
        if dec.get("decision") == "green" and not dec.get("no_decision") and (gs.get(C) or {}).get("state", "pending") in ("pending", "applied"):
            need.add(C)
    return need


def _pending_by_candidate(C):
    for p in sorted(PENDING.glob("e*.json")):
        d = rj(p) or {}
        if d.get("candidate") == C:
            return p, d
    return None


# ---------------------------------------------------------------- dispatch
def live_spec(e, pend):
    sys.path.insert(0, str(GATE_CODE))
    import baseline_v1 as BV
    import prior_gate as PG
    sm = rj(EP / str(e) / "SAMPLE.private.json")
    if not sm:
        return None, "no SAMPLE for e%d (gate has not planned it)" % e
    if sm.get("head") != pend["training_head"]:
        return None, "SAMPLE head %s != training head %s" % (str(sm.get("head"))[:8], pend["training_head"][:8])
    r = rj(HELDOUT / sm["round_id"] / "ROUND.private.json") or {}
    if not r.get("indices") or r.get("seed") is None:
        return None, "round %s draw not ready" % sm["round_id"]
    if list(r["indices"]) != list(sm["indices"]):
        return None, "round draw != epoch sample"
    pv = r["cfg"]
    return dict(id="live-e%d" % e, kind="live", epoch=e, candidate=pend["candidate"], head=pend["training_head"],
                round_id=sm["round_id"], indices=list(r["indices"]), plan_seed=BV.cand_seed(r["seed"], e),
                k=int(pv["k_cand"]), max_tokens=int(PG._mt(pv)), env_id=pv["probe_env"],
                files=(pend.get("checkpoint") or {}).get("files"), max_num_seqs=512), None


def _pit_flag():
    """Load parallel_eval.pit from the PE flag JSON (best-effort)."""
    try:
        flag = Path(os.environ.get("PE_FLAG") or "")
        if not flag or not flag.exists():
            flag = Path(os.environ.get("PE_FLAG") or "/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json")
        pe = (json.loads(flag.read_text()).get("parallel_eval") or {})
        return pe.get("pit") or {}
    except Exception:
        return {}


def _pit_enabled():
    """When parallel_eval.pit.enabled (and not revert), PIT owns dequeue/train."""
    pit = _pit_flag()
    if pit.get("revert") is True:
        return False
    return pit.get("enabled") is True


def _pit_cutover():
    """Live cutover: trainer dequeue / warm / standing lanes off (SPEC §7)."""
    pit = _pit_flag()
    if pit.get("revert") is True:
        return False
    return pit.get("enabled") is True and pit.get("mode") == "live" and pit.get("cutover") is True


def dispatch_tick(submit):
    # Stop live PE dispatch ONLY when pit.enabled AND pit.cutover are both true.
    # Shadow must never affect live dispatch.
    if _pit_cutover():
        return {"pit": "dispatch_noop", "cutover": True}
    out = {}
    for p in sorted(PENDING.glob("e*.json")):
        pend = rj(p) or {}
        e = int(pend.get("epoch", -1))
        if (JOBS / ("live-e%d" % e) / "JOB.json").exists() or (EP / str(e) / "DECISION.json").exists():
            continue
        # PIT host owns this batch (skip marker); never PE-lane double-dispatch.
        from pathlib import Path as _P
        _pit = os.environ.get("PIT_STATE")
        _pe = os.environ.get("PE_STATE") or str(ST)
        _pe_pit = str(_P(_pe) / "pit")
        if _pit and _P(_pit).resolve() != _P(_pe_pit).resolve():
            raise RuntimeError("PIT_STATE=%s disagrees with PE_STATE/pit=%s" % (_pit, _pe_pit))
        _claim_roots = [LST, _P(_pit or _pe_pit), _P(_pit or _pe_pit) / "claims"]
        if any((_r / ("PIT-CLAIM-e%d.json" % e)).exists() for _r in _claim_roots):
            out[e] = "pit-claimed"; continue
        spec, why = live_spec(e, pend)
        if spec is None:
            if pend.get("dispatch_wait") != why:
                pend["dispatch_wait"] = why; wj(p, pend); ev(event="dispatch_wait", epoch=e, why=why)
            out[e] = "wait: " + why; continue
        HOOK.claim(e, pend["candidate"], pend["training_head"])
        submit(spec); ev(event="live_dispatched", epoch=e, candidate=pend["candidate"][:12]); out[e] = "dispatched"
    return out


# ---------------------------------------------------------------- apply
def apply_tick():
    cur, N, H = _cur()
    c = PA.chain()
    if c.get("genesis") is None:
        PA.set_genesis(H); c = PA.chain()
    head_chain = PA.current_head_of(c)
    if head_chain != H and not c.get("applied") and not any(g.get("state") in ("pending", "applied") for g in PA.greens()) \
            and not any((rj(x) or {}).get("state") in ("ready", "offered") for x in APPLY.glob("*.json")):
        c = PA.reset_genesis(H, "CURRENT moved before any PE apply (serial pre-cutover epoch) %s -> %s" % (str(head_chain)[:8], str(H)[:8]))
        head_chain = H; ev(event="pe_regenesis", head=H[:12])
    out = {}
    for p in sorted(PENDING.glob("e*.json")):
        pend = rj(p) or {}; e = int(pend.get("epoch", -1))
        dec = rj(EP / str(e) / "DECISION.json") or {}
        if dec.get("decision") == "green" and not dec.get("no_decision") and dec.get("candidate") == pend.get("candidate"):
            if PA.add_green(e, pend["candidate"], pend["training_head"], dec.get("decided_at") or time.time(),
                            extra=dict(pending=str(p))):
                ev(event="green_queued", epoch=e, candidate=pend["candidate"][:12])
    if head_chain != H:
        # CURRENT moved outside PE (operator rollback etc.) or an apply is in flight -> never offer anything new
        inflight = [a for a in (rj(x) for x in APPLY.glob("*.json")) if a and a.get("state") in ("ready", "offered")]
        out["frozen"] = "chain head %s != CURRENT %s (inflight %d)" % (str(head_chain)[:8], str(H)[:8], len(inflight))
        return out
    if any((rj(x) or {}).get("state") in ("ready", "offered") for x in APPLY.glob("*.json")):
        out["inflight"] = True; return out
    while True:
        item = PA.next_green()
        if not item:
            break
        kind, s = PA.decide_apply(item)
        pend_path = item.get("pending")
        if not pend_path:
            PA.mark(item["candidate"], state="unrestorable", staleness=s, why="missing pending path on GREEN row")
            ev(event="green_unrestorable", epoch=item["epoch"], why="no-pending"); continue
        pend = rj(pend_path) or {}
        if kind == "fresh":
            if pend.get("pe_keep") is False:
                PA.mark(item["candidate"], state="unrestorable", staleness=s, why="pe-keep failed; applying would strand the trainer")
                ev(event="green_unrestorable", epoch=item["epoch"]); continue
            # Defense in depth: never write APPLY ready without a complete fp32 master.
            if not PA.master_index_is_complete(item["candidate"]):
                PA.mark_apply_blocked_incomplete(item["candidate"], epoch=item.get("epoch"))
                ev(event="green_apply_blocked_incomplete_master", epoch=item["epoch"],
                   candidate=str(item["candidate"])[:12])
                out["blocked_incomplete_master"] = item["epoch"]
                continue
            ap = APPLY / ("e%d.json" % item["epoch"])
            if not ap.exists():
                # Re-check immediately before the write (fail closed).
                if not PA.master_index_is_complete(item["candidate"]):
                    PA.mark_apply_blocked_incomplete(item["candidate"], epoch=item.get("epoch"))
                    out["blocked_incomplete_master"] = item["epoch"]
                    continue
                wj(ap, dict(state="ready", epoch=item["epoch"], candidate=item["candidate"], base_head=item["training_head"],
                            green_at=item["green_at"], checkpoint=pend["checkpoint"], optimizer_step=pend["optimizer_step"],
                            checkpoint_authorization=pend["checkpoint_authorization"],
                            publication_receipt_sha256=pend.get("publication_receipt_sha256"), kind="fresh", staleness=0, at_cr=cr()))
                ev(event="apply_ready", epoch=item["epoch"], candidate=item["candidate"][:12])
            out["ready"] = item["epoch"]; break
        if kind == "merge":
            PA.mark(item["candidate"], state="merge_blocked", staleness=s,
                            why="fresh-only phase 1 (Jacob 2026-11-07 07:27 CR Option B): stale GREEN credited/paid via K1, NOT applied until the trainer-signed merged publication is built")
            ev(event="green_merge_blocked", epoch=item["epoch"], staleness=s); continue
        PA.record_stale(item, s); ev(event="green_stale_overflow", epoch=item["epoch"], staleness=s)
    return out


def reconcile_tick():
    out = []
    for ap in sorted(APPLY.glob("*.json")):
        a = rj(ap) or {}
        if a.get("state") != "consumed" or a.get("chain_recorded"):
            continue
        item = next((g for g in PA.greens() if g["candidate"] == a["candidate"]), None)
        if item is None:
            continue
        PA.record_applied(item, "fresh", 0, a["checkpoint"]["id"], a["base_head"])
        a["chain_recorded"] = time.time(); wj(ap, a); ev(event="apply_recorded", epoch=a.get("epoch"), new_head=a["checkpoint"]["id"][:12])
        out.append(a.get("epoch"))
    return out


def tick(submit, *, skip_dispatch: bool = False):
    """Controller tick. Under PIT cutover set skip_dispatch=True so legacy
    PE-lane job submit is skipped while apply_tick + reconcile_tick still run.
    """
    _sync_life_paths()
    LST.mkdir(parents=True, exist_ok=True)
    PENDING.mkdir(parents=True, exist_ok=True)
    APPLY.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, fn in (("restore", restore_tick), ("dispatch", lambda: dispatch_tick(submit)), ("reconcile", reconcile_tick),
                     ("apply", apply_tick)):
        if skip_dispatch and name == "dispatch":
            res[name] = dict(skipped="warm-off-or-pit-cutover")
            continue
        try:
            res[name] = fn()
        except Exception as ex:
            res[name] = "error: %s" % str(ex)[:200]; ev(event="pe_live_error", part=name, err=str(ex)[:300])
    return res
