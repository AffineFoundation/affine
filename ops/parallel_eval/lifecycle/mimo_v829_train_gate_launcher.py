#!/usr/bin/env python3
"""MiMo host v8.2.9 train authenticity gate over signed v8.2.8.

Jacob [pt545u] 2026-10-23: TOPLOC/sampling fail or unverified => no GRPO pack.
- Re-enable audit scheduler (scaled) so all credit/train batches can verify.
- When TRAIN-GATE.flag.json enabled: verified-or-nothing — do not pack unaudited.
- Publishes mimo/train-pack.json when any verified_valid rows exist for trainer host-v829.

Does not loosen TOPLOC tolerances. Does not strip nonce/mixed-outcome.

v8.2.9c (Jacob 2026-10-23): main path gates on the CHEAP check only. Every sealed
row already passed structural admission (incl. TOPLOC proof framing) + native
re-grade; it trains unless (a) its miner is hard-banned in ops/uid-ban.json or
(b) the offline sampling audit already recorded a fail for that batch. The
expensive GPU audit never blocks the seal or the train gate. Also restores the
v8.2.8 invariant that FINALIZED waits for the off-path AUDIT-POPULATION.

v8.2.9e (Jacob t569u/t570u 2026-10-24): operator TRAIN-CONTROL.json next to the gate flag:
rollback-then-pause (finalize onto a past FINALIZED checkpoint, e.g. e95 39f6304d) / pause / train.
Gate, sidecar, bans, TOPLOC, nonces unchanged.

v8.2.9f (Jacob pt595u 2026-10-25): trainer job lifetime 7200 s -> TRAIN_JOB_SECONDS (14400 s).
e135 (39.6% capped, 5220 tok/rollout) outgrew the 2 h signed deadline. Sets only the training
adapter job_seconds (= signed TRAINING-JOB expires_at - created_at) before each _training call.
Worker/coordinator bound stays 86400. Nothing else changed.

v8.2.9g (Jacob 2026-10-25): expensive TOPLOC audit (verify) job lifetime 1800 s -> AUDIT_JOB_SECONDS
(7200 s; lifecycle audit_scheduler.job_seconds is already 7200, v829d capped it to 1800). Groups take
60-80 min on the H200 verifiers, so every 1800 s job expired (e136: 0 completed audits). To keep the
v829d "newest epoch only, never a backlog" behaviour with 2 h jobs, each tick (before the base tick):
  - still-queued (unclaimed) audit jobs of any non-newest epoch are marked expired in the queue
    (same status the queue itself sets at expiry; an event 'expired' reason v829g-stale-epoch is logged),
    so no verifier ever picks up a stale epoch;
  - still-leased jobs of a non-newest epoch are left to finish (a verifier cannot be pre-empted) but
    count against the 2/2 capacity, so newest-epoch jobs are not queued behind them and left to age.
Trainer job lifetime (14400), gate, train control, bans, TOPLOC, nonces, roster: unchanged.

v8.2.9h (Jacob 2026-10-26 13:38 CR "Cut gather to 20 minutes"): miner collection window
(lifecycle collection_seconds, signed manifest deadline = start + collection_seconds) 1800 s ->
COLLECTION_SECONDS. Same epoch-BOUNDARY pattern as the v8.2.4 max_tasks 128<->256 switch: CURRENT.json
is re-stamped to COLLECTION_SECONDS only while the current epoch has no PLAN yet; an in-flight epoch
keeps its planned window. Only values in V829H_COLLECTION_SECONDS are accepted (fail closed otherwise).
A sibling release with COLLECTION_SECONDS = 1800 is the one-step revert. max_tasks (256), trainer,
gate, sidecar, bans, TOPLOC, nonces, roster: unchanged.

v8.2.9i (Jacob pt682u 2026-10-26 14:25 CR; t684u 14:39 CR "go"): checkpoint PROMOTION GATE, operator flag
staged/PROMOTION-GATE.json (sibling of TRAIN-CONTROL.json; absent / enabled!=true = exact v8.2.9h behaviour).
When enabled, an epoch whose training produced candidate C (status updated) is held (tick returns "training")
until the promotion-gate service writes its decision for (head H, candidate C):
  green -> the original updated result (C promoted; next epoch mines from C);
  red   -> no-update onto H, exactly like an operator pause (next epoch mines from H; flip lineage trained=False, so the
           epoch's miners earn zero flip credit).
No decision within decision_timeout_seconds after TRAINING-RESULT -> fail open (promote, marker GATE-TIMEOUT).
After a live RED the next epoch's trainer job is held until the pod trainer is restored to H (TRAINER-RESTORED), at
most restore_wait_seconds after its seal, then that epoch is a no-update (never trains on a wrong base).
Trainer math, gate, sidecar, bans, TOPLOC, nonces, collection window, roster: unchanged.
v8.2.9j (Jacob pt696u 2026-10-26 19:35 CR "bring the epoch timing back under 45 mins"): evaluator opening memo.
The independent evaluation scheduler thread re-verified EVERY epoch's signed OPENING (~12.5 MB each, 156+ epochs:
canonical JSON + ed25519 verify + validate_manifest + validate_sets deep copies) on every 10 s tick, holding the GIL
~45% of wall time and starving the capture/seal thread (py-spy e156 capture: eval thread 1070 vs capture 687 samples).
An opening whose exact (signer, signature, epoch, checkpoint id, start, deadline) already passed the ORIGINAL tick in
this process (its checkpoint key is then registered in evaluator state.json, which never removes keys) is not passed
to the original tick again. New openings, and the whole job/queue/receipt logic, still run the original code each tick.
Operator switch: staged/EPOCH-TIME-LIFECYCLE.json {"eval_memo": false} (or absent file = ON) -> exact v8.2.9i (read
every tick, no restart). Scoring, credit, weights, bans, TOPLOC/nonces, trainer, gate, collection window: unchanged.
v8.2.9k (Jacob pt696u, same pass): seal document read-ahead. py-spy on the e157 seal: 65% of the seal thread's wall
time is spent waiting on SERIAL R2 GETs of the frozen population's training documents (original inputs() generator,
one _get per document). While an original Coordinator.seal runs, the seal module's _get serves each document from
a bounded read-ahead (10 threads, 80 documents ahead) that calls the SAME original _get with the same bucket, key and
limit, in the order listed by the already-frozen POPULATION.json (hint only). The original code still checks every
document's sha256, decodes, grades (same pool, same order), selects, excludes and signs. Any miss, error or unexpected
key falls back to the original serial _get. Switch: EPOCH-TIME-LIFECYCLE.json {"seal_prefetch": false} -> exact v8.2.9j.
v8.2.9m (Jacob t739u 2026-10-27 16:28 CR "Let's go for a 15 min gather not 12"): v8.2.9k with the collection window
accept-list widened to (900, 1200, 1800) and COLLECTION_SECONDS = 900. Same epoch-boundary re-stamp (v8.2.9h): an
in-flight epoch keeps its planned window. Everything else byte-identical to v8.2.9k.
v8.2.9n (Jacob t739u, same pass): (1) capture: inside an original capture() the private snapshot copies of the
training documents are PUT concurrently (original conditional _immutable, same keys/bytes, v8.2.7 write-skip memo)
and all must succeed before that capture's commitment.json PUT and ledger admission (py-spy e175: 59% of the
post-deadline capture pass was these serial PUTs). (2) seal: while an epoch collects, a background pool (24) grades
captured documents with the original mimo_native_pool._grade; the seal's pool returns a cached grade only for the
identical (pool args, manifest, miner, batch) and grades every miss with the original ordered pool. Seal order,
checks, exclusions, de-dup, seed and selection are the original Coordinator.seal. Switches (no restart):
EPOCH-TIME-LIFECYCLE.json {"capture_put_fanout": false} / {"seal_pregrade": false} -> exact v8.2.9m.
v8.2.9l (Jacob t712u 2026-10-27 11:52 CR "build it and then run it immediately"): K=1 per-UID turns. Operator flag
staged/K1.json (absent / enabled!=true / epoch < from_epoch = exact v8.2.9k). When on, each epoch trains ONE UID's batch:
queue = registered UIDs (UID 0 excluded) ordered by (turns taken by this uid+hotkey, registration block, uid); the first
UID in that order whose admitted batch this epoch has >= min_tasks distinct tasks is the turn (no waiting: UIDs without a
ready batch are skipped, not charged). Batch = all of that miner's admitted rows (sealed selected + rows excluded only by
the pooled 256-task budget or cross-miner collision), in submission order (R2 received_at, slot, index), one row per task,
trimmed at max_rollouts. Bans, the verify denylist and the offline TOPLOC ledger are NOT applied on this path. The trainer
job is a lifecycle-signed filtered dispatch of the original shape (same mechanism as the v8.2.9c partial strip), trained by
the original path; promotion gate (v8.2.9i), train control (v8.2.9e), collection window, seal and capture are unchanged.
A turn counts when its job is dispatched (pass or fail). No ready UID -> no-update epoch (K1-SKIP).
v8.2.9o (Jacob t751u 2026-10-28 09:47 CR "Cut over asap"): the v8.2.9l K=1 block above rebased unchanged onto
v8.2.9n (900 s collection, accept (900, 1200, 1800), capture PUT fan-out, seal pre-grade cache).
v8.2.9q (Jacob t758u 2026-10-28 12:39 CR "the next in line is the next registration with a valid batch ... miners needing to
register to submit", modelled on teutonic.ai "one submission per hotkey"): queue rule "registration-once" (default). A
registration = (uid, hotkey, BlockAtRegistration) on the metagraph snapshot; it gets at most ONE turn, ever. Order = registration
block ascending, then uid (UID 0 excluded); registrations that already had a turn (K1-TURNS.jsonl; legacy records without
reg_block resolved from epochs/N/K1-TURN.json, else matched on uid+hotkey) are removed from the queue. First unserved
registration with a ready batch (>= min_tasks distinct admitted tasks) takes the turn; the rest are skipped, not charged.
To submit again a miner must register again (new hotkey, or the same hotkey re-registered at a new block). Everything else
(batch, caps, step, gate, credit) is the v8.2.9o K=1 block unchanged. staged/K1.json "queue_rule": "fewest-turns" -> exact v8.2.9o
order (no restart).
v8.2.9q2 (Jacob t761u 2026-10-28 13:06 CR "yes one per hotkey forever"): queue rule "hotkey-once" (new default), exactly
teutonic.ai's "Each hotkey gets exactly ONE submission, permanently": a HOTKEY that has had a turn (any UID, any registration
block; K1-TURNS.jsonl) never gets another. Order unchanged (registration block, uid; UID 0 excluded; skip without charge).
staged/K1.json "queue_rule": "registration-once" -> exact v8.2.9q; "fewest-turns" -> exact v8.2.9o order (no restart).
v8.2.9s (Jacob 2026-10-29 "Yes, ship the UID+reg_block block now"; swap_hotkey loophole reported by duongna): queue rule
"hotkey-registration-once" (new default) = hotkey-once PLUS: a registration (uid, BlockAtRegistration) that has had a turn
never gets another, whatever hotkey sits on it now (swap_hotkey keeps uid and BlockAtRegistration, so a swapped-in hotkey
on a served registration stays out). A genuine re-registration (new BlockAtRegistration) under a never-served hotkey stays
eligible. Order unchanged. staged/K1.json "queue_rule": "hotkey-once" -> exact v8.2.9q2/r2 (no restart).
v8.2.9t (Jacob t771u / pt771u 2026-10-29 "We should only wait any duration here if the previous batch was green because only
then the model has changed"): RED-skip early seal. Operator switch staged/EPOCH-TIME-LIFECYCLE.json {"red_skip": true}
(absent / not true = exact v8.2.9s). When on, K=1 on and TRAIN-CONTROL=train, an epoch N whose predecessor N-1 FINALIZED
onto its own input checkpoint (RED, K1-skip or any no-update: the head did not change) and that mines from that same head
does not wait for its signed 900 s deadline: after each original capture pass the K=1 rule (k1_select, unchanged) is run on
the rows the seal would admit as candidates according to the v8.2.9n pre-grade cache (cached native grade meets the group
minima); as soon as that names a turn the epoch is sealed (original Coordinator.seal, real time; only its "before the
collection deadline" refusal is waived for this one call) and the original tick continues to the K=1 dispatch/training. The
real seal + real k1_select decide the turn exactly as before (the prediction only picks WHEN to seal). No ready batch ->
keep polling every tick until the signed deadline (then the original seal). After a GREEN (new head) the original 900 s
window is unchanged. If N-1 was itself early-sealed but had no K=1 turn (prediction mismatch), N keeps the full window.
"red_skip": "all" (Jacob t778u 2026-10-29 9:58 CR "drop the submission window entirely, including after GREEN"): the same early
seal in EVERY epoch with a finalized predecessor (after GREEN too); the gate's head pre-probe of the new head already starts at
PLAN on the probe pod and runs in parallel with training (needed only at DECISION). Marker epochs/N/RED-SKIP-SEAL.json.
v8.2.9t3 (liveness fix, e217): the early seal records the signed deadline as POPULATION.sealed_at (= AUDIT-POPULATION
committed_at, which mimo_audit_bridge requires >= deadline); the real seal time is RED-SKIP-SEAL.json sealed_at. Epochs
early-sealed by t2 (sealed_at < deadline) get that one lower bound waived for their exact (sealed_at, deadline) pair. Pinned to the v822 mimo_lifecycle.py / mimo_coordinator.py (else inert).
"""
from __future__ import annotations
import argparse, ast, importlib.util, json, os, sys, textwrap, time
from pathlib import Path

P = Path(__file__).resolve().parent
VERSION = "affine-host-v829-train-authenticity-gate-v1"
SCOPE = "host-train-authenticity-gate-and-full-verify-v1"
V828_DIR = Path("/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/release-v828-signed")
FLAG_PATH = Path("/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/TRAIN-GATE.flag.json")
LEDGER_PATH = Path("/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/verify_ledger.json")
DENY_PATH = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/ops/train-verify-fail-denylist.json")
BAN_PATH = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/ops/uid-ban.json")
ASSURANCE_CHEAP = "natively-graded-cheap-check-pass-sampling-audit-offline"
OUTCOME_CHEAP = "cheap-check-pass"
AUDIT_MAX_INFLIGHT = 2  # v829d: one in-flight audit job per H200 verifier (2 pods)
AUDIT_BUDGET_PER_TICK = 2
# v8.2.9e (Jacob t569u/t570u 2026-10-24): operator train control. Sibling of the signed flag_path
# (no new operator config field, so the mining.json observer field set is unchanged). Absent file or
# mode "train" = exact v8.2.9d behaviour. Corrupt/unknown file = pause (fail closed: no trainer update).
CONTROL_PATH = FLAG_PATH.parent / "TRAIN-CONTROL.json"
EPOCHS_ROOT = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs")
CONTROL_MODES = ("train", "pause", "rollback-then-pause")
# v8.2.9f (Jacob pt595u 2026-10-25): trainer job lifetime (was lifecycle adapters.job_seconds = 7200).
TRAIN_JOB_SECONDS = 14400
# v8.2.9g (Jacob 2026-10-25): expensive-audit (verify role) job lifetime cap (was 1800 in v829d).
AUDIT_JOB_SECONDS = 7200
# v8.2.9h (Jacob 2026-10-26): miner collection window, switched only at an epoch boundary.
# v8.2.9m (Jacob t739u 2026-10-27 16:28 CR "15 min gather"): 900 accepted; sibling c1200 release is the revert.
COLLECTION_SECONDS = 900
V829H_COLLECTION_SECONDS = (900, 1200, 1800)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def load_auth():
    name = "train_gate_authenticity"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, P / "train_gate_authenticity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gate_enabled():
    try:
        return json.loads(FLAG_PATH.read_bytes()).get("enabled") is True
    except Exception:
        return False


def load_ledger():
    try:
        return json.loads(LEDGER_PATH.read_bytes())
    except Exception:
        return {}


def denylist_batch_set():
    auth = load_auth()
    keys, batches = set(), set()
    if not DENY_PATH.is_file():
        return keys, batches
    doc = json.loads(DENY_PATH.read_bytes())
    try:
        keys |= auth.denylist_keys_from_document(doc)
    except Exception:
        pass
    for b in doc.get("batch_sha256") or []:
        if isinstance(b, str) and len(b) == 64:
            batches.add(b)
    return keys, batches


def banned_miners():
    """Hard UID bans (offline audit TOPLOC/sampling fail). Fail closed on a corrupt file."""
    if not BAN_PATH.is_file():
        return set()
    doc = json.loads(BAN_PATH.read_bytes())
    out = set()
    for row in doc.get("banned") or []:
        m = row.get("miner") if isinstance(row, dict) else row
        need(type(m) is str and len(m) == 64, "uid-ban miner public key")
        out.add(m)
    return out


def classify_selected(sealed, ledger):
    auth = load_auth()
    by_batch = ledger.get("by_batch_sha256") or {}
    deny_keys, deny_batches = denylist_batch_set()
    banned = banned_miners()
    eligible, excluded = [], []
    for row in list(sealed.get("selected") or []):
        miner, batch = row.get("miner"), row.get("batch_sha256")
        if type(miner) is not str or type(batch) is not str:
            excluded.append(dict(row, reason="malformed-row"))
            continue
        key = auth.batch_key(miner, batch)
        st = by_batch.get(batch, "pending")
        if isinstance(st, dict):
            st = st.get("outcome") or st.get("status") or "pending"
        if miner in banned:
            excluded.append(dict(row, reason="uid-banned", sampling_audit_outcome=st or "pending"))
            continue
        if key in deny_keys or batch in deny_batches:
            excluded.append(dict(row, reason="sampling-verify-denylist",
                                 sampling_audit_outcome=st or "confirmed_invalid"))
            continue
        disp = auth.classify_for_train(st)
        if disp == "eligible":
            packed = dict(row)
            packed["assurance"] = auth.ASSURANCE_VERIFIED
            packed["sampling_audit_outcome"] = "verified_valid"
            eligible.append(packed)
        elif disp == "deny":
            excluded.append(dict(row, reason="sampling-verify-fail", sampling_audit_outcome=st))
        else:
            # v829c: cheap check passed at seal; expensive audit is offline (not awaited).
            packed = dict(row)
            packed["assurance"] = ASSURANCE_CHEAP
            packed["sampling_audit_outcome"] = OUTCOME_CHEAP
            eligible.append(packed)
    return eligible, excluded


def train_control():
    """None (file absent) -> normal. Otherwise a dict with a valid mode; unreadable -> pause."""
    if not CONTROL_PATH.exists():
        return None
    try:
        doc = json.loads(CONTROL_PATH.read_bytes())
        need(type(doc) is dict and doc.get("mode") in CONTROL_MODES, "train control mode")
        if doc["mode"] == "rollback-then-pause":
            need(type(doc.get("rollback_epoch")) is int and doc["rollback_epoch"] > 0
                 and type(doc.get("target_checkpoint")) is str and len(doc["target_checkpoint"]) == 64,
                 "rollback target")
        return doc
    except Exception as exc:  # fail closed
        return dict(mode="pause", error=str(exc)[:200])


def rollback_result(self, doc, current):
    """Exact ROOT-signed result of a past FINALIZED epoch (checkpoint, step, original authorization)."""
    from subnet.backend_jobs import signed
    path = EPOCHS_ROOT / str(doc["rollback_epoch"]) / "FINALIZED.json"
    need(path.is_file() and not path.is_symlink(), "rollback source FINALIZED")
    final = signed(json.loads(path.read_bytes()), self.identity.id)
    need(final.get("policy_sha256") == self.policy.policy_sha256, "rollback source under the active release policy")
    res = final["result"]
    need(res.get("status") == "updated" and res["checkpoint"]["id"] == doc["target_checkpoint"],
         "rollback source is a real update to the target checkpoint")
    auth = signed(res["checkpoint_authorization"], self.identity.id)
    need(auth.get("policy_sha256") == self.policy.policy_sha256 and auth.get("checkpoint") == res["checkpoint"]
         and auth.get("optimizer_step") == res["optimizer_step"], "rollback authorization binding")
    return res


def join_audit_population(self, directory):
    """v8.2.8 invariant: FINALIZED never precedes its off-path AUDIT-POPULATION."""
    job = self.__dict__.get("_v828_audit_jobs", {}).get(str(directory))
    if job is not None and job.thread is not None and job.thread.is_alive():
        job.thread.join()
    need(job is not None and job.done, "original audit population not yet complete; finalize deferred")


def reload_original_scheduler(module):
    path = Path(module.__file__)
    tree = ast.parse(path.read_text())
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "archive_scoped_audit_scheduler":
            fn = node
            break
    need(fn is not None, "disk archive_scoped_audit_scheduler")
    lines = path.read_text().splitlines(keepends=True)
    src = textwrap.dedent("".join(lines[fn.lineno - 1:fn.end_lineno]))
    ns = dict(module.__dict__)
    exec(compile(src, str(path), "exec"), ns)
    return ns["archive_scoped_audit_scheduler"]


def publish_train_pack(self, sealed, m, eligible, excluded, auth):
    from subnet.storage import canonical
    from subnet.mimo_release_policy import sign, digest
    from subnet.complete_group_capture import _immutable
    adapter = getattr(self.services.make_training_job, "__self__", None)
    if adapter is None:
        return None
    pack_payload = dict(
        version=sealed["version"], epoch=sealed["epoch"], run_id=sealed["run_id"],
        checkpoint=sealed["checkpoint"], manifest_sha256=sealed["manifest_sha256"],
        population_sha256=sealed.get("population_sha256"), selection=sealed.get("selection"),
        selected=eligible, excluded=list(sealed.get("excluded") or []) + excluded,
        training_requires_sampling_audit=True, authenticity_gate=auth.VERSION,
        sealed_dispatch_selected_task_count=len(sealed.get("selected") or []),
        selected_task_count=len(eligible), selected_rollout_count=len(eligible) * 8,
        sampling_verify_stripped=len(excluded),
    )
    pack = sign(self.identity, pack_payload)
    pack_key = self.policy.namespace() + "/epochs/" + m["epoch"] + "/mimo/train-pack.json"
    _immutable(adapter.bucket, pack_key, canonical(pack))
    print(json.dumps(dict(component="train-pack", status="published", key=pack_key,
                          eligible=len(eligible), stripped=len(excluded),
                          sha256=digest(pack))), flush=True)
    return pack_key


# ---- v8.2.9d sidecar (inlined from sidecar_current_epoch.py sha256 67d06d58...) ----
PARK_OLD = '_parked_old_epoch_v829d'


def current_epoch_scheduler(sched_cls, *, max_inflight, budget_per_tick, job_seconds_cap=1800):
    import importlib
    mas = importlib.import_module('subnet.mimo_audit_scheduler')

    class CurrentEpoch(sched_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.maximum, self.budget = max_inflight, budget_per_tick
            self.seconds = min(self.seconds, job_seconds_cap)
            print(json.dumps(dict(component='audit-sidecar', status='current-epoch-only-v829d',
                                  max_inflight=self.maximum, budget_per_tick=self.budget,
                                  job_seconds=self.seconds)), flush=True)

        def _raw_state(self):
            # Original AuditScheduler._load (no archive parking, no v829d parking).
            for klass in type(self).__mro__:
                if '_load' in klass.__dict__ and not hasattr(klass, '_excluded') and klass is not CurrentEpoch:
                    return klass.__dict__['_load'](self)
            raise RuntimeError('base _load')

        def _newest(self, state):
            excluded = self._excluded(state) if hasattr(self, '_excluded') else {}
            eps = sorted(e for e in state['populations'] if e not in excluded)
            return eps[-1] if eps else None

        def admit(self, document):
            try:
                checksum = mas.digest(document)
                epoch = document['payload']['manifest_document']['payload']['epoch']
                if self._raw_state()['populations'].get(epoch) == checksum and (self.root/(checksum+'.population.json')).is_file():
                    return checksum
            except Exception:
                pass
            return super().admit(document)

        def populations(self):
            state = self._raw_state()
            newest = self._newest(state)
            if newest is None:
                return []
            checksum = state['populations'][newest]
            path = self.root/(checksum+'.population.json')
            mas.need(not path.is_symlink() and path.stat().st_size <= 256*1024**2, 'original audit population file')
            doc = json.loads(path.read_bytes()); mas.need(mas.digest(doc) == checksum, 'immutable population hash')
            # Authenticated once per process per immutable checksum (~130 s each for a 24 MB population);
            # the content hash above is still re-checked on every tick.
            seen = self.__dict__.setdefault('_v829d_authenticated', {})
            if seen.get(checksum) != newest:
                p = mas.authenticate_population(doc, self.identity.id, manifest_guard=self.guard)
                mas.need(p['manifest_document']['payload']['epoch'] == newest, 'population epoch directory binding')
                seen[checksum] = newest
            if hasattr(self, '_report'):
                self._report(state)
            return [doc]

        def _v829g_stale_epochs(self, now):
            # v8.2.9g: with 2 h jobs a stale epoch must not keep (or start) occupying a verifier.
            state = self._raw_state()
            newest = self._newest(state)
            dropped, busy = [], 0
            if newest is not None:
                for key, d in state['draws'].items():
                    jid = d.get('job_id')
                    if not jid or d['row']['epoch'] == newest or not d.get('expires_at', 0) > now:
                        continue
                    if not str(jid).startswith('mimo-audit-'):
                        continue
                    q = self._queue_status(jid)
                    if q is None:
                        continue
                    if q['status'] == 'queued':
                        with self.queue.transaction() as db:
                            cur = db.execute("UPDATE jobs SET status='expired' WHERE id=? AND role='verify' AND status='queued'", (jid,))
                            if cur.rowcount == 1:
                                self.queue.event(db, jid, 'expired', reason='v829g-stale-epoch',
                                                 epoch=d['row']['epoch'], newest=newest)
                                dropped.append(jid)
                    elif q['status'] == 'leased':
                        busy += 1
            self.maximum = max(0, max_inflight - busy)
            if dropped or busy:
                print(json.dumps(dict(component='audit-sidecar', status='stale-epoch-v829g', newest=newest,
                                      dropped_queued=dropped, stale_leased_running=busy,
                                      capacity_max=self.maximum)), flush=True)

        def tick(self, now=None):
            self.__dict__['_v829d_tick'] = True
            try:
                self._v829g_stale_epochs(time.time() if now is None else now)
                return super().tick(now)
            finally:
                self.maximum = max_inflight
                self.__dict__['_v829d_tick'] = False

        def _load(self):
            state = super()._load()
            if not self.__dict__.get('_v829d_tick'):
                return state
            newest = self._newest(state)
            old = {k: d for k, d in state['draws'].items() if d['row']['epoch'] != newest}
            state['draws'] = {k: d for k, d in state['draws'].items() if k not in old}
            state[PARK_OLD] = old
            return state

        def _save(self, state):
            value = dict(state)
            old = value.pop(PARK_OLD, {})
            mas.need(not set(old) & set(value['draws']), 'parked-old and active draws disjoint')
            value['draws'] = dict(value['draws'], **old)
            super()._save(value)

    CurrentEpoch.__name__ = sched_cls.__name__
    CurrentEpoch.__qualname__ = sched_cls.__qualname__
    return CurrentEpoch


def install_v829(module, Lifecycle):
    auth = load_auth()
    inner_training = Lifecycle._training
    INNER['training'] = inner_training  # v8.2.9l: K=1 path builds on the original inner training
    original_factory = reload_original_scheduler(module)

    def archive_scoped_audit_scheduler(base, auditable_archives, *, report_name='UNAUDITABLE-BY-DESIGN.json'):
        sched_cls = original_factory(base, auditable_archives, report_name=report_name)

        # v8.2.9d: expensive-audit sidecar = newest auditable epoch only, no backlog, limits on the
        # attributes the base reads (self.maximum/self.budget), job lifetime <= AUDIT_JOB_SECONDS
        # (v829g: 7200 s, was 1800). Inlined (the discovery observer pins exactly two host files).
        return current_epoch_scheduler(sched_cls, max_inflight=AUDIT_MAX_INFLIGHT,
                                       budget_per_tick=AUDIT_BUDGET_PER_TICK, job_seconds_cap=AUDIT_JOB_SECONDS)

    def _training_v829b(self, directory, opening, dispatch, current, now):
        from subnet.backend_jobs import signed
        from subnet.storage import canonical
        from subnet.mimo_release_policy import sign, digest
        from subnet.mimo_lifecycle import save_manifest
        from subnet.complete_group_capture import _immutable
        import time as _time
        path = Path(directory) / 'TRAINING-JOB.json'
        if not gate_enabled():
            return inner_training(self, directory, opening, dispatch, current, now)
        # In-flight pre-gate (unaudited) jobs: never rewrite; let original path finish.
        if path.exists():
            from subnet.backend_jobs import signed as _signed
            from subnet.mimo_release_policy import digest as _digest
            from subnet.mimo_lifecycle import save_manifest
            from subnet.storage import canonical as _canonical
            import time as _time
            existing_env = json.loads(path.read_bytes())
            existing = _signed(existing_env, self.identity.id)
            if existing.get('training', {}).get('training_requires_sampling_audit') is not True:
                return inner_training(self, directory, opening, dispatch, current, now)
            m = _signed(opening, self.identity.id)
            result_path = Path(directory) / 'TRAINING-RESULT.json'
            if result_path.exists():
                result_envelope = json.loads(result_path.read_bytes())
            else:
                result_envelope = self.services.training_result(existing['job_id'])
                if result_envelope is None:
                    return None
                save_manifest(result_path, result_envelope)
            request = _signed(result_envelope, existing['trainer_worker'])
            need(request.get('action') == 'report' and request.get('job_id') == existing['job_id'],
                 'original authorized trainer report request')
            report = request['report']
            from subnet.mimo_worker import validate_report
            validate_report(report, existing, m, self.identity.id, _digest(existing), _time.time())
            need(report['training']['status'] == 'updated', 'nonempty verified train pack requires a real trainer update')
            receipt = report['publication_receipt']
            published = _signed(receipt, existing['training']['publisher_id'])
            from subnet.mimo_release_policy import descriptor
            actual = descriptor(self.services.readback_publication(published))
            need(actual == published['descriptor'], 'full stored model readback must match signed publication')
            authorization = self.policy.authorize_checkpoint(
                actual, input_checkpoint=current['checkpoint']['id'],
                optimizer_step=report['training']['optimizer_step'],
                publication_receipt_sha256=_digest(receipt), identity=self.identity)
            join_audit_population(self, directory)
            return dict(status='updated', checkpoint=actual,
                        optimizer_step=report['training']['optimizer_step'],
                        checkpoint_authorization=authorization,
                        publication_receipt_sha256=_digest(receipt))
        m = signed(opening, self.identity.id)
        sealed = signed(dispatch, self.identity.id)
        if sealed.get('selected_task_count') == 0:
            return inner_training(self, directory, opening, dispatch, current, now)
        ledger = load_ledger()
        eligible, excluded = classify_selected(sealed, ledger)
        print(json.dumps(dict(
            component='train-authenticity-gate', epoch=m.get('epoch'),
            sealed=len(sealed.get('selected') or []), eligible=len(eligible),
            stripped=len(excluded), gate='cheap-check-v829c',
            strip_reasons={r: sum(1 for x in excluded if x.get('reason') == r)
                           for r in sorted({x.get('reason') for x in excluded})},
        )), flush=True)
        if not eligible:
            marker = dict(
                version='sn120-train-gate-skip-v1', status='no-update',
                reason='no-sampling-verified-train-pack',
                authenticity_gate=auth.VERSION, sampling_verify_stripped=len(excluded),
                eligible=0, sealed=len(sealed.get('selected') or []),
                epoch=m.get('epoch'), optimizer_step=current.get('optimizer_step'),
            )
            (Path(directory) / 'TRAIN-GATE-SKIP.json').write_text(
                json.dumps(marker, indent=2, sort_keys=True) + chr(10))
            join_audit_population(self, directory)
            return dict(
                status='no-update', checkpoint=current['checkpoint'],
                optimizer_step=current['optimizer_step'],
                checkpoint_authorization=current.get('checkpoint_authorization'),
                reason='no-sampling-verified-train-pack',
            )
        adapter = getattr(self.services.make_training_job, '__self__', None)
        need(adapter is not None, 'training adapter')
        pack_payload = dict(
            version=sealed['version'], epoch=sealed['epoch'], run_id=sealed['run_id'],
            checkpoint=sealed['checkpoint'], manifest_sha256=sealed['manifest_sha256'],
            population_sha256=sealed.get('population_sha256'), selection=sealed.get('selection'),
            selected=eligible, excluded=list(sealed.get('excluded') or []) + excluded,
            training_requires_sampling_audit=True, authenticity_gate=auth.VERSION,
            sealed_dispatch_selected_task_count=len(sealed.get('selected') or []),
            selected_task_count=len(eligible), selected_rollout_count=len(eligible) * 8,
            sampling_verify_stripped=len(excluded),
        )
        pack = sign(self.identity, pack_payload)
        pack_key = self.policy.namespace() + '/epochs/' + m['epoch'] + '/mimo/train-pack.json'
        _immutable(adapter.bucket, pack_key, canonical(pack))
        print(json.dumps(dict(component='train-pack', status='published', key=pack_key,
                              eligible=len(eligible), stripped=len(excluded),
                              sha256=digest(pack))), flush=True)
        documents = {}
        for row in eligible:
            doc = row['document']
            response = adapter.bucket.client.head_object(Bucket=adapter.bucket.name, Key=doc['key'])
            need(type(response['ContentLength']) is int and 0 < response['ContentLength'] <= 128 * 1024 ** 2,
                 'bounded actual captured training document')
            documents[doc['sha256']] = dict(
                url=adapter.bucket.presign(doc['key'], 'get_object', 86400),
                size=response['ContentLength'])
        from subnet.mimo_worker import VERSION as WORKER
        now_ts = _time.time()
        value = dict(
            schema=1, worker_policy=WORKER, job_id='mimo-train-' + digest(pack)[:48], role='train',
            created_at=now_ts, expires_at=now_ts + adapter.job_seconds, manifest=opening,
            source_files=adapter.source_files, runtime_versions=adapter.runtime_versions,
            trainer_worker=adapter.trainer_worker,
            checkpoint_read_urls=adapter.checkpoint_urls(m['checkpoint']),
            training=dict(
                run_id=m['run_id'], optimizer_step=current['optimizer_step'], config=adapter.trainer_config,
                dispatch=dict(url=adapter.bucket.presign(pack_key, 'get_object', 86400),
                              sha256=digest(pack), size=len(canonical(pack))),
                sealed_dispatch_sha256=digest(dispatch), documents=documents,
                publication_namespace=self.policy.namespace(), publisher_id=adapter.publisher_id,
                role_config_file=adapter.role_config, training_requires_sampling_audit=True,
                authenticity_gate=auth.VERSION, sampling_verify_stripped=len(excluded),
            ),
        )
        envelope = sign(self.identity, value)
        job = signed(envelope, self.identity.id)
        training = job.get('training', {})
        need(job.get('role') == 'train' and job.get('manifest') == opening
             and training.get('optimizer_step') == current['optimizer_step']
             and training.get('publication_namespace') == self.policy.namespace()
             and training.get('training_requires_sampling_audit') is True
             and training.get('sealed_dispatch_sha256') == digest(dispatch)
             and type(training.get('dispatch')) is dict and training['dispatch'].get('sha256'),
             'trainer job must consume authenticity-filtered train pack of this sealed dispatch')
        if not training.get('documents'):
            save_manifest(path, envelope)
            return dict(status='no-update', checkpoint=current['checkpoint'],
                        optimizer_step=current['optimizer_step'],
                        checkpoint_authorization=current.get('checkpoint_authorization'),
                        reason='no-sampling-verified-train-pack')
        save_manifest(path, envelope)
        need(self.services.enqueue_training(envelope) == job['job_id'], 'original trainer queue acknowledgement')
        print(json.dumps(dict(component='train-authenticity-gate', status='job-enqueued',
                              job_id=job['job_id'], eligible=len(eligible))), flush=True)
        return None

    def _training(self, directory, opening, dispatch, current, now):
        """v8.2.9c cheap-check gate. Never waits on the expensive audit.

        - gate flag off: original v8.2.8 path.
        - existing TRAINING-JOB that requires sampling audit (v829b era): v829b result path.
        - existing TRAINING-JOB (original shape): original path with the exact dispatch it was built from.
        - new: rows of banned miners / offline-audit fails are stripped. Nothing stripped -> original
          path on the sealed dispatch. Some stripped -> a lifecycle-signed filtered dispatch (original
          shape, original per-row assurance) is published + persisted and the original path trains it.
          Everything stripped -> TRAIN-GATE-SKIP no-update.
        """
        from subnet.backend_jobs import signed
        from subnet.storage import canonical
        from subnet.mimo_release_policy import sign, digest
        from subnet.mimo_lifecycle import save_manifest
        from subnet.complete_group_capture import _immutable
        import time as _time
        if not gate_enabled():
            return inner_training(self, directory, opening, dispatch, current, now)
        path = Path(directory) / 'TRAINING-JOB.json'
        filtered_path = Path(directory) / 'TRAIN-DISPATCH-V829C.json'
        if path.exists():
            existing = signed(json.loads(path.read_bytes()), self.identity.id)
            if existing.get('training', {}).get('training_requires_sampling_audit') is True:
                return _training_v829b(self, directory, opening, dispatch, current, now)
            use = json.loads(filtered_path.read_bytes()) if filtered_path.exists() else dispatch
            return inner_training(self, directory, opening, use, current, now)
        m = signed(opening, self.identity.id)
        sealed = signed(dispatch, self.identity.id)
        if sealed.get('selected_task_count') == 0:
            return inner_training(self, directory, opening, dispatch, current, now)
        ledger = load_ledger()
        eligible, excluded = classify_selected(sealed, ledger)
        reasons = {r: sum(1 for x in excluded if x.get('reason') == r)
                   for r in sorted({x.get('reason') for x in excluded})}
        print(json.dumps(dict(
            component='train-authenticity-gate', epoch=m.get('epoch'),
            sealed=len(sealed.get('selected') or []), eligible=len(eligible),
            stripped=len(excluded), strip_reasons=reasons, gate='cheap-check-v829c',
        )), flush=True)
        if not eligible:
            marker = dict(
                version='sn120-train-gate-skip-v1', status='no-update',
                reason='every-sealed-group-banned-or-audit-failed', gate='cheap-check-v829c',
                authenticity_gate=load_auth().VERSION, sampling_verify_stripped=len(excluded),
                strip_reasons=reasons, eligible=0, sealed=len(sealed.get('selected') or []),
                epoch=m.get('epoch'), optimizer_step=current.get('optimizer_step'),
            )
            (Path(directory) / 'TRAIN-GATE-SKIP.json').write_text(
                json.dumps(marker, indent=2, sort_keys=True) + chr(10))
            join_audit_population(self, directory)
            return dict(
                status='no-update', checkpoint=current['checkpoint'],
                optimizer_step=current['optimizer_step'],
                checkpoint_authorization=current.get('checkpoint_authorization'),
                reason='every-sealed-group-banned-or-audit-failed',
            )
        if not excluded:
            return inner_training(self, directory, opening, dispatch, current, now)
        # Partial strip: original-shape filtered dispatch, original rows (no relabelled assurance).
        stripped_batches = {x.get('batch_sha256') for x in excluded}
        rows = [r for r in sealed['selected'] if r.get('batch_sha256') not in stripped_batches]
        need(len(rows) == len(eligible), 'filtered rows match eligible')
        group = (sealed['selected_rollout_count'] // sealed['selected_task_count'])
        payload = dict(sealed, selected=rows, selected_task_count=len(rows),
                       selected_rollout_count=len(rows) * group)
        (Path(directory) / 'TRAIN-GATE-STRIP.json').write_text(json.dumps(dict(
            version='sn120-train-gate-strip-v1', gate='cheap-check-v829c', epoch=m.get('epoch'),
            sealed=len(sealed['selected']), eligible=len(rows), strip_reasons=reasons,
            stripped=[dict(miner=x.get('miner'), batch_sha256=x.get('batch_sha256'),
                           reason=x.get('reason')) for x in excluded]), indent=2, sort_keys=True) + chr(10))
        filtered = sign(self.identity, payload)
        adapter = getattr(self.services.make_training_job, '__self__', None)
        need(adapter is not None, 'training adapter')
        key = self.policy.namespace() + '/epochs/' + m['epoch'] + '/mimo/dispatch-v829c.json'
        _immutable(adapter.bucket, key, canonical(filtered))
        documents = {}
        for row in rows:
            doc = row['document']
            response = adapter.bucket.client.head_object(Bucket=adapter.bucket.name, Key=doc['key'])
            need(type(response['ContentLength']) is int and 0 < response['ContentLength'] <= 128 * 1024 ** 2,
                 'bounded actual captured training document')
            documents[doc['sha256']] = dict(url=adapter.bucket.presign(doc['key'], 'get_object', 86400),
                                            size=response['ContentLength'])
        need(len(documents) == len(rows), 'one original compact document per training group')
        from subnet.mimo_worker import VERSION as WORKER
        t0 = _time.time()
        value = dict(schema=1, worker_policy=WORKER, job_id='mimo-train-' + digest(filtered)[:48], role='train',
            created_at=t0, expires_at=t0 + adapter.job_seconds, manifest=opening,
            source_files=adapter.source_files, runtime_versions=adapter.runtime_versions,
            trainer_worker=adapter.trainer_worker, checkpoint_read_urls=adapter.checkpoint_urls(m['checkpoint']),
            training=dict(run_id=m['run_id'], optimizer_step=current['optimizer_step'], config=adapter.trainer_config,
                dispatch=dict(url=adapter.bucket.presign(key, 'get_object', 86400), sha256=digest(filtered),
                              size=len(canonical(filtered))),
                documents=documents, publication_namespace=self.policy.namespace(),
                publisher_id=adapter.publisher_id, role_config_file=adapter.role_config))
        envelope = sign(self.identity, value)
        save_manifest(filtered_path, filtered)
        save_manifest(path, envelope)
        print(json.dumps(dict(component='train-authenticity-gate', status='filtered-dispatch',
                              key=key, eligible=len(rows), stripped=len(excluded))), flush=True)
        return inner_training(self, directory, opening, filtered, current, now)

    def _training_v829e(self, directory, opening, dispatch, current, now):
        """v8.2.9e operator control over the v8.2.9c/d path.

        - no TRAIN-CONTROL.json / mode train: v8.2.9c/d path unchanged.
        - an existing TRAINING-JOB.json: never abandoned; v8.2.9c/d path unchanged.
        - rollback-then-pause and current != target: finalize this epoch as no-update onto the target
          checkpoint, reusing the target epoch's exact ROOT-signed result (checkpoint, optimizer_step,
          checkpoint_authorization). No trainer job, no new signature over model bytes.
        - pause (or rollback already applied): no-update, checkpoint unchanged.
        """
        from subnet.backend_jobs import signed
        adapter = getattr(self.services.make_training_job, "__self__", None)
        need(adapter is not None and type(getattr(adapter, "job_seconds", None)) is int, "v829f training adapter")
        if adapter.job_seconds != TRAIN_JOB_SECONDS:
            print(json.dumps(dict(component="train-job-seconds-v829f", previous=adapter.job_seconds, job_seconds=TRAIN_JOB_SECONDS)), flush=True)
            adapter.job_seconds = TRAIN_JOB_SECONDS
        doc = train_control()
        if doc is None or doc["mode"] == "train" or (Path(directory) / "TRAINING-JOB.json").exists():
            return _training(self, directory, opening, dispatch, current, now)
        m = signed(opening, self.identity.id)
        marker = dict(version="sn120-train-control-v829e", epoch=m.get("epoch"), mode=doc["mode"],
                      input_checkpoint=current["checkpoint"]["id"], input_optimizer_step=current["optimizer_step"],
                      control=doc)
        if doc["mode"] == "rollback-then-pause" and current["checkpoint"]["id"] != doc["target_checkpoint"]:
            res = rollback_result(self, doc, current)
            marker.update(status="no-update", action="rollback", checkpoint=res["checkpoint"]["id"],
                          optimizer_step=res["optimizer_step"], source_epoch=doc["rollback_epoch"])
            (Path(directory) / "TRAIN-CONTROL-ROLLBACK.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + chr(10))
            print(json.dumps(dict(component="train-control-v829e", **{k: v for k, v in marker.items() if k != "control"})), flush=True)
            join_audit_population(self, directory)
            return dict(status="no-update", checkpoint=res["checkpoint"], optimizer_step=res["optimizer_step"],
                        checkpoint_authorization=res["checkpoint_authorization"],
                        reason="operator-rollback-to-e%d-%s" % (doc["rollback_epoch"], doc["target_checkpoint"][:8]))
        marker.update(status="no-update", action="pause", checkpoint=current["checkpoint"]["id"],
                      optimizer_step=current["optimizer_step"])
        (Path(directory) / "TRAIN-CONTROL-PAUSE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + chr(10))
        print(json.dumps(dict(component="train-control-v829e", **{k: v for k, v in marker.items() if k != "control"})), flush=True)
        join_audit_population(self, directory)
        return dict(status="no-update", checkpoint=current["checkpoint"], optimizer_step=current["optimizer_step"],
                    checkpoint_authorization=current.get("checkpoint_authorization"), reason="operator-train-pause")

    _training_v829e.__qualname__ = inner_training.__qualname__
    _training_v829e.__module__ = inner_training.__module__
    Lifecycle._training = _training_v829e
    module.archive_scoped_audit_scheduler = archive_scoped_audit_scheduler
    return dict(
        v829=True, v829c=True, v829d=True, v829e=True, v829f=True, v829g=True, train_job_seconds=TRAIN_JOB_SECONDS, audit_job_seconds=AUDIT_JOB_SECONDS, train_gate='cheap-check-ban-deny-v829c',
        train_control=(train_control() or dict(mode='train', file='absent')).get('mode'),
        audit_scheduler='current-epoch-sidecar-v829d',
        audit_max_inflight=AUDIT_MAX_INFLIGHT,
        audit_budget_per_tick=AUDIT_BUDGET_PER_TICK,
        gate_enabled=gate_enabled(),
    )

# v8.2.9d: two replacement H200 verify workers (Lium affine-verifier-h200-1/-2, Jacob pt566u 2026-10-23).
# v8.2.9r (Jacob 2026-10-28 12:51 CR): the 10-23 pods (calm-shark-c4, zesty-shark-c7) went dark at 11:12 CR and were
# released; their on-pod seeds are gone. Ids below are the fresh pods noble-raven-aa (74563cf0) and gentle-fox-2a (af29cca2).
# Their seeds were generated on the pods and never left them; only public ids are pinned here.
# Added to the coordinator roster with role "verify" only; existing roster entries are unchanged.
V829D_VERIFY_WORKERS = {
    "d4483df3329295d015c1c1a482c4ceaa22d7769fe1324df88e54fe4bf45ac619": ["verify"],  # affine-verifier-h200-1 (noble-raven-aa)
    "eccfec2e7a5f17a3c0cb7e768ad8c75da6845b9028d6735225f18a2f1c84ce6a": ["verify"],  # affine-verifier-h200-2 (gentle-fox-2a)
}


def install_v829d_verifier_roster(Coordinator):
    for k, v in V829D_VERIFY_WORKERS.items():
        need(type(k) is str and len(k) == 64 and int(k, 16) >= 0 and v == ["verify"], "exact v829d verify worker id")
    original_init = Coordinator.__init__

    def __init__(self, path, authority, workers, *a, **kw):
        need(type(workers) is dict and not set(V829D_VERIFY_WORKERS) & set(workers), "v829d verifier ids are new")
        merged = dict(workers)
        merged.update({k: list(v) for k, v in V829D_VERIFY_WORKERS.items()})
        original_init(self, path, authority, merged, *a, **kw)

    __init__.__qualname__ = original_init.__qualname__
    Coordinator.__init__ = __init__
    return sorted(V829D_VERIFY_WORKERS)


def install_v829h_collection(Lifecycle):
    """v8.2.9h: boundary re-stamp of CURRENT.collection_seconds (pattern of v8.2.4 max_tasks)."""
    need(COLLECTION_SECONDS in V829H_COLLECTION_SECONDS, "v829h collection window")
    from subnet.controller import save_manifest
    prior_current = Lifecycle._current

    def _current(self):
        target = COLLECTION_SECONDS
        path = self.root / 'CURRENT.json'
        if path.is_file() and not path.is_symlink():
            current = json.loads(path.read_bytes())
            stamped = current.get('collection_seconds')
            if stamped in V829H_COLLECTION_SECONDS and stamped != target:
                planned = (self.root / 'epochs' / str(current['epoch_index']) / 'PLAN.json').exists()
                if planned:
                    self.collection_seconds = stamped              # in-flight epoch keeps its planned window
                else:
                    save_manifest(path, dict(current, collection_seconds=target))  # epoch-boundary re-stamp
                    print(json.dumps(dict(component="collection-seconds-v829h", epoch_index=current['epoch_index'],
                                          previous=stamped, collection_seconds=target)), flush=True)
                    self.collection_seconds = target
            elif stamped == target:
                self.collection_seconds = target
        return prior_current(self)

    _current.__qualname__, _current.__module__ = prior_current.__qualname__, prior_current.__module__
    Lifecycle._current = _current
    return dict(v829h=True, collection_seconds=COLLECTION_SECONDS, accepted=list(V829H_COLLECTION_SECONDS))


# ---- v8.2.9i promotion gate ----
PG_FLAG = FLAG_PATH.parent / "PROMOTION-GATE.json"
PG_STATE = Path("/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs")
PG_DEFAULTS = dict(decision_timeout_seconds=3600, restore_wait_seconds=1200)


def pg_flag():
    try:
        doc = json.loads(PG_FLAG.read_bytes())
        return dict(PG_DEFAULTS, **doc) if type(doc) is dict and doc.get("enabled") is True else None
    except Exception:
        return None


def _pg_read(path):
    try:
        return json.loads(Path(path).read_bytes())
    except Exception:
        return None


def install_v829i_promotion_gate(Lifecycle):
    prior = Lifecycle._training

    def _training(self, directory, opening, dispatch, current, now):
        f = pg_flag()
        if f is None:
            return prior(self, directory, opening, dispatch, current, now)
        d = Path(directory)
        e = int(d.name)
        H = current["checkpoint"]["id"]
        ctl = train_control()
        # (1) after a live RED, never enqueue a trainer job before the pod trainer is back on H
        if not (d / "TRAINING-JOB.json").exists() and (ctl is None or ctl.get("mode") == "train"):
            prev = PG_STATE / str(e - 1)
            pdec = _pg_read(prev / "DECISION.json") or {}
            pfin = _pg_read(d.parent / str(e - 1) / "FINALIZED.json") or {}
            pres = (pfin.get("payload") or pfin).get("result") or {}
            if (pdec.get("decision") == "red" and pdec.get("head") == H and str(pres.get("reason", "")).startswith("promotion-gate-red")
                    and not (prev / "TRAINER-RESTORED.json").exists()):
                first = self.__dict__.setdefault("_v829i_restore_wait", {}).setdefault(e, time.time())
                if time.time() - first < float(f["restore_wait_seconds"]):
                    if self.__dict__.setdefault("_v829i_logged", set()).isdisjoint({("restore", e)}):
                        self.__dict__["_v829i_logged"].add(("restore", e))
                        print(json.dumps(dict(component="promotion-gate-v829i", epoch=e, status="waiting-trainer-restore", head=H[:12])), flush=True)
                    return None
                m = __import__("subnet.backend_jobs", fromlist=["signed"]).signed(opening, self.identity.id)
                (d / "PROMOTION-GATE-RESTORE-PENDING.json").write_text(json.dumps(dict(
                    version="sn120-promotion-gate-v829i", epoch=m.get("epoch"), status="no-update", head=H,
                    reason="trainer not restored to head within restore_wait_seconds"), indent=2, sort_keys=True) + chr(10))
                join_audit_population(self, directory)
                return dict(status="no-update", checkpoint=current["checkpoint"], optimizer_step=current["optimizer_step"],
                            checkpoint_authorization=current.get("checkpoint_authorization"), reason="promotion-gate-restore-pending")
        cache = self.__dict__.setdefault("_v829i_results", {})
        res = cache.get(str(d))
        if res is None:
            res = prior(self, directory, opening, dispatch, current, now)
            if res is None:
                return None
        if res.get("status") != "updated":
            return res
        cache[str(d)] = res
        C = res["checkpoint"]["id"]
        dec = _pg_read(PG_STATE / str(e) / "DECISION.json") or {}
        marker = dict(version="sn120-promotion-gate-v829i", epoch_index=e, head=H, candidate=C)
        if dec.get("head") == H and dec.get("candidate") == C and dec.get("decision") in ("green", "red"):
            ev = dec.get("evaluated") or {}
            marker.update(decision=dec["decision"], reasons=dec.get("reasons"), paired_z=ev.get("z"), paired_tasks=ev.get("n"),
                          candidate_cap_rate=ev.get("c_cap_rate"), head_pass=ev.get("h_pass"), candidate_pass=ev.get("c_pass"),
                          red_credit=dec.get("red_credit", "zero"))
            (d / "PROMOTION-GATE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + chr(10))
            print(json.dumps(dict(component="promotion-gate-v829i", **{k: v for k, v in marker.items() if k != "reasons"})), flush=True)
            cache.pop(str(d), None)
            if dec["decision"] == "green":
                return res
            return dict(status="no-update", checkpoint=current["checkpoint"], optimizer_step=current["optimizer_step"],
                        checkpoint_authorization=current.get("checkpoint_authorization"),
                        reason="promotion-gate-red-e%d-%s-kept-%s" % (e, C[:8], H[:8]))
        rp = d / "TRAINING-RESULT.json"
        t0 = rp.stat().st_mtime if rp.exists() else time.time()
        if time.time() - t0 > float(f["decision_timeout_seconds"]):
            marker.update(decision="timeout-promote", reason="no gate decision within %ss" % f["decision_timeout_seconds"])
            (d / "PROMOTION-GATE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + chr(10))
            print(json.dumps(dict(component="promotion-gate-v829i", **marker)), flush=True)
            cache.pop(str(d), None)
            return res
        if self.__dict__.setdefault("_v829i_logged", set()).isdisjoint({("hold", e)}):
            self.__dict__["_v829i_logged"].add(("hold", e))
            print(json.dumps(dict(component="promotion-gate-v829i", epoch=e, status="holding-for-gate", head=H[:12], candidate=C[:12])), flush=True)
        return None

    _training.__qualname__, _training.__module__ = prior.__qualname__, prior.__module__
    Lifecycle._training = _training
    return dict(v829i=True, flag=str(PG_FLAG), enabled_now=pg_flag() is not None)



# ---------------- v8.2.9j evaluator opening memo (Jacob pt696u) ----------------
LIFE_OPT_PATH = FLAG_PATH.parent / "EPOCH-TIME-LIFECYCLE.json"


def _v829j_opt(name):
    try:
        d = json.loads(LIFE_OPT_PATH.read_text())
    except FileNotFoundError:
        return True
    except Exception:
        return False  # unreadable -> exact v829i behaviour
    return not (isinstance(d, dict) and d.get(name) is False)


def install_v829j_eval_memo():
    import subnet.mimo_evaluation_scheduler as M
    cls = M.EvaluationScheduler
    original = cls.tick
    if getattr(original, "_v829j", False):
        return dict(v829j=True, eval_memo="already")
    seen = set()

    def fp(o):
        try:
            p = o["payload"]
            return (o["signer"], o["signature"], p.get("epoch"), (p.get("checkpoint") or {}).get("id"), p.get("start"), p.get("deadline"))
        except Exception:
            return None

    def tick(self, openings, *, now=None):
        if not _v829j_opt("eval_memo"):
            return original(self, openings, now=now)
        fresh = [o for o in openings if fp(o) is None or fp(o) not in seen]
        result = original(self, fresh, now=now)
        # reached only if the original tick verified every opening in `fresh` (it raises before any job logic otherwise)
        seen.update(k for k in map(fp, fresh) if k is not None)
        return result
    tick._v829j = True
    cls.tick = tick
    return dict(v829j=True, eval_memo="opening-signature-memo", switch=str(LIFE_OPT_PATH), on_now=_v829j_opt("eval_memo"))



# ---------------- v8.2.9k seal document read-ahead (Jacob pt696u) ----------------
def install_v829k_seal_prefetch(Coordinator):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    mod = sys.modules[Coordinator.__module__]   # subnet.mimo_coordinator: the module whose seal resolves _get
    need(getattr(mod, "Coordinator", None) is Coordinator and callable(getattr(mod, "_get", None)), "v829k seal module")
    g = mod.__dict__
    original_get = g["_get"]
    original_seal = Coordinator.seal
    if getattr(original_seal, "_v829k", False):
        return dict(v829k=True, seal_prefetch="already")
    WORKERS, AHEAD = 10, 80  # botocore default pool = 10 connections
    local = threading.local()

    def _order(root):
        try:
            env = json.loads((Path(root) / "POPULATION.json").read_bytes())
            keys = []
            for cap in env["payload"]["captures"]:
                for item in cap["payload"]["documents"]:
                    keys.append(item["key"])
            return keys
        except Exception:
            return []

    def _get(bucket, key, limit):
        st = getattr(local, "seal", None)
        if st is None:
            return original_get(bucket, key, limit)
        try:
            if st["keys"] is None:
                st["keys"] = _order(st["root"]); st["index"] = {k: i for i, k in enumerate(st["keys"])}
            i = st["index"].get(key)
            if i is not None:
                for j in range(st["next"], min(len(st["keys"]), i + AHEAD)):
                    k = st["keys"][j]
                    st["fut"][(k, limit)] = st["pool"].submit(original_get, bucket, k, limit)
                st["next"] = max(st["next"], min(len(st["keys"]), i + AHEAD))
            f = st["fut"].pop((key, limit), None)
        except Exception:
            f = None
        if f is not None and f.done() is False:
            st["waits"] += 1
        if f is not None:
            try:
                r = f.result(); st["hits"] += 1
                return r
            except Exception:
                st["errors"] += 1
        st["misses"] += 1
        return original_get(bucket, key, limit)

    def seal(self, *a, **kw):
        if not _v829j_opt("seal_prefetch"):
            return original_seal(self, *a, **kw)
        pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="v829k-seal-read")
        local.seal = dict(root=str(self.root), keys=None, index={}, next=0, fut={}, pool=pool, hits=0, misses=0, errors=0, waits=0)
        t0 = time.time()
        try:
            return original_seal(self, *a, **kw)
        finally:
            st = local.seal; local.seal = None
            for f in st["fut"].values():
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            if st["hits"] or st["misses"]:
                print(json.dumps(dict(component="seal-prefetch-v829k", hits=st["hits"], misses=st["misses"], errors=st["errors"],
                                      waits=st["waits"], seconds=round(time.time() - t0, 1))), flush=True)
    seal._v829k = True
    g["_get"] = _get
    Coordinator.seal = seal
    return dict(v829k=True, seal_prefetch="population-order-read-ahead", workers=WORKERS, ahead=AHEAD,
                module=g.get("__name__"), on_now=_v829j_opt("seal_prefetch"))


# ---------------- v8.2.9n capture snapshot-PUT fan-out + seal pre-grade cache (Jacob t739u) ----------------
# Switches (read on every use, no restart): staged/EPOCH-TIME-LIFECYCLE.json
#   {"capture_put_fanout": false} -> exact v8.2.9m capture;  {"seal_pregrade": false} -> exact v8.2.9m seal grading.
V829N_PUT_WORKERS = 9          # one miner commitment holds at most 9 groups; botocore default pool = 10
V829N_PREGRADE_WORKERS = 24    # background native graders while an epoch is collecting (seal pool stays 32)
V829N_PREGRADE_FETCH = 8


def install_v829n_capture_put():
    """Inside an original capture() (capture thread, CapturePass memo active only), the private snapshot
    copies `.../complete-group-snapshots/<miner>/<commitment-sha>/training/<slot>.json` are PUT concurrently
    with the ORIGINAL conditional _immutable. Before the same capture's commitment.json PUT, every one of
    them must have succeeded (first failure, in slot order, is re-raised there, before ledger admission).
    Same objects, same bytes, same write-once checks, same v8.2.7 write-skip memo; only latency changes."""
    import re, threading
    from concurrent.futures import ThreadPoolExecutor
    from subnet.storage import sha
    cache_mod = sys.modules.get("affine_host_capture_cache")
    need(cache_mod is not None and callable(getattr(cache_mod, "_install_incremental_memo", None))
         and hasattr(cache_mod, "_active"), "v829n capture cache module loaded")
    import subnet.complete_group_capture as cgc
    cache_mod._install_incremental_memo()   # idempotent; identical to the first call every CapturePass makes
    inner = cgc._immutable
    if getattr(inner, "_v829n", False):
        return dict(v829n_put=True, capture_put_fanout="already")
    raw = getattr(inner, "_v827_original", None)
    need(callable(raw) and inner.__module__ == cgc.__name__, "v829n v8.2.7 _immutable wrapper present")
    snap = re.compile(r"^(private/[^/]+/complete-group-snapshots/[0-9a-f]{64}/[0-9a-f]{64})/training/[0-9]+\.json$")
    pool = ThreadPoolExecutor(max_workers=V829N_PUT_WORKERS, thread_name_prefix="v829n-snapshot-put")
    pending = {}
    stats = dict(fanout=0, joined=0, failed=0)
    owner = threading.get_ident

    def _immutable(bucket, key, data, _inner=inner):
        memo = getattr(cache_mod._active, "memo", None)
        if memo is None or type(key) is not str or type(data) is not bytes or not _v829j_opt("capture_put_fanout"):
            return _inner(bucket, key, data)
        m = snap.match(key)
        if m is not None:
            mark = (key, sha(data))
            if mark in memo["incremental"]["written"]:
                memo["counters"]["write_skips"] += 1
                return None
            if len(pending) > 64:   # never expected (each capture joins at its commitment); bound memory anyway
                for p in list(pending)[:32]:
                    for _, f in pending.pop(p):
                        f.cancel()
            pending.setdefault((owner(), m.group(1)), []).append((mark, pool.submit(raw, bucket, key, data)))
            stats["fanout"] += 1
            return None
        if key.endswith("/commitment.json"):
            jobs = pending.pop((owner(), key[:-len("/commitment.json")]), [])
            first = None
            for mark, f in jobs:
                try:
                    f.result()
                except BaseException as e:   # original: first failing PUT (slot order) propagates
                    stats["failed"] += 1
                    first = first if first is not None else e
                    continue
                stats["joined"] += 1
                written = memo["incremental"]["written"]
                if len(written) < 262144:
                    written.add(mark)
            if first is not None:
                raise first
        return _inner(bucket, key, data)

    for a in ("__name__", "__qualname__", "__module__", "__doc__"):
        setattr(_immutable, a, getattr(inner, a))
    _immutable._v827_original = raw
    _immutable._v829n = True
    _immutable._v829n_stats = stats
    cgc._immutable = _immutable
    return dict(v829n_put=True, capture_put_fanout="snapshot-training-put-join-before-commitment",
                put_workers=V829N_PUT_WORKERS, put_on_now=_v829j_opt("capture_put_fanout"))


class V829nPregrade:
    """Seal grade cache filled while the epoch is still collecting.

    Key = (sha256 of the native pool's own (tokenizer dir, checkpoint files, template, task assets),
           sha256(canonical(manifest)), miner, sha256(canonical(batch))): exactly the inputs of the original
    mimo_native_pool._grade(request) = grade_batch(batch, manifest, tokenizer, miner). Values are produced by
    that same original _grade in the same kind of spawn worker (same _initialize). Only a returned grade dict
    or an InvalidSample (what the original pool yields) is cached; any worker/transport exception is not.
    The seal's original Coordinator.seal still fetches, sha-checks and decodes every frozen document, checks
    every grade binding, excludes, de-duplicates, seeds and selects; only the source of a byte-identical
    grade changes (cache hit, or the original ordered pool on a miss)."""

    def __init__(self):
        import threading
        self.cache = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.pool = None
        self.pool_key = None
        self.seen = set()
        self.failures = {}
        self.stats = dict(graded=0, invalid=0, errors=0, hits=0, misses=0, epoch=None)

    @staticmethod
    def args_key(native_pool):
        from subnet.storage import canonical, sha
        return sha(canonical(list(native_pool.args)))

    @staticmethod
    def key(ak, ms, miner, batch):
        from subnet.storage import canonical, sha
        return (ak, ms, miner, sha(canonical(batch)))

    # ---- background fill (lifecycle process, own pool, only while the epoch has no POPULATION) ----
    def _close_pool(self):
        pool, self.pool, self.pool_key = self.pool, None, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def fill_once(self, coordinator_root, bucket, native_pool, *, sample=None, limit=None):
        """One sweep over the current captured records; returns number of new grades. Used by the loop and
        by the offline replay (sample: optional predicate on (miner, item) to leave some documents uncached)."""
        import copy, multiprocessing
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        import subnet.mimo_native_pool as NP
        import subnet.complete_group_capture as cgc
        from subnet.complete_group_codec import limits, loads
        from subnet.audit_policy import InvalidSample
        from subnet.storage import canonical, sha
        root = Path(coordinator_root)
        env = json.loads((root / "OPENING.json").read_bytes())
        manifest = env["payload"]
        ms = sha(canonical(manifest))
        ak = self.args_key(native_pool)
        if self.stats["epoch"] != manifest.get("epoch"):
            with self.lock:   # keep only the current opening's grades
                self.cache = {k: v for k, v in self.cache.items() if k[1] == ms}
                self.meta = {m: v for m, v in getattr(self, "meta", {}).items() if v[0][1] == ms}   # v8.2.9t
            self.seen = set()
            self.failures = {}
            self.stats.update(epoch=manifest.get("epoch"))
        todo = []
        for path in sorted((root / "captured").glob("*.json")):
            if self.stop.is_set():
                return 0
            try:
                rec = json.loads(path.read_bytes())["payload"]
            except Exception:
                continue
            if rec.get("manifest_sha256") != ms:
                continue
            for item in rec.get("documents", []):
                mark = (rec["miner"], item["sha256"])
                if mark in self.seen or self.failures.get(mark, 0) >= 3 or (sample is not None and not sample(rec["miner"], item)):
                    continue
                todo.append((rec["miner"], item))
        if limit is not None:
            todo = todo[:limit]
        if not todo:
            return 0
        bound = limits(manifest).metadata_bytes

        def fetch(job):
            miner, item = job
            data, _ = cgc._get(bucket, item["key"], bound)   # original reader (no write)
            need(sha(data) == item["sha256"], "v829n pregrade: captured document changed")
            return miner, item, loads(data, bound)["batch"]

        directory, files, template, assets = native_pool.args
        if self.stop.is_set():
            return 0
        if self.pool is None or self.pool_key != (ak, ms):
            self._close_pool()
            self.pool = ProcessPoolExecutor(max_workers=V829N_PREGRADE_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                                            initializer=NP._initialize,
                                            initargs=(directory, files, template, copy.deepcopy(manifest), assets))
            self.pool_key = (ak, ms)
        new = 0
        with ThreadPoolExecutor(max_workers=V829N_PREGRADE_FETCH, thread_name_prefix="v829n-pregrade-read") as reader:
            fetched = [reader.submit(fetch, j) for j in todo]
            graded = []
            for job, f in zip(todo, fetched):
                if self.stop.is_set():
                    break
                try:
                    request = f.result()
                except Exception:
                    self.stats["errors"] += 1
                    m = (job[0], job[1]["sha256"]); self.failures[m] = self.failures.get(m, 0) + 1
                    continue
                k = self.key(ak, ms, request[0], request[2])
                with self.lock:   # v8.2.9t: document -> grade key (read-only use by the RED-skip readiness check)
                    self.__dict__.setdefault("meta", {})[(job[0], job[1]["sha256"])] = (k, request[2].get("env_id"), request[2].get("index"))
                if k in self.cache:
                    self.seen.add((job[0], job[1]["sha256"]))
                    continue
                graded.append((job, k, self.pool.submit(NP._grade, request)))
            for job, k, f in graded:
                try:
                    result = f.result()
                except Exception:
                    self.stats["errors"] += 1   # native/infra failure: not cached, the seal grades it
                    m = (job[0], job[1]["sha256"]); self.failures[m] = self.failures.get(m, 0) + 1
                    continue
                if isinstance(result, InvalidSample):
                    self.stats["invalid"] += 1
                elif type(result) is not dict:
                    self.stats["errors"] += 1
                    continue
                with self.lock:
                    self.cache[k] = result
                self.seen.add((job[0], job[1]["sha256"]))
                self.stats["graded"] += 1
                new += 1
        return new

    def loop(self, lifecycle):
        import time as _t
        while True:
            _t.sleep(5)
            try:
                if not _v829j_opt("seal_pregrade") or self.stop.is_set():
                    self._close_pool()
                    if self.stop.is_set():
                        self.stop.wait(5)
                    continue
                cur = json.loads((lifecycle.root / "CURRENT.json").read_bytes())
                root = lifecycle.root / "epochs" / str(cur["epoch_index"]) / "coordinator"
                if (not (root / "OPENING.json").is_file() or (root / "POPULATION.json").exists()
                        or (root / "DISPATCH.json").exists() or not (root / "captured").is_dir()):
                    self._close_pool()
                    continue
                n = self.fill_once(root, lifecycle.bucket, lifecycle.grade_batches)
                if n:
                    print(json.dumps(dict(component="seal-pregrade-v829n", epoch=self.stats["epoch"], new=n,
                                          cached=len(self.cache), graded=self.stats["graded"], invalid=self.stats["invalid"],
                                          errors=self.stats["errors"])), flush=True)
            except Exception as e:
                self.stats["errors"] += 1
                print(json.dumps(dict(component="seal-pregrade-v829n", error=repr(e)[:300])), flush=True)
                try:
                    self._close_pool()
                except Exception:
                    pass
                _t.sleep(30)

    def halt(self):
        """Seal is about to grade: stop background grading and free its CPUs (bounded by running grades)."""
        self.stop.set()
        try:
            self._close_pool()
        finally:
            pass

    def resume(self):
        self.stop.clear()

    # ---- seal side ----
    def cached_call(self, original_call):
        from collections import deque
        import copy
        from subnet.storage import canonical, sha
        pre = self

        def __call__(native_pool, requests, manifest):
            if not _v829j_opt("seal_pregrade"):
                yield from original_call(native_pool, requests, manifest)
                return
            pre.halt()
            try:
                ak, ms = pre.args_key(native_pool), sha(canonical(manifest))
                order = deque()

                def feed():
                    for request in requests:
                        miner, _, batch = request
                        with pre.lock:
                            hit = pre.cache.get(pre.key(ak, ms, miner, batch))
                        order.append([request, hit])
                        if hit is None:
                            pre.stats["misses"] += 1
                            yield request
                        else:
                            pre.stats["hits"] += 1
                graded = original_call(native_pool, feed(), manifest)
                try:
                    while True:
                        while order and order[0][1] is not None:
                            request, hit = order.popleft()
                            yield request, copy.deepcopy(hit)
                        try:
                            request, result = next(graded)
                        except StopIteration:
                            break
                        while order[0][1] is not None:          # hits that preceded this miss, in order
                            r, hit = order.popleft()
                            yield r, copy.deepcopy(hit)
                        head = order.popleft()
                        need(head[0] is request, "v829n ordered merge")
                        yield request, result
                    need(all(h is not None for _, h in order), "v829n every miss graded")
                    while order:
                        request, hit = order.popleft()
                        yield request, copy.deepcopy(hit)
                finally:
                    graded.close()
                    print(json.dumps(dict(component="seal-pregrade-v829n", seal=True, hits=pre.stats["hits"],
                                          misses=pre.stats["misses"])), flush=True)
            finally:
                pre.resume()
        __call__.__qualname__ = original_call.__qualname__
        __call__._v829n = True
        return __call__


V829N = V829nPregrade()


def install_v829n_pregrade(Lifecycle):
    import threading
    import subnet.mimo_native_pool as NP
    cls = NP.NativeGradingPool
    if getattr(cls.__call__, "_v829n", False):
        return dict(v829n_pregrade=True, seal_pregrade="already")
    cls.__call__ = V829N.cached_call(cls.__call__)
    prior_tick = Lifecycle.tick

    def tick(self, *a, **kw):
        if V829N.thread is None and isinstance(getattr(self, "grade_batches", None), cls):
            V829N.thread = threading.Thread(target=V829N.loop, args=(self,), name="v829n-pregrade", daemon=True)
            V829N.thread.start()
        return prior_tick(self, *a, **kw)
    tick.__qualname__, tick.__module__ = prior_tick.__qualname__, prior_tick.__module__
    Lifecycle.tick = tick
    return dict(v829n_pregrade=True, seal_pregrade="collection-time-grade-cache", pregrade_workers=V829N_PREGRADE_WORKERS,
                pregrade_on_now=_v829j_opt("seal_pregrade"))


# ---------------- v8.2.9l K=1 per-UID turns (Jacob t712u) ----------------
INNER = {}
K1_FLAG = FLAG_PATH.parent / "K1.json"
K1_STATE = Path("/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/state")
K1_CHAIN = K1_STATE / "K1-CHAIN.json"
K1_TURNS = K1_STATE / "K1-TURNS.jsonl"
K1_REASONS = ("training-task-budget", "cross-miner-task-collision")
K1_DEFAULT = dict(max_rollouts=1000, min_tasks=8, min_rollouts_per_task=2, max_rollouts_per_task=64, chain_max_age_seconds=3 * 3600,
                  queue_rule="hotkey-registration-once")
K1_RULES = ("hotkey-registration-once", "hotkey-once", "registration-once", "fewest-turns")   # v8.2.9s; anything else -> default
K1_DEFAULT_RULE = "hotkey-registration-once"


def k1_served_hotkeys(turns):
    """v8.2.9q2: every hotkey that has had a K=1 turn (forever, whatever UID / registration block)."""
    return {t["hotkey"] for t in turns if isinstance(t, dict) and type(t.get("hotkey")) is str}


def k1_flag():
    try:
        d = json.loads(K1_FLAG.read_bytes())
    except FileNotFoundError:
        return None
    except Exception:
        return None  # unreadable -> exact v8.2.9k (pooled) behaviour
    if not (isinstance(d, dict) and d.get("enabled") is True and type(d.get("from_epoch")) is int):
        return None
    return dict(K1_DEFAULT, **d)


def k1_turns(before_epoch=None):
    out = []
    try:
        for line in K1_TURNS.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict) and type(r.get("epoch_index")) is int and (before_epoch is None or r["epoch_index"] < before_epoch):
                out.append(r)
    except FileNotFoundError:
        pass
    return out


def k1_rows_by_miner(sealed):
    rows = list(sealed.get("selected") or []) + [x for x in (sealed.get("excluded") or []) if x.get("reason") in K1_REASONS]
    by = {}
    for r in rows:
        if not (isinstance(r, dict) and type(r.get("miner")) is str and isinstance(r.get("document"), dict)
                and isinstance(r.get("native_grades"), dict) and type(r.get("batch_sha256")) is str):
            continue
        clean = {k: v for k, v in r.items() if k != "reason"}
        by.setdefault(r["miner"], []).append(clean)
    for m in by:
        by[m].sort(key=lambda r: (float(r["document"].get("received_at") or 0), int(r["document"].get("slot") or 0), int(r.get("index") or 0), r["batch_sha256"]))
    return by


def k1_served(turns):
    """v8.2.9q: registrations that already had a turn -> (set of (uid, hotkey, reg_block), set of legacy (uid, hotkey))."""
    full, legacy = set(), set()
    for t in turns:
        try:
            u, hk = int(t["uid"]), t["hotkey"]
        except Exception:
            continue
        rb = t.get("reg_block")
        if type(rb) is not int:
            try:
                rb = json.loads((EPOCHS_ROOT / str(int(t["epoch_index"])) / "K1-TURN.json").read_bytes()).get("reg_block")
            except Exception:
                rb = None
        if type(rb) is int and rb > 0:
            full.add((u, hk, rb))
        else:
            legacy.add((u, hk))
    return full, legacy


def k1_served_registrations(turns):
    """v8.2.9s: every registration (uid, reg_block) that has had a K=1 turn (reg_block as recorded / resolved by k1_served)."""
    full, _ = k1_served(turns)
    return {(u, rb) for (u, _hk, rb) in full}


def k1_select(sealed, chain, turns, f, epoch_index):
    """Pure: -> dict(status=turn|skip, ...). chain = {uids: {uid: {hotkey, miner_hex, reg_block}}}."""
    group = (sealed["selected_rollout_count"] // sealed["selected_task_count"]) if sealed.get("selected_task_count") else 8
    need(f["min_rollouts_per_task"] <= group <= f["max_rollouts_per_task"], "group size within K1 per-task caps")
    counts = {}
    for t in turns:
        k = (int(t["uid"]), t["hotkey"]); counts[k] = counts.get(k, 0) + 1
    rule = f.get("queue_rule") if f.get("queue_rule") in K1_RULES else K1_DEFAULT_RULE
    full, legacy = k1_served(turns) if rule == "registration-once" else (set(), set())
    hks = k1_served_hotkeys(turns) if rule in ("hotkey-once", "hotkey-registration-once") else set()
    regs = k1_served_registrations(turns) if rule == "hotkey-registration-once" else set()
    uids, served = [], 0
    for u, row in (chain.get("uids") or {}).items():
        u = int(u)
        if u == 0 or not isinstance(row, dict) or type(row.get("miner_hex")) is not str:
            continue
        reg = int(row.get("reg_block") or 0)
        if rule == "hotkey-registration-once":
            if row["hotkey"] in hks or (reg > 0 and (u, reg) in regs):
                served += 1
                continue
            uids.append((0, reg, u, row))
        elif rule == "hotkey-once":
            if row["hotkey"] in hks:
                served += 1
                continue
            uids.append((0, reg, u, row))
        elif rule == "registration-once":
            if (u, row["hotkey"], reg) in full or (u, row["hotkey"]) in legacy:
                served += 1
                continue
            uids.append((0, reg, u, row))
        else:
            uids.append((counts.get((u, row["hotkey"]), 0), reg, u, row))
    uids.sort(key=lambda x: (x[0], x[1], x[2]))
    by = k1_rows_by_miner(sealed)
    skipped = []
    max_tasks = f["max_rollouts"] // group
    for pos, (n, reg, u, row) in enumerate(uids):
        mine = by.get(row["miner_hex"], [])
        seen, batch, dropped = set(), [], 0
        for r in mine:
            t = (r.get("env_id"), r.get("index"))
            if t in seen:
                dropped += 1; continue
            seen.add(t); batch.append(r)
        trimmed = max(0, len(batch) - max_tasks)
        batch = batch[:max_tasks]
        if len(batch) >= f["min_tasks"]:
            return dict(status="turn", epoch_index=epoch_index, uid=u, hotkey=row["hotkey"], miner=row["miner_hex"],
                        reg_block=reg, turns_before=n, queue_position=pos, tasks=len(batch), rollouts=len(batch) * group,
                        rollouts_per_task=group, trimmed_tasks=trimmed, duplicate_tasks_dropped=dropped,
                        queue_rule=rule, served_registrations_excluded=served, queue_length=len(uids),
                        task_indices=[int(r["index"]) for r in batch], batch_sha256=[r["batch_sha256"] for r in batch],
                        skipped_not_ready=skipped[:50], skipped_not_ready_count=len(skipped), rows=batch)
        skipped.append(dict(uid=u, tasks=len(batch)))
    return dict(status="skip", epoch_index=epoch_index,
                reason=({"hotkey-registration-once": "no-unserved-hotkey-or-registration-with-ready-batch", "hotkey-once": "no-unserved-hotkey-with-ready-batch", "registration-once": "no-unserved-registration-with-ready-batch"}.get(rule, "no-registered-uid-with-ready-batch")),
                skipped_not_ready_count=len(skipped), queue_length=len(uids), queue_rule=rule, served_registrations_excluded=served)


def install_v829l_k1(Lifecycle):
    prior = Lifecycle._training

    def _training(self, directory, opening, dispatch, current, now):
        f = k1_flag(); d = Path(directory)
        try:
            e = int(d.name)
        except ValueError:
            return prior(self, directory, opening, dispatch, current, now)
        if f is None or e < f["from_epoch"]:
            return prior(self, directory, opening, dispatch, current, now)
        from subnet.backend_jobs import signed
        from subnet.storage import canonical
        from subnet.mimo_release_policy import sign, digest
        from subnet.mimo_lifecycle import save_manifest
        from subnet.complete_group_capture import _immutable
        inner = INNER["training"]
        adapter = getattr(self.services.make_training_job, "__self__", None)
        need(adapter is not None and type(getattr(adapter, "job_seconds", None)) is int, "v829l training adapter")
        adapter.job_seconds = TRAIN_JOB_SECONDS
        path = d / "TRAINING-JOB.json"; kp = d / "TRAIN-DISPATCH-K1.json"; tp = d / "K1-TURN.json"
        if path.exists():
            if kp.exists():
                return inner(self, directory, opening, json.loads(kp.read_bytes()), current, now)
            return prior(self, directory, opening, dispatch, current, now)   # job built before K1 (old path)
        doc = train_control()
        if doc is not None and doc["mode"] != "train":
            return prior(self, directory, opening, dispatch, current, now)   # v8.2.9e pause / rollback unchanged
        m = signed(opening, self.identity.id); sealed = signed(dispatch, self.identity.id)
        if tp.exists():
            turn = json.loads(tp.read_bytes())
        else:
            try:
                chain = json.loads(K1_CHAIN.read_bytes())
                need(time.time() - float(chain.get("taken_at", 0)) < f["chain_max_age_seconds"], "K1 chain snapshot fresh")
            except Exception as exc:
                chain = None; chain_err = str(exc)[:200]
            if chain is None:
                turn = dict(status="skip", epoch_index=e, reason="k1-chain-snapshot-unavailable", error=chain_err)
            elif not sealed.get("selected_task_count"):
                turn = dict(status="skip", epoch_index=e, reason="no-admitted-tasks")
            else:
                turn = k1_select(sealed, chain, k1_turns(before_epoch=e), f, e)
                turn["chain_block"] = chain.get("block")
                turn["sealed_commitment_sha256"] = v823_sealed_commitment(d, turn.get("miner"))   # v8.2.3 pt765u
            turn.update(version="sn120-k1-turn-v1", epoch=m.get("epoch"), head=current["checkpoint"]["id"],
                        optimizer_step=current["optimizer_step"], at=time.time(),
                        caps=dict(max_rollouts=f["max_rollouts"], min_tasks=f["min_tasks"],
                                  rollouts_per_task=[f["min_rollouts_per_task"], f["max_rollouts_per_task"]]))
            tmp = tp.with_name(".K1-TURN.json.tmp"); tmp.write_text(json.dumps(turn, indent=1, sort_keys=True) + chr(10)); os.replace(tmp, tp)
            if turn["status"] == "turn" and not any(t.get("epoch_index") == e for t in k1_turns()):
                K1_STATE.mkdir(parents=True, exist_ok=True)
                with open(K1_TURNS, "a") as fh:
                    rec = {k: turn[k] for k in ("epoch_index", "epoch", "uid", "hotkey", "miner", "tasks", "rollouts",
                                                "queue_position", "turns_before", "head", "at")}
                    rec.update(reg_block=turn.get("reg_block"), queue_rule=turn.get("queue_rule"))   # v8.2.9q: registration key
                    rec.update(sealed_commitment_sha256=turn.get("sealed_commitment_sha256"))   # v8.2.3 pt765u
                    fh.write(json.dumps(rec, sort_keys=True) + chr(10))
        print(json.dumps(dict(component="k1-v829l", epoch=e, **{k: v for k, v in turn.items() if k in
                              ("status", "uid", "tasks", "rollouts", "queue_position", "reason", "skipped_not_ready_count", "queue_rule", "reg_block")})), flush=True)
        if turn["status"] != "turn":
            join_audit_population(self, directory)
            return dict(status="no-update", checkpoint=current["checkpoint"], optimizer_step=current["optimizer_step"],
                        checkpoint_authorization=current.get("checkpoint_authorization"), reason="k1-skip-" + turn.get("reason", "none"))
        want = turn["batch_sha256"]
        by = {r["batch_sha256"]: r for r in turn["rows"]}
        rows = [by[b] for b in want]
        group = turn["rollouts_per_task"]
        payload = dict(sealed, selected=rows, selected_task_count=len(rows), selected_rollout_count=len(rows) * group,
                       excluded=[x for x in (sealed.get("excluded") or []) if x.get("batch_sha256") not in set(want)])
        filtered = sign(self.identity, payload)
        key = self.policy.namespace() + "/epochs/" + m["epoch"] + "/mimo/dispatch-k1.json"
        _immutable(adapter.bucket, key, canonical(filtered))
        documents = {}
        for row in rows:
            dd = row["document"]
            response = adapter.bucket.client.head_object(Bucket=adapter.bucket.name, Key=dd["key"])
            need(type(response["ContentLength"]) is int and 0 < response["ContentLength"] <= 128 * 1024 ** 2,
                 "bounded actual captured training document")
            documents[dd["sha256"]] = dict(url=adapter.bucket.presign(dd["key"], "get_object", 86400), size=response["ContentLength"])
        need(len(documents) == len(rows), "one original compact document per training group")
        from subnet.mimo_worker import VERSION as WORKER
        t0 = time.time()
        value = dict(schema=1, worker_policy=WORKER, job_id="mimo-train-" + digest(filtered)[:48], role="train",
            created_at=t0, expires_at=t0 + adapter.job_seconds, manifest=opening,
            source_files=adapter.source_files, runtime_versions=adapter.runtime_versions,
            trainer_worker=adapter.trainer_worker, checkpoint_read_urls=adapter.checkpoint_urls(m["checkpoint"]),
            training=dict(run_id=m["run_id"], optimizer_step=current["optimizer_step"], config=adapter.trainer_config,
                dispatch=dict(url=adapter.bucket.presign(key, "get_object", 86400), sha256=digest(filtered),
                              size=len(canonical(filtered))),
                documents=documents, publication_namespace=self.policy.namespace(),
                publisher_id=adapter.publisher_id, role_config_file=adapter.role_config))
        envelope = sign(self.identity, value)
        save_manifest(kp, filtered)
        save_manifest(path, envelope)
        print(json.dumps(dict(component="k1-v829l", status="filtered-dispatch", epoch=e, key=key, uid=turn["uid"], tasks=len(rows))), flush=True)
        return inner(self, directory, opening, filtered, current, now)

    _training.__qualname__, _training.__module__ = prior.__qualname__, prior.__module__
    Lifecycle._training = _training
    f = k1_flag() or {}
    return dict(v829l=True, v829q=True, v829q2=True, v829s=True, k1_flag=str(K1_FLAG), on_now=k1_flag() is not None, state=str(K1_STATE),
                queue_rule=(f.get("queue_rule") if f.get("queue_rule") in K1_RULES else K1_DEFAULT_RULE))


# ---------------- v8.2.9t RED-skip early seal (Jacob t771u / pt771u 2026-10-29) ----------------
V829T_LIFECYCLE_SHA = "4842ec9d02f28260656207ccc773128e85da21e88408fb92982a3f966a70b3d9"     # v822 subnet/mimo_lifecycle.py
V829U_LIFECYCLE_SHA = "98d6a6e7e9798b3ee8f6e1dd42cc6aa6ab67e2b568ff43af873ea9cc7412c81c"     # v8.2.3u (= v822 + head-seed branch in _plan)
V829T_COORDINATOR_SHA = "dee0a6f987ed0ac2166daf28762e7b1d76d3e9d12cb34bbe1de60d26075b8d6a"   # v822 subnet/mimo_coordinator.py
V829T_DEADLINE_WHY = "training population cannot seal before the collection deadline"
V829T_MARKER = "RED-SKIP-SEAL.json"
V829T_AUDIT_BRIDGE_SHA = "9c7825da57a1a3345fed446146831eae7ffde2f143ee182f6d8b75ff916e3247"  # v822 subnet/mimo_audit_bridge.py
# v8.2.9t3: the closed population's protocol time (POPULATION.sealed_at = AUDIT-POPULATION.committed_at) must be >= the signed
# deadline (mimo_audit_bridge.build_population: finite(committed_at, deadline, ...)), so the early seal now records the signed
# deadline as the population's sealed_at; the real wall-clock seal time is in RED-SKIP-SEAL.json (sealed_at).


class _V829tLocal(__import__("threading").local):
    early = None


V829T_TLS = _V829tLocal()


def v829t_on():
    """Explicit opt-in, EPOCH-TIME-LIFECYCLE.json "red_skip": true -> "red" (early seal only after an epoch that kept its head);
    "all" -> early seal in every epoch (Jacob t778u: no submission window after GREEN either). Absent / unreadable / anything
    else -> None = off (exact v8.2.9s)."""
    try:
        d = json.loads(LIFE_OPT_PATH.read_text())
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    v = d.get("red_skip")
    return "red" if v is True else ("all" if v == "all" else None)


def _v829t_rj(p):
    try:
        return json.loads(Path(p).read_bytes())
    except Exception:
        return None


def v829t_head_unchanged(root, e, head):
    """True iff epoch e-1 FINALIZED onto its own input checkpoint (RED / K1-skip / any no-update) and that checkpoint is head."""
    fin = _v829t_rj(Path(root) / "epochs" / str(e - 1) / "FINALIZED.json")
    if not isinstance(fin, dict):
        return False
    p = fin.get("payload") or fin
    res = p.get("result") or {}
    return type(head) is str and p.get("input_checkpoint") == head and (res.get("checkpoint") or {}).get("id") == head


def v829t_prev_early_skip(root, e):
    """Anti-churn: e-1 was early-sealed but produced no K=1 turn (readiness prediction mismatch) -> e keeps the full window."""
    d = Path(root) / "epochs" / str(e - 1)
    return (d / V829T_MARKER).exists() and (_v829t_rj(d / "K1-TURN.json") or {}).get("status") != "turn"


def v829t_state(lifecycle, now):
    """None = original tick. Else dict(mode=watch|sealed|reseal, ...) for the CURRENT epoch while now < its signed deadline."""
    rmode = v829t_on()
    if rmode is None:
        return None
    f = k1_flag()
    if f is None:
        return None
    ctl = train_control()
    if ctl is not None and ctl.get("mode") != "train":
        return None
    root = Path(lifecycle.root)
    cur = _v829t_rj(root / "CURRENT.json")
    if not isinstance(cur, dict) or type(cur.get("epoch_index")) is not int:
        return None
    e = cur["epoch_index"]
    if e < f["from_epoch"]:
        return None
    d = root / "epochs" / str(e); co = d / "coordinator"
    plan = _v829t_rj(d / "PLAN.json")
    if not isinstance(plan, dict) or not (co / "OPENING.json").exists():
        return None
    plan = plan.get("payload") or plan
    deadline = (plan.get("manifest") or {}).get("deadline")
    if type(deadline) is not int or not now < deadline:
        return None
    st = dict(epoch=e, dir=d, coordinator=co, deadline=deadline, head=cur["checkpoint"]["id"], flag=f, rmode=rmode)
    if (d / V829T_MARKER).exists():
        return dict(st, mode="sealed" if (co / "POPULATION.json").exists() else "reseal")
    if (co / "POPULATION.json").exists() or (co / "DISPATCH.json").exists():
        return None
    if ((plan.get("input") or {}).get("checkpoint") or {}).get("id") != st["head"]:
        return None
    if not (root / "epochs" / str(e - 1) / "FINALIZED.json").exists():
        return None        # first epoch / no finalized predecessor: original signed window
    if rmode == "red" and not v829t_head_unchanged(root, e, st["head"]):
        return None        # "red" mode, after a GREEN (new head): original signed window
    if v829t_prev_early_skip(root, e):
        return None
    return dict(st, mode="watch")


def v829t_ready_rows(coordinator_root, manifest, pre):
    """Candidate rows the seal would admit, from the v8.2.9n pre-grade cache only (cached native grade dict with one result per
    attempt that meets the group minima: the seal's own candidate test). Reads the CURRENT captured/*.json records, so a
    replaced commitment counts as replaced (replace-until-seal)."""
    from collections import Counter
    from subnet.storage import canonical, sha
    from subnet.complete_group_contract import meets_minima
    ms = sha(canonical(manifest)); group = manifest["group_contract"]; rows = []
    records = []
    for path in sorted((Path(coordinator_root) / "captured").glob("*.json")):
        rec = (_v829t_rj(path) or {}).get("payload") or {}
        if rec.get("manifest_sha256") == ms and type(rec.get("miner")) is str:
            records.append(rec)
    with pre.lock:
        meta = dict(getattr(pre, "meta", {}) or {})
        cache = pre.cache
        for rec in records:
            for item in rec.get("documents", []):
                m = meta.get((rec["miner"], item.get("sha256")))
                if m is None or m[0][1] != ms or m[0][2] != rec["miner"]:
                    continue
                g = cache.get(m[0])
                if type(g) is not dict or type(g.get("results")) is not list or len(g["results"]) != group["group_size"]:
                    continue
                if not meets_minima(Counter(r.get("classification") for r in g["results"] if isinstance(r, dict)), group):
                    continue
                rows.append(dict(miner=rec["miner"], env_id=m[1], index=m[2], document=item, batch_sha256=m[0][3],
                                 native_grades={"v829t": "pregrade-cache"}))
    return rows


def v829t_predict(st, manifest, pre):
    """-> the K=1 turn (dict) the unchanged k1_select would give on the pre-graded candidate rows, or None."""
    rows = v829t_ready_rows(st["coordinator"], manifest, pre)
    if not rows:
        return None
    chain = json.loads(K1_CHAIN.read_bytes())
    f = st["flag"]
    if not time.time() - float(chain.get("taken_at", 0)) < f["chain_max_age_seconds"]:
        return None
    g = manifest["group_contract"]["group_size"]
    pseudo = dict(selected=rows, excluded=[], selected_task_count=len(rows), selected_rollout_count=len(rows) * g)
    t = k1_select(pseudo, chain, k1_turns(before_epoch=st["epoch"]), f, st["epoch"])
    return t if t.get("status") == "turn" else None


def install_v829t_red_skip(Lifecycle, Coordinator):
    import fcntl, hashlib
    lm = sys.modules[Lifecycle.__module__]; cm = sys.modules[Coordinator.__module__]
    pins = dict(lifecycle=hashlib.sha256(Path(lm.__file__).read_bytes()).hexdigest(),
                coordinator=hashlib.sha256(Path(cm.__file__).read_bytes()).hexdigest())
    if pins not in (dict(lifecycle=V829T_LIFECYCLE_SHA, coordinator=V829T_COORDINATOR_SHA), dict(lifecycle=V829U_LIFECYCLE_SHA, coordinator=V829T_COORDINATOR_SHA)):
        return dict(v829t=False, reason="source pin mismatch: red-skip inert", pins=pins)
    if getattr(Lifecycle.tick, "_v829t", False):
        return dict(v829t=True, red_skip="already")
    orig_need = cm._need

    def _need(value, why):
        # waive ONLY the seal's deadline refusal, ONLY inside v829t's own early seal call
        if not value and why == V829T_DEADLINE_WHY and V829T_TLS.early is not None:
            return None
        return orig_need(value, why)
    cm._need = _need
    # v8.2.9t3 legacy: epochs early-sealed by v8.2.9t2 (POPULATION.sealed_at = real time < signed deadline, marker without
    # protocol_sealed_at) can never build their AUDIT-POPULATION, so FINALIZED would wait forever. Waive ONLY the
    # 'closed original commitment time' lower bound, ONLY for a committed_at/deadline pair that exactly matches such a marker
    # in the last 8 epochs.
    import importlib
    ab = importlib.import_module("subnet.mimo_audit_bridge")
    if hashlib.sha256(Path(ab.__file__).read_bytes()).hexdigest() != V829T_AUDIT_BRIDGE_SHA:
        return dict(v829t=False, reason="audit bridge pin mismatch: red-skip inert")
    orig_finite = ab.finite

    def finite(v, low, high, name):
        if name == "closed original commitment time" and type(v) in (int, float) and type(low) in (int, float) and v < low:
            try:
                n = json.loads((EPOCHS_ROOT.parent / "CURRENT.json").read_bytes())["epoch_index"]
            except Exception:
                n = None
            for e in (range(n, n - 8, -1) if type(n) is int else ()):
                mk = _v829t_rj(EPOCHS_ROOT / str(e) / V829T_MARKER) or {}
                if ("protocol_sealed_at" not in mk and mk.get("sealed_at") == v and mk.get("signed_deadline") == low
                        and v < low <= v + 900):
                    print(json.dumps(dict(component="red-skip-v829t", epoch=e, status="legacy-early-seal-audit-population",
                                          committed_at=v, signed_deadline=low)), flush=True)
                    return float(v)
        return orig_finite(v, low, high, name)
    ab.finite = finite
    prior = Lifecycle.tick
    logged = set()

    def say(**kw):
        print(json.dumps(dict(component="red-skip-v829t", **kw), default=str), flush=True)

    def tick(self, *, now):
        try:
            st = v829t_state(self, now)
        except Exception as exc:
            say(error="state " + repr(exc)[:200]); st = None
        if st is None:
            return prior(self, now=now)
        e = st["epoch"]
        if st["mode"] == "sealed":
            # population frozen at the early seal; the original tick continues as if the deadline had passed (its only use of
            # `now` past this point is that comparison and the no-op re-seal of the existing DISPATCH)
            return prior(self, now=max(now, st["deadline"]))
        if st["mode"] == "watch":
            r = prior(self, now=now)          # original capture pass (replace-until-seal)
            if not (isinstance(r, dict) and r.get("status") == "collecting"):
                return r
            if ("watch", e) not in logged:
                logged.add(("watch", e)); say(epoch=e, status="watching", head=st["head"][:12], deadline=st["deadline"])
            try:
                manifest = json.loads((st["coordinator"] / "OPENING.json").read_bytes())["payload"]
                turn = v829t_predict(st, manifest, V829N)
            except Exception as exc:
                say(epoch=e, error="predict " + repr(exc)[:200]); return r
            if turn is None:
                return r                      # no ready batch yet: poll again next tick
        else:
            turn = (_v829t_rj(st["dir"] / V829T_MARKER) or {}).get("predicted") or {}
        fd = os.open(Path(self.root) / "lifecycle.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return dict(status="collecting", epoch=e, deadline=st["deadline"])
            mp = st["dir"] / V829T_MARKER
            if not mp.exists():
                rec = dict(version="sn120-red-skip-v829t", epoch_index=e, head=st["head"], signed_deadline=st["deadline"],
                           sealed_at=now, protocol_sealed_at=st["deadline"], seconds_before_deadline=st["deadline"] - now,
                           reason=("previous epoch finalized onto the same head (no model change)" if v829t_head_unchanged(self.root, e, st["head"])
                                   else "immediate seal after GREEN (red_skip=all, Jacob t778u); head probe on the new head runs in parallel"),
                           predicted={k: turn.get(k) for k in ("uid", "hotkey", "reg_block", "tasks", "queue_position", "queue_rule")})
                tmp = mp.with_name("." + mp.name + ".tmp"); tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + chr(10)); os.replace(tmp, mp)
            # protocol close time = signed deadline (see V829T_AUDIT_BRIDGE_SHA note); the capture pass already ran at `now`
            self._coordinator(st["dir"]).seal(max_tasks=self.max_tasks, now=st["deadline"])
        finally:
            os.close(fd)
        say(epoch=e, status="early-sealed", seconds_before_deadline=st["deadline"] - now, predicted_uid=turn.get("uid"),
            predicted_tasks=turn.get("tasks"))
        return prior(self, now=max(now, st["deadline"]))

    tick.__qualname__, tick.__module__ = prior.__qualname__, prior.__module__
    tick._v829t = True
    Lifecycle.tick = tick
    return dict(v829t=True, v829t3=True, red_skip_on_now=v829t_on(), switch=str(LIFE_OPT_PATH) + ' {"red_skip": true|"all"}', pins=pins)


# ---------------- v8.2.9p teutonic-style cutoff (Jacob t761u 2026-10-28 1:06 PM CR) ----------------
# Once a HOTKEY has had its K=1 turn (any record in K1-TURNS.jsonl, the k1 queue's own ledger), it is left out of the
# signed epoch opening: its registration row is dropped from the snapshot this epoch's PLAN is built from, so the
# coordinator mints no capability / upload grant for it and capture rejects it as "registered epoch identity".
# Everyone else is unchanged. Per hotkey forever (a new hotkey on the same UID is a new participant).
# Off switch without restart: staged/K1.json "served_cutoff": false (or K1 disabled) -> exact v829q/v829r opening.
# Never changes an already written PLAN.json (in-flight epochs keep their opening).
V829P_CUTOFF_RECORD = "K1-CUTOFF.json"


def v829p_served_hotkeys():
    """Served set = v8.2.9q2 k1_served_hotkeys over the full K1-TURNS.jsonl ledger (one turn per hotkey forever)."""
    return {h for h in k1_served_hotkeys(k1_turns()) if h}


def v829s_regblock_by_uid():
    """v8.2.9s: {uid: (hotkey, reg_block)} from the k1 chain snapshot (K1-CHAIN.json, BlockAtRegistration); {} if unreadable."""
    try:
        rows = json.loads(K1_CHAIN.read_bytes()).get("uids") or {}
        return {int(u): (r["hotkey"], int(r.get("reg_block") or 0)) for u, r in rows.items() if isinstance(r, dict)}
    except Exception:
        return {}


def v829p_filter_snapshot(snapshot, served, served_regs=frozenset(), regblocks=None):
    """Pure: drop served hotkeys, and (v8.2.9s) hotkeys on a served registration (uid, reg_block), from a registration
    snapshot -> (snapshot', excluded hotkeys). reg_block comes from regblocks[uid] only when it names the same hotkey."""
    if not (isinstance(snapshot, dict) and isinstance(snapshot.get("registrations"), dict)):
        return snapshot, []
    rows = snapshot["registrations"]
    regblocks = regblocks or {}

    def on_served_reg(h, r):
        try:
            u = int(r["uid"])
        except Exception:
            return False
        hk, rb = regblocks.get(u, (None, 0))
        return hk == h and rb > 0 and (u, rb) in served_regs
    excluded = sorted(h for h, r in rows.items() if h in served or on_served_reg(h, r))
    if not excluded:
        return snapshot, []
    ex = set(excluded)
    return dict(snapshot, registrations={h: r for h, r in rows.items() if h not in ex}), excluded


def install_v829p_served_cutoff(Lifecycle):
    import dataclasses
    prior = Lifecycle._plan

    def _plan(self, current, directory, now):
        d = Path(directory)
        f = k1_flag()
        try:
            e = int(d.name)
        except ValueError:
            e = None
        if (d / "PLAN.json").exists() or f is None or e is None or e < f["from_epoch"] or f.get("served_cutoff", True) is not True:
            return prior(self, current, directory, now)
        served = v829p_served_hotkeys()
        regs = k1_served_registrations(k1_turns()) if (f.get("queue_rule") if f.get("queue_rule") in K1_RULES else K1_DEFAULT_RULE) == "hotkey-registration-once" else set()
        if not served:
            return prior(self, current, directory, now)
        base = self.services
        seen = {}

        def registrations():
            snap, excluded = v829p_filter_snapshot(base.registrations(), served, regs, v829s_regblock_by_uid())
            seen.update(excluded=excluded, snapshot_block=(snap or {}).get("snapshot_block") if isinstance(snap, dict) else None,
                        kept=len(snap["registrations"]) if isinstance(snap, dict) and isinstance(snap.get("registrations"), dict) else None)
            return snap

        self.services = dataclasses.replace(base, registrations=registrations)
        try:
            plan = prior(self, current, directory, now)
        finally:
            self.services = base
        if plan is not None and "excluded" in seen:
            from subnet.controller import save_manifest
            rec = d / V829P_CUTOFF_RECORD
            if not rec.exists():
                save_manifest(rec, dict(version="affine-host-v829p-served-cutoff-v1", epoch_index=e, rule="one-k1-turn-per-hotkey-forever",
                                        served_hotkeys=sorted(served), excluded_from_opening=seen["excluded"],
                                        snapshot_block=seen["snapshot_block"], opened_identities=len(plan["miners"]), at=time.time()))
            print(json.dumps(dict(component="k1-cutoff-v829p", epoch_index=e, served=len(served), excluded=len(seen["excluded"]),
                                  opened=len(plan["miners"]))), flush=True)
        return plan

    _plan.__qualname__ = prior.__qualname__
    _plan.__module__ = prior.__module__
    Lifecycle._plan = _plan
    f = k1_flag()
    return dict(v829p_cutoff=True, rule="one-k1-turn-per-hotkey-forever", source=str(K1_TURNS),
                on_now=bool(f is not None and f.get("served_cutoff", True) is True), served_now=len(v829p_served_hotkeys()))


# ---------------- v8.2.3 batch lifetime host side (Jacob pt765u) ----------------
# Source (subnet.batch_lifetime) does replace-until-seal and head-bound draws. The host adds:
#  * K1-TURN / K1-TURNS: the sealed (final) commitment hash of the turn's hotkey, for k1.json.
#  * after the seal: delete write-once upload slots that the sealed commitment does not name
#    (superseded uploads). Snapshots and the ledger history stay. Off switch: K1.json
#    "superseded_cleanup": false. Runs in a daemon thread; never in the tick/gate path.
V823_CLEANUP_NAME = "V823-SUPERSEDED-CLEANUP.json"


def v823_sealed_commitment(directory, miner):
    """Final commitment SHA-256 the seal froze for `miner` (None if not in the population)."""
    try:
        pop = json.loads((Path(directory) / "coordinator" / "POPULATION.json").read_bytes())["payload"]
        for envelope in pop["captures"]:
            p = envelope.get("payload", {})
            if p.get("miner") == miner:
                return p.get("commitment_sha256")
    except Exception:
        return None
    return None


def v823_superseded_keys(bucket, epoch_label, miner, kept_slots):
    """Upload-slot keys under this epoch/miner not named by the sealed commitment."""
    out = []
    for prefix in (f"private/{epoch_label}/staging/{miner}/", f"private/{epoch_label}/training/{miner}/"):
        token = None
        while True:
            kw = dict(Bucket=bucket.name, Prefix=prefix, MaxKeys=1000)
            if token:
                kw["ContinuationToken"] = token
            page = bucket.client.list_objects_v2(**kw)
            for row in page.get("Contents", []):
                name = row["Key"][len(prefix):]
                stem, _, ext = name.partition(".")
                if ext in ("zip", "json") and stem.isdigit() and int(stem) not in kept_slots:
                    out.append(row["Key"])
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
    return out


def v823_cleanup_epoch(lifecycle, directory):
    from subnet.backend_jobs import signed
    try:
        from subnet.batch_lifetime import policy
    except ImportError:   # pre-v8.2.3 source: nothing to clean
        return None
    d = Path(directory); coord = d / "coordinator"; rec = d / V823_CLEANUP_NAME
    if rec.exists() or not (coord / "DISPATCH.json").exists() or not (coord / "POPULATION.json").exists():
        return None
    opening = signed(json.loads((coord / "OPENING.json").read_bytes()), lifecycle.identity.id)
    try:
        from subnet.dual_accept import policy as dual_policy
    except ImportError:
        dual_policy = None
    if policy(opening) is None and (dual_policy is None or dual_policy(opening) is None):
        return None
    pop = json.loads((coord / "POPULATION.json").read_bytes())["payload"]
    deleted, kept = [], {}
    for envelope in pop["captures"]:
        p = envelope["payload"]; miner = p["miner"]
        if policy(opening, miner) is None:   # legacy identity: append-only slots, nothing superseded
            continue
        slots = {int(x["slot"]) for x in p.get("documents", [])}
        kept[miner] = sorted(slots)
        for key in v823_superseded_keys(lifecycle.bucket, opening["epoch"], miner, slots):
            lifecycle.bucket.client.delete_object(Bucket=lifecycle.bucket.name, Key=key)
            deleted.append(key)
    out = dict(version="v823-superseded-cleanup-v1", epoch=opening["epoch"], at=time.time(), kept_slots=kept,
               deleted=len(deleted), deleted_keys=deleted[:4096])
    tmp = rec.with_name("." + V823_CLEANUP_NAME + ".tmp"); tmp.write_text(json.dumps(out, indent=1, sort_keys=True) + chr(10))
    os.replace(tmp, rec)
    return out


def install_v823_lifetime(Lifecycle):
    import threading
    state = dict(thread=None)
    prior_tick = Lifecycle.tick

    def loop(lifecycle):
        import time as _t
        while True:
            _t.sleep(60)
            try:
                f = k1_flag() or {}
                if f.get("superseded_cleanup", True) is False:
                    continue
                cur = json.loads((lifecycle.root / "CURRENT.json").read_bytes())["epoch_index"]
                for n in range(max(0, int(cur) - 3), int(cur) + 1):
                    r = v823_cleanup_epoch(lifecycle, lifecycle.root / "epochs" / str(n))
                    if r:
                        print(json.dumps(dict(component="v823-superseded-cleanup", epoch=r["epoch"], deleted=r["deleted"])), flush=True)
            except Exception as e:
                print(json.dumps(dict(component="v823-superseded-cleanup", error=repr(e)[:300])), flush=True)
                _t.sleep(240)

    def tick(self, *a, **kw):
        if state["thread"] is None:
            state["thread"] = threading.Thread(target=loop, args=(self,), name="v823-superseded-cleanup", daemon=True)
            state["thread"].start()
        return prior_tick(self, *a, **kw)
    tick.__qualname__, tick.__module__ = prior_tick.__qualname__, prior_tick.__module__
    Lifecycle.tick = tick
    return dict(v823_lifetime=True, superseded_cleanup="post-seal-daemon", k1_sealed_commitment=True)


# ---------------- v8.2.9v pre-open (Jacob t791u / parent decision 1, 2026-10-29) ----------------
# "One opening per head": while epoch N trains/probes, the NEXT epoch N+1 is planned + opened + discovered on the SAME head,
# with N+1's input = the exact successor a no-update (RED) finalize of N writes (verified on e217->e218 and e219->e220).
# Miners see N+1's opening right away and upload during N's train+probe, so after a RED N+1 is current with a full ready
# population and seals at once (its signed deadline has passed; or v829t seals it at the first ready batch).
# After a GREEN (or any finalize whose successor differs, e.g. a TRAIN-CONTROL rollback) the speculative N+1 is abandoned
# (signed ABANDONED-SPECULATIVE.json, local + immutable R2, like the original abandoned-before-opening), CURRENT moves to
# N+2 chained to N+1's discovery, and N+2 opens normally for the new head. Switch: staged/EPOCH-TIME-LIFECYCLE.json
# "pre_open": true (absent/false = exact v8.2.9t3). Marker epochs/N+1/V829V-PREOPEN.json.
V829V_MARKER = "V829V-PREOPEN.json"
V829V_ABANDON = "ABANDONED-SPECULATIVE.json"


def v829v_on():
    try:
        return json.loads(LIFE_OPT_PATH.read_bytes()).get("pre_open") is True
    except Exception:
        return False


def install_v829v_preopen(Lifecycle):
    import importlib, fcntl
    lm = importlib.import_module(Lifecycle.__module__)
    prior = Lifecycle.tick
    if getattr(prior, "_v829v", False):
        return dict(v829v=True, already=True)

    def say(**kw):
        print(json.dumps(dict(component="preopen-v829v", **kw)), flush=True)

    def locked(self):
        fd = os.open(Path(self.root) / "lifecycle.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd); return None
        return fd

    def abandon_if_stale(self):
        """CURRENT points at a pre-opened epoch whose plan input is not the real successor -> abandon it, move on."""
        cur = self._current(); d = Path(self.root) / "epochs" / str(cur["epoch_index"])
        if not (d / V829V_MARKER).exists() or not (d / "PLAN.json").exists():
            return None
        plan = self._load(d / "PLAN.json")
        if plan["input"] == cur:
            return None
        need((d / "coordinator" / "POPULATION.json").exists() is False, "v829v: stale pre-open was never sealed")
        epoch = plan["manifest"]["epoch"]
        ab = lm.sign(self.identity, dict(version=lm.VERSION, policy_sha256=self.policy.policy_sha256, epoch=epoch,
                     reason="speculative-successor-mismatch", manifest_sha256=lm.digest(plan["manifest"]),
                     checkpoint=plan["input"]["checkpoint"]["id"], current_checkpoint=cur["checkpoint"]["id"]))
        if not (d / V829V_ABANDON).exists():
            lm._immutable(self.bucket, self.policy.namespace() + "/epochs/" + epoch + "/" + V829V_ABANDON, lm.canonical(ab))
            lm.save_manifest(d / V829V_ABANDON, ab)
        disc = d / "DISCOVERY.json"
        prev = self._load(disc)["discovery_sha256"] if disc.exists() else cur["previous_discovery_sha256"]
        lm.save_manifest(Path(self.root) / "CURRENT.json", dict(cur, epoch_index=cur["epoch_index"] + 1, previous_discovery_sha256=prev))
        say(epoch=cur["epoch_index"], status="abandoned-speculative", head=cur["checkpoint"]["id"][:12],
            planned_head=plan["input"]["checkpoint"]["id"][:12])
        return dict(status="abandoned-speculative", epoch=epoch)

    def preopen(self, now):
        ctl = train_control()
        if ctl is not None and ctl.get("mode") != "train":
            return None
        cur = self._current(); n = cur["epoch_index"]; root = Path(self.root)
        d = root / "epochs" / str(n); nd = root / "epochs" / str(n + 1)
        if not ((d / "coordinator" / "DISPATCH.json").exists() and (d / "DISCOVERY.json").exists()) or (d / "FINALIZED.json").exists():
            return None
        if (nd / "PLAN.json").exists() and (nd / "coordinator" / "OPENING.json").exists() and (nd / "DISCOVERY.json").exists():
            return None
        succ = dict(cur, epoch_index=n + 1, round=cur["round"] + 1,
                    previous_discovery_sha256=self._load(d / "DISCOVERY.json")["discovery_sha256"])
        nd.mkdir(parents=True, exist_ok=True)
        if not (nd / V829V_MARKER).exists():
            lm.save_manifest(nd / V829V_MARKER, dict(version="sn120-preopen-v829v", epoch_index=n + 1, during_epoch=n,
                             head=cur["checkpoint"]["id"], at=now))
        plan = self._plan(succ, nd, now)
        if plan is None:
            return None
        co = self._coordinator(nd)
        opening = co.open(plan["manifest"], plan["miners"])
        dp = nd / "DISCOVERY.json"
        if not dp.exists():
            receipt = self.services.discovery(opening, succ["previous_discovery_sha256"])
            from subnet.mimo_release_policy import _hash
            need(type(receipt) is dict and receipt.get("sha256") == lm.digest(opening) and _hash(receipt.get("discovery_sha256")),
                 "v829v discovery readback of exact pre-opening")
            lm.save_manifest(dp, receipt)
        say(epoch=n + 1, status="pre-opened", during_epoch=n, head=cur["checkpoint"]["id"][:12],
            deadline=plan["manifest"]["deadline"])
        return True

    def tick(self, *, now):
        # abandon runs whenever a pre-open marker exists (also with the switch off), so turning pre_open off can never strand
        # a speculative epoch whose successor changed
        fd = locked(self)
        if fd is not None:
            try:
                r = abandon_if_stale(self)
            finally:
                os.close(fd)
            if r is not None:
                return r
        r = prior(self, now=now)
        if not isinstance(r, dict):
            return r
        fd = locked(self)
        if fd is None:
            return r
        try:
            if r.get("status") == "finalized":
                abandon_if_stale(self)
            elif r.get("status") == "training" and v829v_on():
                try:
                    preopen(self, now)
                except Exception as exc:
                    say(status="pre-open-error", error=repr(exc)[:300])
        finally:
            os.close(fd)
        return r

    tick.__qualname__, tick.__module__ = prior.__qualname__, prior.__module__
    tick._v829v = True
    Lifecycle.tick = tick
    return dict(v829v=True, pre_open_on_now=v829v_on(), switch=str(LIFE_OPT_PATH) + ' {"pre_open": true}')


# ---------------- v8.2.9u dual accept host side (Jacob: 125-task release, old clients keep working) ----------------
# Source (subnet.dual_accept) makes ONE signed opening carry two modes per identity: legacy (exact v8.2.2: cap 9,
# inline capability, epoch-bound draws) and extended (cap 125, sealed upload grant, replace-until-seal, head-bound draws).
# The host adds:
#  * open(): identities in the sticky registry are minted extended; everyone else legacy (default: nobody extended).
#  * collect(): an admitted legacy commitment carrying client_upgrade=dual-accept-extended-v1 adds its miner to the registry
#    (effective from the NEXT opening; never removed -> the draw mode of a hotkey changes at most once).
# Off switch (no restart), staged/EPOCH-TIME-LIFECYCLE.json "dual_extended": false -> every identity legacy in new openings
# (v8.2.3u clients then simply mine 9 tasks on the legacy path); "dual_optin": false -> registry frozen.
# In-flight openings are never changed. Registry: state/mimo-mainnet-v1/DUAL-ACCEPT-EXTENDED.json (append-only history).
V829U_REGISTRY = EPOCHS_ROOT.parent / "DUAL-ACCEPT-EXTENDED.json"
V829U_OPENING_RECORD = "DUAL-ACCEPT-OPENING.json"


def v829u_opt(key):
    try:
        v = json.loads(LIFE_OPT_PATH.read_bytes()).get(key, True)
        return v is not False
    except Exception:
        return True


def v829u_registry():
    """{miner_hex: {since_epoch, at}}; {} if absent/unreadable (fail safe: nobody extended)."""
    try:
        doc = json.loads(V829U_REGISTRY.read_bytes())
        rows = doc.get("extended") if isinstance(doc, dict) else None
        return {m: r for m, r in (rows or {}).items() if isinstance(m, str) and len(m) == 64}
    except Exception:
        return {}


def v829u_register(miner, epoch_label):
    import fcntl
    lock = os.open(str(V829U_REGISTRY) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            doc = json.loads(V829U_REGISTRY.read_bytes())
        except FileNotFoundError:
            doc = dict(version="v829u-dual-accept-extended-registry-v1", extended={}, history=[])
        if miner in doc["extended"]:
            return False
        doc["extended"][miner] = dict(since_epoch=epoch_label, at=time.time())
        doc["history"].append(dict(miner=miner, epoch=epoch_label, at=time.time(), event="opt-in"))
        tmp = V829U_REGISTRY.with_name("." + V829U_REGISTRY.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + chr(10)); os.chmod(tmp, 0o644)
        os.replace(tmp, V829U_REGISTRY)
        return True
    finally:
        os.close(lock)


def install_v829u_dual_accept(Coordinator):
    try:
        from subnet import dual_accept as DA
    except ImportError:   # pre-v8.2.3u source: nothing to do
        return dict(v829u=False, reason="source without subnet.dual_accept")
    if getattr(Coordinator.open, "_v829u", False):
        return dict(v829u=True, dual_accept="already")
    prior_open, prior_collect = Coordinator.open, Coordinator.collect

    def open(self, manifest, miners):
        if DA.policy(manifest) is None:
            return prior_open(self, manifest, miners)
        ext = set(v829u_registry()) if v829u_opt("dual_extended") else set()
        with DA.opening(ext):
            envelope = prior_open(self, manifest, miners)
        rec = Path(self.root).parent / V829U_OPENING_RECORD
        if not rec.exists():
            caps = (envelope.get("payload") or {}).get("capabilities") or {}
            got = sorted(m for m, v in caps.items() if isinstance(v, str) and v.startswith(DA.TAG))
            from subnet.controller import save_manifest
            save_manifest(rec, dict(version="v829u-dual-accept-opening-v1", epoch=manifest.get("epoch"), identities=len(caps),
                                    extended=len(got), extended_identities=got, legacy=len(caps) - len(got), at=time.time()))
            print(json.dumps(dict(component="dual-accept-v829u", epoch=manifest.get("epoch"), identities=len(caps), extended=len(got))), flush=True)
        return envelope

    def collect(self, miner, *, now=None):
        record = prior_collect(self, miner, now=now)
        try:
            p = record.get("payload", record) if isinstance(record, dict) else {}
            if p.get("client_upgrade") == DA.UPGRADE and v829u_opt("dual_optin"):
                if v829u_register(miner, p.get("epoch")):
                    print(json.dumps(dict(component="dual-accept-v829u", event="opt-in", miner=miner[:16], epoch=p.get("epoch"))), flush=True)
        except Exception as e:   # registry trouble never blocks capture; the miner stays legacy
            print(json.dumps(dict(component="dual-accept-v829u", error=repr(e)[:300])), flush=True)
        return record

    for f, p in ((open, prior_open), (collect, prior_collect)):
        f.__qualname__, f.__module__ = p.__qualname__, p.__module__
    open._v829u = True
    Coordinator.open, Coordinator.collect = open, collect
    return dict(v829u=True, dual_accept=DA.VERSION, registry=str(V829U_REGISTRY), extended_now=len(v829u_registry()),
                switches=str(LIFE_OPT_PATH) + ' {"dual_extended": false, "dual_optin": false}')


# ---------------- v8.2.9x tunable per-miner batch cap (Jacob t796u, 2026-10-29) ----------------
# One staged value, staged/BATCH-CAP.json, sets how much of ONE miner's batch a K=1 turn trains:
#   {"max_tasks_per_miner": 125, "rollouts_per_task": 8, "max_rollouts_per_miner": 1000, "min_tasks": 8}
# Bounds (fail-safe; anything else -> the default below, which equals the pre-v829x K=1 trim of 1,000 rollouts):
#   rollouts_per_task == 8 (the signed N8 group contract), 8 <= max_tasks_per_miner <= the signed dual-accept ceiling
#   (extended_max_batches, 125 in v8.2.3u), 64 <= max_rollouts_per_miner <= 1000 (trainer cap), min_tasks == 8.
# Applied at K=1 selection (k1_flag max_rollouts -> deterministic trim in submission order, unchanged k1_select), recorded per
# opening in a signed epochs/N/BATCH-CAP.json, published by training_params.py in mining.json training.batch_cap. No restart:
# the file is read on every selection/opening. Upload caps (what a client may send) stay the signed opening values
# (legacy 9 / extended 125); lowering the staged value trims, it never rejects a batch.
V829X_CAP_PATH = FLAG_PATH.parent / "BATCH-CAP.json"
V829X_RECORD = "BATCH-CAP.json"
V829X_GROUP = 8
V829X_TRAINER_MAX_ROLLOUTS = 1000
V829X_CEILING = 125
V829X_DEFAULT = dict(max_tasks_per_miner=125, rollouts_per_task=8, max_rollouts_per_miner=1000, min_tasks=8)


def v829x_cap(ceiling=None):
    """-> dict(effective cap + source/error). Pure read of the staged file; never raises."""
    ceiling = ceiling if type(ceiling) is int and 8 <= ceiling <= 256 else V829X_CEILING
    try:
        d = json.loads(V829X_CAP_PATH.read_bytes())
    except FileNotFoundError:
        return dict(V829X_DEFAULT, max_tasks_per_miner=min(V829X_DEFAULT["max_tasks_per_miner"], ceiling), source="default-absent", ceiling=ceiling)
    except Exception as exc:
        return dict(V829X_DEFAULT, max_tasks_per_miner=min(V829X_DEFAULT["max_tasks_per_miner"], ceiling), source="default-unreadable", error=repr(exc)[:200], ceiling=ceiling)
    try:
        t, r, m, mn = (d.get(k) for k in ("max_tasks_per_miner", "rollouts_per_task", "max_rollouts_per_miner", "min_tasks"))
        need(all(type(x) is int for x in (t, r, m, mn)), "integer cap fields")
        need(r == V829X_GROUP, "rollouts_per_task must equal the signed group size 8")
        need(8 <= t <= ceiling, "max_tasks_per_miner within 8..signed ceiling %d" % ceiling)
        need(64 <= m <= V829X_TRAINER_MAX_ROLLOUTS, "max_rollouts_per_miner within 64..1000 (trainer cap)")
        need(mn == 8, "min_tasks stays 8")
        eff = min(t, m // r)
        need(eff >= 8, "effective cap >= 8 tasks")
        return dict(max_tasks_per_miner=eff, rollouts_per_task=r, max_rollouts_per_miner=eff * r, min_tasks=mn,
                    source=str(V829X_CAP_PATH), ceiling=ceiling, set_at=d.get("set_at"), authorized_by=d.get("authorized_by"))
    except Exception as exc:
        return dict(V829X_DEFAULT, max_tasks_per_miner=min(V829X_DEFAULT["max_tasks_per_miner"], ceiling), source="default-invalid", error=str(exc)[:200], ceiling=ceiling)


def install_v829x_batch_cap(Coordinator):
    g = globals()
    if g.get("_V829X_INSTALLED"):
        return dict(v829x=True, already=True)
    prior_flag = g["k1_flag"]

    def k1_flag():
        f = prior_flag()
        if f is None:
            return None
        cap = v829x_cap()
        return dict(f, max_rollouts=min(int(f["max_rollouts"]), cap["max_tasks_per_miner"] * cap["rollouts_per_task"]),
                    batch_cap=cap)
    g["k1_flag"] = k1_flag      # every caller in this module (k1 selection, v829t ready prediction) reads the global

    prior_open = Coordinator.open

    def open(self, manifest, miners):
        envelope = prior_open(self, manifest, miners)
        try:
            rec = Path(self.root).parent / V829X_RECORD
            if not rec.exists():
                try:
                    from subnet import dual_accept as DA
                    dp = DA.policy(manifest)
                except Exception:
                    dp = None
                ceiling = dp["extended_max_batches"] if dp else manifest.get("max_batches")
                cap = v829x_cap(ceiling if dp else None)
                from subnet.mimo_release_policy import sign
                from subnet.controller import save_manifest
                doc = dict(version="sn120-batch-cap-v1", epoch=manifest.get("epoch"), at=time.time(),
                           legacy_max_tasks=manifest.get("max_batches"), signed_ceiling_max_tasks=ceiling,
                           signed_upload_slots=(dp or {}).get("extended_upload_slots"), dual_accept=bool(dp),
                           effective=cap, trainer_max_rollouts=V829X_TRAINER_MAX_ROLLOUTS)
                save_manifest(rec, sign(self.identity, doc) if getattr(self, "identity", None) is not None else doc)
                print(json.dumps(dict(component="batch-cap-v829x", epoch=manifest.get("epoch"), max_tasks=cap["max_tasks_per_miner"],
                                      rollouts_per_task=cap["rollouts_per_task"], ceiling=ceiling, source=cap["source"])), flush=True)
        except Exception as exc:     # the record is informational; never block an opening
            print(json.dumps(dict(component="batch-cap-v829x", error=repr(exc)[:300])), flush=True)
        return envelope

    open.__qualname__, open.__module__ = prior_open.__qualname__, prior_open.__module__
    for k in ("_v829u",):
        if getattr(prior_open, k, False):
            setattr(open, k, True)
    Coordinator.open = open
    g["_V829X_INSTALLED"] = True
    c = v829x_cap()
    return dict(v829x=True, cap_file=str(V829X_CAP_PATH), max_tasks_now=c["max_tasks_per_miner"],
                rollouts_per_task=c["rollouts_per_task"], source_now=c["source"])


def main():
    spec = importlib.util.spec_from_file_location("affine_v828_launcher", V828_DIR / "mimo_v828_launcher.py")
    v828 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v828
    if str(V828_DIR) not in sys.path:
        sys.path.insert(0, str(V828_DIR))
    spec.loader.exec_module(v828)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-config", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--v828-operator-config",
                        default=str(V828_DIR / "v828-operator.ROOT-ADMITTED.private.json"))
    args = parser.parse_args()
    uid, parent, operator, module, Lifecycle, Coordinator, methods, workers = v828.checked(
        v828._paths.private_json(args.v828_operator_config))
    v824, v828_meta = v828.compose(uid, parent, operator, Lifecycle, Coordinator, methods, module, workers)
    v829_meta = install_v829(module, Lifecycle)
    v829_meta['v829h'] = install_v829h_collection(Lifecycle)
    v829_meta['v829l'] = install_v829l_k1(Lifecycle)   # between v829e (train control) and v829i (gate)
    v829_meta['v829p_cutoff'] = install_v829p_served_cutoff(Lifecycle)
    v829_meta['v823_lifetime'] = install_v823_lifetime(Lifecycle)   # v8.2.3 pt765u: sealed hash + superseded cleanup   # v8.2.9p: served hotkeys get no opening grant
    v829_meta['v829i'] = install_v829i_promotion_gate(Lifecycle)
    v829_meta['v829j'] = install_v829j_eval_memo()
    v829_meta['v829k'] = install_v829k_seal_prefetch(Coordinator)
    v829_meta['v829u'] = install_v829u_dual_accept(Coordinator)   # v8.2.9u: dual accept (legacy 9 / extended 125)
    v829_meta['v829x'] = install_v829x_batch_cap(Coordinator)     # v8.2.9x: tunable per-miner batch cap (staged/BATCH-CAP.json)
    v829_meta['v829n'] = dict(install_v829n_capture_put(), **install_v829n_pregrade(Lifecycle))
    v829_meta['v829t'] = install_v829t_red_skip(Lifecycle, Coordinator)
    v829_meta['v829v'] = install_v829v_preopen(Lifecycle)   # outermost tick wrapper (pre-open next epoch on the same head)
    import subnet.distributed_roles as _job_queue  # the role/job queue (NOT the lifecycle mimo_coordinator)
    v829_meta['v829d_verify_workers'] = install_v829d_verifier_roster(_job_queue.Coordinator)
    v829_meta['v829d_roster_class'] = _job_queue.Coordinator.__module__ + '.' + _job_queue.Coordinator.__name__
    _ = args.operator_config
    if args.check:
        print(json.dumps(dict(
            version=VERSION, scope=SCOPE, v828=v828_meta, v829=v829_meta,
            gate_enabled=gate_enabled(), production_services_started=False,
            operator_admission_passed=True, v829_check=True,
        )))
        return
    print(json.dumps(dict(component="host-v824", **v824)), flush=True)
    print(json.dumps(dict(component="host-v828", **v828_meta)), flush=True)
    print(json.dumps(dict(component="host-v829", **v829_meta)), flush=True)
    sys.argv = [operator["base"]["entry"]["path"]] + operator["base"]["argv"]
    module.main()


if __name__ == "__main__":
    main()
