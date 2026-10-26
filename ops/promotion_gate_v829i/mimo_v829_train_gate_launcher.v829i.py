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
"""
from __future__ import annotations
import argparse, ast, importlib.util, json, sys, textwrap, time
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
COLLECTION_SECONDS = 1200
V829H_COLLECTION_SECONDS = (1200, 1800)


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
# Their seeds were generated on the pods and never left them; only public ids are pinned here.
# Added to the coordinator roster with role "verify" only; existing roster entries are unchanged.
V829D_VERIFY_WORKERS = {
    "387835e568ff94c9fdaebc9d8e00545cc5b71b9e6dce266a65d8df28299fc2c8": ["verify"],  # affine-verifier-h200-1
    "5d89b881679a36f9c44dbf09fbf4f88cdc2be200f58938e06d9d7811ae0d0172": ["verify"],  # affine-verifier-h200-2
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
    v829_meta['v829i'] = install_v829i_promotion_gate(Lifecycle)
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
