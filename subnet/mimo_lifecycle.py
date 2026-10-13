"""Restartable MiMo epoch orchestration; no chain writer or live entrypoint.

The host supplies read-only registration/discovery adapters, its existing role
queue, and full model readback. All external mutations receive stable signed
records or job IDs for retry. Sampling audits are deliberately independent.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import fcntl
import json
import os
from pathlib import Path
import secrets
import time

from .backend_jobs import signed
from .controller import save_manifest
from .storage import canonical
from .mimo_release_policy import (ReleasePolicy, POLICY_KEY, CHECKPOINT_KEY,
    descriptor, digest, need, sign)
from .mimo_coordinator import Coordinator
from .complete_group_capture import _immutable, _get

VERSION = 'mimo-independent-epoch-lifecycle-v1'


class TrainingRecoveryRequired(RuntimeError):
    def __init__(self, job_id, queue_status):
        self.job_id, self.queue_status = job_id, queue_status
        super().__init__('original training job requires authenticated recovery')


@dataclass(frozen=True)
class Services:
    # registrations() -> {snapshot_block, registrations:{hotkey:{uid,public_key,snapshot_block}}}
    registrations: object
    # committed_miners(manifest) must perform a complete bounded listing.
    committed_miners: object
    # discovery(opening, previous_sha256) returns opening and discovery readback digests.
    discovery: object
    # checkpoint_urls(descriptor) returns GET-only capabilities for the exact file inventory.
    checkpoint_urls: object
    # make_training_job(opening, dispatch, optimizer_step, namespace) -> authority-signed job.
    make_training_job: object
    enqueue_training: object
    # training_result(job_id) -> original worker-signed queue report request, or None.
    training_result: object
    # readback_publication(receipt_payload) -> exact descriptor, after FULL object digest checks.
    readback_publication: object
    # Optional only for original private diagnostic callbacks. The concrete
    # host always renews this exact source archive's GET capability per epoch.
    source_bundle: object = None


class Lifecycle:
    def __init__(self, root, bucket, identity, policy, services, *, resolve_task, grade_batch,
                 allow_public=False, collection_seconds=1200, max_tasks=512, first_round=0,
                 previous_discovery_sha256=None, grade_batches=None):
        self.root = Path(root).absolute()
        need(self.root.resolve() == self.root, 'ordinary lifecycle state ancestry')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.policy = ReleasePolicy(policy, identity.id, allow_public=allow_public)
        need(type(collection_seconds) is int and 1 <= collection_seconds <= 3600
             and type(max_tasks) is int and 1 <= max_tasks <= 512
             and type(first_round) is int and first_round >= 0, 'bounded explicit lifecycle limits')
        need(self.policy.mode != 'public' or first_round > 0,
             'public lifecycle requires the explicit next global network round')
        need(all(callable(getattr(services, k)) for k in Services.__dataclass_fields__ if k != 'source_bundle')
             and (services.source_bundle is None or callable(services.source_bundle)),
             'explicit lifecycle service adapters')
        need(self.policy.mode != 'public' or callable(services.source_bundle),
             'public lifecycle needs renewable exact source capabilities')
        self.bucket, self.identity, self.services = bucket, identity, services
        self.resolve_task, self.grade_batch = resolve_task, grade_batch
        self.grade_batches = grade_batches
        self.collection_seconds, self.max_tasks, self.first_round = collection_seconds, max_tasks, first_round
        self.allow_public = allow_public
        from .mimo_release_policy import _hash
        need(previous_discovery_sha256 is None or _hash(previous_discovery_sha256),
             'explicit original discovery CAS value')
        self.previous_discovery_sha256 = previous_discovery_sha256

    def _load(self, path):
        need(path.is_file() and not path.is_symlink(), 'ordinary lifecycle record')
        return json.loads(path.read_bytes())

    def _current(self):
        path = self.root / 'CURRENT.json'
        if not path.exists():
            save_manifest(path, dict(version=VERSION, policy_sha256=self.policy.policy_sha256,
                epoch_index=0, round=self.first_round, checkpoint=self.policy.scope['initial_checkpoint'],
                optimizer_step=0, checkpoint_authorization=None, previous_discovery_sha256=self.previous_discovery_sha256,
                collection_seconds=self.collection_seconds, max_tasks=self.max_tasks))
        current = self._load(path)
        need(current.get('version') == VERSION and current.get('policy_sha256') == self.policy.policy_sha256
             and current.get('collection_seconds') == self.collection_seconds
             and current.get('max_tasks') == self.max_tasks, 'same original lifecycle release/configuration')
        descriptor(current['checkpoint'])
        return current

    def _coordinator(self, directory):
        return Coordinator(directory / 'coordinator', self.bucket, self.identity,
            resolve_task=self.resolve_task, grade_batch=self.grade_batch, release_policy=self.policy,
            grade_batches=self.grade_batches)

    def _plan(self, current, directory, now):
        path = directory / 'PLAN.json'
        if path.exists():
            plan = self._load(path)
            need(plan['input'] == current, 'epoch plan belongs to the current checkpoint cursor')
            return plan
        snapshot = self.services.registrations()
        need(type(snapshot) is dict and set(snapshot) == {'snapshot_block','registrations'}
             and type(snapshot['snapshot_block']) is int and snapshot['snapshot_block'] >= 0
             and type(snapshot['registrations']) is dict, 'complete current registration snapshot')
        rows = snapshot['registrations']; keys=[]; uids=set()
        need(len(rows) <= 1000, 'bounded network registration set')
        for hotkey, row in rows.items():
            need(type(hotkey) is str and hotkey and type(row) is dict and
                 set(row) == {'uid','public_key','snapshot_block'} and type(row['uid']) is int
                 and 0 <= row['uid'] < 1000 and row['uid'] not in uids
                 and row['snapshot_block'] == snapshot['snapshot_block'], 'unique current UID snapshot')
            key=row['public_key']
            need(type(key) is str and len(key) == 64 and all(c in '0123456789abcdef' for c in key)
                 and key not in keys, 'unique registered Ed25519 key')
            keys.append(key);uids.add(row['uid'])
        if not keys:
            return None
        epoch = ('nonpayable-' if self.policy.mode == 'private' else '') + self.policy.run_id + '-e' + str(current['epoch_index']).zfill(6)
        m = copy.deepcopy(self.policy.scope['immutable_manifest'])
        m.update(epoch=epoch, start=now, deadline=now+self.collection_seconds,
                 checkpoint=copy.deepcopy(current['checkpoint']))
        if self.services.source_bundle is not None:
            m['source_bundle'] = self.services.source_bundle(m['source_bundle'])
        urls = self.services.checkpoint_urls(current['checkpoint'])
        need(type(urls) is dict and set(urls) == set(current['checkpoint']['files']),
             'full checkpoint download capabilities')
        from .backend_jobs import r2_url
        for url in urls.values(): r2_url(url, 'GET')
        m['checkpoint']['read_urls'] = urls
        m['sampling_contract']['randomness'] = secrets.token_hex(32)
        m['sampling_contract']['calibration']['checkpoint'] = current['checkpoint']['id']
        m[POLICY_KEY] = self.policy.envelope
        if current['checkpoint_authorization'] is not None:
            m[CHECKPOINT_KEY] = current['checkpoint_authorization']
        self.policy.validate_manifest(m)
        regs = sign(self.identity, dict(epoch=epoch, **snapshot))
        plan = dict(input=current, manifest=m, miners=sorted(keys), registrations=regs)
        save_manifest(path, plan)
        return plan

    def _capture(self, coordinator, manifest, now):
        from .complete_group_capture import SubmissionRejected
        miners = self.services.committed_miners(manifest)
        need(type(miners) is list and len(miners) <= len(manifest['capabilities'])
             and len(set(miners)) == len(miners) and set(miners) <= set(manifest['capabilities']),
             'complete registered commitment listing')
        rejected = {}
        for miner in sorted(miners):
            try:
                coordinator.collect(miner, now=now)
            except SubmissionRejected as error:
                # Malformed signed submissions do not stall every honest miner.
                # Infrastructure errors propagate and cannot finalize a partial listing.
                rejected[miner] = str(error)[:1024]
        return rejected

    def _audit_population(self, directory, coordinator, opening, dispatch, plan, current):
        path = directory / 'AUDIT-POPULATION.json'
        if path.exists():
            document = self._load(path)
        else:
            from .mimo_audit_bridge import build_population
            sealed = signed(self._load(coordinator.root / 'POPULATION.json'), self.identity.id)
            captures = {x['payload']['miner']:x for x in sealed['captures']}
            commitments = {}
            for miner, envelope in captures.items():
                capture = signed(envelope, self.identity.id)
                raw, _ = _get(self.bucket, capture['snapshot'] + '/commitment.json', 16*1024**2)
                from .storage import sha
                need(sha(raw) == capture['commitment_sha256'], 'original captured commitment digest')
                commitments[miner] = json.loads(raw)
            value = build_population(opening, captures, commitments, dispatch, plan['registrations'],
                round=current['round'], committed_at=sealed['sealed_at'], authority=self.identity.id,
                manifest_guard=self.policy.validate_manifest)
            document = sign(self.identity, value)
            save_manifest(path, document)
        from .mimo_audit_bridge import authenticate_population
        authenticate_population(document, self.identity.id, manifest_guard=self.policy.validate_manifest)
        _immutable(self.bucket, self.policy.namespace()+'/audit-populations/'+plan['manifest']['epoch']+'.json',
                   canonical(document))
        return document

    def _training(self, directory, opening, dispatch, current, now):
        m = signed(opening, self.identity.id); d = signed(dispatch, self.identity.id)
        if d['selected_task_count'] == 0:
            return dict(status='no-update', checkpoint=current['checkpoint'],
                        optimizer_step=current['optimizer_step'], checkpoint_authorization=current['checkpoint_authorization'])
        path = directory / 'TRAINING-JOB.json'
        if path.exists(): envelope=self._load(path)
        else:
            envelope=self.services.make_training_job(opening,dispatch,current['optimizer_step'],self.policy.namespace())
            # Save only after validation below; a caller cannot substitute another opening or dispatch.
        job=signed(envelope,self.identity.id)
        need(job.get('role')=='train' and job.get('manifest')==opening
             and job.get('training',{}).get('optimizer_step')==current['optimizer_step']
             and job['training'].get('publication_namespace')==self.policy.namespace()
             and job['training'].get('dispatch',{}).get('sha256')==digest(dispatch),
             'trainer job must consume this original opening and frozen dispatch')
        from .mimo_training_worker import validate_job
        validate_job(job,m,self.identity.id,allow_public=self.allow_public)
        if not path.exists(): save_manifest(path,envelope)
        # Stable original signed job: retry after a lost queue ACK cannot create another update.
        need(self.services.enqueue_training(envelope)==job['job_id'], 'original trainer queue acknowledgement')
        result_path=directory/'TRAINING-RESULT.json'
        if result_path.exists(): result_envelope=self._load(result_path)
        else:
            result_envelope=self.services.training_result(job['job_id'])
            if result_envelope is None:return None
        request=signed(result_envelope,job['trainer_worker'])
        need(request.get('action')=='report' and request.get('job_id')==job['job_id'],
             'original authorized trainer report request')
        report=request['report']
        from .mimo_worker import validate_report
        # Fetching and authenticating a queued result can outlast the tick's
        # starting timestamp. Judge its completion time when we receive it;
        # the original signed job deadline still bounds allowed completion.
        validate_report(report,job,m,self.identity.id,digest(job),time.time())
        need(report['training']['status']=='updated', 'nonempty dispatch requires a real trainer update')
        receipt=report['publication_receipt']; published=signed(receipt,job['training']['publisher_id'])
        actual=descriptor(self.services.readback_publication(published))
        need(actual==published['descriptor'], 'full stored model readback must match signed publication')
        authorization=self.policy.authorize_checkpoint(actual,input_checkpoint=current['checkpoint']['id'],
            optimizer_step=report['training']['optimizer_step'],publication_receipt_sha256=digest(receipt),identity=self.identity)
        if not result_path.exists():save_manifest(result_path,result_envelope)
        return dict(status='updated',checkpoint=actual,optimizer_step=report['training']['optimizer_step'],
            checkpoint_authorization=authorization,publication_receipt_sha256=digest(receipt))

    def tick(self, *, now):
        """One bounded orchestration pass; the host schedules calls and records errors.

        A slow/failed trainer preserves the current epoch/checkpoint and every
        prior reward assessment. The audit scheduler may keep consuming durable
        populations independently while this method waits for training.
        """
        need(type(now) is int and now >= 0, 'explicit integer coordinator time')
        fd=os.open(self.root/'lifecycle.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from .mimo_source_migration import boundary_gate
            held=boundary_gate(self.root,self.policy,self.identity,now=now)
            if held is not None:return held
            current=self._current(); directory=self.root/'epochs'/str(current['epoch_index'])
            directory.mkdir(parents=True,exist_ok=True)
            plan=self._plan(current,directory,now)
            if plan is None:return dict(status='waiting-for-registrations',checkpoint=current['checkpoint']['id'])
            coordinator=self._coordinator(directory)
            if now >= plan['manifest']['deadline'] and not (coordinator.root/'OPENING.json').exists():
                # The signed opening is saved BEFORE its first bucket write, so
                # this precise state never issued capabilities or discovery.
                # Do not reuse its expired upload window after a process crash.
                abandoned=sign(self.identity,dict(version=VERSION,policy_sha256=self.policy.policy_sha256,
                    epoch=plan['manifest']['epoch'],reason='expired-before-opening',
                    manifest_sha256=digest(plan['manifest']),checkpoint=current['checkpoint']['id']))
                _immutable(self.bucket,self.policy.namespace()+'/epochs/'+plan['manifest']['epoch']+
                           '/ABANDONED-BEFORE-OPENING.json',canonical(abandoned))
                save_manifest(directory/'ABANDONED-BEFORE-OPENING.json',abandoned)
                save_manifest(self.root/'CURRENT.json',dict(current,epoch_index=current['epoch_index']+1))
                return dict(status='abandoned-before-opening',epoch=plan['manifest']['epoch'])
            opening=coordinator.open(plan['manifest'],plan['miners']);manifest=signed(opening,self.identity.id)
            discovery_path=directory/'DISCOVERY.json'
            if not discovery_path.exists():
                receipt=self.services.discovery(opening,current['previous_discovery_sha256'])
                from .mimo_release_policy import _hash
                need(type(receipt)is dict and receipt.get('sha256')==digest(opening)
                     and _hash(receipt.get('discovery_sha256')), 'discovery readback of exact opening and descriptor')
                save_manifest(discovery_path,receipt)
            if not (coordinator.root/'POPULATION.json').exists():
                rejected=self._capture(coordinator,manifest,now)
                save_manifest(directory/'CAPTURE-REJECTIONS.json',sign(self.identity,dict(epoch=manifest['epoch'],rejected=rejected)))
            if now < manifest['deadline']:
                return dict(status='collecting',epoch=manifest['epoch'],deadline=manifest['deadline'])
            dispatch=coordinator.seal(max_tasks=self.max_tasks,now=now)
            self._audit_population(directory,coordinator,opening,dispatch,plan,current)
            try:
                result=self._training(directory,opening,dispatch,current,now)
            except TrainingRecoveryRequired as error:
                status=dict(status='recovery-required',epoch=manifest['epoch'],
                    job_id=error.job_id,queue_status=error.queue_status,
                    checkpoint=current['checkpoint']['id'])
                save_manifest(directory/'RECOVERY-REQUIRED.json',status)
                return status
            if result is None:return dict(status='training',epoch=manifest['epoch'])
            final=sign(self.identity,dict(version=VERSION,policy_sha256=self.policy.policy_sha256,
                epoch=manifest['epoch'],round=current['round'],opening_sha256=digest(opening),
                dispatch_sha256=digest(dispatch),input_checkpoint=current['checkpoint']['id'],result=result))
            _immutable(self.bucket,self.policy.namespace()+'/epochs/'+manifest['epoch']+'/FINALIZED.json',canonical(final))
            save_manifest(directory/'FINALIZED.json',final)
            successor=dict(current,epoch_index=current['epoch_index']+1,round=current['round']+1,
                checkpoint=result['checkpoint'],optimizer_step=result['optimizer_step'],
                checkpoint_authorization=result['checkpoint_authorization'],
                previous_discovery_sha256=self._load(discovery_path)['discovery_sha256'])
            save_manifest(self.root/'CURRENT.json',successor)
            return dict(status='finalized',epoch=manifest['epoch'],training=result['status'],checkpoint=result['checkpoint']['id'])
        finally:
            os.close(fd)
