"""Default-off, ROOT-policy-bound scheduler for the same independent 128 tasks.

The CPU adapter stages a qualified source, observes originals, publishes full
readback ACKs and retires only each group's private model. A dispatch intent is
durable before its one launch; uncertain transport never grants another launch.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

from subnet.backend_jobs import canonical, signed, file_map
from ops.owned_cached_group_operator import private_json, save
from ops.owned_cached_larger_cohort import SOURCE

VERSION = 'continuous-owned-heldout128-policy-v1'
JOURNAL_VERSION = 'continuous-owned-heldout128-journal-v1'


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_policy(envelope, authority, now):
    p = signed(envelope, authority)
    if p.get('version') != VERSION or p.get('execute_allowed') is not True:
        raise ValueError('explicit ROOT continuous128 policy required')
    if any(type(p.get(k)) not in (int, float) or not math.isfinite(p[k]) for k in ('created_at', 'expires_at')):
        raise ValueError('finite service policy clocks')
    if not p['created_at'] <= now < p['expires_at'] or p['expires_at']-p['created_at'] > 90*86400:
        raise ValueError('bounded fresh service policy')
    if p.get('source_sha256') != SOURCE or len(p.get('source_files', {})) != 177:
        raise ValueError('original qualified4db scientific source')
    if type(p.get('first_optimizer_step')) is not int or p['first_optimizer_step'] < 1:
        raise ValueError('prospective optimizer step floor')
    bootstrap=p.get('bootstrap_completed',[])
    if (not bootstrap or len({v['checkpoint'] for v in bootstrap})!=len(bootstrap) or
        any(type(v['optimizer_step']) is not int or not 1<=v['optimizer_step']<p['first_optimizer_step'] for v in bootstrap) or
        max(v['optimizer_step'] for v in bootstrap)!=p['first_optimizer_step']-1):
        raise ValueError('completed predecessor watermark and exact historical adoption')
    if (type(p.get('per_checkpoint_job_count')) is not int or p['per_checkpoint_job_count'] != 4 or
        type(p.get('per_original_task_count')) is not int or p['per_original_task_count'] != 32):
        raise ValueError('four distinct32 originals only')
    if type(p.get('group_lifetime_seconds')) is not int or not 0 < p['group_lifetime_seconds'] <= 7200:
        raise ValueError('bounded original group lifetime')
    groups = p['groups']
    if (len(groups) != 4 or [g['group'] for g in groups] != list(range(4)) or
        digest(groups) != p['cohort_sha256']):
        raise ValueError('fixed ordered128 cohort')
    indices = []
    for g in groups:
        if (type(g['group']) is not int or g['env_id'] != 'affine_math' or len(g['indices']) != 32 or
            type(g['harness'].get('top_p')) not in (int,float) or type(g['harness'].get('max_output_tokens')) is not int or
            any(type(i) is not int or not 0 <= i < 7496 for i in g['indices']) or
            g['seeds'] != [20261002+i*1000 for i in g['indices']] or
            g['harness'] != {'version':'text-tools-long-kv-v3','policy':'autoregressive',
                             'max_output_tokens':1024,'temperature':.7,'top_p':1.}):
            raise ValueError('original cached1024 tasks and seeds')
        indices += g['indices']
    mining = p['mining_indices']
    excluded = p['old32_indices']
    if (len(set(indices)) != 128 or len(mining) != 6746 or len(set(mining)) != 6746 or
        any(type(i) is not int or not 0 <= i < 7496 for i in mining) or
        len(excluded) != 32 or len(set(excluded)) != 32 or
        set(indices) & (set(mining) | set(excluded)) or set(excluded) & set(mining)):
        raise ValueError('no mining or previous32 leakage')
    for name in ('production_directory', 'journal_directory', 'adapter_path'):
        path = Path(p[name])
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError('canonical service-owned paths')
    if p.get('production_services_stopped') is not False or p.get('normal32_replaced') is not False:
        raise ValueError('independent default-off rollout preserves normal32')
    return p


def discover(directory, authority, now):
    """Only immutable original ROOT-signed learner completions admit a model."""
    out = {}
    for path in sorted(Path(directory).glob('*-signed-learner-completion.json')):
        raw = path.read_bytes()
        value = signed(json.loads(raw), authority)
        step = value.get('round')
        when = value.get('completed_at')
        cp = value.get('next_checkpoint')
        if (type(step) is not int or step < 0 or type(when) not in (int,float) or
            not math.isfinite(when) or when > now or not isinstance(cp,str) or
            len(cp) != 64 or any(c not in '0123456789abcdef' for c in cp) or
            path.name != value['epoch']+'-signed-learner-completion.json'):
            raise ValueError('authentic finite original learner completion')
        row = dict(checkpoint=cp, round=step,
                   completed_at=when, completion_sha256=hashlib.sha256(raw).hexdigest(),
                   completion=envelope_copy(json.loads(raw)))
        if cp in out and out[cp]['completion_sha256'] != row['completion_sha256']:
            raise ValueError('conflicting original completion for checkpoint')
        out[cp] = row
    return sorted(out.values(), key=lambda r:(r['completed_at'],r['round'],r['checkpoint']), reverse=True)


def envelope_copy(value):
    return copy.deepcopy(value)


def scope_checkpoint(packet):
    return packet['scope']['payload']['checkpoint']


class Continuous128:
    """One exclusively owned durable outbox; originals survive every restart.

    Adapter methods: publication(row), prepare(policy, publication, identity),
    launch(packet), observe(packet), archive_complete(packet, result), idle().
    prepare is CPU-only and returns exact four signed originals plus scope.
    publication must full-read/authenticate checkpoint+optimizer publication.
    observe performs existing passive ACK relay; it never issues a GPU job.
    """
    def __init__(self, envelope, authority, adapter, *, clock=time.time):
        self.envelope=envelope; self.authority=authority; self.adapter=adapter; self.clock=clock
        self.policy=validate_policy(envelope,authority,clock())
        self.root=Path(self.policy['journal_directory']); self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.is_symlink(): raise ValueError('owned service journal directory')
        self.fd=None

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:
            if self.fd is not None: os.close(self.fd);self.fd=None
            raise

    def _enter(self):
        self.fd=os.open(self.root/'exclusive.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try: fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException: os.close(self.fd);self.fd=None;raise
        self.path=self.root/'outbox.json'
        binding=digest(self.policy)
        if self.path.exists():
            self.state=private_json(self.path)
            if self.state.get('version')!=JOURNAL_VERSION or self.state.get('policy_sha256')!=binding:
                raise ValueError('immutable original service-policy outbox binding')
        else:
            self.state=dict(version=JOURNAL_VERSION,policy_sha256=binding,checkpoints={})
            # Adapter full-reads/authenticates historical archives and all ACKs.
            # In-flight CP14 cannot satisfy this bootstrap; prospective policy
            # starts at15 only after its existing originals genuinely complete.
            for entry in self.policy['bootstrap_completed']:
                envelope=self.adapter.bootstrap_complete(entry)
                if digest(envelope)!=entry['summary_sha256']:
                    raise ValueError('exact original completed bootstrap summary')
                value=signed(envelope,self.authority)
                if value.get('version')=='owned-cached-heldout128-checkpoint-actual-v1':
                    cp=value['checkpoint']['id'] if isinstance(value['checkpoint'],dict) else value['checkpoint']
                    complete=(value.get('all_four_genuine_full_R2_ACKs') is True and
                              value.get('group_owned_model_retired') is True and value.get('task_count')==128)
                elif value.get('version')=='owned-cached-heldout128-paired-actual-v1':
                    # Adapter authenticates this scoped group's original4 ACKs
                    # to the explicit checkpoint; the paired envelope stays intact.
                    cp=entry['checkpoint']
                    complete=(entry.get('group_label') in value.get('groups',{}) and
                              value.get('all_eight_genuine_full_R2_ACKs') is True and
                              value.get('all_group_owned_models_retired') is True and value.get('task_count_per_checkpoint')==128)
                else:complete=False;cp=None
                if (cp!=entry['checkpoint'] or value.get('source_sha256')!=SOURCE or
                    value.get('cohort_sha256')!=self.policy['cohort_sha256'] or not complete):
                    raise ValueError('complete original predecessor128 only')
                self.state['checkpoints'][cp]=dict(checkpoint=cp,phase='complete',summary=envelope,bootstrap=True)
            save(self.path,self.state)
        return self

    def __exit__(self,*args):
        if self.fd is not None: os.close(self.fd);self.fd=None

    def persist(self): save(self.path,self.state)

    def step(self):
        if self.fd is None: raise ValueError('exclusive service lease required')
        validate_policy(self.envelope,self.authority,self.clock())
        if private_json(self.path)!=self.state: raise ValueError('service outbox changed')
        rows=self.state['checkpoints']
        for row in rows.values():
            if row['phase']=='complete' and row.get('projection_pending'):
                self.adapter.publish_pointer(row['packet'],row['summary'])
                row['projection_pending']=False;self.persist()
                return dict(status='complete128-projection-published',checkpoint=row['checkpoint'])
        if any(r['phase']=='infrastructure-pending' for r in rows.values()):
            return dict(status='original-infrastructure-reconciliation-required',new_job_started=False)
        active=[r for r in rows.values() if r['phase'] in ('prepared','dispatch_attempted','observing')]
        if len(active)>1: raise ValueError('one independent GPU group at a time')
        if active:
            row=active[0];packet=row['packet']
            if row['phase']=='prepared':
                if not self.adapter.idle():return dict(status='physical-reservation-deferred')
                if self.clock()>=packet['expires_at']:
                    row['phase']='expired-unissued';self.persist();return dict(status='expired-unissued')
                row.update(phase='dispatch_attempted',attempted_at=self.clock());self.persist()
                self.adapter.launch(packet)
                row['phase']='observing';self.persist()
                return dict(status='observing-original',checkpoint=row['checkpoint'])
            if row['phase']=='dispatch_attempted' and hasattr(self.adapter,'reconcile_launch'):
                continuation=self.adapter.reconcile_launch(packet)
                if continuation['status'] in ('expired-unissued','physical-reservation-deferred'):
                    return continuation
                row['phase']='observing';self.persist()
            result=self.adapter.observe(packet)
            if result['status']=='complete':
                if (result.get('durable_ACK_count')!=4 or result.get('owned_model_retired') is not True or
                    result.get('count')!=128 or type(result.get('successes')) is not int or
                    not 0<=result['successes']<=128):
                    raise ValueError('no partial score or absent retirement')
                summary=self.adapter.archive_complete(packet,result)
                value=signed(summary,self.authority)
                if (value.get('version')!='owned-cached-heldout128-checkpoint-actual-v1' or
                    value.get('checkpoint') not in (row['checkpoint'],scope_checkpoint(packet)) or
                    value.get('cohort_sha256')!=self.policy['cohort_sha256'] or value.get('source_sha256')!=SOURCE or
                    value.get('all_four_genuine_full_R2_ACKs') is not True or value.get('group_owned_model_retired') is not True or
                    value.get('task_count')!=128 or value.get('successes')!=result['successes']):
                    raise ValueError('actual signed full128 archive summary')
                row.update(phase='complete',summary=summary,projection_pending=True);self.persist()
            elif result['status'] in ('original-infrastructure-failure','expired-unissued'):
                # No fabricated zero and no replacement. Ownership/cleanup
                # remains journaled for explicit automated reconciliation.
                row.update(phase='infrastructure-pending',failure=result);self.persist()
            return result
        for original in discover(self.policy['production_directory'],self.authority,self.clock()):
            cp=original['checkpoint']
            if cp in rows:continue
            publication=self.adapter.publication(original)
            descriptor=signed(publication['checkpoint_descriptor'],self.authority)
            if descriptor['id']!=cp or file_map(descriptor['files'])!=cp:
                raise ValueError('full original published checkpoint identity')
            state=signed(publication['optimizer_publication'],self.authority)
            descriptor=state['descriptor']
            if (state.get('version')!='authority-persistent-trainer-state-v1' or
                digest(descriptor)!=state['descriptor_sha256'] or descriptor['inference_checkpoint']!=cp or
                descriptor['epoch']!=signed(original['completion'],self.authority)['epoch'] or
                type(descriptor['optimizer_steps']) is not int):
                raise ValueError('original optimizer/model lineage publication')
            if descriptor['optimizer_steps']<self.policy['first_optimizer_step']:continue
            identity=digest([digest(self.policy),cp,self.policy['cohort_sha256']])
            packet=self.adapter.prepare(self.policy,publication,identity)
            if (packet.get('checkpoint')!=cp or packet.get('cohort_sha256')!=self.policy['cohort_sha256'] or
                packet.get('identity')!=identity or len(packet.get('original_jobs',[]))!=4 or
                packet.get('optimizer_step')!=descriptor['optimizer_steps'] or
                not self.clock()<packet.get('expires_at',0)<=min(self.policy['expires_at'],self.clock()+self.policy['group_lifetime_seconds'])):
                raise ValueError('exact bounded per-checkpoint four-original packet')
            from ops.owned_cached_group_retention import validate_scope
            from ops.owned_cached_group_operator import validate_originals
            scope=validate_scope(packet['scope'],self.authority,packet['workspace'],self.policy['source_files'],now=self.clock())
            if scope['groups']!=self.policy['groups'] or scope['runtime_versions']!=self.policy['runtime_versions'] or scope['checkpoint']['id']!=cp:
                raise ValueError('exact pinned service scientific scope')
            validate_originals(scope,packet['original_jobs'],self.authority)
            from ops.owned_cached_group_ack_relay import QualifiedGroupObserver
            QualifiedGroupObserver(scope,packet['original_jobs'],self.authority)
            payloads=[signed(j,self.authority) for j in packet['original_jobs']]
            if len({j['job_id'] for j in payloads})!=4:
                raise ValueError('four distinct immutable original identities')
            rows[cp]=dict(checkpoint=cp,phase='prepared',original=original,packet=packet)
            self.persist()
            return dict(status='prepared-originals',checkpoint=cp)
        return dict(status='waiting-published-checkpoint')


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--authority',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    envelope=private_json(a.policy);policy=validate_policy(envelope,a.authority,time.time())
    if not a.execute:
        print(json.dumps(dict(dispatch_allowed=False,reason='default-off, ROOT signed service policy and --execute required')));return
    path=Path(policy['adapter_path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=policy['adapter_sha256']:raise ValueError('qualified CPU adapter pin')
    spec=importlib.util.spec_from_file_location('root_owned128_adapter',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    adapter=module.Adapter(policy,a.authority)
    with Continuous128(envelope,a.authority,adapter) as service:
        while True:
            print(json.dumps(service.step(),sort_keys=True),flush=True)
            if a.once:return
            time.sleep(5)


if __name__=='__main__':main()
