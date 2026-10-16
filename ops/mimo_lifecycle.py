#!/usr/bin/env python3
"""Explicit opt-in host runner for a prepared MiMo lifecycle; never writes weights.

No default configuration, public key, service, or live pointer is selected.
An operator must provide an enabled owner-only configuration and qualified
signed release. Public mode additionally requires --allow-public-mimo.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def private_json(path):
    path=Path(path).absolute();st=path.stat()
    if path.resolve()!=path or not path.is_file() or st.st_uid!=os.getuid() or st.st_mode&0o077 or st.st_size>32*1024**2:
        raise ValueError('owner-only ordinary bounded lifecycle configuration required')
    return json.loads(path.read_bytes())


def archive_execution_resolver(release, archive_execution, live):
    """Per-epoch worker execution metadata for audit/evaluation jobs.

    manifest -> its signed release(-chain) member -> that member's qualified source
    archive -> the ROOT-approved runtime inventory for that archive. A job for an
    epoch produced under an earlier chain member must carry that member's inventory
    (workers route by archive and refuse any other inventory), never the live
    deployment's. The active member's archive always maps to the live closure.
    """
    import copy
    def exact(entry):
        return (type(entry) is dict and set(entry) == {'source_files', 'runtime_versions'}
                and type(entry['source_files']) is dict and entry['source_files']
                and type(entry['runtime_versions']) is dict)
    if not exact(live):
        raise ValueError('exact live worker execution metadata')
    members = [policy for _, policy in getattr(release, 'members', [(None, release)])]
    archives = {m.policy_sha256: m.scope['immutable_manifest']['source_bundle']['sha256'] for m in members}
    active = archives[release.policy_sha256]
    table = {} if archive_execution is None else archive_execution
    if type(table) is not dict or not all(type(k) is str and exact(v) for k, v in table.items()):
        raise ValueError('exact per-archive execution metadata')
    if not set(table) <= set(archives.values()):
        raise ValueError('archive execution limited to signed release chain archives')
    if hasattr(release, 'members') and set(table) != set(archives.values()):
        raise ValueError('every signed release chain archive needs ROOT-approved execution metadata')
    if active in table and table[active] != live:
        raise ValueError('active archive execution must equal the deployed worker closure')
    table = copy.deepcopy(table); table[active] = copy.deepcopy(live)
    auditable = frozenset(a for a, e in table.items() if e == live)
    def resolve(manifest):
        if hasattr(release, 'policy_for_manifest'):
            policy = release.policy_for_manifest(manifest)
        else:
            release.validate_manifest(manifest); policy = release
        archive = archives[policy.policy_sha256]
        if (manifest.get('source_bundle') or {}).get('sha256') != archive:
            raise ValueError('manifest source archive differs from its signed release member')
        return copy.deepcopy(table[archive])
    # Archives whose ROOT-approved inventory equals the closure the warm workers run:
    # only these can pass both source_for_job and mimo_worker.source_guard.
    resolve.auditable_archives = auditable
    return resolve



def archive_scoped_audit_scheduler(base, auditable_archives, *, report_name='UNAUDITABLE-BY-DESIGN.json'):
    """Audit scheduler that only selects, renews and dispatches rows whose epoch archive the
    running workers can execute (registry inventory == warm worker closure).

    Draws of other epochs are parked: hidden from the base scheduler on load (never renewed or
    dispatched) and written back byte-for-byte on save (history kept). Their populations are not
    re-authenticated per tick. Excluded epochs are reported, never silently dropped.
    """
    import json
    import time
    from subnet import mimo_audit_scheduler as mas
    auditable = frozenset(auditable_archives)
    if not auditable or not all(type(a) is str and len(a) == 64 for a in auditable):
        raise ValueError('explicit auditable archive set')
    PARK = '_parked_unauditable'

    class ArchiveScopedAuditScheduler(base):
        def _population_facts(self, checksum):
            cache = self.__dict__.setdefault('_facts', {})
            if checksum not in cache:
                path = self.root/(checksum+'.population.json')
                mas.need(not path.is_symlink() and path.stat().st_size <= 256*1024**2, 'original audit population file')
                doc = json.loads(path.read_bytes())
                mas.need(mas.digest(doc) == checksum, 'immutable population hash')
                payload = doc['payload']
                cache[checksum] = dict(archive=payload['manifest_document']['payload']['source_bundle']['sha256'],
                                       records=len(payload['records']))
            return cache[checksum]

        def _excluded(self, state):
            out = {}
            for epoch, checksum in state['populations'].items():
                facts = self._population_facts(checksum)
                if facts['archive'] not in auditable: out[epoch] = facts
            return out

        def _load(self):
            state = super()._load()
            mas.need(PARK not in state, 'parked draws never persisted under their own key')
            excluded = self._excluded(state)
            parked = {k: d for k, d in state['draws'].items() if d['row']['epoch'] in excluded}
            state['draws'] = {k: d for k, d in state['draws'].items() if k not in parked}
            state[PARK] = parked
            return state

        def _save(self, state):
            value = dict(state); parked = value.pop(PARK, {})
            mas.need(not set(parked) & set(value['draws']), 'parked and active draws disjoint')
            value['draws'] = dict(value['draws'], **parked)
            super()._save(value)

        def populations(self):
            state = super()._load(); result = []
            for epoch, checksum in sorted(state['populations'].items()):
                if self._population_facts(checksum)['archive'] not in auditable: continue
                path = self.root/(checksum+'.population.json')
                doc = json.loads(path.read_bytes()); mas.need(mas.digest(doc) == checksum, 'immutable population hash')
                p = mas.authenticate_population(doc, self.identity.id, manifest_guard=self.guard)
                mas.need(p['manifest_document']['payload']['epoch'] == epoch, 'population epoch directory binding')
                result.append(doc)
            self._report(state)
            return result

        def _report(self, state):
            excluded = self._excluded(state); epochs = {}
            for epoch, facts in sorted(excluded.items()):
                draws = [d for d in state['draws'].values() if d['row']['epoch'] == epoch]
                epochs[epoch] = dict(archive=facts['archive'], records=facts['records'], draws_parked=len(draws),
                                     draws_with_job=sum(1 for d in draws if d.get('job_id')))
            value = dict(version='mimo-audit-unauditable-by-design-v1', auditable_archives=sorted(auditable),
                reason=('MiMo verify jobs run in the warm worker; source_for_job requires the archive registry inventory and '
                        'mimo_worker.source_guard requires the worker running closure. Rows of other archives cannot pass both '
                        'on the running workers, so they are never selected, renewed or dispatched.'),
                excluded_epochs=epochs, excluded_records=sum(e['records'] for e in epochs.values()),
                excluded_draws_parked=sum(e['draws_parked'] for e in epochs.values()))
            path = self.root/report_name
            old = json.loads(path.read_bytes()) if path.exists() else None
            if old is None or {k: v for k, v in old.items() if k != 'updated_at'} != value:
                mas.atomic(path, dict(value, updated_at=time.time()))

    return ArchiveScopedAuditScheduler

def run(args):
    config=private_json(args.config)
    if config.get('enabled')is not True:
        raise ValueError('lifecycle configuration is disabled')
    from ops.mimo_training_rehearsal import identity
    actor=identity(config['authority_seed_file'])
    from subnet.mimo_release_policy import ReleasePolicy
    policy=ReleasePolicy(config['release_policy'],actor.id,allow_public=args.allow_public_mimo)
    release_succession=config.get('release_succession')
    release_chain=None
    if release_succession is not None:
        from subnet.mimo_release_succession import ReleaseChain
        release_chain=ReleaseChain(release_succession,actor.id,allow_public=args.allow_public_mimo)
        if release_chain.active.policy_sha256!=policy.policy_sha256:
            raise ValueError('configured release policy must be the active succession member')
    release=release_chain if release_chain is not None else policy
    release_guard=release.validate_manifest
    from subnet.storage import Bucket
    from subnet.distributed_roles import Coordinator as Queue,CoordinatorServer
    from subnet.chain import ChainAdapter
    from subnet.mimo_lifecycle import Lifecycle
    from subnet.mimo_lifecycle_adapters import R2Services
    from subnet.complete_group_native import resolve_task,grade_batch
    from subnet.mimo_native_pool import NativeGradingPool,authenticated_tokenizer
    from subnet.mimo_runtime import file_hash
    from subnet.mimo_grpo import TrainerConfig
    from subnet.mimo_worker import source_inventory
    if source_inventory()!=config['adapters']['source_files']:
        raise ValueError('host source differs from explicit prepared worker closure')
    source_root=Path(__file__).resolve().parents[1]
    for name,expected in policy.scope['task_assets'].items():
        path=source_root/name
        if (path.resolve()!=path or not path.is_file() or path.stat().st_size!=expected['bytes']
                or file_hash(path)!=expected['sha256']):
            raise ValueError('native task asset differs from qualified release')
    state=Path(config['state_root']).absolute()
    if state.resolve()!=state:raise ValueError('ordinary lifecycle root required')
    # Tokenizer only: validator never loads GPU model weights or optimizer state.
    tokenizer_root=Path(config['tokenizer_directory']).absolute()
    files=policy.scope['initial_checkpoint']['files']
    tokenizer=authenticated_tokenizer(tokenizer_root,files,
        policy.scope['immutable_manifest']['mimo_runtime']['chat_template_sha256'])
    native_pool=NativeGradingPool(tokenizer_root,files,
        policy.scope['immutable_manifest']['mimo_runtime']['chat_template_sha256'],
        policy.scope['task_assets'],**config.get('native_grading',{}))
    bucket=Bucket(config['bucket'])
    queue=Queue(config['queue_database'],actor.id,config['workers'],allow_public_mimo=args.allow_public_mimo)
    chain=ChainAdapter(state/'registration-observations',**config.get('chain',{}))
    training=TrainerConfig(**config['trainer_config']).validate().document()
    adapter=R2Services(bucket,actor,policy,queue,chain,**config['adapters'],trainer_config=training,
        readback_state=state/'model-readbacks')
    lifecycle=Lifecycle(state,bucket,actor,config['release_policy'],adapter.services(),
        resolve_task=lambda m,e,i:resolve_task(m,e,i,tokenizer),
        grade_batch=lambda b,m,miner:grade_batch(b,m,tokenizer,miner=miner),
        grade_batches=native_pool,release_succession=release_succession,
        allow_public=args.allow_public_mimo,**config.get('lifecycle',{}))
    # This runner owns one explicit loopback API for this exact DB/roster. The
    # operator supplies its private authenticated tunnel; no public bind occurs.
    server_config=config['queue_server']
    if (set(server_config)!={'host','port'} or server_config['host']!='127.0.0.1'
            or type(server_config['port'])is not int or not 1024<=server_config['port']<=65535):
        raise ValueError('explicit loopback queue API address required')
    from subnet.mimo_release_policy import sign
    import threading
    # Audit/evaluation jobs carry the inventory of their own epoch's archive.
    execution=archive_execution_resolver(release,config.get('archive_execution'),
        dict(source_files=adapter.source_files,runtime_versions=adapter.runtime_versions))
    stopped=threading.Event();audit_thread=None;eval_thread=None
    audit_config=dict(config.get('audit_scheduler',{}))
    audit_enabled=audit_config.pop('enabled',False)
    if type(audit_enabled)is not bool or(policy.mode=='public'and not audit_enabled):
        raise ValueError('public lifecycle requires its independent audit scheduler')
    if audit_enabled:
        from subnet.mimo_audit_scheduler import AuditScheduler
        AuditScheduler=archive_scoped_audit_scheduler(AuditScheduler,execution.auditable_archives)
        scheduler=AuditScheduler(bucket=bucket,queue=queue,identity=actor,
            manifest_guard=release_guard,metadata=lambda m:dict(execution(m),
                checkpoint_read_urls=adapter.checkpoint_urls(m['checkpoint'])),**audit_config)
        def audit_loop():
            admitted=set()
            while not stopped.is_set():
                try:
                    for path in sorted((state/'epochs').glob('*/AUDIT-POPULATION.json')):
                        if path in admitted:continue
                        if path.is_symlink()or not 0<path.stat().st_size<=256*1024**2:
                            raise ValueError('bounded original audit population file')
                        scheduler.admit(json.loads(path.read_bytes()));admitted.add(path)
                    scheduler.tick()
                except Exception as error:
                    print(json.dumps(dict(component='auditor',status='retrying',error_type=type(error).__name__)),flush=True)
                stopped.wait(args.interval)
        audit_thread=threading.Thread(target=audit_loop,name='mimo-audit-scheduler',daemon=True)
    eval_config=dict(config.get('evaluation_scheduler',{}))
    eval_enabled=eval_config.pop('enabled',False)
    if type(eval_enabled)is not bool or(policy.mode=='public'and not eval_enabled):
        raise ValueError('public lifecycle requires its independent evaluator scheduler')
    if eval_enabled:
        from subnet.mimo_evaluation_scheduler import EvaluationScheduler
        evaluator=EvaluationScheduler(bucket=bucket,queue=queue,identity=actor,policy=release,
            metadata=lambda m:dict(execution(m),
                checkpoint_read_urls=adapter.checkpoint_urls(m['checkpoint'])),**eval_config)
        def evaluation_loop():
            while not stopped.is_set():
                try:
                    openings=[]
                    for path in sorted((state/'epochs').glob('*/coordinator/OPENING.json')):
                        if path.is_symlink()or not 0<path.stat().st_size<=32*1024**2:
                            raise ValueError('bounded original evaluation opening')
                        openings.append(json.loads(path.read_bytes()))
                    evaluator.tick(openings)
                except Exception as error:
                    print(json.dumps(dict(component='evaluator',status='retrying',error_type=type(error).__name__)),flush=True)
                stopped.wait(args.interval)
        eval_thread=threading.Thread(target=evaluation_loop,name='mimo-evaluation-scheduler',daemon=True)
    server=CoordinatorServer((server_config['host'],server_config['port']),queue,
                             lambda payload:sign(actor,payload))
    thread=threading.Thread(target=server.serve_forever,name='mimo-role-queue',daemon=True)
    thread.start()
    if audit_thread is not None:audit_thread.start()
    if eval_thread is not None:eval_thread.start()
    try:
        while True:
            try:
                result=lifecycle.tick(now=int(time.time()))
                print(json.dumps(result,sort_keys=True),flush=True)
            except Exception as error:
                # Do not print presigned URLs or credential-bearing adapter errors.
                print(json.dumps(dict(status='retrying',error_type=type(error).__name__)),flush=True)
                if args.once:raise RuntimeError('MiMo lifecycle pass failed: '+type(error).__name__)from None
            if args.once:return
            time.sleep(args.interval)
    finally:
        stopped.set()
        if audit_thread is not None:audit_thread.join(timeout=5)
        if eval_thread is not None:eval_thread.join(timeout=5)
        server.shutdown();server.server_close();thread.join(timeout=5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--allow-public-mimo',action='store_true')
    p.add_argument('--once',action='store_true')
    p.add_argument('--interval',type=int,default=10)
    args=p.parse_args()
    if not 1<=args.interval<=60:p.error('interval must be between 1 and 60 seconds')
    run(args)


if __name__=='__main__':main()
