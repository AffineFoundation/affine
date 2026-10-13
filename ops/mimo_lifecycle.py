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


def run(args):
    config=private_json(args.config)
    if config.get('enabled')is not True:
        raise ValueError('lifecycle configuration is disabled')
    from ops.mimo_training_rehearsal import identity
    actor=identity(config['authority_seed_file'])
    from subnet.mimo_release_policy import ReleasePolicy
    policy=ReleasePolicy(config['release_policy'],actor.id,allow_public=args.allow_public_mimo)
    from subnet.mimo_source_migration import SourceReleaseRegistry
    registry=SourceReleaseRegistry(config['source_release_registry'],actor.id,allow_public=args.allow_public_mimo) if 'source_release_registry' in config else None
    if registry is not None and (registry.active.policy_sha256!=policy.policy_sha256):
        raise ValueError('exact source-only active policy and trainer configuration')
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
    queue=Queue(config['queue_database'],actor.id,config['workers'],allow_public_mimo=args.allow_public_mimo,
        source_registry=registry,worker_execution=config.get('worker_execution'))
    chain=ChainAdapter(state/'registration-observations',**config.get('chain',{}))
    training=TrainerConfig(**config['trainer_config']).validate().document()
    if registry is not None and registry.payload['trainer_config']!=training:raise ValueError('unchanged exact trainer objective configuration')
    adapter=R2Services(bucket,actor,policy,queue,chain,**config['adapters'],trainer_config=training,
        readback_state=state/'model-readbacks')
    def execution(m):
        return registry.execution_for(m) if registry else dict(source_files=adapter.source_files,runtime_versions=adapter.runtime_versions)
    if registry is not None and registry.payload['execution'][policy.scope['immutable_manifest']['source_bundle']['sha256']] != dict(source_files=adapter.source_files,runtime_versions=adapter.runtime_versions):
        raise ValueError('active source execution profile')
    lifecycle=Lifecycle(state,bucket,actor,config['release_policy'],adapter.services(),
        resolve_task=lambda m,e,i:resolve_task(m,e,i,tokenizer),
        grade_batch=lambda b,m,miner:grade_batch(b,m,tokenizer,miner=miner),
        grade_batches=native_pool,
        allow_public=args.allow_public_mimo,**config.get('lifecycle',{}))
    # This runner owns one explicit loopback API for this exact DB/roster. The
    # operator supplies its private authenticated tunnel; no public bind occurs.
    server_config=config['queue_server']
    if (set(server_config)!={'host','port'} or server_config['host']!='127.0.0.1'
            or type(server_config['port'])is not int or not 1024<=server_config['port']<=65535):
        raise ValueError('explicit loopback queue API address required')
    from subnet.mimo_release_policy import sign
    import threading
    stopped=threading.Event();audit_thread=None;eval_thread=None
    audit_config=dict(config.get('audit_scheduler',{}))
    audit_enabled=audit_config.pop('enabled',False)
    if type(audit_enabled)is not bool or(policy.mode=='public'and not audit_enabled):
        raise ValueError('public lifecycle requires its independent audit scheduler')
    if audit_enabled:
        from subnet.mimo_audit_scheduler import AuditScheduler
        scheduler=AuditScheduler(bucket=bucket,queue=queue,identity=actor,
            manifest_guard=registry.resolve if registry else policy.validate_manifest,metadata=lambda m:dict(
                **execution(m),
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
        evaluator=EvaluationScheduler(bucket=bucket,queue=queue,identity=actor,policy=policy,
            metadata=lambda m:dict(**execution(m),
                checkpoint_read_urls=adapter.checkpoint_urls(m['checkpoint'])),source_registry=registry,**eval_config)
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
