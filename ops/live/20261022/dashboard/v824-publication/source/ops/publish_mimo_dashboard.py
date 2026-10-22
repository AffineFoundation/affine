#!/usr/bin/env python3
"""Run the existing dashboard HTTP/export service with a MiMo-only data view.

No HTML/CSS changes; old database rows and signed history remain intact.
The standard server keeps its routes and canonical guide export; closed legacy
records are read from the existing database without rescanning every artifact.
"""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True


def database_view(original,projection,history_path=None,authority=None):
    class MiMoDatabase(original):
        def refresh(self):
            # The existing server calls this at startup and in its poll thread.
            # Do not rescan legacy artifacts or verify audit reports in HTTP handlers.
            prior=super().snapshot(current_only=False)
            now=time.time();result=projection.project(prior,now)
            if history_path is not None:
                from dashboard.mimo_live_projection import read,evidence_availability
                try:history=read(history_path)
                except (OSError,ValueError):history=None
                result=evidence_availability(result,history,authority,now)
            self._mimo_snapshot=result  # atomically replace only a complete projection
        def snapshot(self,current_only=False):
            if not current_only:return super().snapshot(current_only=False)
            result=getattr(self,'_mimo_snapshot',None)
            if result is None:raise RuntimeError('MiMo dashboard refresh has not completed')
            return copy.deepcopy(result)
    return MiMoDatabase


def install_host_validator(ref,authority,pinned):
    """Current release validator: the ROOT-admitted v8.2.4 host override that the live
    lifecycle runs (signed uid-operator admission -> pinned mimo_uid_capacity.py).
    Installs ONLY the v1+v2 model-publication receipt check; no lifecycle/trainer methods."""
    import importlib.util,inspect
    from subnet.backend_jobs import signed
    from subnet.storage import canonical,sha
    operator_path=pinned(ref['operator_config']);operator=json.loads(operator_path.read_bytes())
    proposal={k:v for k,v in operator.items() if k!='admission'}
    if signed(operator['admission'],authority)!=dict(version=operator['version'],configuration_sha256=sha(canonical(proposal)),
            source_unchanged=True,scope='host-registration-capacity-only',capacity=operator['capacity']):
        raise ValueError('dashboard exact ROOT host override admission')
    helper_path=pinned(ref['helper'])
    if helper_path.parent!=operator_path.parent or operator['host_files'].get(helper_path.name)!=ref['helper']['sha256']:
        raise ValueError('dashboard host override file not ROOT-admitted')
    spec=importlib.util.spec_from_file_location('affine_dashboard_host_validator',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import subnet.mimo_training_worker as worker
    import subnet.mimo_grpo.publication as publication
    if helper.method_sha(worker.validate_result)!=helper.V824_PINS['mimo_training_worker.validate_result']:
        raise ValueError('dashboard exact original validate_result required')
    source=helper.v824_validate_result_source(inspect.getsource(worker.validate_result))
    exec(compile(helper.V824_PUBLICATION_VERIFIED,inspect.getsourcefile(publication),'exec'),publication.__dict__)
    exec(compile(source,inspect.getsourcefile(worker),'exec'),worker.__dict__)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--projection-config',required=True)
    parser.add_argument('--preview',action='store_true')
    args,remaining=parser.parse_known_args(argv)
    config_path=Path(args.projection_config);cfg=json.loads(config_path.read_bytes())
    def pinned(ref):
        p=Path(ref['path'])
        if p.resolve()!=p or p.is_symlink()or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('dashboard exact source/config pin')
        return p
    legacy=pinned(cfg['legacy_server']);runtime=Path(cfg['runtime_root'])
    for ref in cfg['runtime_files']:pinned(ref)
    for ref in cfg['projection_files']:pinned(ref)
    lifecycle=json.loads(pinned(cfg['lifecycle_config']).read_bytes())
    sys.path.insert(0,str(runtime));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from subnet.mimo_release_policy import ReleasePolicy
    from subnet.mimo_worker import validate_report
    if 'host_validator' in cfg:install_host_validator(cfg['host_validator'],cfg['authority'],pinned)
    from dashboard.mimo_live_projection import Projection
    from dashboard.mimo_audit_projection import AuditProjection
    from dashboard import server
    if Path(server.__file__).resolve()!=legacy:
        raise ValueError('dashboard actual legacy module source differs')
    server.PUBLIC=Path(cfg['public_root'])
    original_export=server.export_snapshot
    server.export_snapshot=lambda snapshot,destination:original_export(snapshot,destination,public=server.PUBLIC)
    active=ReleasePolicy(lifecycle['release_policy'],cfg['authority'],allow_public=True)
    from subnet.mimo_release_succession import ReleaseChain
    # Every epoch/finalized result/eval receipt/audit is validated against the signed
    # chain member governing its own round (v7 r189 -> v8.2 r228 -> v8.2.1 r229 -> v8.2.2/3/4 r231).
    policy=ReleaseChain(lifecycle['release_succession'],cfg['authority'],allow_public=True)
    if policy.active.policy_sha256!=active.policy_sha256:
        raise ValueError('configured release policy must be the active succession member')
    if policy.chain_sha256!=cfg['release_chain_sha256']:raise ValueError('dashboard exact release chain pin')
    if policy.mode!='public':raise ValueError('public dashboard requires public release')
    # Per-archive audit execution exactly as the lifecycle derives it: ROOT-approved
    # inventory per chain archive, with the active archive bound to the live closure.
    archives={p.policy_sha256:p.scope['immutable_manifest']['source_bundle']['sha256'] for _,p in policy.members}
    live={k:lifecycle['adapters'][k] for k in ('source_files','runtime_versions')}
    execution=copy.deepcopy(lifecycle['archive_execution'])
    if set(execution)!=set(archives.values()):raise ValueError('every signed release chain archive needs execution metadata')
    if execution[archives[policy.active.policy_sha256]]!=live:raise ValueError('active archive execution must equal the live closure')
    projection=Projection(lifecycle['state_root'],policy,cfg['authority'],lifecycle['workers'],validate_report,
        first_round=lifecycle['lifecycle']['first_round'],started_at=cfg['started_at'],model=cfg['model'],
        audit_projection=AuditProjection(lifecycle['state_root'],lifecycle['queue_database'],cfg['authority'],
            lifecycle['workers'],policy.validate_manifest,execution))
    # Explicit path arguments preserve original database/source/static assets when
    # operator stages these modules in a separate immutable source closure.
    server.Database=database_view(server.Database,projection,
        cfg.get('history_export_path'),cfg['authority'])
    if args.preview:
        prior=json.loads(Path(cfg['prior_snapshot']).read_bytes())
        print(json.dumps(projection.project(prior,time.time()),allow_nan=False));return
    sys.argv=[str(legacy)]+remaining
    server.main()


if __name__=='__main__':main()
