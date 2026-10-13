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
    from dashboard.mimo_live_projection import Projection
    from dashboard.mimo_audit_projection import AuditProjection
    from dashboard import server
    if Path(server.__file__).resolve()!=legacy:
        raise ValueError('dashboard actual legacy module source differs')
    server.PUBLIC=Path(cfg['public_root'])
    original_export=server.export_snapshot
    server.export_snapshot=lambda snapshot,destination:original_export(snapshot,destination,public=server.PUBLIC)
    policy=ReleasePolicy(lifecycle['release_policy'],cfg['authority'],allow_public=True)
    if policy.mode!='public':raise ValueError('public dashboard requires public release')
    from subnet.mimo_source_migration import SourceReleaseRegistry
    registry=SourceReleaseRegistry(lifecycle['source_release_registry'],cfg['authority'],allow_public=True) if 'source_release_registry' in lifecycle else None
    projection=Projection(lifecycle['state_root'],policy,cfg['authority'],lifecycle['workers'],validate_report,
        first_round=lifecycle['lifecycle']['first_round'],started_at=cfg['started_at'],model=cfg['model'],source_registry=registry,
        audit_projection=AuditProjection(lifecycle['state_root'],lifecycle['queue_database'],cfg['authority'],
            lifecycle['workers'],registry.resolve if registry else policy.validate_manifest,
            registry.payload['execution'] if registry else {policy.scope['immutable_manifest']['source_bundle']['sha256']:
                {k:lifecycle['adapters'][k] for k in ('source_files','runtime_versions')}}))
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
