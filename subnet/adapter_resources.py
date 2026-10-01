"""Next-version signed execution resources. Not wired into the live adapter.

The coordinator must authenticate the containing epoch before using a contract.
Download authorization is separate from byte identity; private grader archives
must never be sent to a miner. Run provider verification in a fresh worker.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib import util
from pathlib import Path

from .environment_resources import (canonical, digest, _descriptor, safe_relative,
    materialize, verify_dependencies, canonical_task_identity, bind_task_resources)

SCHEMA = 'environment-execution-contract-v2'
SHA = re.compile(r'^[0-9a-f]{64}$')
IMAGE = re.compile(r'^sha256:[0-9a-f]{64}$')


def _sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError('invalid execution resource hash')
    return value


def _reference(ref):
    if set(ref) != {'descriptor_id', 'archive_sha256', 'object_key'}:
        raise ValueError('unexpected execution resource reference fields')
    _sha(ref['descriptor_id']); _sha(ref['archive_sha256'])
    key = safe_relative(ref['object_key'])
    if any(c in key for c in ('?', '#', ':', '@')):
        raise ValueError('resource references are object keys, not credentials or URLs')


def validate_contract(contract):
    if not isinstance(contract, dict): raise ValueError('execution contract must be a mapping')
    body = {k:v for k,v in contract.items() if k != 'id'}
    if set(body) != {'schema', 'environment_version', 'dependencies', 'dependency_scope', 'dependency_closure_reviewed', 'native_platform_descriptor_id', 'task_resources', 'docker_image'}:
        raise ValueError('unexpected execution contract fields')
    if body['schema'] != SCHEMA or digest(canonical(body)) != contract.get('id'):
        raise ValueError('execution contract identity mismatch')
    if not isinstance(body['environment_version'],str) or not body['environment_version']:
        raise ValueError('missing environment version')
    scope=body['dependency_scope']
    if scope=='reviewed-full-closure':
        if body['dependency_closure_reviewed'] is not True:raise ValueError('full dependency closure must be explicitly reviewed')
        _sha(body['native_platform_descriptor_id'])
    elif scope=='provider-namespace-controlled':
        if body['dependency_closure_reviewed'] is not False or body['native_platform_descriptor_id'] is not None:
            raise ValueError('controlled provider scope must disclose incomplete closure')
        if not body['environment_version'].startswith('prime-resource-controlled-'):
            raise ValueError('controlled provider scope needs distinct controlled version')
    else:raise ValueError('unknown dependency verification scope')
    _reference(body['dependencies'])
    if not isinstance(body['task_resources'],list): raise ValueError('resource list required')
    seen=set(); bindings=set()
    for row in body['task_resources']:
        if set(row) != {'audience', 'ref', 'identity_resource'} or row['audience'] not in ('miner-setup','verifier'):
            raise ValueError('invalid task resource role')
        _reference(row['ref']); key=row['ref']['descriptor_id']
        if key in seen: raise ValueError('duplicate task resource')
        seen.add(key)
        if not isinstance(row['identity_resource'],bool): raise ValueError('resource identity flag required')
        if row['identity_resource']: bindings.add(key)
    if len(bindings)>1: raise ValueError('ambiguous canonical task resource')
    image = body['docker_image']
    if image is not None:
        if set(image) != {'image_id','repo_digest'} or not isinstance(image['image_id'],str) or not IMAGE.fullmatch(image['image_id']):
            raise ValueError('Docker image must bind immutable content ID')
        ref=image['repo_digest']
        if ref is not None and (not isinstance(ref,str) or not re.fullmatch(r'[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}',ref)):
            raise ValueError('Docker pull reference must be an immutable repository digest')
    return body


def build_contract(*, environment_version, dependencies, task_resources=(), docker_image=None, dependency_closure_reviewed=False,
                   dependency_scope='reviewed-full-closure',native_platform_descriptor_id=None):
    body={'schema':SCHEMA,'environment_version':environment_version,'dependencies':dependencies,
          'dependency_scope':dependency_scope,'dependency_closure_reviewed':dependency_closure_reviewed,
          'native_platform_descriptor_id':native_platform_descriptor_id,'task_resources':list(task_resources),'docker_image':docker_image}
    result={**body,'id':digest(canonical(body))};validate_contract(result);return result


def descriptor_for(contract, ref, descriptors, kind):
    validate_contract(contract)
    descriptor=descriptors.get(ref['descriptor_id'])
    if descriptor is None or descriptor.get('id')!=ref['descriptor_id']:
        raise ValueError('missing authenticated resource descriptor')
    _descriptor(descriptor,kind)
    return descriptor


def prepare_environment(contract, descriptors, archives, cache, *, audience,native_platform_verifier=None):
    """Materialize role-appropriate resources; caller downloads archives first.

    This never modifies sys.path or imports an environment provider. Spawn a
    fresh worker with dependency_root before site-packages, then invoke
    verify_provider_before_import. Cache destinations are immutable.
    """
    body=validate_contract(contract)
    if body['dependency_scope']=='reviewed-full-closure':
        descriptor=descriptors.get(body['native_platform_descriptor_id'])
        if not descriptor or descriptor.get('schema')!='trusted-native-platform-v1' or digest(canonical({k:v for k,v in descriptor.items() if k!='id'}))!=body['native_platform_descriptor_id']:
            raise ValueError('native platform descriptor missing or unauthenticated')
        if native_platform_verifier is None or native_platform_verifier(descriptor) is not True:
            raise ValueError('approved native platform verifier required for full closure')
    if audience not in ('miner','verifier'): raise ValueError('unknown execution role')
    refs=[(body['dependencies'],'dependencies')]
    for row in body['task_resources']:
        if row['audience']=='miner-setup' or audience=='verifier':
            descriptor=descriptor_for(contract,row['ref'],descriptors,'task-resources')
            if descriptor['audience']!=row['audience']: raise ValueError('signed resource audience mismatch')
            refs.append((row['ref'],'task-resources'))
    roots={}
    for ref,kind in refs:
        descriptor=descriptor_for(contract,ref,descriptors,kind)
        archive=archives.get(ref['descriptor_id'])
        if archive is None: raise ValueError('missing authorized resource archive')
        dest=Path(cache)/ref['descriptor_id']
        if digest(Path(archive).read_bytes()) != ref['archive_sha256']:
            raise ValueError('resource archive hash mismatch')
        # Reuse requires exact byte membership. Extraction mode is immutable,
        # but owner processes can still change it; do not trust file modes.
        if dest.exists():
            verify_materialized(descriptor,dest)
        else:
            materialize(descriptor,archive,dest,archive_sha256=ref['archive_sha256'])
        roots[ref['descriptor_id']]=str(dest.resolve())
    return {'contract_id':contract['id'],'dependency_root':roots[body['dependencies']['descriptor_id']],
            'resource_roots':roots,'audience':audience}


def verify_materialized(descriptor, root):
    from .environment_resources import _package_metadata
    kind=descriptor.get('kind');_descriptor(descriptor,kind)
    if kind=='task-resources': expected=descriptor['files']
    elif kind=='dependencies':
        expected={}
        for row in descriptor['packages']:
            expected.update(row['files']);expected.update({k:digest(v) for k,v in _package_metadata(row).items()})
    else: raise ValueError('unknown resource kind')
    base=Path(root)
    if base.is_symlink(): raise ValueError('resource root symlink')
    actual={}
    for path in base.rglob('*'):
        if path.is_symlink(): raise ValueError('resource bytefile symlink')
        if path.is_file(): actual[path.relative_to(base).as_posix()]=digest(path.read_bytes())
    if actual!=expected: raise ValueError('materialized resource footprint mismatch')
    return True


def verify_provider_before_import(contract, descriptors, dependency_root):
    """Must run in the fresh, isolated provider worker before constructing tasks."""
    body=validate_contract(contract)
    descriptor=descriptor_for(contract,body['dependencies'],descriptors,'dependencies')
    # -B prevents generating caches, but alone does NOT prevent loading existing
    # forged bytecode. Exact extracted membership rejects .pyc/native additions.
    # Reject preloaded providers too: their in-memory code may predate byte pins.
    if not sys.dont_write_bytecode:
        raise ValueError('provider worker must start with Python -B')
    verify_materialized(descriptor,dependency_root)
    base=Path(dependency_root).resolve()
    for row in descriptor['packages']:
        for name,module in row['modules'].items():
            if any(key==name or key.startswith(name+'.') for key in sys.modules):
                raise ValueError('provider already imported before authentication: '+name)
            resolved=util.find_spec(name)
            expected=base/module['path']
            if module['kind']=='module':
                matched=resolved is not None and resolved.origin is not None and Path(resolved.origin).resolve()==expected.resolve()
            elif (expected/'__init__.py').exists():
                matched=resolved is not None and resolved.origin is not None and Path(resolved.origin).resolve()==(expected/'__init__.py').resolve()
            else:
                matched=resolved is not None and resolved.origin is None and {Path(p).resolve() for p in resolved.submodule_search_locations or []}=={expected.resolve()}
            if not matched: raise ValueError('provider is outside authenticated isolated root: '+name)
    return verify_dependencies(descriptor)


def inspect_docker(image_id):
    result=subprocess.run(['docker','image','inspect',image_id],check=True,capture_output=True,text=True,timeout=30)
    rows=json.loads(result.stdout)
    if len(rows)!=1: raise ValueError('ambiguous Docker image')
    return rows[0]


def verify_docker_before_start(contract, *, inspect=inspect_docker):
    """Return immutable image ID for docker run; never start via a mutable tag."""
    image=validate_contract(contract)['docker_image']
    if image is None: return None
    actual=inspect(image['image_id'])
    if actual.get('Id')!=image['image_id']:
        raise ValueError('Docker image content ID mismatch')
    if image['repo_digest'] is not None and image['repo_digest'] not in actual.get('RepoDigests',[]):
        raise ValueError('Docker image repository digest mismatch')
    return image['image_id']


def prepare_task(contract, descriptors, roots, data, task_config, *, audience):
    """Canonical identity precedes path mapping; absent private bytes are allowed.

    The miner must not construct a host-side private grader. It receives the
    signed canonical identity and miner setup resources; only the verifier may
    bind the identity resource if its audience is verifier.
    """
    body=validate_contract(contract)
    if audience not in ('miner','verifier'): raise ValueError('unknown execution role')
    identities=[row for row in body['task_resources'] if row['identity_resource']]
    if not identities:
        identity=digest(canonical({'schema':'portable-task-identity-v1','environment_version':body['environment_version'],
                                   'data':data,'task_config':task_config}))
        return {'task_hash':identity,'runtime_data':dict(data),'private_grader_available':False}
    row=identities[0];descriptor=descriptor_for(contract,row['ref'],descriptors,'task-resources')
    if descriptor['audience']!=row['audience']: raise ValueError('resource audience mismatch')
    identity=canonical_task_identity(data,task_config,descriptor,environment_version=body['environment_version'])
    if audience=='miner' and row['audience']=='verifier':
        return {'task_hash':identity,'runtime_data':None,'private_grader_available':False}
    root=roots.get(descriptor['id'])
    if root is None: raise ValueError('missing task resource root')
    verify_materialized(descriptor,root)
    return {'task_hash':identity,'runtime_data':bind_task_resources(data,descriptor,root),'private_grader_available':row['audience']=='verifier'}
