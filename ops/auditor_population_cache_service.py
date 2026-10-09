"""Pinned CPU-only wrapper for idempotent continuous-audit population admission."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

AUTHORITY='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    from nacl.signing import VerifyKey
    document=json.loads(Path(args.policy).read_bytes())
    if document.get('signer')!=AUTHORITY:raise ValueError('authority')
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
    p=document['payload']
    if p['version']!='idempotent-auditor-population-operator-v1' or p['execute_allowed']is not True:raise ValueError('operator scope')
    if digest(__file__)!=p['runner_sha256']:raise ValueError('operator runner drift')
    for row in (p['original_runner'],p['original_policy'],p['cache_module']):
        if digest(row['path'])!=row['sha256']:raise ValueError('original pin drift')
    original=load('original_completed_math_service',p['original_runner']['path'])
    base_document=json.loads(Path(p['original_policy']['path']).read_bytes());raw=base_document['payload']
    if raw['kind']!='auditor':raise ValueError('auditor only')
    sys.path.insert(0,raw['baseline_runtime'])
    from ops import durable_audit_services as guards
    base=guards.signed(base_document);config=original.validate(base,guards)
    service=guards.prepare_runtime(guards.signed(guards.read(base['baseline_policy']['path'])))
    cache=load('authenticated_population_cache',p['cache_module']['path']);cache.install(service)
    if args.check:
        print(json.dumps(dict(checked=True,kind='auditor',source=base['source_sha256'],jobs_dispatched=False,cache='authenticated-idempotent-v1')));return
    with guards.singleton(base['singleton_lock']):
        guards.no_predecessors(base)
        service.main(['--config',base['config']['path']])

if __name__=='__main__':main()
