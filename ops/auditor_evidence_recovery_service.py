"""ROOT-pinned CPU recovery of acknowledged missing audit queue evidence."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(document):
    from nacl.signing import VerifyKey
    if document.get('signer') != AUTHORITY:
        raise ValueError('authority')
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(canonical(document['payload']),
                                             base64.b64decode(document['signature'], validate=True))
    return document['payload']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(document):
    p = signed(document)
    if p['version'] != 'auditor-acknowledged-missing-evidence-CPU-v1' or p['execute_allowed'] is not True:
        raise ValueError('operator scope')
    if digest(__file__) != p['runner_sha256']:
        raise ValueError('operator runner drift')
    names = ('population_policy', 'original_operator', 'registry', 'previous_assessment',
             'deferral_module', 'reviewed_numerical_module')
    for name in names:
        row = p[name]
        if set(row) != {'path', 'sha256'} or Path(row['path']).is_symlink() or digest(row['path']) != row['sha256']:
            raise ValueError('exact original recovery input: ' + name)
    previous = signed(read(p['population_policy']['path']))
    if previous['version'] != 'idempotent-auditor-population-operator-v1' or previous['execute_allowed'] is not True:
        raise ValueError('original idempotent auditor scope')
    if digest(p['original_operator']['path']) != previous['runner_sha256']:
        raise ValueError('original population operator pin')
    for name in ('original_runner', 'original_policy', 'cache_module'):
        row = previous[name]
        if digest(row['path']) != row['sha256']:
            raise ValueError('original population policy pin')
    original = load('original_completed_math_service', previous['original_runner']['path'])
    base_document = read(previous['original_policy']['path'])
    base = signed(base_document)
    if base['kind'] != 'auditor':
        raise ValueError('auditor only')
    sys.path.insert(0, base['baseline_runtime'])
    from ops import durable_audit_services as guards
    original.validate(base, guards)
    service = guards.prepare_runtime(guards.signed(guards.read(base['baseline_policy']['path'])))
    cache = load('authenticated_population_cache', previous['cache_module']['path'])
    cache.install(service)
    helper = load('acknowledged_missing_audit_originals', p['deferral_module']['path'])
    from subnet import numerical_resolution
    reviewed = load('subnet._reviewed_missing_original_numerical_resolution', p['reviewed_numerical_module']['path'])
    config = read(base['config']['path'])
    directory = Path(config['state']) / 'continuous-audit'
    registry = helper.Registry(read(p['registry']['path']), read(p['previous_assessment']['path']),
                               AUTHORITY, service.authenticate, directory)
    receipt = helper.install(service, registry, numerical_resolution, reviewed.apply)
    return service, base, guards, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    service, base, guards, receipt = prepare(read(args.policy))
    if args.check:
        print(json.dumps(dict(checked=True, jobs_dispatched=False, **receipt)))
        return
    with guards.singleton(base['singleton_lock']):
        guards.no_predecessors(base)
        service.main(['--config', base['config']['path']])


if __name__ == '__main__':
    main()
