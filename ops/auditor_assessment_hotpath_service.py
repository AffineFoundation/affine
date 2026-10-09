"""ROOT-pinned CPU acceleration of the existing authenticated auditor."""
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
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(
        canonical(document['payload']), base64.b64decode(document['signature'], validate=True))
    return document['payload']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install(service, original_policy, optimized_policy, reviewed_numerical):
    """Keep all existing admission/deferral guards and scope memo to one snapshot."""
    original = service.ContinuousAuditor.hourly_snapshot
    if getattr(original, '_authenticated_hotpath_v1', False):
        raise ValueError('assessment hotpath already installed')
    if not getattr(original, '_acknowledged_missing_originals_v1', False):
        raise ValueError('original missing-evidence guard required')
    original_policy.valid_digest = optimized_policy.valid_digest

    def hourly_snapshot(self, *args, **kwargs):
        with reviewed_numerical.authenticated_reference_cache():
            return original(self, *args, **kwargs)

    hourly_snapshot._authenticated_hotpath_v1 = True
    hourly_snapshot._acknowledged_missing_originals_v1 = True
    service.ContinuousAuditor.hourly_snapshot = hourly_snapshot


def prepare(document):
    p = signed(document)
    if p['version'] != 'authenticated-auditor-assessment-hotpath-CPU-v1' or p['execute_allowed'] is not True:
        raise ValueError('operator scope')
    if digest(__file__) != p['runner_sha256']:
        raise ValueError('operator runner drift')
    for name in ('original_runner', 'original_recovery_policy', 'optimized_policy', 'optimized_numerical'):
        row = p[name]
        if set(row) != {'path', 'sha256'} or Path(row['path']).is_symlink() or digest(row['path']) != row['sha256']:
            raise ValueError('exact original optimization input: ' + name)
    previous = read(p['original_recovery_policy']['path'])
    base_policy = signed(previous)
    if (base_policy['runner_sha256'] != p['original_runner']['sha256']
            or base_policy['reviewed_numerical_module'] != p['optimized_numerical']):
        raise ValueError('exact reviewed numerical implementation binding')
    original = load('original_guarded_auditor_recovery', p['original_runner']['path'])
    service, base, guards, receipt = original.prepare(previous)
    from subnet import continuous_audit_policy
    reviewed = sys.modules['subnet._reviewed_missing_original_numerical_resolution']
    if digest(reviewed.__file__) != p['optimized_numerical']['sha256']:
        raise ValueError('actual reviewed numerical origin')
    optimized = load('subnet._assessment_optimized_continuous_policy', p['optimized_policy']['path'])
    install(service, continuous_audit_policy, optimized, reviewed)
    receipt = dict(receipt, authenticated_reference_cache='one-hourly-snapshot',
                   digest_language='exact-lowercase-64-hex', scoring_changes=False)
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
