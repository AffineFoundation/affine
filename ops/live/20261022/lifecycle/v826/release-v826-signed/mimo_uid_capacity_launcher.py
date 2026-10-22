#!/usr/bin/env python3
"""ROOT-admitted host roster capacity over the existing capture launcher (+ v8.2.4 host overrides)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
VERSION = 'affine-host-uid-capacity-launcher-v1'
SCOPE = 'host-registration-capacity-only'
# Keep the qualified import hygiene in spawned native grading children.
_spec = importlib.util.spec_from_file_location('affine_uid_host_paths', P/'mimo_host_path_launcher.py')
_paths = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _paths
_spec.loader.exec_module(_paths)
if __name__ in ('__main__', '__mp_main__'):
    _paths.install()


def pinned(ref):
    if type(ref) is not dict or set(ref) != {'path', 'sha256'}:
        raise ValueError('exact pinned host reference')
    path = Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('ordinary absolute host file')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('host file digest differs')
    return path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked(config):
    if set(config) != {'version', 'capacity', 'parent_entry', 'parent_operator',
                       'host_files', 'original_methods', 'admission'}:
        raise ValueError('exact host capacity configuration')
    if config['version'] != VERSION or type(config['capacity']) is not int or config['capacity'] != 1000:
        raise ValueError('explicit authorized 1000 UID capacity')
    if set(config['host_files']) != {'mimo_uid_capacity.py', 'mimo_uid_capacity_launcher.py', 'mimo_host_path_launcher.py'}:
        raise ValueError('exact host capacity files')
    for name, checksum in config['host_files'].items():
        pinned(dict(path=str(P/name), sha256=checksum))
    parent_path = pinned(config['parent_entry'])
    operator_path = pinned(config['parent_operator'])
    parent = load(parent_path, 'affine_original_capture_launcher')
    parent.paths.install()
    operator = parent.paths.private_json(operator_path)
    module, Lifecycle = parent.checked(operator)
    from subnet.backend_jobs import signed
    from subnet.storage import canonical, sha
    authority = parent.paths.private_json(operator['base']['lifecycle_config']['path'])['release_policy']['signer']
    proposal = {k: v for k, v in config.items() if k != 'admission'}
    expected = dict(version=VERSION, configuration_sha256=sha(canonical(proposal)),
                    source_unchanged=True, scope=SCOPE, capacity=1000)
    if signed(config['admission'], authority) != expected:
        raise ValueError('exact ROOT host capacity admission')
    from subnet.mimo_coordinator import Coordinator
    helper = load(P/'mimo_uid_capacity.py', 'affine_host_uid_capacity')
    methods = helper.replacements(Lifecycle, Coordinator, config['original_methods'])
    return parent, operator, module, Lifecycle, Coordinator, methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-config', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = _paths.private_json(args.operator_config)
    parent, operator, module, Lifecycle, Coordinator, methods = checked(config)
    if args.check:
        helper = sys.modules['affine_host_uid_capacity']
        from subnet.mimo_training_worker import validate_result
        ok = (helper.method_sha(validate_result) == helper.V824_PINS['mimo_training_worker.validate_result'] and
              helper.method_sha(Lifecycle._training) == helper.V824_PINS['Lifecycle._training'] and
              helper.method_sha(Lifecycle._current) == helper.V824_PINS['Lifecycle._current'])
        helper.v824_validate_result_source(__import__('inspect').getsource(validate_result))
        helper.v824_training_source(__import__('inspect').getsource(Lifecycle._training))
        print(json.dumps(dict(version=VERSION, capacity=1000, original_source_guard_passed=True,
                              operator_admission_passed=True, v824_override_pins_passed=ok,
                              production_services_started=False)))
        return
    parent.install_capture(Lifecycle, operator['workers'])
    Lifecycle._plan, Coordinator.open = methods
    helper = sys.modules['affine_host_uid_capacity']
    print(json.dumps(dict(component='host-v824', **helper.v824_install(Lifecycle))), flush=True)
    sys.argv = [operator['base']['entry']['path']] + operator['base']['argv']
    module.main()


if __name__ == '__main__':
    main()
