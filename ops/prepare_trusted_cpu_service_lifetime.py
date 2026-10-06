"""Prepare an unsigned exact-current API/auditor lifetime package. No mutations."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.running_api_validation import authenticate, canonical, digest, file_bytes
from ops.trusted_cpu_service_lifetime import VERSION, STATUS, current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--api-unit', default='affine-independent-source-aware173-queue-api-v1.service')
    parser.add_argument('--auditor-unit', default='affine-continuous-group4-six-source-auditor-v1.service')
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute() or out != out.resolve() or out.exists():
        raise ValueError('fresh absolute review-only directory')
    out.mkdir(mode=0o700)
    package = out/'operator-package/ops'
    package.mkdir(parents=True)
    (package/'__init__.py').write_bytes(b'')
    for name in ['trusted_cpu_service_lifetime.py', 'running_api_validation.py', 'readonly_coordinator_status.py']:
        shutil.copyfile(Path(__file__).parent/name, package/name)
    entrypoint = str(package/'trusted_cpu_service_lifetime.py')
    policy_path = str(out/'scope.TRUSTED-CPU-SERVICE-LIFETIME.ROOT-SIGNED.private.json')
    files = {str(p): hashlib.sha256(file_bytes(p)).hexdigest() for p in package.iterdir()}
    prepared = {}
    services = {}
    history = {}
    queue = None
    seed = None
    for role, name in [('api', args.api_unit), ('auditor', args.auditor_unit)]:
        observed = current(dict(unit=name))
        if not observed['process'] or observed['systemd']['ActiveState'] != 'active':
            raise ValueError('exact existing active services required')
        argv = observed['process']['argv']
        config_path = argv[argv.index('--config')+1]
        scope_path = argv[argv.index('--scope')+1]
        config = json.loads(file_bytes(config_path))
        old = authenticate(json.loads(file_bytes(scope_path)), args.authority)
        if not old['created_at'] <= observed['process']['started_at'] < old['expires_at']:
            raise ValueError('actual original service startup provenance')
        if role == 'api':
            tree, inventory = old['operator_tree'], old['operator_files']
        else:
            transition = old['api173_operational_transition']
            legacy = Path(transition['legacy_launcher_path'])
            if hashlib.sha256(file_bytes(legacy)).hexdigest() != transition['legacy_launcher_sha256']:
                raise ValueError('pinned auditor legacy provenance')
            tree = str(legacy.parent/'auditor-GROUP4-current9f-exact-operator-tree')
            inventory_path = legacy.parent/'auditor-GROUP4-full-operator-file-inventory.private.json'
            if hashlib.sha256(file_bytes(inventory_path)).hexdigest() != old['operator_inventory_sha256']:
                raise ValueError('pinned full2102 auditor inventory')
            inventory = json.loads(file_bytes(inventory_path))
        for relative, pin in inventory.items():
            path = str(Path(tree)/relative)
            if hashlib.sha256(file_bytes(path)).hexdigest() != pin:
                raise ValueError('full original operator tree unchanged')
            files[path] = pin
        for path in [config_path, scope_path, config['bucket']['credentials_file']]:
            files[path] = hashlib.sha256(file_bytes(path)).hexdigest()
        state = Path(config['state'])
        queue_path = str(state/'roles/verifier-queue.sqlite3')
        if queue is not None and queue != queue_path:
            raise ValueError('same actual queue in both services')
        queue = queue_path
        seed = str(state/'authority.seed')
        properties = dict(row.split('=', 1) for row in subprocess.check_output(
            ['systemctl', '--user', 'show', name, '-p', 'FragmentPath', '-p', 'DropInPaths'], text=True).splitlines())
        fragment = properties['FragmentPath']
        if not fragment.startswith('/home/const/.config/systemd/user/'):
            raise ValueError('persistent original unit required')
        dropins = properties['DropInPaths'].split()
        for path in [fragment, *dropins]:
            files[path] = hashlib.sha256(file_bytes(path)).hexdigest()
            saved = out/'original-service-records'/role/Path(path).name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(file_bytes(path))
        expected = [sys.executable, '-I', '-B', entrypoint, 'run-service', '--policy', policy_path,
                    '--authority', args.authority, '--role', role]
        new_dropin = str(Path(fragment).with_name(name+'.d')/'99-trusted-cpu-service-lifetime.conf')
        if Path(new_dropin).exists():
            raise ValueError('fresh lifetime drop-in path')
        text = '[Service]\nType=exec\nExecStart=\nExecStart='+' '.join(expected)+'\nRestart=no\n'
        prepared_path = out/(role+'.lifetime.dropin.conf')
        prepared_path.write_text(text)
        files[new_dropin] = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
        prepared[new_dropin] = str(prepared_path)
        supervisor_unit = 'affine-trusted-'+role+'-lifetime-supervisor-v1.service'
        supervisor_path = str(Path(fragment).with_name(supervisor_unit))
        if Path(supervisor_path).exists():
            raise ValueError('fresh supervisor unit')
        supervision_argv = list(expected)
        supervision_argv[4] = 'supervise'
        supervisor = ('[Unit]\nDescription=Bounded trusted '+role+' CPU lifetime supervisor\n'
                      'StartLimitIntervalSec=300\nStartLimitBurst=3\n[Service]\nType=exec\nExecStart='+
                      ' '.join(supervision_argv)+'\nRestart=on-failure\nRestartSec=30\n')
        supervisor_prepared = out/(role+'.supervisor.service')
        supervisor_prepared.write_text(supervisor)
        files[supervisor_path] = hashlib.sha256(supervisor_prepared.read_bytes()).hexdigest()
        prepared[supervisor_path] = str(supervisor_prepared)
        service = dict(role=role, unit=name, config_path=config_path, config_sha256=digest(config),
                       operator_tree=tree, operator_files=inventory, unit_fragment=fragment,
                       dropins=[*dropins, new_dropin], environment_files=[], initial_instance=observed,
                       expected_argv=expected, supervisor_unit=supervisor_unit, supervisor_unit_path=supervisor_path,
                       max_failures=3, circuit_seconds=1800, backoff_seconds=5, max_backoff_seconds=300,
                       readiness_seconds=120 if role == 'api' else 900)
        if role == 'api':
            registry = authenticate(json.loads(file_bytes(old['registry_path'])), args.authority)
            service.update(registry_path=old['registry_path'], source_trees=old['source_trees'],
                           readiness_url='http://127.0.0.1:'+str(config['remote']['verifier_queue']['port'])+'/request')
            files[old['registry_path']] = hashlib.sha256(file_bytes(old['registry_path'])).hexdigest()
            if registry['version'] != 'source-specific-sampling-api-admission-v1':
                raise ValueError('exact API registry version')
            for source, row in registry['sources'].items():
                pins = row['runtime_files']
                for relative, pin in pins.items():
                    path = str(Path(old['source_trees'][source])/relative)
                    if hashlib.sha256(file_bytes(path)).hexdigest() != pin:
                        raise ValueError('full genuine scientific source maps')
                    files[path] = pin
        else:
            service.update(state_path=str(state/'continuous-audit/audit-state.json'),
                           health_path=str(state/'continuous-audit-health.json'))
        services[role] = service
        history[role] = dict(path=scope_path, sha256=files[scope_path])
    st = Path(queue).stat()
    policy = dict(version=VERSION, execute_allowed=False, authority=args.authority, created_at=time.time(),
                  not_after=None, authorization_status_path=str(out/'authorization-status.ROOT-SIGNED.private.json'),
                  host_uid=os.getuid(), host_machine_id_sha256=hashlib.sha256(file_bytes('/etc/machine-id')).hexdigest(),
                  job_permissions_unchanged=True, scientific_contracts_unchanged=True, GPU_execution_allowed=False,
                  python=sys.executable, entrypoint=entrypoint, policy_path=policy_path,
                  authority_seed_path=seed, database=queue, queue_inode=[st.st_dev, st.st_ino],
                  files=files, services=services, historical_execution_scopes=history,
                  record_directory=str(out/'immutable-instance-events'))
    (out/'scope.TRUSTED-CPU-SERVICE-LIFETIME.UNSIGNED.private.json').write_bytes(canonical(
        dict(payload=policy, signature=None, unsigned=True)))
    (out/'authorization-status.UNSIGNED.private.json').write_bytes(canonical(dict(
        payload=dict(version=STATUS, active_policy_sha256=digest(policy)), signature=None, unsigned=True)))
    (out/'prepared-service-files.private.json').write_bytes(canonical(prepared))
    print(json.dumps(dict(package=str(out), unsigned=True, services_changed=False,
                          full_API173=True, full_auditor2102=True, source_rows=len(registry['sources']))))


if __name__ == '__main__':
    main()
