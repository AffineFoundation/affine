"""Exact CPU import-root repair; scientific grading bytes stay pinned."""
from pathlib import Path

VERSION = 'native-grader-exact-CPU-root-rebinding-v1'
ALLOWED = {'subnet/backend_jobs.py', 'subnet/remote_backend.py',
           'subnet/persistent_training_controller.py', 'subnet/training_startup_recovery.py'}
ADDED = {'tests/test_training_cache_ack_recovery.py'}


def validate_declaration(repair, old, new, cpu_root, cpu_files):
    fields = {'version', 'previous_authorization', 'previous_execution_root', 'previous_files', 'changed_files'}
    if set(repair) != fields or repair['version'] != VERSION:
        raise ValueError('explicit native CPU-root repair schema')
    if old.get('execution_root') != repair['previous_execution_root'] or new.get('execution_root') != cpu_root:
        raise ValueError('native actual approved CPU import root')
    projected = dict(new); projected['execution_root'] = old['execution_root']
    if projected != old:
        raise ValueError('native grading authority changes only import root')
    before = repair['previous_files']
    if set(cpu_files) != set(before) | ADDED or set(before) & ADDED:
        raise ValueError('same complete CPU deployment closure')
    changes = {name: value for name, value in cpu_files.items() if before.get(name) != value}
    if set(changes) != ALLOWED | ADDED or repair['changed_files'] != changes:
        raise ValueError('only exact previously reviewed orchestration recovery bytes differ')
    if any(cpu_files.get(name) != expected for name, expected in new['source_files'].items()
           if name in ('subnet/native_math_prompt.py', 'subnet/math_completion.py',
                       'subnet/native_math_grader.py', 'subnet/environments.py',
                       'subnet/protocol.py', 'subnet/trajectory_identity.py')):
        raise ValueError('unchanged native scientific dependencies')


def validate(p, guards, inventory):
    repair = p['native_execution_rebinding']
    row = p['native_training_eligibility']['authorization']
    if guards.file_hash(row['path']) != row['file_sha256']:
        raise ValueError('exact native authorization file')
    old = guards.signed(repair['previous_authorization'])
    new = guards.signed(guards.read(row['path']))
    validate_declaration(repair, old, new, p['cpu_overlay']['root'], p['cpu_overlay']['files'])
    if inventory(repair['previous_execution_root']) != repair['previous_files']:
        raise ValueError('complete immutable prior CPU deployment')
    return new


def assert_loader(native_prompt, authorization):
    actual = Path(native_prompt.__file__).resolve()
    expected = (Path(authorization['execution_root']) / 'subnet/native_math_prompt.py').resolve()
    if actual != expected:
        raise ValueError('native filter approved source loader preflight')
