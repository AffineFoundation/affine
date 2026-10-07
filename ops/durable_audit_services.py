"""Durable ROOT-pinned CPU API/auditor startup; never renew job capabilities."""
import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from contextlib import contextmanager

VERSION = 'durable-pinned-audit-service-v1'
AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def read(path):
    return json.loads(Path(path).read_bytes())

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def private_file(path):
    p = Path(path)
    if not p.is_absolute():
        raise ValueError('absolute operator file required')
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1 or s.st_mode & 0o002:
        raise ValueError('owned regular single-link operator file required')
    if any(parent.is_symlink() for parent in (p, *p.parents)):
        raise ValueError('symlink operator path')
    return p

def file_hash(path):
    return hashlib.sha256(private_file(path).read_bytes()).hexdigest()

def signed(document, authority=AUTHORITY):
    from nacl.signing import VerifyKey
    if set(document) != {'payload', 'signature', 'signer'} or document['signer'] != authority:
        raise ValueError('ROOT execution policy authority')
    signature = base64.b64decode(document['signature'], validate=True)
    if len(signature) != 64:
        raise ValueError('BASE64 ROOT signature required')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']), signature)
    return document['payload']

def pinned_files(root, files):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir() or not files:
        raise ValueError('pinned immutable runtime root')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_hash(root / relative) != expected:
            raise ValueError('pinned runtime file drift')

def queue_schema(connection):
    return connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()

def validate_queue(row):
    path = private_file(row['path'])
    s = path.stat()
    if [s.st_dev, s.st_ino] != row['inode']:
        raise ValueError('original queue inode changed')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as db:
        if db.execute('PRAGMA journal_mode').fetchone()[0].lower() != 'wal' or db.execute('PRAGMA synchronous').fetchone()[0] != 2:
            raise ValueError('queue must preserve WAL/FULL')
        if digest(queue_schema(db)) != row['schema_sha256']:
            raise ValueError('queue schema changed')

def validate_additions(old, new, source):
    if len(old) != 8 or set(new) != set(old) | {source} or any(new[k] != v for k, v in old.items()):
        raise ValueError('eight historical source rows must remain exact')

def verify_document(row, authority):
    if file_hash(row['path']) != row['file_sha256']:
        raise ValueError('signed document bytes changed')
    payload = signed(read(row['path']), authority)
    if digest(payload) != row['payload_sha256']:
        raise ValueError('signed document payload changed')
    return payload

def validate_policy(document, authority=AUTHORITY):
    p = signed(document, authority)
    fields = {'version','kind','execute_allowed','authority','identity','config','queue','singleton_lock',
              'excluded_units','runner_file_sha256','operator','admission','historical_admission','source','source_trees','authority_seed','API_dependency'}
    if set(p) != fields or p['version'] != VERSION or p['kind'] not in ('API','auditor') or p['execute_allowed'] is not True or p['authority'] != authority:
        raise ValueError('exact durable execution policy')
    # A stable policy has no expiration/PID/boot bindings. Jobs and capability
    # validators retain their original signed deadlines and source contracts.
    if file_hash(Path(__file__).resolve()) != p['runner_file_sha256']:
        raise ValueError('durable runner drift')
    if not isinstance(p['excluded_units'],list) or not p['excluded_units'] or any(not isinstance(v,str) or not v.endswith('.service') for v in p['excluded_units']):
        raise ValueError('explicit predecessor services required')
    identity = p['identity']
    if set(identity) != {'uid','machine_id_sha256'} or identity['uid'] != os.getuid() or hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest() != identity['machine_id_sha256']:
        raise ValueError('approved local physical identity')
    if set(p['config']) != {'path','file_sha256'} or file_hash(p['config']['path']) != p['config']['file_sha256']:
        raise ValueError('pinned config drift')
    validate_queue(p['queue'])
    op = p['operator']
    if set(op) != {'root','files','retry_helper','overlay'}:
        raise ValueError('exact CPU operator inventory')
    pinned_files(op['root'], op['files'])
    old = verify_document(p['historical_admission'], authority)
    new = verify_document(p['admission'], authority)
    cfg = read(p['config']['path'])
    if p['kind'] == 'API':
        validate_additions(old['sources'], new['sources'], p['source'])
        if set(p['source_trees']) != set(new['sources']):
            raise ValueError('exact nine source roots')
        for source, row in new['sources'].items():
            root = p['source_trees'][source]
            pinned_files(root, row['runtime_files'])
            actual = {str(f.relative_to(root)) for f in (Path(root)/'subnet').glob('*.py')}
            if actual != set(row['runtime_files']):
                raise ValueError('complete admitted runtime inventory')
        if p['API_dependency'] is not None:
            raise ValueError('API must not depend on itself')
        if op['retry_helper'] is not None or op['overlay'] is not None:
            raise ValueError('API runtime remains unchanged')
    else:
        dependency=p['API_dependency']
        if set(dependency) != {'unit','policy_path','policy_file_sha256'} or not dependency['unit'].endswith('.service') or file_hash(dependency['policy_path']) != dependency['policy_file_sha256']:
            raise ValueError('exact durable API dependency')
        api=signed(read(dependency['policy_path']),authority)
        if api['kind']!='API' or api['version']!=VERSION or api['queue']!=p['queue'] or api['source']!=p['source'] or api['authority']!=authority or api['execute_allowed'] is not True:
            raise ValueError('same admitted queue/API authority')
        for key in ('approved_sources','job_metadata'):
            validate_additions(old[key], new[key], p['source'])
        before = old['execution_evidence_policy']; after = new['execution_evidence_policy']
        if before['effective_cutoff'] != after['effective_cutoff'] or any(after['sources'].get(k) != v for k,v in before['sources'].items()):
            raise ValueError('historical report interpretation changed')
        if cfg['continuous_audit_service']['source_admission'] != read(p['admission']['path']):
            raise ValueError('auditor must use exact signed admission')
        helper = op['retry_helper']
        if set(helper) != {'path','file_sha256'} or file_hash(helper['path']) != helper['file_sha256']:
            raise ValueError('reviewed retry helper drift')
        overlay = op['overlay']
        if set(overlay) != {'root','files'}:
            raise ValueError('exact frozen auditor overlay')
        pinned_files(overlay['root'], overlay['files'])
        if p['source_trees']:
            raise ValueError('auditor does not execute scientific source trees')
    seed = p['authority_seed']
    if set(seed) != {'path','file_sha256'} or file_hash(seed['path']) != seed['file_sha256']:
        raise ValueError('authority identity changed')
    path = private_file(seed['path'])
    if path.stat().st_mode & 0o077:
        raise ValueError('private authority seed permissions')
    from nacl.signing import SigningKey
    if SigningKey(bytes.fromhex(path.read_text().strip())).verify_key.encode().hex() != authority:
        raise ValueError('authority seed mismatch')
    if Path(cfg['state']).resolve() / 'authority.seed' != path.resolve():
        raise ValueError('original config authority path')
    return p

@contextmanager
def singleton(path):
    p = Path(path)
    if not p.is_absolute() or p.parent.resolve() != p.parent or p.parent.stat().st_uid != os.getuid():
        raise ValueError('owned singleton directory')
    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1 or s.st_mode & 0o077:
            raise ValueError('private singleton lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (p.stat().st_dev,p.stat().st_ino) != (s.st_dev,s.st_ino):
            raise ValueError('singleton path replaced')
        yield
    finally:
        os.close(fd)

def no_predecessors(p):
    for name in p['excluded_units']:
        values = dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',name,'-p','MainPID','-p','ActiveState'],text=True).splitlines())
        if int(values['MainPID']) != 0 or values['ActiveState'] in ('active','activating'):
            raise ValueError('competing CPU service owner')

def live_api_dependency(p):
    row=p['API_dependency']
    if row is None:return
    values=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',row['unit'],'-p','MainPID','-p','ActiveState'],text=True).splitlines())
    pid=int(values['MainPID'])
    if pid<=0 or values['ActiveState']!='active':raise ValueError('durable API dependency not active')
    argv=[v.decode() for v in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if v]
    if not any(argv[i:i+2]==['-m','ops.durable_audit_services'] for i in range(len(argv)-1)):raise ValueError('actual durable API entrypoint required')
    if argv.count('--policy')!=1 or argv[argv.index('--policy')+1]!=row['policy_path'] or '--check' in argv:raise ValueError('actual durable API policy owner')

def load_module(path):
    spec = importlib.util.spec_from_file_location('_durable_reviewed_auditor_retry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_optional_numerical_overlay(op):
    # The frozen legacy retry finder owns exactly its original three modules.
    # A separately ROOT-pinned optional fourth CPU module is loaded explicitly;
    # no legacy finder or scientific package is broadened or edited.
    relative = 'subnet/numerical_resolution.py'
    if relative not in op['overlay']['files']:
        return False
    path = Path(op['overlay']['root']) / relative
    if file_hash(path) != op['overlay']['files'][relative]:
        raise ValueError('exact optional numerical overlay pin')
    name = 'subnet.numerical_resolution'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return True


def prepare_runtime(p):
    # A fresh CPU import graph is required; no scientific loader/source aliases.
    for name in list(sys.modules):
        if name == 'subnet' or name.startswith('subnet.'):
            del sys.modules[name]
    op = p['operator']; sys.path.insert(0,op['root'])
    if p['kind'] == 'API':
        from subnet import coordinator_api_service as service, source_sampling_admission as gate
        admission = gate.SamplingAdmission(read(p['admission']['path']), p['authority'], p['source_trees'])
        service.Coordinator = gate.guarded_coordinator(service.Coordinator, admission)
    else:
        retry = load_module(op['retry_helper']['path'])
        sys.meta_path.insert(0,retry.ScopedAuditorFinder(op['overlay']['root'],op['overlay']['files']))
        from subnet import distributed_roles
        retry.install_status_retry(distributed_roles)
        retry.install_transaction_retry(distributed_roles,p['queue']['inode'])
        numerical_loaded = load_optional_numerical_overlay(op)
        if read(p['config']['path'])['continuous_audit_service'].get('numerical_resolution') is not None and not numerical_loaded:
            raise ValueError('numerical configuration requires its exact pinned CPU module')
        from subnet import continuous_audit_service as service
        service.admitted_service_config(read(p['config']['path'])['continuous_audit_service'],p['authority'])
    if Path(service.__file__).resolve().is_relative_to(Path(op['root']).resolve()) is False and p['kind']=='API':
        raise ValueError('unexpected API import origin')
    expected=Path(op['root'])/'subnet/coordinator_api_service.py' if p['kind']=='API' else Path(op['overlay']['root'])/'subnet/continuous_audit_service.py'
    if Path(service.__file__).resolve()!=expected.resolve():raise ValueError('unexpected pinned service origin')
    if p['kind']=='API':
        if Path(gate.__file__).resolve()!=Path(op['root'])/'subnet/source_sampling_admission.py':raise ValueError('unexpected admission module origin')
    else:
        if Path(distributed_roles.__file__).resolve()!=Path(op['root'])/'subnet/distributed_roles.py':raise ValueError('unexpected coordinator origin')
        for name in ('subnet.continuous_audit_policy','subnet.audit_queue_snapshot','subnet.numerical_resolution'):
            module=sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve()!=Path(op['overlay']['root'])/(name.replace('.','/')+'.py'):raise ValueError('unexpected frozen overlay origin')
    return service

def main():
    if sys.flags.optimize:
        raise ValueError('frozen scientific guards require unoptimized Python')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',required=True);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    p=validate_policy(read(args.policy))
    with singleton(p['singleton_lock']):
        no_predecessors(p)
        live_api_dependency(p)
        service=prepare_runtime(p)
        validate_queue(p['queue'])
        if args.check:
            print(json.dumps({'checked':True,'kind':p['kind'],'sources':9,'queue_preserved':True}));return
        if p['kind']=='API':
            service.main(['--config',p['config']['path'],'--authority-seed',p['authority_seed']['path'],'--expected-authority',p['authority']])
        else:service.main(['--config',p['config']['path']])

if __name__=='__main__':main()
