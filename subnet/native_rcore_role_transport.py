"""Prospective operator-only common role transport; no private fields in job DTO."""
import hashlib,json,os,tempfile
from contextlib import contextmanager
from pathlib import Path
from .native_rcore_boundary import canonical
from .native_rcore_common import admit_descriptor,BINDINGS
ROLE_AUDIENCE={'mine':'miner','verify':'verifier','train':'trainer','evaluate':'evaluator'}
PINNED=('subnet/environments.py','subnet/native_rcore_common.py','subnet/native_rcore_boundary.py','subnet/native_rcore_role_transport.py')

def admit_role_binding(envelope,authority,spec,descriptor,now=None):
    from .backend_jobs import validate
    job,manifest=validate(envelope,authority,now=now)
    if job['role']not in ROLE_AUDIENCE or descriptor.get('audience')!=ROLE_AUDIENCE[job['role']]:raise ValueError('signed common role/local audience mismatch')
    if authority!=spec.config['terminal_public_binding']['authority']:raise ValueError('signed job/terminal authority mismatch')
    if any(name in job for name in ('env','environment_overrides','terminal_bindings','terminal_binding_path','resource_descriptor')):raise ValueError('uploaded job cannot choose private transport paths/environment')
    definitions=manifest.get('environments')
    if definitions is None:definitions=[dict(env_id=manifest['environment']['id'],spec=manifest['environment'],indices=manifest['indices'])]
    matched=[row for row in definitions if row.get('env_id')==spec.id]
    if len(matched)!=1:raise ValueError('exact signed environment registry identity')
    definition=matched[0]
    if not isinstance(definition.get('indices'),list)or len(definition['indices'])!=len(set(definition['indices']))or any(type(i)is not int or not 0<=i<spec.num_samples for i in definition['indices']):raise ValueError('signed mining sample geometry')
    if canonical(definition['spec'])!=canonical(spec.to_dict()):raise ValueError('signed common environment identity')
    heldout=manifest.get('heldout_indices',{}).get(spec.id)
    if not isinstance(heldout,list) or not heldout or len(heldout)!=len(set(heldout)) or any(type(i)is not int or not 0<=i<spec.num_samples for i in heldout) or set(heldout)&set(definition['indices']):raise ValueError('signed disjoint heldout registry required')
    if job['role']=='evaluate':
        suites=[row for row in job['heldout'] if row['env_id']==spec.id]
        if not suites or any(not set(row['indices'])<=set(heldout) for row in suites):raise ValueError('signed evaluation heldout exclusion')
    resource=job.get('terminal_resource_binding')
    expected=dict(revision='rcore-terminal-job-binding-v1',environment_sha256=hashlib.sha256(canonical(spec.to_dict())).hexdigest(),profile_sha256=spec.config['terminal_public_binding']['profile_sha256'],guard_sha256=spec.config['terminal_public_binding']['guard_sha256'],role_local_descriptor_sha256=hashlib.sha256(canonical(descriptor)).hexdigest())
    if resource!=expected:raise ValueError('signed role-local terminal identity')
    root=Path(__file__).resolve().parents[1]
    for name in PINNED:
        if job['source_files'].get(name)!=hashlib.sha256((root/name).read_bytes()).hexdigest():raise ValueError('signed common transport source pin')
    admit_descriptor(spec,descriptor)
    return job,manifest

@contextmanager
def role_bindings(envelope,authority,spec,descriptor,now=None):
    admit_role_binding(envelope,authority,spec,descriptor,now)
    old=os.environ.get(BINDINGS)
    with tempfile.TemporaryDirectory(prefix='affine-rcore-role-')as folder:
        path=Path(folder)/'operator-binding.json';path.write_bytes(canonical(descriptor));path.chmod(0o600);os.environ[BINDINGS]=str(path)
        try:yield
        finally:
            if old is None:os.environ.pop(BINDINGS,None)
            else:os.environ[BINDINGS]=old

def execute_role(envelope,authority,spec,descriptor,workspace,**kwargs):
    """Later qualified GPU role entrypoint; CPU tests never invoke model execution."""
    from .backend_jobs import execute
    with role_bindings(envelope,authority,spec,descriptor):return execute(envelope,authority,workspace,**kwargs)
