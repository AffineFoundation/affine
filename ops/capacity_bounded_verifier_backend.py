"""Pinned operator bootstrap; only tightens authenticated transport byte limits.

The original source, main(), full digest checks, model and sampler run unchanged.
"""
import base64,hashlib,importlib.util,json,subprocess,sys
from types import SimpleNamespace
from pathlib import Path


def verify_envelope_limit(job,authority):
    """Byte allowance only; source, capacity and proof checks remain mandatory."""
    import base64
    from nacl.signing import VerifyKey
    envelope=job['manifest']
    if envelope.get('signer')!=authority:raise ValueError('original manifest authority')
    raw=json.dumps(envelope['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    VerifyKey(bytes.fromhex(authority)).verify(raw,base64.b64decode(envelope['signature'],validate=True))
    manifest=envelope['payload'];cap=manifest.get('training_task_capacity')
    if (job.get('role')=='verify' and type(cap)is dict and set(cap)=={'version','max_tasks'}
            and cap['version']=='signed-training-task-capacity-v1' and type(cap['max_tasks'])is int and cap['max_tasks']==512
            and type(manifest.get('max_batches'))is int and manifest['max_batches']==9
            and type(manifest.get('K'))is int and manifest['K']==4 and type(manifest.get('L'))is int and manifest['L']==4
            and manifest.get('training_policy')=='bf16-cpu-fp32-master-task-normalized-persistent-v4'
            and manifest.get('training_input_policy')=='committed-unaudited-training-v1'
            and ('samples_per_batch'not in manifest or type(manifest['samples_per_batch'])is int and manifest['samples_per_batch']==8)
            and 'training_startup_recovery'not in manifest):return 32_000_000
    return 4_000_000

def bind_transport(backend,job,authority,workspace,policy,admission):
    _,cp,sizes,_,_=admission.budget(job,policy,authority)
    manifest=job['manifest']['payload'];limits=admission.input_limits(job,manifest)
    approved={}
    root=Path(workspace).absolute()
    for name,size in sizes.items():approved[(cp['read_urls'][name],cp['files'][name],str(root/'checkpoints'/cp['id']/name))]=size
    for index,(obj,size)in enumerate(zip(job['submissions'],limits,strict=True)):
        approved[(obj['url'],obj['sha256'],str(root/'jobs'/job['job_id']/('submission-'+str(index)+'.zip')))]=size
    original=backend.get_object
    def bounded(url,expected,destination,limit,**kwargs):
        key=(url,expected,str(Path(destination).absolute()))
        # An approved URL appearing under another path/digest cannot fall back
        # to the larger scientific transport cap.
        if key not in approved and any(url==binding[0]for binding in approved):raise ValueError('exact approved capacity transport mapping')
        if key in approved:limit=min(limit,approved[key])
        return original(url,expected,destination,limit,**kwargs)
    backend.get_object=bounded
    return original

_TRANSPORT_ADMISSION_CODE=r"""
import importlib.util,json,sys
from pathlib import Path
source=Path(sys.argv[1]);helper=Path(sys.argv[2]);sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('ops.verifier_capacity_admission',helper)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
v=json.load(sys.stdin)
if v.get('validate_runtime',False):module.validate_runtime_inventory(v['job'],v['policy'],v['authority'],source)
result=module.budget(v['job'],v['policy'],v['authority']);limits=module.input_limits(v['job'],v['job']['manifest']['payload'])
if any(n=='torch'or n.startswith('torch.')for n in sys.modules):raise ValueError('capacity validation must remain CPU metadata only')
print(json.dumps({'sizes':result[2],'limits':limits},separators=(',',':'),allow_nan=False))
"""

def isolated_transport_admission(job,policy,authority,source,helper,*,validate_runtime=False):
    raw=json.dumps(dict(job=job,policy=policy,authority=authority,validate_runtime=validate_runtime),separators=(',',':'),allow_nan=False).encode()
    if len(raw)>verify_envelope_limit(job,authority):raise ValueError('bounded signed capacity input')
    result=subprocess.run([sys.executable,'-I','-B','-c',_TRANSPORT_ADMISSION_CODE,str(source),str(helper)],input=raw,capture_output=True,timeout=30,check=False,cwd=source)
    if result.returncode or len(result.stdout)>1_000_000:raise ValueError('isolated transport capacity validation refused')
    value=json.loads(result.stdout);cp=job['manifest']['payload']['checkpoint']
    if set(value)!={'sizes','limits'}or not isinstance(value['sizes'],dict)or set(value['sizes'])!=set(cp['files'])or not isinstance(value['limits'],list)or len(value['limits'])!=len(job['submissions'])or any(type(x)is not int or x<=0 for x in [*value['sizes'].values(),*value['limits']]):raise ValueError('exact isolated capacity result')
    return SimpleNamespace(budget=lambda *args:(None,cp,value['sizes'],None,None),input_limits=lambda *args:value['limits'])

def isolated_runtime_inventory(job,policy,authority,source,helper):
    code="""import importlib.util,json,sys
from pathlib import Path
source=Path(sys.argv[1]);sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('operator_capacity_inventory',sys.argv[2])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
v=json.load(sys.stdin);module.validate_runtime_inventory(v['job'],v['policy'],v['authority'],source)
if any(n=='torch'or n.startswith('torch.')for n in sys.modules):raise ValueError('CPU inventory only')
print('checked')
"""
    raw=json.dumps(dict(job=job,policy=policy,authority=authority),separators=(',',':'),allow_nan=False).encode()
    if len(raw)>verify_envelope_limit(job,authority):raise ValueError('bounded signed capacity input')
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(source),str(helper)],input=raw,capture_output=True,timeout=30,cwd=source)
    if result.returncode or result.stdout!=b'checked\n':raise ValueError('isolated runtime inventory refused')

def main():
    argv=list(sys.argv)
    at=argv.index('--capacity-policy');policy_path=Path(argv[at+1]);del argv[at:at+2]
    authority=argv[argv.index('--authority')+1];workspace=argv[argv.index('--workspace')+1]
    source=Path.cwd().resolve(strict=True);sys.path.insert(0,str(source))
    from nacl.signing import VerifyKey
    def authenticate(document,signer):
        if document['signer']!=signer:raise ValueError('original authority')
        raw=json.dumps(document['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        VerifyKey(bytes.fromhex(signer)).verify(raw,base64.b64decode(document['signature'],validate=True))
        return document['payload']
    path=Path(argv[1])
    with path.open('rb')as stream:raw=stream.read(32_000_001)
    if len(raw)>32_000_000:raise ValueError('absolute verify envelope size')
    envelope=json.loads(raw);job=authenticate(envelope,authority)
    if len(raw)>verify_envelope_limit(job,authority):raise ValueError('signed verify envelope size')
    if job.get('role')!='verify':raise ValueError('capacity bootstrap verify only')
    authenticate(job['manifest'],authority)
    helper=Path(__file__).resolve().parent/'verifier_capacity_admission.py'
    policy=json.loads(policy_path.read_bytes())
    isolated_runtime_inventory(job,policy,authority,source,helper)
    for name,expected in job['source_files'].items():
        p=Path(name)
        if p.is_absolute()or '..'in p.parts or not name.startswith('subnet/')or (source/p).is_symlink()or hashlib.sha256((source/p).read_bytes()).hexdigest()!=expected:raise ValueError('original scientific runtime source hash')
    admission=isolated_transport_admission(job,policy,authority,source,helper)
    from subnet import backend_jobs
    bind_transport(backend_jobs,job,authority,workspace,policy,admission)
    sys.argv=argv;backend_jobs.main()

if __name__=='__main__':main()
