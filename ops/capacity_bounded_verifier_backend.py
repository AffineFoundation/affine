"""Pinned operator bootstrap; only tightens authenticated transport byte limits.

The original source, main(), full digest checks, model and sampler run unchanged.
"""
import base64,hashlib,importlib.util,json,subprocess,sys
from types import SimpleNamespace
from pathlib import Path

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
v=json.load(sys.stdin);result=module.budget(v['job'],v['policy'],v['authority']);limits=module.input_limits(v['job'],v['job']['manifest']['payload'])
if any(n=='torch'or n.startswith('torch.')for n in sys.modules):raise ValueError('capacity validation must remain CPU metadata only')
print(json.dumps({'sizes':result[2],'limits':limits},separators=(',',':'),allow_nan=False))
"""

def isolated_transport_admission(job,policy,authority,source,helper):
    raw=json.dumps(dict(job=job,policy=policy,authority=authority),separators=(',',':'),allow_nan=False).encode()
    if len(raw)>4_000_000:raise ValueError('bounded ordinary capacity input')
    result=subprocess.run([sys.executable,'-I','-B','-c',_TRANSPORT_ADMISSION_CODE,str(source),str(helper)],input=raw,capture_output=True,timeout=30,check=False,cwd=source)
    if result.returncode or len(result.stdout)>1_000_000:raise ValueError('isolated transport capacity validation refused')
    value=json.loads(result.stdout);cp=job['manifest']['payload']['checkpoint']
    if set(value)!={'sizes','limits'}or not isinstance(value['sizes'],dict)or set(value['sizes'])!=set(cp['files'])or not isinstance(value['limits'],list)or len(value['limits'])!=len(job['submissions'])or any(type(x)is not int or x<=0 for x in [*value['sizes'].values(),*value['limits']]):raise ValueError('exact isolated capacity result')
    return SimpleNamespace(budget=lambda *args:(None,cp,value['sizes'],None,None),input_limits=lambda *args:value['limits'])

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
    if path.stat().st_size>4_000_000:raise ValueError('ordinary verify envelope size')
    envelope=json.loads(path.read_bytes());job=authenticate(envelope,authority)
    if job.get('role')!='verify':raise ValueError('capacity bootstrap verify only')
    authenticate(job['manifest'],authority)
    helper=Path(__file__).resolve().parent/'verifier_capacity_admission.py'
    policy=json.loads(policy_path.read_bytes())
    spec=importlib.util.spec_from_file_location('operator_capacity_runtime_inventory',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate_runtime_inventory(job,policy,authority,source)
    for name,expected in job['source_files'].items():
        p=Path(name)
        if p.is_absolute()or '..'in p.parts or not name.startswith('subnet/')or (source/p).is_symlink()or hashlib.sha256((source/p).read_bytes()).hexdigest()!=expected:raise ValueError('original scientific runtime source hash')
    admission=isolated_transport_admission(job,policy,authority,source,helper)
    from subnet import backend_jobs
    bind_transport(backend_jobs,job,authority,workspace,policy,admission)
    sys.argv=argv;backend_jobs.main()

if __name__=='__main__':main()
