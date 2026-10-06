"""Pinned operator bootstrap; only tightens authenticated transport byte limits.

The original source, main(), full digest checks, model and sampler run unchanged.
"""
import base64,hashlib,importlib.util,json,sys
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
    if set(job['source_files'])!={str(p.relative_to(source))for p in (source/'subnet').glob('*.py')}:raise ValueError('complete original runtime inventory')
    for name,expected in job['source_files'].items():
        p=Path(name)
        if p.is_absolute()or '..'in p.parts or not name.startswith('subnet/')or (source/p).is_symlink()or hashlib.sha256((source/p).read_bytes()).hexdigest()!=expected:raise ValueError('original scientific runtime source hash')
    helper=Path(__file__).resolve().parent/'verifier_capacity_admission.py'
    spec=importlib.util.spec_from_file_location('ops.verifier_capacity_admission',helper)
    admission=importlib.util.module_from_spec(spec);sys.modules[spec.name]=admission;spec.loader.exec_module(admission)
    from subnet import backend_jobs
    policy=json.loads(policy_path.read_bytes())
    bind_transport(backend_jobs,job,authority,workspace,policy,admission)
    sys.argv=argv;backend_jobs.main()

if __name__=='__main__':main()
