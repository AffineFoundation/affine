"""Separately pinned CPU metadata peer entry. Default: admission only.

An explicit signed execution grant may call the unchanged frozen backend; the
only override is its source finder for four authenticated CPU metadata modules.
It must be composed with the independently admitted native N8 operator before
production use. This file is not part of the scientific 177-file claim.
"""
import argparse,hashlib,importlib,importlib.abc,importlib.util,json,os,sys,types
from pathlib import Path


def prepare(document,authority,operator_root,scientific_root,entry_path=None):
    from nacl.signing import VerifyKey
    def signed(d):
        if d['signer']!=authority:raise ValueError('ROOT authority')
        import base64
        sig=d['signature']
        try:s=base64.b64decode(sig,validate=True)
        except Exception:s=bytes.fromhex(sig)
        VerifyKey(bytes.fromhex(authority)).verify(json.dumps(d['payload'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode(),s)
        return d['payload']
    job=signed(document);manifest=signed(job['manifest'])
    a=signed(job['learner_selection_operator_admission']);grant=signed(a['authorization_document'])
    entry=Path(entry_path or __file__).resolve()
    if hashlib.sha256(entry.read_bytes()).hexdigest()!=grant['peer_entry_sha256']:raise ValueError('executed separately pinned peer entry SHA')
    operator_root=Path(operator_root);scientific_root=Path(scientific_root)
    package='_authenticated_CPU_selection_peer'
    if any(k==package or k.startswith(package+'.')for k in sys.modules):raise ValueError('fresh CPU peer namespace required')
    mod=types.ModuleType(package);mod.__path__=[str(operator_root/'subnet'),str(scientific_root/'subnet')];sys.modules[package]=mod
    # Compile the bootstrap bridge directly: no editable dependency or pyc.
    spec=importlib.util.spec_from_loader(package+'.learner_selection_operator_bridge',loader=None)
    b=importlib.util.module_from_spec(spec);b.__file__=str(operator_root/'subnet/learner_selection_operator_bridge.py');b.__package__=package
    # Authenticate every operator byte before executing any operator code.
    for name,h in grant['operator_files'].items():
        path=operator_root/name
        if path.resolve()!=path or path.is_symlink()or hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError('peer operator SHA/path')
    class PeerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if not fullname.startswith(package+'.'):return None
            suffix=fullname[len(package)+1:];location=operator_root/'subnet'/Path(*suffix.split('.')).with_suffix('.py')
            if not location.is_file():location=scientific_root/'subnet'/Path(*suffix.split('.')).with_suffix('.py')
            if not location.is_file():return None
            class Loader(importlib.abc.Loader):
                def create_module(self,spec):return None
                def exec_module(self,module):module.__file__=str(location);exec(compile(location.read_bytes(),str(location),'exec'),module.__dict__)
            return importlib.util.spec_from_file_location(fullname,location,loader=Loader())
    finder=PeerFinder();sys.meta_path.insert(0,finder)
    bridge=importlib.import_module(package+'.learner_selection_operator_bridge')
    importlib.import_module(package+'.committed_training_inputs')
    receipt=bridge.admit_peer(document,authority,operator_root=operator_root,scientific_root=scientific_root)
    receipt.update(peer_entry_sha256=grant['peer_entry_sha256'],CPU_override_symbols=['subnet.backend_jobs.FreshSourceFinder'],backend_execution_allowed=grant['backend_execution_allowed'])
    return receipt,grant,bridge


def bind_backend(grant,operator_root,scientific_root):
    """Explicit CPU parser override survives the original fresh-source reload."""
    if any(k=='subnet'or k.startswith('subnet.')for k in sys.modules):raise ValueError('fresh scientific namespace required')
    root=Path(scientific_root);op=Path(operator_root)
    package=types.ModuleType('subnet');package.__path__=[str(root/'subnet')];sys.modules['subnet']=package
    class CPUFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            name=fullname.replace('.','/')+'.py'
            if name not in grant['operator_files']:return None
            location=op/name
            class Loader(importlib.abc.Loader):
                def create_module(self,spec):return None
                def exec_module(self,module):
                    raw=location.read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=grant['operator_files'][name]:raise ValueError('executed CPU parser changed')
                    module.__file__=str(location);exec(compile(raw,str(location),'exec'),module.__dict__)
            return importlib.util.spec_from_file_location(fullname,location,loader=Loader())
    cpu=CPUFinder();sys.meta_path.insert(0,cpu)
    backend_path=root/'subnet/backend_jobs.py';raw=backend_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=grant['scientific_source_files']['subnet/backend_jobs.py']:raise ValueError('original backend SHA')
    backend=types.ModuleType('subnet.backend_jobs');backend.__file__=str(backend_path);backend.__package__='subnet';sys.modules[backend.__name__]=backend
    exec(compile(raw,str(backend_path),'exec'),backend.__dict__)
    Original=backend.FreshSourceFinder
    class DeclaredCPUFinder(Original):
        def find_spec(self,fullname,path=None,target=None):
            return cpu.find_spec(fullname,path,target)or super().find_spec(fullname,path,target)
    backend.FreshSourceFinder=DeclaredCPUFinder
    sys.meta_path.insert(0,DeclaredCPUFinder(root))
    return backend


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--authority',required=True);p.add_argument('--operator-root',required=True);p.add_argument('--scientific-root',required=True);p.add_argument('--execute-backend',action='store_true');p.add_argument('--workspace');p.add_argument('--checkpoint-cache');a=p.parse_args()
    document=json.loads(Path(a.job).read_bytes());receipt,grant,_=prepare(document,a.authority,a.operator_root,a.scientific_root)
    if a.execute_backend:
        if not grant['backend_execution_allowed']or not a.workspace:raise ValueError('explicit separately signed backend execution required')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('original CUBLAS profile required')
        backend=bind_backend(grant,a.operator_root,a.scientific_root)
        result=backend.execute(document,a.authority,a.workspace,a.checkpoint_cache)
        receipt['scientific_operation_started']=True;receipt['backend_result']=dict(success=result['success'],job_id=result['job_id'])
    print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
