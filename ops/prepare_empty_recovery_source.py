"""Create a new qualified controller source with one signed empty-window policy.

Input must be a credential-free verified source archive extracted into a separate
checkout. This utility never changes its input or installs/restarts a service.
"""
import argparse, hashlib, shutil
from pathlib import Path

BASE = {'gpu_service.py':'586fc7eed4dc6399778c57ebc41817123766cec6638a596280f6414c16892b55',
        'remote_backend.py':'92cc5a9dc4bec311122475c518a48deb54fa9ebc72ec00106d1eca3a7bf39164'}

def prepare(source, destination, helper):
    source=Path(source).resolve();destination=Path(destination).resolve();helper=Path(helper).resolve()
    if destination.exists() or destination==source or destination.is_relative_to(source):raise ValueError('new isolated destination required')
    files=list(source.rglob('*'))
    if any(p.is_symlink() for p in files):raise ValueError('source links forbidden')
    for name,digest in BASE.items():
        if hashlib.sha256((source/'subnet'/name).read_bytes()).hexdigest()!=digest:raise ValueError('qualified controller base bytes')
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError('qualified empty policy helper')
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(helper,destination/'subnet/empty_epoch_policy.py')
    p=destination/'subnet/remote_backend.py';text=p.read_text()
    needle="        heldouts=kwargs.pop('heldout_indices',None)\n"
    replacement=needle+"        operator_test_policy=kwargs.pop('operator_test_policy',None)\n        if operator_test_policy is not None:\n            from .empty_epoch_policy import validate\n            validate(operator_test_policy)\n            if not args or not args[0].startswith('nonpayable-'):raise ValueError('nonpayable controlled window')\n"
    if text.count(needle)!=1:raise ValueError('qualified open hook')
    text=text.replace(needle,replacement).replace("        manifest['max_batches']=max_batches\n","        manifest['max_batches']=max_batches\n        if operator_test_policy is not None:manifest['operator_test_policy']=operator_test_policy\n");p.write_text(text)
    p=destination/'subnet/gpu_service.py';text=p.read_text()
    text=text.replace("    return dict(heldout_indices=", "    result=dict(heldout_indices=",1)
    needle="        backend_profile=BACKEND_PROFILE,model_id=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'))\n\ndef heldout"
    replacement="        backend_profile=BACKEND_PROFILE,model_id=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'))\n    from .empty_epoch_policy import selected\n    policy=selected(config,round_number)\n    if policy is not None:result['operator_test_policy']=policy\n    return result\n\ndef heldout"
    if text.count(needle)!=1:raise ValueError('qualified signed contract hook')
    text=text.replace(needle,replacement)
    needle="            if active['phase']=='mine':\n                if time.time()<manifest['deadline']:\n"
    replacement="            if active['phase']=='mine':\n                from .empty_epoch_policy import dispatch_allowed\n                if dispatch_allowed(manifest) and time.time()<manifest['deadline']:\n"
    if text.count(needle)!=1:raise ValueError('qualified dispatch hook')
    text=text.replace(needle,replacement)
    needle="                result,reports=controller.finalize(manifest,status['checkpoint_path']);save(state/(epoch+'-verified.json'),reports)\n"
    replacement=needle+"                from .empty_epoch_policy import validate_empty_completion\n                validate_empty_completion(manifest,result,reports)\n"
    if text.count(needle)!=1:raise ValueError('qualified empty finality hook')
    text=text.replace(needle,replacement);p.write_text(text)
    return {str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [destination/'subnet/gpu_service.py',destination/'subnet/remote_backend.py',destination/'subnet/empty_epoch_policy.py']}

HELPER_SHA='307ba7d8aeb093c4570b9d3c58b94dc74179cd1da297fedb84bf2fb9c33dfd3c'
if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path);parser.add_argument('--helper',type=Path,default=Path('subnet/empty_epoch_policy.py'));args=parser.parse_args();print(json.dumps(prepare(args.source,args.destination,args.helper)))
