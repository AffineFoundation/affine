"""Prospective single-job operator launcher; CPU admission mode executes no model."""
import argparse,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace

def main():
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('fresh -I -B role launcher required')
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True,type=Path);parser.add_argument('--authority',required=True);parser.add_argument('--operator-descriptor',required=True,type=Path);parser.add_argument('--workspace',required=True);parser.add_argument('--cpu-admission-only',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from subnet.backend_jobs import validate,SOURCE_FILES
    envelope=json.loads(args.job.read_bytes());job,manifest=validate(envelope,args.authority)
    # Qualify source bytes before importing the prospective terminal adapter.
    for name,expected in job['source_files'].items():
        path=root/name
        if path.is_symlink()or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('signed role source bytes')
    rows=manifest.get('environments') or [dict(env_id=manifest['environment']['id'],spec=manifest['environment'])]
    matched=[r for r in rows if r['env_id']=='affine_rcore']
    if len(matched)!=1:raise ValueError('one signed RCore definition')
    raw=matched[0]['spec'];spec=SimpleNamespace(**raw);spec.to_dict=lambda:raw
    from subnet.native_rcore_role_transport import admit_role_binding,execute_role
    descriptor=json.loads(args.operator_descriptor.read_bytes())
    if args.cpu_admission_only:
        admit_role_binding(envelope,args.authority,spec,descriptor)
        loaded=[name for name in SOURCE_FILES if name!='subnet/backend_jobs.py'and name[:-3].replace('/','.')in sys.modules]
        if loaded:raise ValueError('runtime imported before job source admission')
        print(json.dumps(dict(admitted=True,model_execution=False,gpu_allocation=False,preloaded_runtime_modules=loaded)));return
    execute_role(envelope,args.authority,spec,descriptor,args.workspace)
if __name__=='__main__':main()
