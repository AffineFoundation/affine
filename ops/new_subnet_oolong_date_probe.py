"""Bounded genuine candidate rollouts and independent full computation/replay audit."""
import hashlib,json,os,time,copy,argparse
from pathlib import Path
import numpy as np
from subnet.model import Runtime,model_files

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--harness',default='state/multi-environment/oolong-date-harness.json');parser.add_argument('--output',default='state/multi-environment/oolong-date-proof-pilot');args=parser.parse_args()
    checkpoint='/root/miner-state/2b80bfabb0b54a409d8fb4df832112d208773d40391a76e5e2475d3306f31166'
    spec=json.loads(Path('state/original-task-snapshots/oolong-date-fixed4.spec.json').read_text())
    harness=json.loads(Path(args.harness).read_text())
    files=model_files(checkpoint)
    binding={'id':Path(checkpoint).name,'files':files,'filemap_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    miner=Runtime(checkpoint,files,threads=2,environment=spec,harness=harness)
    verifier=Runtime(checkpoint,files,threads=2,environment=spec,harness=harness)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    profile={'device':'cpu','dtype':'float32','attention':'eager','torch_threads':2,'toploc_parts_threads':getattr(miner,'toploc_threads',None),
        'native_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_CBWR','ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA')}}
    rows=[];found=set()
    for seed in range(31415,31431):
        start=time.time();rollout,arrays=miner.rollout(0,seed)
        valid=verifier.verify(rollout,arrays)
        name=f'seed-{seed}'
        (out/(name+'.rollout.json')).write_text(json.dumps(rollout)+'\n')
        np.savez_compressed(out/(name+'.probabilities.npz'),**{'turn_'+str(i):v for i,v in enumerate(arrays)})
        changed=copy.deepcopy(rollout);changed['reward']=0.5
        tamper_rejected=False
        try:tamper_rejected=not verifier.verify(changed,arrays)
        except (ValueError,AssertionError):tamper_rejected=True
        row={'seed':seed,'original_index':215,'sample_index':0,'checkpoint':binding,'source_hash':spec['source_hash'],
             'task_hash':rollout['task_hash'],'harness':harness,'runtime_profile':profile,'reward':rollout['reward'],
             'classification':rollout['classification'],'turns':len(rollout['turns']),
             'native_tool_observations':sum(len(t['observations']) for t in rollout['turns']),
             'independent_verified':valid,'claimed_reward_tamper_rejected':tamper_rejected,
             'training':False,'policy':'target-model-weighted public Counter vs Counter+1, not autonomous solving',
             'seconds':round(time.time()-start,2),'artifact':str(out/(name+'.rollout.json'))}
        rows.append(row);found.add(rollout['classification'])
        (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
        if valid and {'positive','negative'}<=found:break
    report={'success':{'positive','negative'}<=found and all(r['independent_verified'] and r['claimed_reward_tamper_rejected'] for r in rows),
            'classes':sorted(found),'rows':len(rows),'training':False,'approved_checkpoint':binding['id']}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['success']:raise RuntimeError('both honest classes were not verified within bounded search')
if __name__=='__main__':main()
