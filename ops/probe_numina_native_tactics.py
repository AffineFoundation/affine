"""Bounded original Numina native controls using public starter files only.

No model, proof fingerprint, optimizer or chain operation runs here. The original
signature guard and Lean compiler determine reward; hidden proofs are never read.
"""
import argparse,hashlib,json,time,shlex
from pathlib import Path
from subnet.environments import EnvironmentSpec,EnvironmentSession

def tactic_command(path):
    # Read only the public starter inside the original sandbox, never task gold.
    program="from pathlib import Path; import re; p=Path("+repr(path)+"); s=p.read_text(); s,n=re.subn(r'\\bsorry\\b', 'first | aesop | omega | nlinarith | ring | norm_num | decide', s); assert n; p.write_text(s)"
    return 'python3 -c '+shlex.quote(program)

def run(spec,output,indices):
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for index in indices:
        session=EnvironmentSession(spec);start=time.time()
        try:
            reset=session.reset(index,index)
            config=session.task.config
            action={'text':'Apply public automated Lean tactics to the starter proof.',
                'tool_calls':[{'name':'bash','arguments':{'command':tactic_command(config.proof_file_path)}}]}
            first=session.step(action)
            outcome=first if first['done'] else session.step({'text':'Finished.'})
            rows.append(dict(index=index,task_hash=session.task.hash,reward=outcome['reward'],
                classification=outcome['classification'],status='native_completed',
                tool_observations=first['observations'],grader_info=dict(session.trace.info),elapsed=time.time()-start))
        except Exception as error:
            rows.append(dict(index=index,status='native_error',error_type=type(error).__name__,error=str(error)[-2000:],elapsed=time.time()-start))
        finally:session.close()
        (output/'native-controls.json').write_text(json.dumps(dict(rows=rows,source_hash=spec.source_hash,
            policy='public-starter-automated-tactics-v1',hidden_proofs_consumed=False,
            model_execution=False,optimizer_performed=False,chain_transactions=False),indent=2,default=str)+'\n')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=16)
    a=p.parse_args();spec=EnvironmentSpec.from_dict(json.loads(a.spec.read_bytes()))
    if spec.id!='affine_numina' or not 1<=a.count<=min(16,spec.num_samples):raise ValueError('bounded original mining controls')
    run(spec,a.output,list(range(a.count)))
if __name__=='__main__':main()
