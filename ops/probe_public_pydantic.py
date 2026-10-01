"""Bounded native controls for public-only Pydantic AST proposals (training only)."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from subnet.environments import EnvironmentSpec, EnvironmentSession
from subnet import public_pydantic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=Path('state/pydantic-tasksets/affine_pydantic.spec.json'))
    p.add_argument('--indices',nargs='+',type=int,default=list(range(16)))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if any(type(i) is not int or not 0<=i<16 for i in a.indices) or len(set(a.indices))!=len(a.indices):
        raise ValueError('original training indices0..15 only; heldout controls excluded')
    spec=EnvironmentSpec.from_dict(json.loads(a.spec.read_bytes()));session=EnvironmentSession(spec);rows=[]
    try:
        for index in a.indices:
            observation=session.reset(index,42)
            try: pair=public_pydantic.proposals(observation['messages'])
            except ValueError as error:
                rows.append({'index':index,'task_hash':observation['task_hash'],'status':'bounded_proposer_unsupported','error':str(error)})
                continue
            results=[]
            for candidate in pair:
                session.reset(index,42);outcome=session.step(candidate)
                results.append({'candidate':candidate,'reward':outcome['reward'],'classification':outcome['classification']})
            rows.append({'index':index,'task_hash':observation['task_hash'],'public_messages':observation['messages'],
                         'same_character_length':len(pair[0])==len(pair[1]),'results':results,'status':'native_controls'})
    finally: session.close()
    report={'timestamp':time.time(),'generator_revision':public_pydantic.REVISION,
            'generator_source_sha256':hashlib.sha256(Path(public_pydantic.__file__).read_bytes()).hexdigest(),
            'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'spec_source_hash':spec.source_hash,'generator_executes_schema_code':False,
            'generator_reads_gold_fields':False,'model_execution':False,'proofs_generated':False,
            'common_training':False,'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{'index':r['index'],'status':r['status'],'rewards':[v['reward'] for v in r.get('results',[])]} for r in rows]))


if __name__=='__main__': main()
