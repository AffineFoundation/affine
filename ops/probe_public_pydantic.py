"""Bounded native controls for public-only Pydantic AST proposals (training only)."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from subnet.environments import EnvironmentSpec, create_session
from subnet import public_pydantic, public_pydantic_type_mutation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=Path('state/pydantic-tasksets/affine_pydantic.spec.json'))
    p.add_argument('--indices',nargs='+',type=int,default=list(range(16)))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generator',choices=['ast-v1','same-key-wrong-type-v2'],default='ast-v1')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if any(type(i) is not int or not 0<=i<16 for i in a.indices) or len(set(a.indices))!=len(a.indices):
        raise ValueError('original training indices0..15 only; heldout controls excluded')
    if a.seed<0:raise ValueError('nonnegative environment initialization seed')
    generator=public_pydantic if a.generator=='ast-v1' else public_pydantic_type_mutation
    spec=EnvironmentSpec.from_dict(json.loads(a.spec.read_bytes()))
    if spec.id!='affine_pydantic' or spec.max_turns!=1 or spec.num_samples<16:
        raise ValueError('original single-turn Pydantic training population required')
    session=create_session(spec);rows=[]
    try:
        for index in a.indices:
            observation=session.reset(index,a.seed)
            try: pair=generator.proposals(observation['messages'])
            except ValueError as error:
                rows.append({'index':index,'task_hash':observation['task_hash'],'status':'bounded_proposer_unsupported','error':str(error)})
                continue
            results=[]
            for candidate in pair:
                assert session.reset(index,a.seed)==observation
                outcome=session.step(candidate)
                if getattr(getattr(session,'trace',None),'info',{}).get('score_error'):
                    raise ValueError('native grader failure cannot become a negative control')
                replay=create_session(spec)
                try:
                    assert replay.reset(index,a.seed)==observation
                    repeated=replay.step(candidate)
                    if getattr(getattr(replay,'trace',None),'info',{}).get('score_error'):
                        raise ValueError('native replay grader failure')
                    if repeated!=outcome:raise ValueError('fresh original terminal replay mismatch')
                finally:replay.close()
                results.append({'candidate':candidate,'reward':outcome['reward'],'classification':outcome['classification']})
            rows.append({'index':index,'task_hash':observation['task_hash'],'public_messages':observation['messages'],
                         'same_character_length':len(pair[0])==len(pair[1]),'results':results,'status':'native_controls',
                         'fresh_original_replays':True,
                         'qualifying_K1L1':{r['classification'] for r in results}=={'positive','negative'}})
    finally: session.close()
    report={'timestamp':time.time(),'generator_revision':generator.REVISION,
            'generator_source_sha256':hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest(),
            'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'spec_source_hash':spec.source_hash,'generator_executes_schema_code':False,
            'generator_reads_gold_fields':False,'model_execution':False,'proofs_generated':False,
            'common_training':False,'environment_seed':a.seed,'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{'index':r['index'],'status':r['status'],'rewards':[v['reward'] for v in r.get('results',[])]} for r in rows]))


if __name__=='__main__': main()
