"""Grade public-only candidate proposals in the original environments.

This runs native environment controls, not model inference or training.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from subnet.environments import EnvironmentSpec, EnvironmentSession
from subnet.public_candidates import VERSION, propose
from subnet import public_candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-directory',type=Path,default=Path('state/original-task-snapshots'))
    parser.add_argument('--sources',nargs='+',default=['affine_science','affine_scitext','affine_logic','affine_unscramble'])
    parser.add_argument('--indices',nargs='+',type=int,default=[0,1])
    parser.add_argument('--output',type=Path,default=Path('state/multi-environment/public-solver-research/root-public-candidate-native-grades.json'))
    args=parser.parse_args();rows=[]
    for source in args.sources:
        spec=EnvironmentSpec.from_dict(json.loads((args.spec_directory/f'fixed4-{source}.spec.json').read_text()))
        session=EnvironmentSession(spec)
        try:
            for index in args.indices:
                observation=session.reset(index,42);values=propose(source,observation['messages']);results=[]
                for value in values:
                    session.reset(index,42);result=session.step(value)
                    results.append(dict(candidate=value,reward=result['reward'],classification=result['classification']))
                rows.append(dict(source=source,index=index,task_hash=observation['task_hash'],public_messages=observation['messages'],
                                 generator_version=VERSION,spec_source_hash=spec.source_hash,results=results))
        finally:session.close()
    result=dict(timestamp=time.time(),used_gold_fields=False,fresh_model_execution=False,rows=rows,
                generator_source_sha256=hashlib.sha256(Path(public_candidates.__file__).read_bytes()).hexdigest(),
                probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps([dict(source=r['source'],index=r['index'],candidates=len(r['results']),
                          positive=sum(v['classification']=='positive' for v in r['results']),
                          negative=sum(v['classification']=='negative' for v in r['results'])) for r in rows]))


if __name__=='__main__':main()
