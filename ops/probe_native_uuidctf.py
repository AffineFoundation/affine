"""Small original UUIDCTF standard-difficulty native execution/replay controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from subnet.backend_jobs import canonical
from subnet.environments import build_spec,create_session,snapshot_spec
from subnet.public_uuidctf import public_solver_command


def execute(spec,index,actions):
    session=create_session(spec)
    try:
        reset=session.reset(index,20261001+index)
        turns=[dict(action=action,result=session.step(action)) for action in actions]
        if not turns[-1]['result']['done']:
            raise ValueError('completed native trajectory required')
        return dict(environment_definition_sha256=hashlib.sha256(canonical(spec.to_dict())).hexdigest(),
                    index=index,task_hash=reset['task_hash'],reset=reset,turns=turns,reward=turns[-1]['result']['reward'])
    finally:session.close()


def replay(spec,artifact):
    if canonical(execute(spec,artifact['index'],[row['action'] for row in artifact['turns']]))!=canonical(artifact):
        raise ValueError('original native trajectory or score mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--count',type=int,default=2);args=parser.parse_args()
    if not 1<=args.count<=4:raise ValueError('bounded original control population')
    args.state.mkdir(parents=True,exist_ok=True)
    config={'taskset':{'tasks':['uuidctf-'+str(i).zfill(5) for i in range(args.count)]}}
    spec=build_spec('affine_uuidctf',config,num_samples=args.count,max_turns=3,max_output_tokens=4096)
    spec=snapshot_spec(spec,args.state/'original-tasks.json');(args.state/'environment.json').write_bytes(canonical(spec.to_dict()))
    controls=[]
    for index in range(args.count):
        row=dict(index=index,positive_reward=None,negative_reward=None,mutations_rejected=[])
        for label,command in [('positive',public_solver_command()),('negative',"printf '%s' '{\"result_uuid\":\"00000000-0000-0000-0000-000000000000\"}' > /workspace/answer.json")]:
            actions=[dict(text='Inspect and process public incident evidence.',tool_calls=[dict(name='bash',arguments=dict(command=command))]),dict(text='Done.')]
            artifact=execute(spec,index,actions);assert artifact['reward']==(1.0 if label=='positive' else 0.0)
            replay(spec,artifact);row[label+'_reward']=artifact['reward'];row['task_hash']=artifact['task_hash']
            (args.state/f'{index}-{label}.json').write_bytes(canonical(artifact))
            for mutation in ('reward','observation'):
                changed=copy.deepcopy(artifact)
                if mutation=='reward':changed['reward']=1-artifact['reward']
                else:changed['turns'][0]['result']['observations'][0]['content']='forged public solver result'
                try:replay(spec,changed)
                except ValueError:row['mutations_rejected'].append(label+'-'+mutation)
                else:raise AssertionError('forged native outcome accepted')
        controls.append(row);(args.state/'controls.json').write_bytes(canonical(controls))
        print(json.dumps(row),flush=True)
    (args.state/'qualification.json').write_bytes(canonical(dict(source='affine_uuidctf',controls=controls,
        original_difficulty='standard',original_index_selection=True,source_hash=spec.source_hash,
        taskset_sha256=hashlib.sha256((args.state/'original-tasks.json').read_bytes()).hexdigest(),
        model_execution=False,TOPLOC_generated=False,optimizer_ran=False,chain_transactions=False)))


if __name__=='__main__':main()
