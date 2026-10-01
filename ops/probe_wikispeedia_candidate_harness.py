"""Native qualification of public graph choices in the shared model harness."""
import argparse
import hashlib
import json
from pathlib import Path
from subnet.backend_jobs import canonical
from subnet.environments import EnvironmentSpec,create_session
from subnet.harness import action,turn_config
from subnet.native_wikispeedia import public_candidate_harness,execute,replay,verify_public_resources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualified-state',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args()
    if args.state.exists():raise ValueError('fresh owned output directory required')
    spec=EnvironmentSpec.from_dict(json.loads((args.qualified_state/'environment.json').read_bytes()))
    # Resolve the approved original adapter namespace before importing its graph.
    session=create_session(spec)
    try:session.reset(0,20261001)
    finally:session.close()
    from wikispeedia_v1.graph import WikiGraph,DEFAULT_CACHE_DIR
    graph=WikiGraph.load(include_text=False)
    resources=verify_public_resources(DEFAULT_CACHE_DIR)
    tasks=json.loads((args.qualified_state/'original-tasks.json').read_bytes())
    if not 1<=len(tasks)<=32:raise ValueError('bounded qualified original task population')
    args.state.mkdir(parents=True)
    records=[]
    for index,task in enumerate(tasks):
        original=json.loads((args.qualified_state/f'{index}-positive.json').read_bytes())
        replay(spec,original)
        data=task['data'];config=public_candidate_harness(data['source'],data['target'],
                original['reset']['tools'],graph.links,spec.max_turns)
        outcomes={}
        for label,choice in [('positive',0),('negative',1)]:
            # Native controls deliberately choose a fixed branch; these are not
            # samples from a model. The subsequent model experiment chooses it.
            actions=[action(turn_config(config['harness'],turn)['candidates'][choice],config['harness'])
                     for turn in range(config['max_turns'])]
            if label=='positive':actions=actions[:-1]  # Target arrival terminates natively.
            trace=execute(spec,index,original['seed'],actions);replay(spec,trace)
            if trace['reward']!=(1. if label=='positive' else 0.):raise ValueError('original native control outcome')
            name=f'{index}-{label}.json';(args.state/name).write_bytes(canonical(trace))
            outcomes[label]=dict(reward=trace['reward'],artifact=name,sha256=hashlib.sha256(canonical(trace)).hexdigest())
        records.append(dict(index=index,task_hash=original['task_hash'],candidate_config=config,outcomes=outcomes))
    report=dict(schema=1,environment=spec.to_dict(),resources=resources,
                helper_sha256=hashlib.sha256(Path('subnet/native_wikispeedia.py').read_bytes()).hexdigest(),
                records=records,model_execution=False,TOPLOC_generated=False,
                optimizer_ran=False,chain_transactions=False,
                scope='Public candidate shared-dialect native controls only; no model admission')
    (args.state/'qualification.json').write_bytes(canonical(report))
    print(json.dumps(dict(tasks=len(records),native_executions=5*len(records),
                          fresh_replays=3*len(records),model_execution=False)))


if __name__=='__main__':main()
