"""Fresh-process original EOG replay and falsification controls."""
import copy
import argparse
import json
from pathlib import Path
from subnet.native_eog_isolation import replay, logical_state, sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--relocation',action='store_true');args=parser.parse_args()
    root=Path('state/native-eog-isolation')
    task=json.loads((root/'original-calendar-relocation.private.json').read_text()) if args.relocation else json.loads((root/'original-calendar-four.private.json').read_text())[3]
    runtime=json.loads((root/'runtime-descriptor.json').read_text())
    artifact=json.loads((root/('relocation-fixed-positive.private.json' if args.relocation else 'alchemy-fixed-attempt.private.json')).read_text())
    events=artifact['events'];reward=artifact['final_grade']['reward'];digest=sha(logical_state(artifact['final_state']))
    honest=replay(task,runtime,events,reward,digest)
    mutations={}
    changed=copy.deepcopy(events);changed[0]['observation']+=' forged'
    mutations['fabricated_observation']=(runtime,changed,reward,digest)
    changed=copy.deepcopy(events)
    if args.relocation:
        index=next(i for i,e in enumerate(changed) if e['name']=='patch_event')
        changed[index]['arguments']['location']='Forged event location'
    else:changed[2]['arguments']['summary']='Forged calendar title'
    mutations['substituted_tool_action']=(runtime,changed,reward,digest)
    mutations['false_reward']=(runtime,events,1 if reward==0 else 0,digest)
    mutations['false_logical_database']=(runtime,events,reward,'0'*64)
    changed_runtime=copy.deepcopy(runtime)
    if args.relocation:changed_runtime['clock']='2026-01-02T00:00:00+00:00'
    else:changed_runtime['seed']='f'*64
    mutations['different_entropy_profile']=(changed_runtime,events,reward,digest)
    results={}
    for name,(profile,trace,claimed,db_hash) in mutations.items():
        try:replay(task,profile,trace,claimed,db_hash)
        except ValueError as exc:results[name]={'rejected':True,'reason':str(exc)}
        else:raise AssertionError('accepted forged native artifact: '+name)
    report={'passed':True,'honest_replay':honest,'mutations':results,
            'scope':'original Calendar API/reward replay only; no model proof or optimizer',
            'runtime':runtime,'chain_submission':False}
    (root/('relocation-replay-controls.json' if args.relocation else 'replay-controls.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'honest_replayed_tools':honest['replayed_tools'],
                      'rejected_mutations':len(results),'original_reward':honest['reward']}))

if __name__=='__main__':main()
