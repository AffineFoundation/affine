"""Bounded original Calendar native controls, without inference/chain claims."""
import argparse
import hashlib
import json
from pathlib import Path
from subnet.native_eog_isolation import NativeEOGSession, IMAGE, REVISION, sha

def payload(observation):
    return json.loads(json.loads(observation)[0]['text'])

def public_relocation_policy(session, wrong_location=False):
    calendars=payload(session.call('get_calendar_list',{}))
    for calendar in calendars['items']:
        listed=payload(session.call('list_events',{'calendarId':calendar['id']}))
        if 'items' not in listed:continue
        for event in listed['items']:
            start=event.get('start',{}).get('dateTime') or event.get('start',{}).get('date','')
            if event.get('location')!='MegaCorp Headquarters, 123 Business Ave, New York, NY' or not '2025-11-01'<=start[:10]<='2025-11-28':continue
            description=(event.get('description') or '').rstrip()
            session.call('patch_event',{'calendarId':calendar['id'],'eventId':event['id'],
                'location':'Wrong office' if wrong_location else 'TechCorp Main Campus - Building 1, Conference Room B',
                'description':description+'\nRelocated due to unavailability at MegaCorp Headquarters.'})

def public_alchemy_policy(session, wrong_role=False):
    listed=payload(session.call('list_events',{'calendarId':'alice-projects'}))
    matches=[x for x in listed['items'] if x['summary']=='Sprint Planning & Architecture Review']
    if len(matches)!=1:raise ValueError('public event lookup must be unique')
    session.call('move_event',{'calendarId':'alice-projects','eventId':matches[0]['id'],'destination':'alice-primary'})
    session.call('patch_calendar',{'calendarId':'alice-primary','summary':'Project Alchemy'})
    for email,role in [('bob.smith@techcorp.com','reader'),('carol.white@techcorp.com','reader'),
                       ('dave.brown@techcorp.com','reader' if wrong_role else 'writer')]:
        session.call('insert_acl_rule',{'calendarId':'alice-primary','scope':{'type':'user','value':email},'role':role})

def public_helios_policy(session, wrong_date=False):
    """Actions transcribed solely from original public task 1 and tool schemas.

    The only runtime identifier is the actual create_calendar observation.
    Neither verifier definitions nor seed state are policy inputs.
    """
    created=payload(session.call('create_calendar',{'summary':'Helios Innovation Roadmap',
        'description':'Strategy and roadmap milestones.','timeZone':'America/New_York'}))
    identifier=created['id']
    session.call('patch_calendar',{'calendarId':identifier,'location':'New York HQ'})
    # Original server's public error asks for colorRgbFormat even for colorId.
    observation=session.call('add_calendar_to_list',{'id':identifier,'colorId':'7'})
    if 'must set colorRgbFormat=true' in observation:
        session.call('add_calendar_to_list',{'id':identifier,'colorId':'7','colorRgbFormat':True})
    date='2026-01-16' if wrong_date else '2026-01-15'
    session.call('create_event',{'calendarId':identifier,'summary':'Helios Kickoff',
         # 11 AM EST is 16:00 UTC; supply canonical UTC RFC3339 inputs.
         'start':{'dateTime':date+'T16:00:00Z','timeZone':'America/New_York'},
         'end':{'dateTime':date+'T17:00:00Z','timeZone':'America/New_York'},
         'attendees':[{'email':'alice.manager@techcorp.com'},{'email':'dave.brown@techcorp.com'}]})

def run_control(task, root, name, policy=None,runtime=None):
    session=NativeEOGSession(task,runtime=runtime)
    try:
        public=session.start(); initial=session.grade()
        if policy: policy(session)
        final=session.grade(); isolation=session.isolation()
        private={'public':public,'initial_grade':initial,'final_grade':final,
                 'initial_state':session.initial_state,'final_state':session.state(),
                 'events':session.events,'isolation':isolation}
        path=root/(name+'.private.json');path.write_text(json.dumps(private,indent=2));path.chmod(0o600)
        return {'initial_reward':initial['reward'],'final_reward':final['reward'],
                'tool_calls':len(session.events),'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'task_id':public['task_id'],'isolation':isolation,
                'all_verifiers_passed':bool(final['results']) and all(x['passed'] for x in final['results'])}
    finally: session.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--state',type=Path,default=Path('state/native-eog-isolation'))
    parser.add_argument('--relocation-controls',action='store_true')
    args=parser.parse_args();root=args.state;tasks=json.loads((root/'original-calendar-four.private.json').read_text())
    if args.relocation_controls:
        task=json.loads((root/'original-calendar-relocation.private.json').read_text())
        runtime=json.loads((root/'runtime-descriptor.json').read_text())
        positive=run_control(task,root,'relocation-fixed-positive',public_relocation_policy,runtime)
        negative=run_control(task,root,'relocation-fixed-negative',lambda s:public_relocation_policy(s,True),runtime)
        assert positive['initial_reward']==negative['initial_reward']==0
        assert positive['final_reward']==1 and negative['final_reward']==0
        report={'passed':True,'positive':positive,'negative':negative,'runtime':runtime,
                'original_reward_pair_verified':True,'policy_source':'public instruction and actual tool observations',
                'model_execution_verified':False,'optimizer_steps':0,'chain_submission':False}
        (root/'relocation-positive-negative-controls.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'passed':True,'positive_reward':1,'negative_reward':0,'tool_calls_each':positive['tool_calls']}))
        return
    results={}
    for index,task in enumerate(tasks):results['reset_'+str(index)]=run_control(task,root,'reset_'+str(index))
    runtime=json.loads((root/'runtime-descriptor.json').read_text())
    results['public_policy_attempt']=run_control(tasks[3],root,'alchemy-fixed-attempt',public_alchemy_policy,runtime)
    results['public_policy_fresh_reexecution']=run_control(tasks[3],root,'alchemy-fixed-replay',public_alchemy_policy,runtime)
    results['wrong_public_role']=run_control(tasks[3],root,'alchemy-fixed-wrong-role',lambda s:public_alchemy_policy(s,True),runtime)
    assert all(results['reset_'+str(i)]['final_reward']==0 for i in range(4))
    assert results['public_policy_attempt']['final_reward']==results['public_policy_fresh_reexecution']['final_reward']
    original=json.loads((root/'alchemy-fixed-attempt.private.json').read_text())
    replay=json.loads((root/'alchemy-fixed-replay.private.json').read_text())
    exact_events=original['events']==replay['events']
    def logical(state):
        # Filesystem last-modified time is operator-only transport metadata,
        # never a tool observation or original SQL grader input.
        return {k:state[k] for k in ('service','database_id','table_counts','table_data')}
    exact_states=logical(original['initial_state'])==logical(replay['initial_state']) and logical(original['final_state'])==logical(replay['final_state'])
    raw_equal=original['initial_state']==replay['initial_state'] and original['final_state']==replay['final_state']
    assert exact_events and exact_states
    session=NativeEOGSession(tasks[1]);rejected=[]
    try:
        public=session.start()
        for name in ('sql-runner','seed-database','delete_event','../../api/sql-runner'):
            try:session.call(name,{'query':'SELECT 1'})
            except ValueError:rejected.append(name)
            else:raise AssertionError('privileged/unselected tool exposed')
        assert 'verifiers' not in public and 'services' not in public and 'seed_file' not in json.dumps(public)
        assert session.grade()['reward']==0
    finally:session.close()
    original={}
    for p in [Path('subnet/vendor/research/environments/tool_use/enterprise_ops_gym_v1/enterprise_ops_gym_v1/toolset.py'),
              Path('subnet/vendor/research/environments/tool_use/enterprise_ops_gym_v1/enterprise_ops_gym_v1/taskset.py'),
              Path('subnet/vendor/legacy/rollouts/envs/affine_eog_v1/affine_eog_v1/taskset.py')]:
        original[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report={'schema':1,'revision':REVISION,'image':IMAGE,'original_source_files':original,
        'fixture_file_sha256':hashlib.sha256((root/'original-calendar-four.private.json').read_bytes()).hexdigest(),
        'runtime':runtime,
        'state_comparison_fields':['service','database_id','table_counts','table_data'],
        'raw_state_metadata_equal':raw_equal,
        'controls':results,'restricted_actor_calls_rejected':rejected,
        'original_database_rewards_verified':True,'native_tool_execution_verified':True,
        'fresh_public_policy_execution_verified':True,'exact_transcript_replay_verified':exact_events and exact_states,
        'positive_reward_control_verified':results['public_policy_attempt']['final_reward']==1,
        'scope':'Controlled original Calendar runtime only; no model execution and no common pipeline admission.',
        'model_proofs_verified':False,'optimizer_steps':0,'chain_submission':False,
        'shared_pipeline_epoch_completed':False,'full_upstream_orchestrator_tested':False}
    (root/'native-controls.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'reset_tasks':4,'public_policy_reward':results['public_policy_attempt']['final_reward'],
                     'restricted_actor_calls_rejected':len(rejected),'exact_transcript_replay_verified':exact_events and exact_states}))

if __name__=='__main__':main()
