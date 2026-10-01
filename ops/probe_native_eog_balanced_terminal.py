"""NEW original Calendar terminal controls; public balanced candidate, no model claim."""
import copy, hashlib, json
from pathlib import Path
from subnet.native_eog_split import OperatorBroker, PublicActor, request
from subnet.native_eog_isolation import sha
from ops.probe_native_eog import public_relocation_policy

class BalancedActor:
    def __init__(self,actor,negative):self.actor=actor;self.negative=negative
    def call(self,name,arguments):
        arguments=copy.deepcopy(arguments)
        if self.negative and name=='patch_event':
            assert arguments['location']=='TechCorp Main Campus - Building 1, Conference Room B'
            arguments['location']='TechCorp Main Campus - Building 2, Conference Room B'
        return self.actor.call(name,arguments)

def run(task,runtime,negative=False,expected=None):
    broker=OperatorBroker(task,runtime)
    try:
        actor=PublicActor(broker.endpoint,broker.actor_capability,sha(broker.public));public=actor.reset()
        if expected:
            for event in expected['events']:
                assert actor.call(event['name'],event['arguments'])==event['observation']
        else:public_relocation_policy(BalancedActor(actor,negative))
        finish=actor.finish();assert actor.finish()==finish
        grade=request(broker.endpoint,broker.operator_capability,'operator',{'operation':'grade'})
        assert finish['reward']==grade['grade']['reward'] and finish['transcript_sha256']==grade['transcript_sha256']
        artifact={'schema':3,'scope':'curated original native terminal control, not sampled model output',
            'public':{'task_id':public['task_id'],'messages':public['messages'],'tools':public['tools']},
            'public_descriptor':public,'public_descriptor_sha256':sha(public),
            'events':copy.deepcopy(broker.session.events),'claimed_reward':finish['reward'],
            'runtime':runtime,'original_seed_sha256':public['seed_sha256'],'source_files':public['source_files'],
            'terminal_text':'DONE','terminal_model_proof_verified':False}
        if expected:assert artifact==expected
        return artifact,{'grader_descriptor':broker.private_grader,'grade':grade,'finish':finish}
    finally:broker.close()

def main():
    base=Path('state/native-eog-isolation');root=Path('state/native-eog-balanced-terminal-v3');root.mkdir(exist_ok=True)
    task=json.loads((base/'original-calendar-relocation.private.json').read_text());runtime=json.loads((base/'runtime-descriptor.json').read_text())
    artifacts={};report={}
    for name,negative in [('positive',False),('negative-building2',True)]:
        artifact,private=run(task,runtime,negative)
        assert artifact['claimed_reward']==(0 if negative else 1)
        replay,replayprivate=run(task,runtime,expected=artifact)
        p=root/(name+'.public.json');p.write_text(json.dumps(artifact,indent=2)+'\n')
        q=root/(name+'.operator.private.json');q.write_text(json.dumps(private,indent=2)+'\n');q.chmod(0o600)
        q=root/(name+'-replay.operator.private.json');q.write_text(json.dumps(replayprivate,indent=2)+'\n');q.chmod(0o600)
        artifacts[name]=artifact;report[name]={'reward':artifact['claimed_reward'],'exact_fresh_replay':True,'public_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'public_descriptor_sha256':sha(artifact['public_descriptor'])}
    a=artifacts['positive'];b=artifacts['negative-building2']
    assert a['public']==b['public'] and a['events'][:3]==b['events'][:3]
    assert a['events'][3]['arguments']['location'].replace('Building 1','Building 2')==b['events'][3]['arguments']['location']
    assert len(a['events'][3]['arguments']['location'])==len(b['events'][3]['arguments']['location'])
    report.update({'passed':True,'same_public_initial_context':True,'same_first_three_tool_events':True,'first_divergence_turn_index':3,'equal_location_character_length':True,'token_length_check_pending':True,'terminal_text':'DONE','terminal_model_proof_verified':False,'chain_submission':False})
    (root/'controls.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
