"""NEW public-location mutant: first patch only, unchanged subsequent actions."""
import copy,hashlib,json,pathlib
from subnet.native_eog_split import OperatorBroker,PublicActor,request
from subnet.native_eog_isolation import sha
from ops.probe_native_eog import public_relocation_policy

class FirstPatchActor:
 def __init__(self,actor):self.actor=actor;self.patches=0
 def call(self,name,arguments):
  arguments=copy.deepcopy(arguments)
  if name=='patch_event':
   self.patches+=1
   if self.patches==1:
    if arguments['location']!='TechCorp Main Campus - Building 1, Conference Room B':raise ValueError('public requested location contract')
    arguments['location']='TechCorp Main Campus - Building 2, Conference Room B'
  return self.actor.call(name,arguments)

def run(task,runtime,expected=None):
 broker=OperatorBroker(task,runtime)
 try:
  actor=PublicActor(broker.endpoint,broker.actor_capability,sha(broker.public));public=actor.reset()
  if expected:
   for event in expected['events']:
    if actor.call(event['name'],event['arguments'])!=event['observation']:raise ValueError('fresh exact native observation')
  else:public_relocation_policy(FirstPatchActor(actor))
  terminal=actor.finish();grade=request(broker.endpoint,broker.operator_capability,'operator',{'operation':'grade'})
  if terminal['reward']!=grade['grade']['reward'] or terminal['reward']!=0 or terminal['transcript_sha256']!=grade['transcript_sha256']:raise ValueError('actual native negative terminal grade')
  artifact={'schema':3,'scope':'public first-location-mutant native control; not sampled model output','public':{'task_id':public['task_id'],'messages':public['messages'],'tools':public['tools']},'events':copy.deepcopy(broker.session.events),'claimed_reward':terminal['reward'],'runtime':runtime,'original_seed_sha256':public['seed_sha256'],'source_files':public['source_files']}
  if len(artifact['events'])!=6 or artifact['events'][3]['arguments']['location']!='TechCorp Main Campus - Building 2, Conference Room B' or any(e['arguments']['location']!='TechCorp Main Campus - Building 1, Conference Room B' for e in artifact['events'][4:]):raise ValueError('only first patch mutant')
  if expected and artifact!=expected:raise ValueError('fresh exact public replay')
  return artifact,{'public_descriptor_sha256':sha(public),'terminal_public_response':terminal,'grade':grade,'probe_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
 finally:broker.close()

def main():
 base=pathlib.Path('state/native-eog-isolation');out=pathlib.Path('state/native-eog-first-patch-v3');out.mkdir(exist_ok=False)
 task=json.loads((base/'original-calendar-relocation.private.json').read_text());runtime=json.loads((base/'runtime-descriptor.json').read_text());artifact,report=run(task,runtime);_,replay=run(task,runtime,artifact)
 p=out/'negative-first-building2-model-input.public.json';p.write_text(json.dumps(artifact,indent=2)+'\n')
 for name,value in [('native-control.json',report),('native-replay.json',replay)]:
  with (out/name).open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
 print(json.dumps({'public':str(p),'public_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_reward':artifact['claimed_reward'],'exact_fresh_replay':True,'model_proof_pending':True}),flush=True)
if __name__=='__main__':main()
