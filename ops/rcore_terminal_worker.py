import sys,json,contextlib
from pathlib import Path
from subnet.environments import create_session
from subnet.native_rcore_boundary import OperatorBroker,REVISION,canonical
root=Path(__file__).resolve().parent
profile=json.loads((root.parent/'signed-resource-profile.json').read_bytes())['payload'];spec=json.loads((root/'operator/environment.json').read_bytes())
admission=dict(revision=REVISION,role='trusted-terminal-grader',environment_version=profile['environment_version'],environment_source_hash=profile['environment_source_hash'],source_bundle_sha256=profile['source_bundle_sha256'],provider_resource_id=profile['provider_resource_id'])
broker=OperatorBroker(admission,spec,create_session);actor=None
for line in sys.stdin:
 try:
  request=json.loads(line)
  with contextlib.redirect_stdout(sys.stderr):
   if request.get('op')=='reset' and set(request)=={'op','index','seed'}:
    actor=broker.actor(request['index'],request['seed']);result=actor.reset()
   elif request.get('op')=='finish' and set(request)=={'op','text'} and actor is not None:result=actor.finish(request['text'])
   elif request=={'op':'close'}:broker.close();break
   else:raise ValueError('terminal protocol')
  print(json.dumps({'ok':True,'result':result},allow_nan=False),flush=True)
 except BaseException as error:
  broker.close();print(json.dumps({'ok':False,'error_type':type(error).__name__}),flush=True);break
broker.close()
