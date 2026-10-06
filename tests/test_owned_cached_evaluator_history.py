import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from subnet.storage import Identity,canonical
from subnet.checkpoint_evaluator import fingerprint
from subnet.owned_cached_evaluation import POLICY
from ops.owned_cached_evaluator_history import inherit_completed
from ops.continuous_owned_cached_evaluator import config_admission
from test_owned_cached_evaluator_1024 import Explicit1024Controls

class CompletedOriginalHistoryControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.old=Path(self.tmp.name)/'original';self.new=Path(self.tmp.name)/'continuing';self.old.mkdir();self.new.mkdir();self.who=Identity(bytes(range(32)));self.raw={};self.acks={};self.files={};self.remote=SimpleNamespace(workspace='/owned-original-workspace',checked=Mock());self.jobs=SimpleNamespace(instance=lambda source:self.remote)
  self.config=Explicit1024Controls().config();self.config.update(before_checkpoint='a'*64,state=str(self.new),version='continuous-owned-cached-checkpoints-1024-v3',stop_after_pair=False);self.config['completed_history']={'state':str(self.old),'workspace':self.remote.workspace,'after_checkpoint':'b'*64,'files':self.files};self.controller=SimpleNamespace(state=self.new,authority=self.who,bucket=SimpleNamespace(get=lambda key:self.acks[key]))
  for phase,cp,steps in [('before','a'*64,10),('after','b'*64,11)]:
   manifest=dict(epoch='original-'+phase,checkpoint={'id':cp,'files':{'config.json':'c'*64,'model.safetensors':'d'*64}},source_bundle={'sha256':self.config['source_sha256'],'format':'tar.gz'},environments=[{'env_id':'math','spec':{}}]);plan=[dict(env_id='math',indices=self.config['heldout'][0]['indices'],seeds=[20261002+i*1000 for i in self.config['heldout'][0]['indices']],harness=self.config['heldout'][0]['harness'])];req=dict(label='original-'+phase,phase=phase,manifest=manifest,public_optimizer_steps=steps,config=self.config.copy(),heldout_plan=plan);req['config'].pop('completed_history');identity=fingerprint(manifest,req['config'],plan);row=dict(status='complete',request=req,request_sha256=hashlib.sha256(canonical(req)).hexdigest(),evaluation_id=identity)
   jid='job-'+phase;job=dict(job_id=jid,role='evaluate',owned_evaluation_policy=POLICY,heldout=plan,manifest=self.sign(manifest));envelope=self.sign(job);jobsha=hashlib.sha256(canonical(job)).hexdigest();prior=dict(job_id=jid,job_sha256=jobsha);report=dict(job_id=jid,checkpoint=cp,job_sha256=jobsha,success=True);key='ACK-'+phase;ack=dict(original_job=envelope,original_report=report,workspace=self.remote.workspace,durable_report_full_readback=True,job_sha256=jobsha);self.acks[key]=canonical(self.sign(ack));disp=dict(durable_ack_key=key,durable_ack_sha256=hashlib.sha256(self.acks[key]).hexdigest(),result={'status':'complete'})
   for path,value in [('checkpoint-evaluations/'+identity+'.json',row),('roles/original-'+phase+'.json',prior),('roles/'+jid+'-job.json',envelope),('roles/'+jid+'-report.json',report),('cache-disposal/'+jid+'.json',disp)]:self.put(path,value)
 def sign(self,v):return dict(payload=v,signer=self.who.id,signature=base64.b64encode(self.who.key.sign(canonical(v)).signature).decode())
 def put(self,name,value):
  raw=canonical(value);p=self.old/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw);self.files[name]=hashlib.sha256(raw).hexdigest()
 def test_originals_import_exact_bytes_backendchecked_without_resampling(self):
  config_admission(self.config);result=inherit_completed(self.controller,self.jobs,self.config);self.assertFalse(result['resampled']);self.assertEqual(self.remote.checked.call_count,2)
  for name in self.files:self.assertEqual((self.old/name).read_bytes(),(self.new/name).read_bytes())
 def test_tampered_file_or_ACK_never_imported(self):
  for change in ['file','ACK']:
   with self.subTest(change=change):
    if change=='file':p=self.old/next(iter(self.files));raw=p.read_bytes();p.write_bytes(raw+b' ')
    else:key=next(iter(self.acks));raw=self.acks[key];self.acks[key]=raw+b' '
    with self.assertRaises(ValueError):inherit_completed(self.controller,self.jobs,self.config)
    self.assertFalse(list(self.new.rglob('*.json')))
    if change=='file':p.write_bytes(raw)
    else:self.acks[key]=raw
 def test_false_disposal_and_wrong_owned_workspace_refused(self):
  name='cache-disposal/job-after.json';v=json.loads((self.old/name).read_bytes());v['result']['status']='deferred';self.put(name,v)
  with self.assertRaisesRegex(ValueError,'disposal'):inherit_completed(self.controller,self.jobs,self.config)
  self.config['completed_history']['workspace']='/external-cache'
  with self.assertRaisesRegex(ValueError,'workspace'):inherit_completed(self.controller,self.jobs,self.config)
 def test_original_request_cannot_be_changed_or_future_parent_imported(self):
  self.config['completed_history']['after_checkpoint']='future'
  with self.assertRaisesRegex(ValueError,'CP10 CP11'):inherit_completed(self.controller,self.jobs,self.config)
  self.assertFalse(list(self.new.rglob('*.json')))
 def test_unrelated_files_and_private_authority_never_imported(self):
  self.put('roles/authority.seed.json',{})
  with self.assertRaisesRegex(ValueError,'allowlist'):inherit_completed(self.controller,self.jobs,self.config)
 def test_continuous_profile_never_stops_or_changes_cap(self):
  config_admission(self.config)
  for change in [dict(stop_after_pair=True),dict(evaluation_token_cap=128),dict(completed_history=None)]:
   with self.subTest(change=change),self.assertRaises(ValueError):config_admission(dict(self.config,**change))

if __name__=='__main__':unittest.main()
