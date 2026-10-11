"""Real CPU Adam fixture, not a production or full7B execution claim."""
import base64,copy,hashlib,json,os,socket,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.persistent_cpu_adamw import PersistentCPUAdamW,genesis,parameter_inventory,sha
from subnet.persistent_training_state import resource_plan,admit_resources,export_state,restore_state,HEADER_RESERVE
from subnet.persistent_training_protocol import state_pointer,validate_parent
from subnet.optimizer_state_cache import StateCache,STAT_VERSION,STAT_VALIDATION,promote,committed_cache_descriptor
from subnet.research_parent_adoption import VERSION,MODULE,validate_successor

class ResearchParentAdoptionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.workspace=self.root/'production';self.workspace.mkdir(mode=0o700)
  self.research=self.root/'research';self.research.mkdir(mode=0o700);self.retained=self.research/'retained-Adam8';self.retained.mkdir(mode=0o700)
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  self.params=[('w',torch.nn.Parameter(torch.arange(1,5,dtype=torch.bfloat16)))];_,self.inventory=parameter_inventory(self.params)
  self.plan=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=HEADER_RESERVE+1024,disk_reserve_bytes=0,ram_reserve_bytes=0,concurrency=1)
  self.admission=admit_resources(self.root,self.plan);self.genesis=genesis(self.inventory,'11'*32)
  self.optimizer=PersistentCPUAdamW(self.params,'11'*32,approved_genesis=self.genesis,approved_genesis_sha256=sha(self.genesis),resource_admission=self.admission)
  for _ in range(8):self.params[0][1].grad=torch.ones_like(self.params[0][1]);self.optimizer.step()
  def retain(name,path,digest,size):
   target=self.retained/name;os.rename(path,target);target.chmod(0o600);return True
  def commit(d):
   self.write(self.retained/'descriptor.json',d)
   return dict(descriptor_sha256=sha(d),local_sha_verified=True,durable_readback_verified=False,authority_committed=False)
  self.descriptor,_=export_state(self.optimizer,epoch='research-original',inference_checkpoint='22'*32,workspace=self.research,
   publish_shard=self.forbidden,readback_shard=self.forbidden,commit_descriptor=commit,resource_admission=self.admission,
   shard_bytes=HEADER_RESERVE+1024,readback_mode='trainer-local-only-v1',retain_shard=retain)
  scope=dict(version='ROOT-original-recipe-full7B-positive-SFT-replication-v1',source_files={'subnet/research_kernel.py':'aa'*32},
   epoch='research-original',original_output_checkpoint='22'*32,parameters=self.inventory,genesis=self.genesis,
   hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),workspace=str(self.research))
  self.scope_json=json.dumps(self.sign(scope),sort_keys=True,indent=2)+'\n'
  self.execution=dict(version='ROOT-original-full7B-SFT-replication-execution-v1',scope_sha256=hashlib.sha256(self.scope_json.encode()).hexdigest(),
   state_descriptor_sha256=sha(self.descriptor),checkpoint='22'*32,optimizer_steps=8,actual_full7B_GPU_execution=True,new_actual_Adam8_retained=True,
   exact_original_checkpoint_equality=True,exact_original_BF16_parameter_equality=True,historical_optimizer_recovery_claim=False)
  self.source_files={MODULE:'bb'*32,'subnet/optimizer_state_cache.py':'cc'*32}
  self.publication=dict(version=VERSION,namespace='private/trainer-state/adoption/research-parent',adoption_id='research-parent',
   descriptor_sha256=sha(self.descriptor),descriptor=self.descriptor,research_scope_json=self.scope_json,research_execution=self.sign(self.execution),
   research_source_files_sha256=sha(scope['source_files']),retained_directory=str(self.retained),
   successor=dict(epoch='first-production',source_sha256='33'*32,execution_source_sha256='55'*32,source_files=self.source_files,trainer_workspace=str(self.workspace),
   hostname=scope['hostname'],boot_id=scope['boot_id'],checkpoint='22'*32,optimizer_step_before=8))
  self.binding=dict(parent=state_pointer(self.publication),input_checkpoint='22'*32,parameters=self.inventory,global_step_before=8,
   genesis_sha256=sha(self.genesis),epoch='first-production',source_sha256='33'*32)
  self.manifest=dict(source_bundle=dict(sha256='33'*32),epoch='first-production',trainer_state_binding=self.binding,
   optimizer_state_export_policy='trainer-local-only-v1',optimizer_state_local_cache=dict(version=STAT_VERSION,validation=STAT_VALIDATION,max_checkpoint_bytes=10*1024**2),
   persistent_publication_policy=dict(version='parallel-persistent-publication-v1',state_readback='trainer-local',checkpoint_readback_workers=1))
  self.job=dict(job_id='actual-first-random123',role='train',source_files=self.source_files,manifest=self.sign(self.manifest),
   persistent_training=dict(parent_publication=self.sign(self.publication),output_shards={'state-000000.safetensors':{}}))
  self.job['unaudited_training_execution']=self.sign(dict(version='unaudited-training-execution-amendment-v6-positive-sft',
   job_id=self.job['job_id'],epoch='first-production',training_source_bundle={'sha256':'55'*32},execution_source_files=self.source_files,
   original_source_bundle_sha256='33'*32,parent_descriptor_sha256=sha(self.descriptor),optimizer_step_before=8))
 def forbidden(self,*a,**kw):raise AssertionError('optimizer network transfer forbidden')
 def sign(self,b):return dict(payload=copy.deepcopy(b),signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(b)).signature).decode())
 def write(self,p,b):p.write_bytes(canonical(b));p.chmod(0o600)
 def cache(self,job=None):return StateCache(self.workspace,job or self.job,self.manifest,self.authority)
 def changed(self,p):
  b=copy.deepcopy(self.binding);b['parent']=state_pointer(p);return self.sign(p),b
 def test_actual_adam8_restore_preserves_original_and_has_no_fake_ack(self):
  self.assertEqual(validate_parent(self.sign(self.publication),self.binding,self.authority),self.descriptor)
  original={p.name:p.read_bytes() for p in self.retained.iterdir()};out=self.workspace/'restore';out.mkdir()
  with self.cache() as cache:
   self.assertGreater(cache.prepare_parent(self.descriptor,'33'*32),0)
   state,evidence=restore_state(self.descriptor,sha(self.descriptor),'22'*32,self.inventory,workspace=out,
    fetch_shard=lambda n,p:cache.fetch(n,p,self.forbidden),resource_admission=admit_resources(out,self.plan),owned_cache=cache)
   self.assertNotIn('ROOT_ack',cache.current)
   for name,row in self.optimizer.rows.items():
    for slot in ('master','exp_avg','exp_avg_sq'):torch.testing.assert_close(row[slot],state[1][name][slot],rtol=0,atol=0)
    self.assertEqual(state[1][name]['step'],8)
  self.assertEqual(original,{p.name:p.read_bytes() for p in self.retained.iterdir()})
  self.assertFalse((self.workspace/'.optimizer-state-cache/current.json').exists());self.assertFalse((self.workspace/(self.job['job_id']+'.json')).exists())
 def test_same_job_retry_different_job_and_modified_job_rejected(self):
  with self.cache() as c:c.prepare_parent(self.descriptor,'33'*32)
  with self.cache() as c:c.prepare_parent(self.descriptor,'33'*32)
  for field,value in [('job_id','different'),('steps',999)]:
   j=copy.deepcopy(self.job);j[field]=value
   if field=='job_id':
    d=j['unaudited_training_execution']['payload'];d['job_id']=value;j['unaudited_training_execution']=self.sign(d)
   with self.cache(j) as c:
    with self.assertRaisesRegex(ValueError,'different original job'):c.prepare_parent(self.descriptor,'33'*32)
 def test_wrong_receipt_and_scope_rejected_even_if_outer_root_signed(self):
  for change in ('scope_sha256','state_descriptor_sha256','checkpoint'):
   p=copy.deepcopy(self.publication);e=copy.deepcopy(self.execution);e[change]='ee'*32;p['research_execution']=self.sign(e)
   env,b=self.changed(p)
   with self.assertRaisesRegex(ValueError,'original research execution'):validate_parent(env,b,self.authority)
 def test_bad_signature_and_source_epoch_checkpoint_counter(self):
  bad=self.sign(self.publication);bad['payload']['successor']['epoch']='forged'
  with self.assertRaises(Exception):validate_parent(bad,self.binding,self.authority)
  for field,value in [('source_sha256','ff'*32),('epoch','second'),('optimizer_step_before',9),('checkpoint','ff'*32)]:
   p=copy.deepcopy(self.publication);p['successor'][field]=value;env,b=self.changed(p)
   with self.assertRaisesRegex(ValueError,'first-successor'):validate_parent(env,b,self.authority)
  j=copy.deepcopy(self.job);j['source_files'][MODULE]='fe'*32
  with self.assertRaisesRegex(ValueError,'local successor source'):validate_successor(self.sign(self.publication),j,self.manifest,self.authority)
 def test_draft_capability_preparation_cannot_consume_local_parent(self):
  draft=copy.deepcopy(self.job);draft.pop('unaudited_training_execution')
  self.assertEqual(validate_successor(self.sign(self.publication),draft,self.manifest,self.authority,preparation=True),self.publication)
  with self.assertRaises(ValueError):validate_successor(self.sign(self.publication),draft,self.manifest,self.authority)
  with self.cache(draft) as cache:
   with self.assertRaises(ValueError):cache.prepare_parent(self.descriptor,'33'*32)
  self.assertFalse((self.workspace/'.optimizer-state-cache/research-parent-use.json').exists())
  wrong=copy.deepcopy(self.job);d=wrong['unaudited_training_execution']['payload'];d['training_source_bundle']['sha256']='ee'*32;wrong['unaudited_training_execution']=self.sign(d)
  with self.assertRaisesRegex(ValueError,'signed retained-SFT'):validate_successor(self.sign(self.publication),wrong,self.manifest,self.authority,preparation=True)
 def test_historical_parent_validation_has_no_adoption_import(self):
  import builtins
  old=dict(version='authority-local-trainer-state-v1',namespace='private/trainer-state/old/original',job_id='original',job_sha256='ab'*32,
   descriptor_sha256=sha(self.descriptor),descriptor=self.descriptor)
  binding=copy.deepcopy(self.binding);binding['parent']=state_pointer(old);original_import=builtins.__import__
  def checked(name,*args,**kwargs):
   if 'research_parent_adoption' in name:raise AssertionError('historical path imported new adoption')
   return original_import(name,*args,**kwargs)
  with patch('builtins.__import__',side_effect=checked):self.assertEqual(validate_parent(self.sign(old),binding,self.authority),self.descriptor)
 def test_loader_replaces_untrusted_bootstrap_adoption_module(self):
  import subprocess,sys
  script=r"""import sys,types,tempfile,pathlib,importlib
sys.path.insert(0,sys.argv[1])
from subnet import backend_jobs
with tempfile.TemporaryDirectory() as td:
 root=pathlib.Path(td);(root/'subnet').mkdir()
 (root/'subnet/research_parent_adoption.py').write_text('MARKER="pinned"\n')
 stale=types.ModuleType('subnet.research_parent_adoption');stale.MARKER='stale'
 sys.modules['subnet.research_parent_adoption']=stale
 backend_jobs.install_source_loader(root,('subnet/research_parent_adoption.py',))
 fresh=importlib.import_module('subnet.research_parent_adoption')
 assert fresh.MARKER=='pinned' and pathlib.Path(fresh.__file__).parent==root/'subnet'
"""
  result=subprocess.run([sys.executable,'-B','-c',script,str(Path(__file__).resolve().parents[1])],capture_output=True,text=True,timeout=30)
  self.assertEqual(result.returncode,0,result.stderr)
 def test_wrong_genesis_and_counter(self):
  for field,value in [('genesis_sha256','ef'*32),('global_step_before',9)]:
   b=copy.deepcopy(self.binding);b[field]=value
   with self.assertRaisesRegex(ValueError,'counter and genesis'):validate_parent(self.sign(self.publication),b,self.authority)
 def test_actual_host_and_workspace_binding(self):
  with patch('subnet.research_parent_adoption.socket.gethostname',return_value='other'):
   with self.cache() as c:
    with self.assertRaisesRegex(ValueError,'actual host'):c.prepare_parent(self.descriptor,'33'*32)
  other=self.root/'other';other.mkdir()
  with StateCache(other,self.job,self.manifest,self.authority) as c:
   with self.assertRaisesRegex(ValueError,'actual host'):c.prepare_parent(self.descriptor,'33'*32)
 def test_shard_mutation_hardlink_descriptor_rejected_and_preserved(self):
  p=next(self.retained.glob('*.safetensors'));original=p.read_bytes();p.write_bytes(original[:-1]+bytes([original[-1]^1]))
  with self.cache() as c:
   with self.assertRaisesRegex(ValueError,'shard bytes changed'):c.prepare_parent(self.descriptor,'33'*32)
  p.write_bytes(original);os.link(p,self.root/'hardlink')
  with self.cache() as c:
   with self.assertRaisesRegex(ValueError,'single-link'):c.prepare_parent(self.descriptor,'33'*32)
  (self.root/'hardlink').unlink();self.write(self.retained/'descriptor.json',{})
  with self.cache() as c:
   with self.assertRaisesRegex(ValueError,'descriptor bytes'):c.prepare_parent(self.descriptor,'33'*32)
  self.assertTrue(p.exists())
 def test_pending_partial_output_or_current_state_never_discarded(self):
  for name in ('pending.json','current.json','promotion.json'):
   with self.cache() as c:
    p=c.root/name;self.write(p,{'original':'keep'})
    with self.assertRaisesRegex(ValueError,'existing production state'):c.prepare_parent(self.descriptor,'33'*32)
    self.assertEqual(json.loads(p.read_bytes()),{'original':'keep'});p.unlink()
 def test_fd_mutation_after_admission_rejected(self):
  with self.cache() as c:
   c.prepare_parent(self.descriptor,'33'*32);row=self.descriptor['shards'][0];target=self.workspace/'fresh';receipt=c.fetch(row['name'],target,self.forbidden)
   try:
    p=self.retained/row['name'];p.write_bytes(p.read_bytes())
    with self.assertRaisesRegex(ValueError,'retained local parent'):receipt.validate(target,row,c)
   finally:receipt.close()
 def test_normal_successor_ack_takes_over_and_blocks_readoption(self):
  original={p.name:p.read_bytes() for p in self.retained.iterdir()}
  with self.cache() as c:
   c.prepare_parent(self.descriptor,'33'*32)
   restore=self.workspace/'before-successor';restore.mkdir()
   state,_=restore_state(self.descriptor,sha(self.descriptor),'22'*32,self.inventory,workspace=restore,
    fetch_shard=lambda n,p:c.fetch(n,p,self.forbidden),resource_admission=admit_resources(restore,self.plan),owned_cache=c)
   self.optimizer=PersistentCPUAdamW(self.params,'22'*32,restored=state,resource_admission=self.admission)
   self.params[0][1].grad=torch.ones_like(self.params[0][1]);self.optimizer.step();c.begin_candidate()
   out=self.workspace/'jobs'/self.job['job_id'];out.mkdir(parents=True)
   d,_=export_state(self.optimizer,epoch='first-production',inference_checkpoint='44'*32,workspace=out,publish_shard=self.forbidden,readback_shard=self.forbidden,
    commit_descriptor=lambda v:dict(descriptor_sha256=sha(v),local_sha_verified=True,durable_readback_verified=False,authority_committed=False),
    resource_admission=self.admission,shard_bytes=HEADER_RESERVE+1024,readback_mode='trainer-local-only-v1',retain_shard=c.retain)
   self.assertEqual(d['parent_state_sha256'],sha(self.descriptor));self.assertEqual(d['optimizer_steps'],9)
   c.finish(d)
  self.write(self.workspace/(self.job['job_id']+'.json'),self.sign(self.job));namespace='private/trainer-state/first-production/'+self.job['job_id']
  report=dict(success=True,new_checkpoint=dict(id='44'*32),persistent_training_state=dict(descriptor=d,descriptor_sha256=sha(d),namespace=namespace));self.write(out/'report.json',report)
  ack=self.sign(dict(version='durable-original-trainer-cache-ACK-v1',job_id=self.job['job_id'],job_sha256=sha(self.job),report_sha256=sha(report),new_checkpoint=report['new_checkpoint'],
   authority_state_committed=True,trainer_state=dict(descriptor_sha256=sha(d),optimizer_steps=9,namespace=namespace)))
  self.assertTrue(promote(ack,self.authority,self.workspace)['promoted'])
  current=json.loads((self.workspace/'.optimizer-state-cache/current.json').read_bytes());self.assertEqual(committed_cache_descriptor(current,self.authority,self.workspace),d)
  with self.cache() as c:
   with self.assertRaisesRegex(ValueError,'existing production state'):c.prepare_parent(self.descriptor,'33'*32)
  j=dict(self.job,job_id='next-real',persistent_training=dict(output_shards={'state-000000.safetensors':{}}))
  with self.cache(j) as c:self.assertGreater(c.prepare_parent(d,'33'*32),0)
  self.assertEqual(original,{p.name:p.read_bytes() for p in self.retained.iterdir()})

if __name__=='__main__':unittest.main()
