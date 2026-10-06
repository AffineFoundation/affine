import base64,copy,hashlib,json,tempfile,time,unittest
from pathlib import Path
from nacl.signing import SigningKey
from subnet.backend_jobs import canonical,file_map
from subnet.cache_lifecycle import CacheLifecycle
from ops.owned_cached_group_retention import GroupRetention,retire_group,validate_scope,capacity_admission,VERSION
from ops.owned_cached_larger_cohort import POLICY,SOURCE
class GroupRetentionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.files={f'subnet/m{i}.py':hashlib.sha256(str(i).encode()).hexdigest()for i in range(177)};self.files.pop('subnet/m0.py');self.files['subnet/cache_lifecycle.py']='86bd543f85d39e0b8e42fbed10ff2c814aaedd55fce0bb2427107bf332227dfa';cpfiles={'model.safetensors':hashlib.sha256(b'owned').hexdigest(),'config.json':hashlib.sha256(b'{}').hexdigest()};self.cp={'files':cpfiles,'id':file_map(cpfiles)};directory=self.root/'checkpoints'/self.cp['id'];directory.mkdir(parents=True);(directory/'model.safetensors').write_bytes(b'owned');(directory/'config.json').write_bytes(b'{}');self.cache=CacheLifecycle(self.root)
  with self.cache.lease_checkpoint(self.cp['id']):self.cache.record_checkpoint(self.cp['id'],cpfiles)
  self.groups=[dict(group=g,indices=list(range(g*32,(g+1)*32)),seeds=[20261002+i*1000 for i in range(g*32,(g+1)*32)],harness={'max_output_tokens':1024})for g in range(4)];self.acks=[];bindings={};(self.root/'runner-status').mkdir()
  for g in self.groups:
   jid='job'+str(g['group']);m={'checkpoint':self.cp,'source_bundle':{'sha256':SOURCE}};job={'job_id':jid,'manifest':self.sign(m),'source_files':self.files,'role':'evaluate','owned_evaluation_policy':POLICY,'heldout':[{k:v for k,v in g.items()if k!='group'}]};sha=hashlib.sha256(canonical(job)).hexdigest();report={'job_id':jid,'job_sha256':sha,'checkpoint':self.cp['id'],'success':True};ack={'version':'owned-cached-evaluation-durable-ack-v1','original_job':self.sign(job),'original_report':report,'durable_report_full_readback':True,'workspace':str(self.root),'checkpoint':self.cp,'job_sha256':sha,'report_sha256':hashlib.sha256(canonical(report)).hexdigest()};self.acks.append(self.sign(ack));bindings[jid]={'group':g['group'],'job_sha256':sha};(self.root/'runner-status'/(jid+'.json')).write_text(json.dumps({'job_id':jid,'phase':'complete','exit_code':0}))
  self.scope={'version':VERSION,'execute_allowed':True,'minimum_free_cold_bytes':20000000000,'minimum_free_warm_margin_bytes':5000000000,'created_at':1,'expires_at':7200,'workspace':str(self.root),'source_sha256':SOURCE,'source_files':self.files,'experiment_id':'owned-cached-native-heldout128-cap1024-v1','groups':self.groups,'original_jobs':bindings,'checkpoint':self.cp}
 def sign(self,p):return {'payload':p,'signer':self.authority,'signature':base64.b64encode(self.key.sign(canonical(p)).signature).decode()}
 def retire(self,acks=None):return retire_group(self.sign(self.scope),self.acks if acks is None else acks,self.authority,self.root,self.files,now=2,live=lambda s:s.get('test_live',False))
 def test_namespace_group_lease_does_not_block_frozen_checkpoint_lease(self):
  with GroupRetention(self.sign(self.scope),self.authority,self.root,self.files,now=2):
   with self.cache.lease_checkpoint(self.cp['id'],blocking=False):pass
   with self.assertRaises(BlockingIOError):
    with GroupRetention(self.sign(self.scope),self.authority,self.root,self.files,now=2):pass
 def test_all_four_real_ACKs_remove_exact_owned_model_once(self):
  result=self.retire();self.assertEqual(result['status'],'complete');self.assertEqual(result['removed'],[self.cp['id']]);self.assertTrue(self.retire()['already_absent'])
 def test_missing_or_duplicate_ACKs_keep_model(self):
  for acks in (self.acks[:3],[self.acks[0]]*4):
   with self.assertRaises(ValueError):self.retire(acks)
   self.assertTrue((self.root/'checkpoints'/self.cp['id']).exists())
 def test_actual_live_original_or_checkpoint_lease_defers(self):
  marker=self.root/'runner-status/job2.json';d=json.loads(marker.read_bytes());d['test_live']=True;marker.write_text(json.dumps(d));self.assertEqual(self.retire()['reason'],'original-still-live');d['test_live']=False;marker.write_text(json.dumps(d))
  with self.cache.lease_checkpoint(self.cp['id']):self.assertEqual(self.retire()['status'],'deferred')
 def test_unsigned_false_expired_wrong_source_and_changed_bytes_fail(self):
  for key,val in [('execute_allowed',False),('expires_at',1),('source_sha256','old')]:
   s=copy.deepcopy(self.scope);s[key]=val
   with self.assertRaises(ValueError):validate_scope(self.sign(s),self.authority,self.root,self.files,now=2)
  (self.root/'checkpoints'/self.cp['id']/'model.safetensors').write_bytes(b'changed-owned-bytes');self.assertEqual(self.retire()['status'],'deferred')
 def test_failed_terminal_original_can_dispose_after_authentic_failure_ACK_no_model_score(self):
  ack=copy.deepcopy(self.acks[2]['payload']);ack['original_report']['success']=False;ack['report_sha256']=hashlib.sha256(canonical(ack['original_report'])).hexdigest();acks=self.acks.copy();acks[2]=self.sign(ack);marker=self.root/'runner-status/job2.json';d=json.loads(marker.read_bytes());d['exit_code']=1;d['phase']='failed';marker.write_text(json.dumps(d));r=self.retire(acks);self.assertEqual(r['failed_originals'],['job2']);self.assertIsNone(r['model_reward']);self.assertEqual(r['status'],'complete')

 def test_owned_capacity_accounting_requires_active_group_exact_receipt(self):
  lease=GroupRetention(self.sign(self.scope),self.authority,self.root,self.files,now=2)
  with self.assertRaises(ValueError):capacity_admission(lease,20000000000)
  with lease:
   r=capacity_admission(lease,20000000000);self.assertEqual(r['existing_owned_model_bytes'],7);self.assertEqual(r['required_free_bytes'],19999999993);self.assertTrue(r['admitted']);self.assertFalse(capacity_admission(lease,10000000000)['admitted'])
   (self.root/'checkpoints'/self.cp['id']/'model.safetensors').write_bytes(b'changed')
   with self.assertRaises(ValueError):capacity_admission(lease,20000000000)
 def test_external_cache_or_changed_ack_report_cannot_retire_owned_files(self):
  for change in ('external','report','falseACK'):
   acks=copy.deepcopy(self.acks);a=acks[0]['payload'];job=a['original_job']['payload']
   if change=='external':job['checkpoint_cache']='/external'
   if change=='report':a['original_report']['job_id']='different'
   if change=='falseACK':a['durable_report_full_readback']=False
   a['original_job']=self.sign(job);acks[0]=self.sign(a)
   with self.assertRaises(ValueError):self.retire(acks)
   self.assertTrue((self.root/'checkpoints'/self.cp['id']).exists())
