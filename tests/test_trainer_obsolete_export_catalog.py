import base64,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from ops import trainer_obsolete_export_catalog as m

canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
class CatalogTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  self.path=self.root/'jobs/old/checkpoint-persistent-final';self.path.mkdir(parents=True)
  self.content={'config.json':b'{}','model.safetensors':b'old-model'}
  for n,b in self.content.items():(self.path/n).write_bytes(b)
  files={n:hashlib.sha256(b).hexdigest()for n,b in self.content.items()};self.cp={'id':hashlib.sha256(canonical(files)).hexdigest(),'files':files}
  ack=dict(version='durable-original-trainer-cache-ACK-v1',authority_state_committed=True,job_id='current',input_checkpoint={'id':'a'*64},new_checkpoint={'id':'b'*64},trainer_state={'optimizer_steps':70,'descriptor_sha256':'c'*64})
  signedack=self.sign(ack)
  self.value=dict(version=m.VERSION,workspace=str(self.root),current_ack=signedack,candidates=[dict(directory=str(self.path),original_job=self.sign({'role':'train','job_id':'old'}),publication_job=self.sign({'role':'upload','manifest':self.sign({'checkpoint':self.cp})}),checkpoint=self.cp,readback=dict(kind='independent-full-R2-export-readback-v1',checkpoint=self.cp,objects={n:dict(full_get=True,sha256=h,size=len(self.content[n]),key='public/checkpoints/'+self.cp['id']+'/'+n)for n,h in files.items()}))])
  (self.root/'.cache-lifecycle').mkdir();self.write('.cache-lifecycle/trainer-current-state.json',ack['trainer_state'])
  (self.root/'.optimizer-state-cache').mkdir();self.write('.optimizer-state-cache/current.json',{'ROOT_ack':signedack,'job_id':'current'});self.write('.optimizer-state-cache/promotion.json',{'ack':signedack,'phase':'complete','child_terminal_confirmed':True});(self.root/'runner-status').mkdir()
  self.guard=patch.object(m,'assert_unreferenced');self.guard.start();self.addCleanup(self.guard.stop)
 def sign(self,d):return dict(payload=d,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(d)).signature).decode())
 def write(self,path,value):(self.root/path).write_bytes(canonical(value))
 def run_catalog(self,apply=False):return m.retire_catalog(self.sign(self.value),self.authority,self.root,apply=apply)
 def test_review_preserves_and_apply_retires_only_export(self):
  sentinel=self.root/'.optimizer-state-cache/state.bin';sentinel.write_bytes(b'master-moments');report=self.path.parent/'original.log';report.write_bytes(b'failure retained')
  before=sentinel.read_bytes();self.assertEqual(self.run_catalog()['status'],'reviewed');self.assertTrue(self.path.exists())
  self.assertEqual(self.run_catalog(True)['removed_checkpoints'],[self.cp['id']]);self.assertFalse(self.path.exists());self.assertEqual(sentinel.read_bytes(),before);self.assertTrue(report.exists());self.assertEqual(self.run_catalog(True)['already_absent'],[self.cp['id']])
 def test_consumed_catalog_noop_after_state_advances(self):
  self.run_catalog(True);self.write('.cache-lifecycle/trainer-current-state.json',{'optimizer_steps':71,'descriptor_sha256':'f'*64});self.write('.optimizer-state-cache/current.json',{'new_state':'must not be touched'});result=self.run_catalog(True);self.assertEqual(result['already_absent'],[self.cp['id']]);self.assertTrue(result['current_checkpoint_not_read'])
 def test_partly_consumed_stale_catalog_requires_refresh(self):
  self.write('.cache-lifecycle/trainer-current-state.json',{'optimizer_steps':71,'descriptor_sha256':'f'*64});self.assertRaises(m.StaleCatalogError,self.run_catalog,True);self.assertTrue(self.path.exists())
 def test_actual_reset_bound_retention_marker(self):
  ack=self.value['current_ack']['payload'];ack['trainer_state'].update(genesis_sha256='d'*64,inference_checkpoint='b'*64);envelope=self.sign(ack);self.value['current_ack']=envelope;self.write('.optimizer-state-cache/current.json',{'ROOT_ack':envelope,'job_id':'current'});self.write('.optimizer-state-cache/promotion.json',{'ack':envelope,'phase':'complete','child_terminal_confirmed':True});self.write('.cache-lifecycle/trainer-current-state.json',{'version':'reset-bound-trainer-retention-v1','ROOT_ack':envelope,'lineage':ack['trainer_state'],'retired_geneses':[],'transition_sha256':'e'*64});self.assertEqual(self.run_catalog()['ready_checkpoints'],[self.cp['id']])
 def test_unknown_marker_is_not_admitted(self):
  self.write('.cache-lifecycle/trainer-current-state.json',{'version':'unknown','optimizer_steps':70,'descriptor_sha256':'c'*64});self.assertRaises(ValueError,self.run_catalog)
 def test_incorrect_local_bytes_refuse_before_adoption(self):
  (self.path/'model.safetensors').write_bytes(b'wrong');self.assertRaises(ValueError,self.run_catalog,True);self.assertTrue(self.path.exists())
 def test_incomplete_r2_readback_refuses(self):
  self.value['candidates'][0]['readback']['objects']['config.json']['full_get']=False;self.assertRaises(ValueError,self.run_catalog,True)
 def test_wrong_archive_key_refuses(self):
  self.value['candidates'][0]['readback']['objects']['config.json']['key']='unrelated';self.assertRaises(ValueError,self.run_catalog)
 def test_wrong_publication_cp_refuses(self):
  self.value['candidates'][0]['publication_job']=self.sign({'role':'upload','manifest':self.sign({'checkpoint':dict(self.cp,id='d'*64)})});self.assertRaises(ValueError,self.run_catalog)
 def test_current_and_input_cp_protected(self):
  for field in ['input_checkpoint','new_checkpoint']:
   old=self.value['current_ack'];ack=dict(old['payload']);ack[field]={'id':self.cp['id']};self.value['current_ack']=self.sign(ack);self.assertRaises(ValueError,self.run_catalog);self.value['current_ack']=old
 def test_pending_optimizer_refuses(self):
  self.write('.optimizer-state-cache/pending.json',{});self.assertRaises(ValueError,self.run_catalog,True)
 def test_stale_step_refuses(self):
  self.write('.cache-lifecycle/trainer-current-state.json',{'optimizer_steps':71,'descriptor_sha256':'c'*64});self.assertRaises(ValueError,self.run_catalog)
 def test_unfinished_promotion_refuses(self):
  p=self.root/'.optimizer-state-cache/promotion.json';d=json.loads(p.read_bytes());d['phase']='pending';p.write_bytes(canonical(d));self.assertRaises(ValueError,self.run_catalog)
 def test_extra_member_refuses(self):
  (self.path/'surprise').write_bytes(b'x');self.assertRaises(ValueError,self.run_catalog)
 def test_symlink_and_hardlink_refuse(self):
  f=self.path/'config.json';b=f.read_bytes();f.unlink();f.symlink_to(self.path/'model.safetensors');self.assertRaises(ValueError,self.run_catalog);f.unlink();f.write_bytes(b);os.link(f,self.root/'alias');self.assertRaises(ValueError,self.run_catalog)
 def test_external_path_refuses(self):
  self.value['candidates'][0]['directory']=str(self.root/'.optimizer-state-cache');self.assertRaises(ValueError,self.run_catalog)
 def test_catalog_bad_signature_refuses(self):
  e=self.sign(self.value);e['payload']['workspace']+='other';self.assertRaises(Exception,m.retire_catalog,e,self.authority,self.root)
 def test_live_original_refuses(self):
  fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split();self.write('runner-status/live.json',{'runner_pid':os.getpid(),'runner_pid_ticks':fields[19]});self.assertRaises(ValueError,self.run_catalog)
 def test_same_bytes_multiple_paths_not_implicitly_evicted(self):
  unrelated=self.root/'checkpoints'/('e'*64);unrelated.mkdir(parents=True);(unrelated/'sentinel').write_bytes(b'retain');self.run_catalog(True);self.assertTrue((unrelated/'sentinel').exists())
 def test_duplicate_cp_refuses(self):
  self.value['candidates']*=2;self.assertRaises(ValueError,self.run_catalog)
 def test_process_reference_refuses(self):
  self.guard.stop()
  proc=self.root/'fake-proc';proc.mkdir();(proc/str(os.getpid())).symlink_to(Path('/proc')/str(os.getpid()),target_is_directory=True)
  with (self.path/'model.safetensors').open('rb'):
   self.assertRaises(ValueError,m.assert_unreferenced,[self.path],proc)
  self.guard.start()
if __name__=='__main__':unittest.main()
