import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from subnet.storage import canonical
from subnet.training_receipts import sha
from ops.verifier_capacity_publication import enqueue,flush,install,backfill,VERSION
from ops.verifier_capacity_admission import admit,VERSION as ADMIT
from test_verifier_capacity_admission import VerifierCapacityAdmission

class CapacityPublication(unittest.TestCase):
 def setUp(self):
  self.fx=VerifierCapacityAdmission();self.fx.setUp();self.addCleanup(self.fx.doCleanups);self.c=SimpleNamespace(authority=SimpleNamespace(id=self.fx.authority),signed=self.fx.sign)
  self.v=dict(version=VERSION,outbox=str(self.fx.root/'outbox'),replicas={k:{}for k in ('1','2','3','4','5','6','8')});self.policy=self.fx.sign(self.v)
  self.cp=dict(id=self.fx.cp,files={'model.safetensors':self.fx.sha});self.staged=dict(checkpoint=self.cp['id'],operator_independent_hashes=True,objects={'model.safetensors':dict(sha256=self.fx.sha,bytes=len(self.fx.bytes))})
 def replicate(self,grant,config):return dict(version='owned-capacity-inventory-install-v1',checkpoint_id=grant['payload']['checkpoint_id'],grant_sha256=sha(grant),installed=True)
 def test_default_off_publication_untouched(self):self.assertIsNone(enqueue(None,None,None,None));self.assertEqual(flush(None,None,replicate=None),[])
 def test_genuine_full_readback_metadata_signed_once_and_all7_replayed(self):
  grant=enqueue(self.c,self.policy,self.cp,self.staged);self.assertEqual(len(flush(self.c,self.policy,replicate=self.replicate)),7);self.assertEqual(flush(self.c,self.policy,replicate=self.replicate),[]);self.assertEqual(enqueue(self.c,self.policy,self.cp,self.staged),grant)
 def test_HEAD_partial_changed_bytes_or_oldmodel_mismatch_refused(self):
  for key,value in [('operator_independent_hashes',False),('checkpoint','f'*64)]:
   staged=copy.deepcopy(self.staged);staged[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):enqueue(self.c,self.policy,self.cp,staged)
  staged=copy.deepcopy(self.staged);staged['objects']['model.safetensors']['sha256']='0'*64
  with self.assertRaises(ValueError):enqueue(self.c,self.policy,self.cp,staged)
 def test_network_failure_pending_after_process_restart_then_7complete(self):
  grant=enqueue(self.c,self.policy,self.cp,self.staged)
  def fail(*a):raise TimeoutError()
  self.assertTrue(all(r['status']=='deferred'for r in flush(self.c,self.policy,replicate=fail)))
  self.assertEqual(json.loads((Path(self.v['outbox'])/(self.cp['id']+'.json')).read_bytes()),grant);self.assertEqual(len(flush(self.c,self.policy,replicate=self.replicate)),7)
 def test_directory_grant_admission_and_wrong_size_origin_refused(self):
  grant=enqueue(self.c,self.policy,self.cp,self.staged);directory=self.fx.root/'inventories';directory.mkdir();path=directory/(self.cp['id']+'.json');path.write_bytes(canonical(grant))
  value=copy.deepcopy(self.fx.value);value.pop('checkpoint_inventories');value['checkpoint_inventory_directory']=str(directory)
  result=admit(self.fx.job,self.fx.sign(value),self.fx.authority,lifecycle=self.fx.cache,free_bytes=lambda:10**11);self.assertEqual(result['model_total_bytes'],len(self.fx.bytes))
  grant['payload']['files']['model.safetensors']['size']+=1;path.write_bytes(canonical(grant))
  with self.assertRaises(Exception):admit(self.fx.job,self.fx.sign(value),self.fx.authority,lifecycle=self.fx.cache)
 def test_hook_original_publication_once_no_remote_shipping_or_resultchange(self):
  seen=[]
  class Remote:
   def commit_remote_checkpoint(self,*args):seen.append(args);return self.cp
  remote=Remote();remote.cp=self.cp;remote.controller=self.c;install(Remote,self.policy)
  self.assertEqual(remote.commit_remote_checkpoint({},self.staged),self.cp);self.assertEqual(len(seen),1)
 def test_idle_observer_backfills_only_completed_owned_publication(self):
  roles=self.fx.root/'roles';roles.mkdir();(roles/'original-checkpoint-publication.json').write_bytes(canonical(self.staged));self.assertEqual(len(backfill(self.c,self.policy,roles=roles)),1);self.assertEqual(len(flush(self.c,self.policy,replicate=self.replicate)),7)
  (roles/'failed-partial.json').write_text('not a complete publication');self.assertEqual(len(backfill(self.c,self.policy,roles=roles)),1)
 def test_eight_remote_calls_maximum_per_observer_tick(self):
  enqueue(self.c,self.policy,self.cp,self.staged);other=dict(id='e'*64,files=self.cp['files']);staged=dict(self.staged,checkpoint=other['id']);enqueue(self.c,self.policy,other,staged);calls=[]
  def fail(grant,config):calls.append(grant);raise TimeoutError()
  self.assertEqual(len(flush(self.c,self.policy,replicate=fail)),8);self.assertEqual(len(calls),8)
 def test_offline_old_grants_do_not_starve_new_checkpoint_replicas(self):
  for cp in ('a'*64,'b'*64,'c'*64):enqueue(self.c,self.policy,dict(self.cp,id=cp),dict(self.staged,checkpoint=cp))
  observed=[]
  def fail(grant,config):observed.append(grant['payload']['checkpoint_id']);raise TimeoutError()
  flush(self.c,self.policy,replicate=fail);flush(self.c,self.policy,replicate=fail);flush(self.c,self.policy,replicate=fail)
  self.assertEqual(set(observed),{'a'*64,'b'*64,'c'*64})
 def test_real_metadata_installer_rejects_bad_signature_and_preserves_existing_grant(self):
  import subprocess,sys
  from ops.verifier_capacity_publication import INSTALL_SCRIPT
  grant=enqueue(self.c,self.policy,self.cp,self.staged);directory=self.fx.root/'remote-metadata'
  payload=dict(grant=grant,authority=self.fx.authority,directory=str(directory));r=subprocess.run([sys.executable,'-c',INSTALL_SCRIPT],input=canonical(payload),capture_output=True)
  self.assertEqual(r.returncode,0,r.stderr);self.assertTrue(json.loads(r.stdout)['installed']);original=(directory/(self.cp['id']+'.json')).read_bytes()
  bad=copy.deepcopy(payload);bad['grant']['payload']['files']['model.safetensors']['size']+=1;r=subprocess.run([sys.executable,'-c',INSTALL_SCRIPT],input=canonical(bad),capture_output=True)
  self.assertNotEqual(r.returncode,0);self.assertEqual((directory/(self.cp['id']+'.json')).read_bytes(),original)
