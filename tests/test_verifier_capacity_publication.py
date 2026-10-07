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
 def test_lifetime_revocation_guard_blocks_before_any_metadata_tick(self):
  from nacl.signing import SigningKey
  from subnet.backend_jobs import signed
  from ops.verifier_capacity_publication import main
  key=SigningKey.generate();seed=self.fx.root/'test-only-seed';seed.write_text(key.encode().hex());seed.chmod(0o600)
  policy=self.fx.root/'test-only-policy';policy.write_bytes(canonical(dict(payload=self.v,signer=key.verify_key.encode().hex(),signature=__import__('base64').b64encode(key.sign(canonical(self.v)).signature).decode())))
  args=SimpleNamespace(policy=str(policy),authority=key.verify_key.encode().hex(),seed_file=str(seed),publication_state='never-read',controller_state='never-read',queue='never-read',once=False)
  def revoked():raise ValueError('revoked')
  with patch('argparse.ArgumentParser.parse_args',return_value=args),patch('ops.verifier_capacity_publication.backfill')as fill,patch('ops.verifier_capacity_publication.flush')as ship:
   with self.assertRaisesRegex(ValueError,'revoked'):main(guard=revoked)
   fill.assert_not_called();ship.assert_not_called()
 def test_additive_eighth_route_preserves_original_seven_receipts(self):
  grant=enqueue(self.c,self.policy,self.cp,self.staged);self.assertEqual(len(flush(self.c,self.policy,replicate=self.replicate)),7)
  receipts=Path(self.v['outbox'])/self.cp['id'];original={p:p.read_bytes()for p in receipts.glob('*.json')}
  extended=copy.deepcopy(self.v);extended['replicas']['9']={'new':True};policy=self.fx.sign(extended)
  self.assertEqual(enqueue(self.c,policy,self.cp,self.staged),grant);self.assertEqual(flush(self.c,policy,replicate=self.replicate),[dict(replica='9',status='complete')]);self.assertEqual(flush(self.c,policy,replicate=self.replicate),[])
  self.assertTrue(all(p.read_bytes()==raw for p,raw in original.items()));self.assertEqual(set(p.stem for p in receipts.glob('*.json')),{'1','2','3','4','5','6','8','9'})
 def test_eighth_route_cannot_remove_original_or_reactivate_retired_seven(self):
  from ops.verifier_capacity_publication import checked_policy
  for remove,extra in [('4','9'),(None,'7')]:
   value=copy.deepcopy(self.v)
   if remove is not None:del value['replicas'][remove]
   value['replicas'][extra]={}
   with self.subTest(remove=remove,extra=extra),self.assertRaises(ValueError):checked_policy(self.fx.sign(value),self.fx.authority)
 def test_eighth_route_requires_authenticated_policy(self):
  from ops.verifier_capacity_publication import checked_policy
  envelope=copy.deepcopy(self.policy);envelope['payload']['replicas']['9']={}
  with self.assertRaises(Exception):checked_policy(envelope,self.fx.authority)
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
  roles=self.fx.root/'roles';roles.mkdir();(self.fx.root/'controller.json').write_bytes(canonical(dict(persistent_state_committed=True,checkpoint=dict(self.cp,descriptor_key='public/checkpoints/'+self.cp['id']+'/authorities/'+self.fx.authority+'/checkpoint.json'))));(roles/'original-checkpoint-publication.json').write_bytes(canonical(self.staged));self.assertEqual(len(backfill(self.c,self.policy,publication_state=roles,controller_state=self.fx.root/'controller.json')),1);self.assertEqual(len(flush(self.c,self.policy,replicate=self.replicate)),7)
  (roles/'failed-partial.json').write_text('not a complete publication');self.assertEqual(len(backfill(self.c,self.policy,publication_state=roles,controller_state=self.fx.root/'controller.json')),1)
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

 def test_current_authority_advances_without_hardcoded_checkpoint_or_ancient_backfill(self):
  roles=self.fx.root/'roles';roles.mkdir()
  other=dict(self.cp,id='e'*64)
  for label,cp in [('old',self.cp),('new',other)]:
   (roles/(label+'-checkpoint-publication.json')).write_bytes(canonical(dict(self.staged,checkpoint=cp['id'])))
  state=self.fx.root/'controller.json'
  def save(cp):state.write_bytes(canonical(dict(persistent_state_committed=True,checkpoint=dict(cp,descriptor_key='public/checkpoints/'+cp['id']+'/authorities/'+self.fx.authority+'/checkpoint.json'))))
  save(self.cp);backfill(self.c,self.policy,publication_state=roles,controller_state=state);self.assertEqual(len(list(Path(self.v['outbox']).glob('*.json'))),1)
  save(other);backfill(self.c,self.policy,publication_state=roles,controller_state=state);self.assertEqual(len(list(Path(self.v['outbox']).glob('*.json'))),2)
 def test_exclusive_publication_crash_beforelink_never_leaves_invalid_final(self):
  from ops.verifier_capacity_publication import publish_exclusive
  target=self.fx.root/'immutable.json'
  with patch('ops.verifier_capacity_publication.os.link',side_effect=RuntimeError('crash before link')):
   with self.assertRaises(RuntimeError):publish_exclusive(target,{'complete':True})
  self.assertFalse(target.exists());self.assertEqual(list(self.fx.root.glob('.immutable.json.writing-*')),[]);publish_exclusive(target,{'complete':True})
  with self.assertRaises(FileExistsError):publish_exclusive(target,{'changed':True})
  self.assertEqual(json.loads(target.read_bytes()),{'complete':True})
 def test_authenticated_live_historical_queue_is_included_expired_and_ancient_are_not(self):
  import sqlite3
  from ops.verifier_capacity_publication import current_published_models
  publications=self.fx.root/'publications';publications.mkdir();state=self.fx.root/'authority.json';old=dict(self.cp,id='e'*64);ancient=dict(self.cp,id='f'*64)
  state.write_bytes(canonical(dict(persistent_state_committed=True,checkpoint=dict(self.cp,descriptor_key='public/checkpoints/'+self.cp['id']+'/authorities/'+self.fx.authority+'/checkpoint.json'))))
  for label,cp in [('current',self.cp),('old',old),('ancient',ancient)]:
   (publications/(label+'-checkpoint-publication.json')).write_bytes(canonical(dict(self.staged,checkpoint=cp['id'])))
  queue=self.fx.root/'queue.sqlite3';db=sqlite3.connect(queue);db.execute('create table jobs(role text,status text,expires real,envelope text)')
  for cp,expiry in [(old,100),(ancient,9)]:
   job=copy.deepcopy(self.fx.job);manifest=copy.deepcopy(job['manifest']['payload']);manifest['checkpoint']=cp;job['manifest']=self.fx.sign(manifest)
   db.execute('insert into jobs values(?,?,?,?)',('verify','queued',expiry,canonical(self.fx.sign(job)).decode()))
  db.commit();db.close();original=queue.read_bytes();rows=list(current_published_models(publication_state=publications,controller_state=state,authority=self.fx.authority,queue_path=queue,now=10))
  self.assertEqual([cp['id']for cp,_ in rows],[self.cp['id'],old['id']]);self.assertEqual(queue.read_bytes(),original)
