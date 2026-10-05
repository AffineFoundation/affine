import base64,copy,json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from test_commitment_freeze_prefetch import BoundedCommitmentFreeze
from subnet import commitment_transport as c
from subnet.capture_status import InfrastructureSkipped,close_epoch
from subnet.remote_backend import RemoteController
from subnet.storage import Identity,canonical
from subnet.backend_jobs import signed

class Controls(unittest.TestCase):
 def fixture(self):
  g,ids,j=BoundedCommitmentFreeze().fixture(4);g.epochs['e']['commitment_binding']['freeze_until']=30
  g.freeze=lambda e:c.freeze(g,e)
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
  controller=RemoteController.__new__(RemoteController);controller.state=Path(temp.name);controller.bucket=g.bucket;controller.gateway=g;controller.authority=Identity();controller.jobs=SimpleNamespace(run=Mock(side_effect=AssertionError('no GPU/inference/audit/training')))
  controller.signed=lambda p:dict(payload=p,signer=controller.authority.id,signature=base64.b64encode(controller.authority.key.sign(canonical(p)).signature).decode())
  manifest=dict(epoch='e',payable=False,start=0,deadline=20,checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64},submission_transport_policy=c.VERSION,hourly_execution_policy=dict(version='bounded-hourly-phases-v1',mine_seconds=20,freeze_seconds=10,audit_seconds=20,train_publication_seconds=0,weight_seconds=0,slack_seconds=0))
  status=dict(active=dict(epoch='e',phase='collect'),round=11,checkpoint={'id':'a'*64},checkpoint_path='/original/CP',training_steps=21,trainer_state=dict(optimizer_steps=3,inference_checkpoint='a'*64))
  return controller,manifest,status,g,ids
 def check_closed(self,controller,manifest,status,g):
  before=copy.deepcopy(status)
  with self.assertRaises(InfrastructureSkipped):controller.finalize(manifest,'/original/CP')
  capture=signed(json.loads((controller.state/'e-capture-status.json').read_text()),controller.authority.id)
  self.assertEqual(capture['status'],'metadata_incomplete');self.assertFalse(capture['complete']);self.assertFalse(capture['verification_claim']);self.assertFalse(capture['rewards_eligible'])
  close_epoch(controller,manifest,status,controller.state/'controller.json','test-stream')
  self.assertIsNone(status['active']);self.assertEqual(status['round'],12)
  for key in ('checkpoint','checkpoint_path','training_steps','trainer_state'):self.assertEqual(status[key],before[key])
  self.assertNotIn('frozen_receipts',g.epochs['e']);controller.jobs.run.assert_not_called()
  for label in ('scores','audit-challenge','audit-manifest','training-metrics','reward-publication-ready','verified'):
   self.assertFalse((controller.state/('e-'+label+'.json')).exists(),label)
  self.assertNotIn('public/e/receipts.json',g.bucket.objects)
  self.assertNotIn('public/e/scores.json',g.bucket.objects)
  return capture
 def test_expired_before_discovery_closes_without_fake_empty_receipts_or_update(self):
  controller,m,status,g,ids=self.fixture()
  with patch('time.time',return_value=31):capture=self.check_closed(controller,m,status,g)
  self.assertFalse(capture['discovery_complete']);self.assertEqual(set(capture['unresolved_miners']),set(i.id for i in ids));self.assertEqual(capture['known_captured'],[])
 def test_actual_service_collect_closes_then_next_run_reaches_fresh_opening(self):
  from subnet.gpu_service import run
  controller,m,status,g,ids=self.fixture();status['initial_published']=True
  (controller.state/'controller.json').write_bytes(canonical(status));(controller.state/'e-manifest.json').write_bytes(canonical(m))
  controller.train=Mock(side_effect=AssertionError('no training'));controller.open=Mock(side_effect=RuntimeError('reached fresh opening'))
  config=dict(state=str(controller.state),bucket={},remote={},source_bundle={},epoch_prefix='nonpayable-capture-control',registration_allowlist=[ids[0].id],owned_miner_dispatch=False)
  chain=SimpleNamespace(registrations=Mock(return_value={'hotkey':dict(public_key=ids[0].id)}))
  with patch('time.time',return_value=31),patch('subnet.gpu_service.Bucket',return_value=g.bucket),patch('subnet.gpu_service.Gateway',return_value=g),patch('subnet.gpu_service.RemoteController',return_value=controller),patch('subnet.gpu_service.ChainAdapter',return_value=chain),patch('subnet.gpu_service.evaluate',side_effect=AssertionError('no eval')),patch('subnet.reward_publication.emit',side_effect=AssertionError('no reward')):
   run(config,once=True)
   closed=json.loads((controller.state/'controller.json').read_text());self.assertIsNone(closed['active']);self.assertEqual(closed['round'],12);self.assertEqual(closed['checkpoint'],status['checkpoint']);self.assertEqual(closed['training_steps'],21);self.assertEqual(closed['trainer_state'],status['trainer_state'])
   with patch('subnet.gpu_service.contract',return_value={'training_policy':'test-prospective'}),patch('subnet.gpu_service.log.exception'),self.assertRaisesRegex(RuntimeError,'reached fresh opening'):run(config,once=True)
  controller.open.assert_called_once();self.assertEqual(controller.open.call_args.args[1],status['checkpoint']);self.assertTrue(controller.open.call_args.args[0].endswith('-12'));controller.train.assert_not_called();controller.jobs.run.assert_not_called()
 def test_list_failure_crossing_cutoff_closes_and_preserves_original_journal(self):
  controller,m,status,g,ids=self.fixture();clock=[29]
  def fail(*a):clock[0]=31;raise RuntimeError('actual listing unavailable')
  g.bucket.complete_commitment_listing=fail
  with patch('time.time',side_effect=lambda:clock[0]):capture=self.check_closed(controller,m,status,g)
  self.assertEqual(capture['metadata_failure']['reason'],'discovery_infrastructure_incomplete')
 def test_partial_GET_failure_after_cutoff_preserves_known_hashes_no_retry_restart(self):
  controller,m,status,g,ids=self.fixture();clock=[29];failed=ids[-1].id;old=g.bucket.get_object;barrier=threading.Barrier(4)
  def read(**kw):
   barrier.wait(timeout=5)
   if failed in kw['Key']:clock[0]=31;raise RuntimeError('actual read unavailable')
   return old(**kw)
  g.bucket.get_object=read
  with patch('time.time',side_effect=lambda:clock[0]):capture=self.check_closed(controller,m,status,g)
  self.assertEqual(len(capture['known_captured']),3);self.assertEqual(capture['unresolved_miners'],[failed]);self.assertEqual(len(g.epochs['e']['commitment_pending']),3)
  with patch('time.time',return_value=40),patch.object(g,'freeze',side_effect=AssertionError('no forever retry')):
   with self.assertRaises(InfrastructureSkipped):controller.finalize(m,'/original/CP')
  self.assertEqual(signed(json.loads((controller.state/'e-capture-status.json').read_text()),controller.authority.id),capture)
 def test_before_cutoff_retries_and_historical_without_policy_is_unchanged(self):
  for historical in (False,True):
   controller,m,status,g,ids=self.fixture()
   if historical:m.pop('hourly_execution_policy');g.epochs['e']['commitment_binding'].pop('freeze_until')
   g.bucket.complete_commitment_listing=lambda *a:(_ for _ in()).throw(RuntimeError('transient'))
   with patch('time.time',return_value=29),self.assertRaises(c.FreezeMetadataIncomplete):controller.finalize(m,'/original/CP')
   self.assertFalse((controller.state/'e-capture-status.json').exists());self.assertEqual(status['round'],11)
 def test_public_failure_keeps_same_original_status_and_only_advances_after_durable_publication(self):
  controller,m,status,g,ids=self.fixture();original=g.bucket.json
  def outage(key,value):
   if key.endswith('infrastructure-skips.json'):raise RuntimeError('public status unavailable')
   return original(key,value)
  with patch('time.time',return_value=31):
   with self.assertRaises(InfrastructureSkipped):controller.finalize(m,'/original/CP')
   g.bucket.json=outage
   with self.assertRaises(RuntimeError):close_epoch(controller,m,status,controller.state/'controller.json','test-stream')
   self.assertEqual(status['round'],11);self.assertEqual(status['active']['epoch'],'e')
   g.bucket.json=original;close_epoch(controller,m,status,controller.state/'controller.json','test-stream')
  self.assertEqual(status['round'],12);self.assertEqual(len(json.loads((controller.state/'infrastructure-skipped-epochs.json').read_text())),1)
