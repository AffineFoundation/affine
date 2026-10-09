"""Real RemoteController.open with selection projection and future capture/timing."""
import copy,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'tests'))
from test_remote_blacklist_opening import Opening
from subnet import controller,training_documents,gpu_service
from subnet.backend_jobs import COVERED_POLICY,signed
from subnet.learner_blacklist_selection import FIELD
from ops.trainer_lifecycle import fair_token_capture as fair
from ops.trainer_lifecycle import prospective_collection_timing as timing
BEFORE=dict(version='bounded-parallel-token-capture-v2',workers=8,max_document_bytes=2000000,max_inflight_bytes=16000000,completion_order='first-completed',journal_version='fsynced-per-epoch-capture-v1',state_checkpoint_documents=16)
OLD=dict(version='bounded-hourly-phases-v1',mine_seconds=600,freeze_seconds=60,audit_seconds=600,train_publication_seconds=2100,weight_seconds=120,slack_seconds=120);CURRENT=dict(OLD,mine_seconds=1800,audit_seconds=60,train_publication_seconds=1500,weight_seconds=60);FUTURE=dict(CURRENT,mine_seconds=1200,slack_seconds=720)
A=dict(version=timing.VERSION,first_round=91,previous_duration=600,duration=1800,previous_hourly_policy=OLD,hourly_policy=CURRENT,successor_first_round=93,successor_duration=1200,successor_hourly_policy=FUTURE)
class CaptureOpening(Opening):
 def exercise(self,n):
  self.epoch='nonpayable-capture-'+str(n);self.c.gateway.epochs={self.epoch:{'start':3700}}
  (self.state/'controller.json').write_text(json.dumps(dict(round=n,active=dict(epoch=self.epoch,phase='opening'))))
  self.policy=self.sign(dict(self.policy['payload'],target_round=n))
  config=dict(source_bundle=self.source,heldout=[dict(env_id='math',indices=[1])],training_policy=COVERED_POLICY,learner_capture_policy=copy.deepcopy(BEFORE),training_input_policy='committed-unaudited-training-v1',submission_transport_policy='small-commitment-pairs-v2',duration=1800,hourly_execution_policy=CURRENT,K=4,L=4,learner_blacklist_selection_policy=self.policy)
  row=dict(spec=controller.legacy_spec().to_dict(),indices=[0],harness=dict(version='text-tools-v1'));config['heldout'][0]['env_id']=row['spec']['id']
  implementation=W/'subnet/training_documents.py';service=SimpleNamespace(contract=gpu_service.contract);original_open=controller.Controller.open;original_globals=original_open.__globals__;original_save=controller.save_manifest;first=[]
  def save(path,value):first.append(copy.deepcopy(value));return original_save(path,value)
  with patch.object(training_documents,'capture',training_documents.capture),patch.object(training_documents,'capture_policy',training_documents.capture_policy),patch.object(training_documents,'freeze_receipts',training_documents.freeze_receipts),patch.object(controller,'save_manifest',save),patch.object(gpu_service,'definitions',return_value=[row]),patch('time.time',return_value=3700):
   fair.install(training_documents,implementation,hashlib.sha256(implementation.read_bytes()).hexdigest(),93,contract_module=service,previous_policy=BEFORE)
   timing.install(service,A)
   self.assertIs(controller.Controller.open,original_open);self.assertIs(controller.Controller.open.__globals__,original_globals)
   kw=service.contract(config,n);m=self.c.open(self.epoch,self.cp,['a'*64],max_batches=3,**kw)
  expected=dict(BEFORE,workers=16,max_inflight_bytes=32000000,state_checkpoint_documents=128)if n>=93 else BEFORE
  duration=1200 if n>=93 else 1800
  self.assertTrue(first);self.assertEqual(first[0][FIELD],self.policy);self.assertEqual(first[0]['max_batches'],3);self.assertEqual(first[0]['learner_capture_policy'],expected);self.assertEqual(first[0]['deadline']-first[0]['start'],duration)
  self.assertEqual(json.loads((self.state/(self.epoch+'-manifest.json')).read_bytes()),m)
  published=[a.args[1]for a in self.c.bucket.json.call_args_list if a.args[0].endswith('/manifest.json')];self.assertEqual(len(published),1);self.assertEqual(signed(published[0],self.auth),m)
  self.assertEqual(m['learner_capture_policy'],expected);self.assertEqual(m[FIELD],self.policy);self.assertEqual((m['K'],m['L'],m['max_batches']),(4,4,3));self.assertEqual(config['learner_capture_policy'],BEFORE)
 def test_real93_first_local_and_public_opening(self):self.exercise(93)
 def test_real92_retains_original_capture_and_window(self):self.exercise(92)
if __name__=='__main__':unittest.main()
