import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from subnet.training_receipts import sha
from ops.failed_training_evidence_queue import enqueue,drain,RENEWAL_VERSION
from ops.failed_training_evidence_retention import retire,VERSION
from test_failed_training_evidence_retention import FailedEvidenceRetention

class DurableEvidenceQueue(unittest.TestCase):
 def setUp(self):
  self.f=FailedEvidenceRetention();self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.c=SimpleNamespace(authority=SimpleNamespace(id=self.f.authority),signed=self.f.sign)
  self.blueprint={k:self.f.value[k]for k in ('version','original_signed_job','original_terminal','directory','files','journal')}
  self.policy=dict(version=VERSION,approved_failures=[self.f.sign(self.blueprint)])
  self.queue=self.f.root/'durable-queue';self.ack=self.f.value['durable_recovery_ACK']
 def enroll(self):return enqueue(self.c,self.ack,self.policy,queue_path=self.queue,now=10)
 def tick(self,dispatch,idle=lambda:True,now=20):return drain(self.c,self.policy,queue_path=self.queue,idle=idle,dispatch=dispatch,now=now)
 def dispatch(self,grant,renewal):
  with patch('ops.training_retention.gpu_processes',return_value=[]),patch('ops.training_retention.processes',return_value=[]):return retire(grant,self.f.authority,workspace=self.f.root,archive=self.f.archive,policy={'version':VERSION},now=4000 if renewal else 20,renewal=renewal)
 def test_default_off_no_queue_reads(self):
  self.assertEqual(enqueue(self.c,None,None,queue_path='/not-owned'),[]);self.assertEqual(drain(self.c,None,queue_path='/not-owned',idle=None,dispatch=None),[])
 def test_restart_preserves_one_grant_and_real_ACK_not_nextcheckpoint(self):
  self.enroll();before=next(self.queue.glob('*.json')).read_bytes();ack=copy.deepcopy(self.ack['payload']);ack['input_checkpoint']={'id':'0'*64}
  enqueue(self.c,self.f.sign(ack),self.policy,queue_path=self.queue,now=200)
  self.assertEqual(before,next(self.queue.glob('*.json')).read_bytes());self.assertEqual(self.tick(self.dispatch)[0]['status'],'complete');self.assertEqual(self.tick(self.dispatch,now=1000),[])
 def test_no_idle_no_dispatch_then_later_observer_retries(self):
  self.enroll();seen=[];self.assertEqual(self.tick(lambda *x:seen.append(x),idle=lambda:False),[]);self.assertEqual(seen,[])
  self.assertEqual(self.tick(self.dispatch)[0]['status'],'complete')
 def test_network_exception_durable_backoff_then_retry(self):
  self.enroll()
  def fail(*args):raise TimeoutError('transport')
  self.assertEqual(self.tick(fail)[0]['error_type'],'TimeoutError');self.assertEqual(self.tick(self.dispatch,now=21),[]);self.assertEqual(self.tick(self.dispatch,now=60)[0]['status'],'complete')
 def test_same_immutable_journal_crash_and_expiry_renewal(self):
  self.enroll();original=Path.unlink
  def crash(p,*a,**k):
   if p.name=='state-000000.safetensors':original(p,*a,**k);raise RuntimeError('post-unlink crash')
   return original(p,*a,**k)
  with patch.object(Path,'unlink',crash):self.assertEqual(self.tick(self.dispatch)[0]['status'],'deferred')
  journal=Path(self.f.value['journal']).read_bytes();self.assertEqual(self.tick(self.dispatch,now=4000)[0]['status'],'complete');self.assertEqual(Path(self.f.value['journal']).read_bytes(),journal)
 def test_wronggrant_renewal_or_extendedduration_refuses(self):
  grant=self.f.sign(self.f.value)
  for delta in [{'grant_sha256':'0'*64},{'expires_at':8000},{'execute_allowed':False}]:
   renewal=dict(version=RENEWAL_VERSION,execute_allowed=True,grant_sha256=sha(grant),created_at=3990,expires_at=5000);renewal.update(delta)
   with self.subTest(delta=delta),self.assertRaises(ValueError):
    retire(grant,self.f.authority,workspace=self.f.root,archive=self.f.archive,policy={'version':VERSION},now=4000,renewal=self.f.sign(renewal))
  self.assertTrue(self.f.transfer.exists())
 def test_simultaneous_observers_dispatch_only_one(self):
  self.enroll();nested=[]
  def run_once(*args):nested.extend(self.tick(self.dispatch));return self.dispatch(*args)
  self.assertEqual(self.tick(run_once)[0]['status'],'complete');self.assertEqual(nested,[])
 def test_changed_blueprint_cannot_take_over_pending_journal(self):
  key=self.enroll()[0];p=self.queue/(key+'.json');doc=json.loads(p.read_bytes());doc['blueprint']['payload']['directory']='/other';p.write_text(json.dumps(doc))
  with self.assertRaises(ValueError):self.tick(self.dispatch)
