import json,os,sys,time,tempfile,subprocess,io,datetime,unittest
from pathlib import Path
from types import SimpleNamespace
import test_bounded_token_capture as fixtures
from subnet.commitment_transport import canonical,sha
from subnet.training_documents import capture,capture_policy
from subnet.capture_journal import CaptureJournal,CommitmentJournal,JournalDurabilityError
V2=dict(fixtures.POLICY,version='bounded-parallel-token-capture-v2',journal_version='fsynced-per-epoch-capture-v1',state_checkpoint_documents=16)

def disk_gateway(root,create=False,count=1):
 root=Path(root);path=root/'gateway.json'
 if create:
  original,state,*_=fixtures.ParallelCaptureTests().gateway(count)
  state['commitment_binding']['learner_capture_policy']=V2.copy()
  state['commitment_binding']['freeze_until']=time.time()+.8
  root.mkdir(exist_ok=True);(root/'objects').mkdir()
  for miner,p in state['commitment_pending'].items():
   key='private/bounded-test/training/'+miner+'/0.json'
   response=original.bucket.client.get_object(Bucket='bucket',Key=key)
   (root/'objects'/sha(key.encode())).write_bytes(response['Body'].read())
  path.write_bytes(canonical(state))
 state=json.loads(path.read_bytes());calls=[];puts=[];persists=[]
 def get(**kw):
  from botocore.exceptions import ClientError
  calls.append(kw['Key']);p=root/'objects'/sha(kw['Key'].encode())
  if not p.exists():raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
  return dict(Body=io.BytesIO(p.read_bytes()),LastModified=datetime.datetime.fromtimestamp(state['start']+1,datetime.timezone.utc))
 def put(key,data):
  p=root/'objects'/sha(key.encode());p.write_bytes(data)
  with p.open('rb')as f:os.fsync(f.fileno())
  puts.append(key)
 def persist():
  persists.append(1);p=root/'gateway.tmp';p.write_bytes(canonical(state));p.replace(path)
 return SimpleNamespace(state_path=path,epochs={'bounded-test':state},bucket=SimpleNamespace(name='bucket',client=SimpleNamespace(get_object=get),put=put),persist=persist),calls,puts,persists

def crash_child(root,stage):
 g,*_=disk_gateway(root)
 if stage=='commit':
  original=CaptureJournal.commit
  def commit(self,miner,receipt):original(self,miner,receipt);os._exit(71)
  CaptureJournal.commit=commit
 else:
  original=g.bucket.put
  def put(key,data):original(key,data);os._exit(72)
  g.bucket.put=put
 capture(g,'bounded-test')

def commitment_gateway(root,create=False):
 g,calls,puts,persists=disk_gateway(root,create);s=g.epochs['bounded-test']
 if create:
  s['miners']=list(s['commitment_pending']);s['max_batches']=3
  s['commitment_binding'].update(version='small-commitment-pairs-v2',source='b'*64)
  for miner,p in s['commitment_pending'].items():
   key='private/bounded-test/commitments/'+miner+'.json'
   (Path(root)/'objects'/sha(key.encode())).write_bytes(canonical(p['document']))
  s['commitment_pending']={};s.pop('commitment_capture_complete');g.persist()
 original=g.bucket.client.get_object
 def get(**kw):r=original(**kw);r['ETag']='original-etag';return r
 g.bucket.client.get_object=get
 return g,calls,puts,persists

def commitment_crash_child(root):
 from subnet.commitment_transport import freeze
 g,*_=commitment_gateway(root);original=CommitmentJournal.accepted
 def accepted(self,miner,value):original(self,miner,value);os._exit(73)
 CommitmentJournal.accepted=accepted;freeze(g,'bounded-test')

class JournalTests(unittest.TestCase):
 def test_real_process_crash_after_fsynced_commit_replays_without_network_after_cutoff(self):
  self.crash_replay('commit',71,False)
 def test_real_process_crash_after_PUT_recovers_exact_frozen_intent_after_cutoff(self):
  self.crash_replay('put',72,True)
 def crash_replay(self,stage,code,recovery_get):
  with tempfile.TemporaryDirectory()as d:
   g,*_=disk_gateway(d,True)
   source=str(Path(__file__).resolve().parents[1]);tests=str(Path(__file__).parent)
   command="import sys;sys.path[:0]=%r;from test_capture_journal import crash_child;crash_child(%r,%r)"%([source,tests],d,stage)
   p=subprocess.run([sys.executable,'-I','-B','-c',command],timeout=5)
   self.assertEqual(p.returncode,code)
   while time.time()<g.epochs['bounded-test']['commitment_binding']['freeze_until']:time.sleep(.02)
   g,calls,puts,persists=disk_gateway(d);capture(g,'bounded-test')
   self.assertEqual(len(calls),int(recovery_get));self.assertTrue(all(k.startswith('public/')for k in calls));self.assertFalse(puts)
   state=g.epochs['bounded-test'];self.assertEqual(sum(len(v)for v in state['training_document_snapshots'].values()),1);self.assertFalse(any(state['training_document_deferred'].values()))
   self.assertEqual(sum(len(v)for v in json.loads(g.state_path.read_bytes())['training_document_snapshots'].values()),1)
 def test_checkpoint_bound_and_final_partial_flush(self):
  with tempfile.TemporaryDirectory()as d:
   g,calls,puts,persists=disk_gateway(d,True,35);g.epochs['bounded-test']['commitment_binding']['freeze_until']=time.time()+5
   capture(g,'bounded-test');self.assertEqual(len(puts),35);self.assertEqual(len(persists),3)
 def test_context_tamper_and_corrupt_row_refuse_before_GET(self):
  with tempfile.TemporaryDirectory()as d:
   g,calls,*_=disk_gateway(d,True);wal=CaptureJournal(g,'bounded-test');path=wal.path;wal.close()
   g.epochs['bounded-test']['commitment_binding']['checkpoint']='c'*64
   with self.assertRaises(ValueError):capture(g,'bounded-test')
   self.assertFalse(calls)
   g,calls,*_=disk_gateway(d);p=json.loads(path.read_bytes());p['sha256']='f'*64;path.write_bytes(canonical(p)+b'\n')
   with self.assertRaises(ValueError):capture(g,'bounded-test')
   self.assertFalse(calls)
 def test_symlink_and_concurrent_owner_refuse(self):
  with tempfile.TemporaryDirectory()as d:
   g,*_=disk_gateway(d,True);wal=CaptureJournal(g,'bounded-test')
   try:
    with self.assertRaises(BlockingIOError):CaptureJournal(g,'bounded-test')
   finally:wal.close()
   path=wal.path;data=path.read_bytes();path.unlink();target=Path(d)/'outside';target.write_bytes(data);path.symlink_to(target)
   with self.assertRaises(OSError):CaptureJournal(g,'bounded-test')
 def test_torn_tail_preserves_prior_durable_intent(self):
  with tempfile.TemporaryDirectory()as d:
   g,*_=disk_gateway(d,True);wal=CaptureJournal(g,'bounded-test');miner=next(iter(g.epochs['bounded-test']['commitment_pending']));p=g.epochs['bounded-test']['commitment_pending'][miner];b=p['document']['payload']['batches'][0]
   receipt=dict(slot=0,sha256=b['training_sha256'],size=b['training_size'],frozen_key=p['root']+'/training/0.json',captured_at=time.time(),assurance='unaudited');wal.intent(miner,receipt);path=wal.path;wal.close()
   with path.open('ab')as f:f.write(b'{"partial":');f.flush();os.fsync(f.fileno())
   wal=CaptureJournal(g,'bounded-test')
   try:self.assertEqual(wal.unresolved(),[(miner,receipt)]);self.assertTrue(path.read_bytes().endswith(b'\n'))
   finally:wal.close()
 def test_wrong_context_does_not_truncate_partial_tail(self):
  with tempfile.TemporaryDirectory()as d:
   g,*_=disk_gateway(d,True);wal=CaptureJournal(g,'bounded-test');path=wal.path;wal.close()
   with path.open('ab')as f:f.write(b'partial')
   before=path.read_bytes();g.epochs['bounded-test']['commitment_binding']['source']='e'*64
   with self.assertRaises(ValueError):CaptureJournal(g,'bounded-test')
   self.assertEqual(path.read_bytes(),before)
 def test_foreign_owned_lock_and_journal_file_each_refuse(self):
  from unittest.mock import patch
  for target in (0,1):
   with tempfile.TemporaryDirectory()as d:
    g,*_=disk_gateway(d,True);original=os.fstat;count=[0]
    def foreign(fd):
     st=original(fd);n=count[0];count[0]+=1
     if n==target:return SimpleNamespace(st_mode=st.st_mode,st_uid=st.st_uid+1,st_nlink=st.st_nlink,st_size=st.st_size)
     return st
    with patch('subnet.capture_journal.os.fstat',side_effect=foreign):
     with self.assertRaises(ValueError):CaptureJournal(g,'bounded-test')
 def test_wrong_declared_hash_and_nonintent_commit_refuse(self):
  with tempfile.TemporaryDirectory()as d:
   g,*_=disk_gateway(d,True);wal=CaptureJournal(g,'bounded-test');miner=next(iter(g.epochs['bounded-test']['commitment_pending']));p=g.epochs['bounded-test']['commitment_pending'][miner]
   receipt=dict(slot=0,sha256='f'*64,size=20,frozen_key=p['root']+'/training/0.json',captured_at=time.time(),assurance='unaudited')
   try:
    with self.assertRaises(JournalDurabilityError):wal.commit(miner,receipt)
   finally:wal.close()
 def test_unresolved_intent_with_corrupted_frozen_bytes_refuses_capture_completion(self):
  with tempfile.TemporaryDirectory()as d:
   g,calls,*_=disk_gateway(d,True);wal=CaptureJournal(g,'bounded-test');miner=next(iter(g.epochs['bounded-test']['commitment_pending']));p=g.epochs['bounded-test']['commitment_pending'][miner];b=p['document']['payload']['batches'][0]
   receipt=dict(slot=0,sha256=b['training_sha256'],size=b['training_size'],frozen_key=p['root']+'/training/0.json',captured_at=time.time(),assurance='unaudited');wal.intent(miner,receipt);wal.close()
   (Path(d)/'objects'/sha(receipt['frozen_key'].encode())).write_bytes(b'corrupt')
   with self.assertRaises(ValueError):capture(g,'bounded-test')
   self.assertNotIn('training_document_capture_complete',g.epochs['bounded-test']);self.assertFalse(g.epochs['bounded-test']['training_document_snapshots'])
 def test_global_checkpoint_conflict_fails_closed(self):
  with tempfile.TemporaryDirectory()as d:
   g,*_=disk_gateway(d,True);capture(g,'bounded-test');miner=next(iter(g.epochs['bounded-test']['training_document_snapshots']));g.epochs['bounded-test']['training_document_snapshots'][miner]['0']['sha256']='f'*64
   with self.assertRaises(ValueError):capture(g,'bounded-test')
 def test_v2_policy_propagates_exactly_into_signed_manifest_and_gateway(self):
  from unittest.mock import patch
  with patch('test_bounded_token_capture.POLICY',V2):
   fixtures.CaptureOpeningTests('test_first_signed_manifest_and_durable_gateway_bind_policy_copy').test_first_signed_manifest_and_durable_gateway_bind_policy_copy()
 def test_policy_batch_bool_or_over128_refuses(self):
  for n in (True,0,129):
   with self.assertRaises(ValueError):capture_policy(dict(V2,state_checkpoint_documents=n))

class CommitmentJournalTests(unittest.TestCase):
 def test_actual_process_crash_preserves_original_signed_commitment_after_cutoff(self):
  from unittest.mock import patch
  from subnet.commitment_transport import freeze
  with tempfile.TemporaryDirectory()as d:
   g,*_=commitment_gateway(d,True);source=str(Path(__file__).resolve().parents[1]);tests=str(Path(__file__).parent)
   code="import sys;sys.path[:0]=%r;from test_capture_journal import commitment_crash_child;commitment_crash_child(%r)"%([source,tests],d)
   p=subprocess.run([sys.executable,'-I','-B','-c',code],timeout=5);self.assertEqual(p.returncode,73)
   while time.time()<g.epochs['bounded-test']['commitment_binding']['freeze_until']:time.sleep(.02)
   g,calls,puts,persists=commitment_gateway(d)
   with patch('subnet.training_documents.capture',return_value={}),patch('subnet.training_documents.freeze_receipts',return_value={}):freeze(g,'bounded-test')
   self.assertFalse(calls);self.assertTrue(g.epochs['bounded-test']['commitment_capture_complete']);self.assertEqual(len(g.epochs['bounded-test']['commitment_pending']),1)
   row=next(iter(g.epochs['bounded-test']['commitment_pending'].values()));self.assertEqual(row['etag'],'original-etag');self.assertEqual(row['received_at'],datetime.datetime.fromtimestamp(g.epochs['bounded-test']['start']+1,datetime.timezone.utc).timestamp())
 def test_original_signature_mutation_rejected_even_rehashed_WAL(self):
  from unittest.mock import patch
  from subnet.commitment_transport import freeze
  with tempfile.TemporaryDirectory()as d:
   g,*_=commitment_gateway(d,True)
   with patch('subnet.training_documents.capture',return_value={}),patch('subnet.training_documents.freeze_receipts',return_value={}):freeze(g,'bounded-test')
   file=next((Path(d)/'capture-journals').glob('*.jsonl'));rows=[json.loads(v)for v in file.read_bytes().splitlines()]
   row=rows[1]['payload']['receipt'];row['document']['payload']['source']='c'*64;row['sha256']=sha(canonical(row['document']));row['size']=len(canonical(row['document']));row['root']='public/bounded-test/submissions/'+rows[1]['payload']['miner']+'/'+row['sha256'];rows[1]['sha256']=sha(canonical(rows[1]['payload']));file.write_bytes(b'\n'.join(canonical(v)for v in rows)+b'\n')
   with self.assertRaises(Exception):CommitmentJournal(g,'bounded-test')
 def test_foreign_owner_refuses_before_any_tail_mutation(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory()as d:
   g,*_=disk_gateway(d,True);wal=CaptureJournal(g,'bounded-test');path=wal.path;wal.close()
   with path.open('ab')as f:f.write(b'partial')
   before=path.read_bytes()
   with patch('subnet.capture_journal.os.getuid',return_value=os.getuid()+1):
    with self.assertRaises(ValueError):CaptureJournal(g,'bounded-test')
   self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
