"""Owned and external miners actually upload the same signed v2 framing."""
import json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from training_receipt_fixtures import transport_fixture
from subnet.storage import Identity
from subnet.batches import unpack
from subnet.backend_jobs import owned_commitment_upload
from subnet.commitment_transport import VERSION2,validate,sha
from subnet.training_documents import validate as validate_document
from subnet.miner import Miner

class TokenUpload(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name);self.identity=Identity();f=transport_fixture(SigningKey.generate());self.rows=unpack(f['data']);self.manifest=dict(f['manifest'],source_bundle={'sha256':'b'*64},deadline=time.time()+120,submission_transport_policy=VERSION2,max_batches=1)
  self.cap=dict(put_url='COMMIT',batch_put_urls=['PROOF'],training_put_urls=['TOKENS'],headers={});self.objects={};self.calls=[]
  self.seed=self.path/'seed';self.seed.write_text(self.identity.key.encode().hex());self.seed.chmod(0o600)
 def put(self,url,**kw):self.calls.append(url);self.objects[url]=kw['data'];return SimpleNamespace(status_code=200,raise_for_status=lambda:None)
 def assert_complete(self):
  claim=validate(self.objects['COMMIT'],self.manifest['epoch'],self.identity.id,1)['payload']['batches'][0]
  validate_document(self.objects['TOKENS'],self.manifest['epoch'],self.manifest['checkpoint']['id'],self.identity.id,claim)
  self.assertEqual(claim['sha256'],sha(self.objects['PROOF']));self.assertEqual(self.calls,['PROOF','TOKENS','COMMIT'])
 def test_owned_real_prepared_upload_and_restart_do_not_rewrite_slots(self):
  job=dict(miner_id=self.identity.id,miner_identity_file=str(self.seed),capability=self.cap);journal=self.path/'owned-journal.json'
  with patch('requests.put',side_effect=self.put):
   upload=owned_commitment_upload(job,self.manifest,journal);prepared=[(b,upload.prepare_pair(b,a))for b,a in self.rows];upload.upload_pairs(prepared,60)
  self.assert_complete();self.calls=[]
  with patch('requests.put',side_effect=self.put):owned_commitment_upload(job,self.manifest,journal).upload_pairs(prepared,60)
  self.assertEqual(self.calls,['COMMIT'])
 def test_public_real_prepared_upload_same_framing_and_failed_token_no_commit(self):
  with patch('subnet.miner.check_runtime_profile'):miner=Miner(self.identity,self.manifest,'unused',capability=self.cap,state_path=self.path/'public-state')
  miner.batches=self.rows
  def fail(url,**kw):
   if url=='TOKENS':raise TimeoutError('token transport')
   return self.put(url,**kw)
  with patch('requests.put',side_effect=fail),self.assertRaises(TimeoutError):miner.upload()
  self.assertNotIn('COMMIT',self.objects);self.calls=[]
  with patch('requests.put',side_effect=self.put):miner.upload()
  self.assertEqual(self.calls,['TOKENS','COMMIT']);self.calls=['PROOF',*self.calls];self.assert_complete()
if __name__=='__main__':unittest.main()
