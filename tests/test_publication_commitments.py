"""Real freeze/controller/history boundary, with no invented child hash audit."""
import copy,json,unittest
from types import SimpleNamespace
import test_commitment_production as production
from subnet import commitment_transport as transport
from subnet.controller import Controller
from subnet.publication import history,publish_history
from subnet.storage import canonical

class CommitmentHistoryTests(unittest.TestCase):
 def setUp(self):
  self.f=production.Tests();self.f.setUp();self.addCleanup(self.f.doCleanups)
  f=self.f;f.submit(0);self.miner=f.ids[0].id
  self.manifest=dict(f.m,checkpoint=dict(f.m['checkpoint'],files={}),source_bundle=dict(f.m['source_bundle'],key='public/sources/test/source.tar.gz'))
  self.c=Controller.__new__(Controller);self.c.state=f.state;self.c.bucket=f.b;self.c.authority=f.ids[2]
  self.c.bucket.get=lambda key:self.c.bucket.objects[key][0]
  (f.state/'e-manifest.json').write_bytes(canonical(self.manifest))
 def freeze(self):
  receipts=transport.freeze(self.f.g,'e');return dict(epoch_id='e',receipts=receipts,payable=False)
 def test_real_frozen_receipt_and_signed_history_without_heavy_reads(self):
  result=self.freeze();saved=copy.deepcopy(result);doc=history(self.c,[result]);row=doc['epochs'][0]['frozen'][self.miner]
  self.assertEqual(row['commitment']['sha256'],result['receipts'][self.miner]['sha256'])
  child=row['artifacts'][0];original=result['receipts'][self.miner]['artifacts'][0]
  for field in ('slot','env_id','index','batch_sha256','sha256','size'):self.assertEqual(child[field],original[field])
  self.assertEqual(child['url'],self.c.bucket.presign(original['frozen_key']))
  self.assertEqual(child['hash_assurance'],'declared-payload-hash-until-selected-verifier')
  self.assertNotIn('fully_audited',child);self.assertNotIn('verified',child)
  publish_history(self.c,'stream',[result]);envelope=json.loads(self.c.bucket.get('public/streams/stream/history.json'))
  self.assertEqual(envelope['signer'],self.c.authority.id)
  from subnet.backend_jobs import signed
  self.assertEqual(signed(envelope,self.c.authority.id)['epochs'],doc['epochs'])
  self.assertEqual(self.f.b.heavy_reads,0);self.assertEqual(result,saved)
 def test_original_noncanonical_commitment_bytes_are_preserved(self):
  key='private/e/commitments/'+self.miner+'.json';env=json.loads(self.f.b.objects[key][0]);data=json.dumps(env,indent=2).encode();self.f.b.put(key,data)
  result=self.freeze();row=history(self.c,[result])['epochs'][0]['frozen'][self.miner]
  self.assertEqual(row['commitment']['size'],len(data));self.assertEqual(row['commitment']['sha256'],transport.sha(data))
 def test_inventory_and_scope_corruption_fail_before_publication(self):
  original=self.freeze()
  changes=[lambda r:r['artifacts'][0].update(sha256='f'*64),lambda r:r['artifacts'][0].update(slot=1),lambda r:r['artifacts'][0].update(size=1),lambda r:r['artifacts'].clear(),lambda r:r['artifacts'][0].update(frozen_key='private/e/staging/x.zip'),lambda r:r.update(commitment_key='public/other/commitment.json'),lambda r:r['commitment_document']['payload'].update(source='c'*64)]
  for change in changes:
   with self.subTest(change=change):
    result=copy.deepcopy(original);change(result['receipts'][self.miner])
    with self.assertRaises(ValueError):publish_history(self.c,'stream',[result])
    self.assertNotIn('public/streams/stream/history.json',self.f.b.objects)
 def test_corrupt_actual_commitment_or_model_binding_refused(self):
  result=self.freeze();key=result['receipts'][self.miner]['commitment_key'];saved=self.f.b.objects[key]
  self.f.b.put(key,b'corrupt')
  with self.assertRaisesRegex(ValueError,'content mismatch'):history(self.c,[result])
  self.f.b.objects[key]=saved
  manifest=dict(self.manifest,checkpoint=dict(id='f'*64,files={}))
  (self.f.state/'e-manifest.json').write_bytes(canonical(manifest))
  with self.assertRaisesRegex(ValueError,'model/source'):history(self.c,[result])
 def test_explicit_policy_required(self):
  result=self.freeze();manifest=dict(self.manifest);manifest.pop('submission_transport_policy');(self.f.state/'e-manifest.json').write_bytes(canonical(manifest))
  with self.assertRaisesRegex(ValueError,'signed transport policy'):history(self.c,[result])
 def test_actual_remote_finalize_to_history_contract(self):
  # Exercises real selection, reports, scoring and freeze with one test verifier.
  self.f.test_zero_allocations_no_job_and_no_heavy_download()
  result=json.loads((self.f.state/'e-scores.json').read_bytes())
  doc=history(self.c,[result]);self.assertEqual(len(doc['epochs'][0]['frozen']),3)
  self.assertEqual(sum(result['points'].values()),1);self.assertEqual(self.f.b.heavy_reads,0)
