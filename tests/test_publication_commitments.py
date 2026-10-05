"""Real freeze/controller/history boundary, with no invented child hash audit."""
import copy,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
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

 def test_repeat_history_uses_exact_private_admission_without_network(self):
  result=self.freeze();original=self.f.b.get_object
  with patch.object(self.f.b,'get_object',wraps=original)as read:
   first=history(self.c,[result]);second=history(self.c,[result]);publish_history(self.c,'stream',[result])
   self.assertEqual(read.call_count,1)
  self.assertEqual(first['epochs'],second['epochs'])
  journal=next((self.f.state/'history-commitment-admissions').glob('*.json'))
  self.assertEqual(journal.stat().st_mode & 0o777,0o600)
  saved=json.loads(journal.read_bytes());saved['binding']['epoch']='other';journal.write_bytes(canonical(saved))
  with patch.object(self.f.b,'get_object',side_effect=AssertionError('no fallback from corrupt admission')):
   with self.assertRaisesRegex(ValueError,'journal binding'):history(self.c,[result])
 def test_manifest_or_receipt_change_never_reuses_old_admission(self):
  result=self.freeze();history(self.c,[result]);original=self.f.b.get_object
  changed=copy.deepcopy(result);changed['receipts'][self.miner]['artifacts'][0]['size']+=1
  with patch.object(self.f.b,'get_object',wraps=original)as read:
   with self.assertRaisesRegex(ValueError,'inventory'):history(self.c,[changed])
   self.assertEqual(read.call_count,1)
  manifest=dict(self.manifest,checkpoint=dict(id='f'*64,files={}))
  (self.f.state/'e-manifest.json').write_bytes(canonical(manifest))
  with patch.object(self.f.b,'get_object',wraps=original)as read:
   with self.assertRaisesRegex(ValueError,'model/source'):history(self.c,[result])
   self.assertEqual(read.call_count,1)
  self.assertEqual(len(list((self.f.state/'history-commitment-admissions').glob('*.json'))),1)

 def test_incomplete_capture_history_has_no_fabricated_scientific_routes(self):
  from subnet.storage import sha
  payload=dict(version='commitment-capture-status-v1',epoch='e',status='metadata_incomplete',complete=False,
   manifest_sha256=sha(canonical(self.manifest)),source_sha256=self.manifest['source_bundle']['sha256'],
   checkpoint=self.manifest['checkpoint']['id'],verification_claim=False,audits_started=False,
   accepted_batches=0,rewards_eligible=False,known_captured=[],unresolved_miners=[self.miner])
  envelope=self.c.signed(payload);raw=canonical(envelope)
  (self.f.state/'e-capture-status.json').write_bytes(raw)
  closure=dict(epoch='e',status='infrastructure_skipped_metadata_incomplete',capture_status_sha256=sha(raw),
   checkpoint=self.manifest['checkpoint']['id'],training_updates=0,verification_claim=False,payable=False,chain_transactions=False)
  path=self.f.state/'infrastructure-skipped-epochs.json';path.write_bytes(canonical([closure]))
  with patch.object(self.f.b,'get_object',side_effect=AssertionError('no remote scientific read')):
   doc=history(self.c,[])
  self.assertEqual(doc['epochs'],[]);row=doc['infrastructure_skips'][0]
  self.assertEqual(row['capture_status']['sha256'],sha(raw));self.assertNotIn('objects',row)
  self.assertNotIn('frozen',row);self.assertNotIn('audits',row)
  closure['training_updates']=1;path.write_bytes(canonical([closure]))
  with self.assertRaisesRegex(ValueError,'incomplete infrastructure'):history(self.c,[])
