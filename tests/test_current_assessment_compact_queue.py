"""Actual signed admission fixtures and SQLite byte/race refusal controls."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sqlite3,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from ops import current_assessment_compact_queue as m,current_assessment_admission_cache as cache,current_assessment_evidence as candidate
import test_current_assessment_admission_cache as old
import test_current_assessment_evidence as fx
baseline=candidate  # Default-off loader retains the original full-row path.

class CompactControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.path=self.root/'jobs.sqlite';self.directory=self.root/'cache';self.directory.mkdir(mode=0o700)
  self.x,self.q,self.records,self.context=old.fixture();self.key=SigningKey.generate();self.policy=old.make_policy(self.directory,self.x.authority,self.key,self.x.root)
  self.handle=cache.AdmissionCache(self.policy,authority=self.x.root,context=self.context,signing_key=self.key);self.addCleanup(self.handle.close);self.write(self.q)
 def write(self,q):
  with sqlite3.connect(self.path)as db:
   db.execute('DROP TABLE IF EXISTS jobs');db.execute('CREATE TABLE jobs ('+','.join('"'+k+'"' for k in q)+')');db.execute('INSERT INTO jobs VALUES ('+','.join('?'for _ in q)+')',list(q.values()))
 def scan(self):return m.scan(SimpleNamespace(path=self.path),[self.q['id']])
 def invoke(self,view=None):
  view=view or self.scan();return m.admit_bound(self.handle,view.bindings[self.q['id']],self.records,cutoff=30,full_row=lambda:view.fetch(self.q['id']))
 def test_exact_existing_fingerprint_and_cache_key_reuse(self):
  first=self.handle.admit(self.q,self.records,cutoff=30);v=self.scan();self.assertEqual(v.bindings[self.q['id']]['fingerprint'],cache.queue_fingerprint(self.q))
  with patch.object(v,'fetch',side_effect=AssertionError('hit must not load original')):self.assertEqual(first,self.invoke(v))
  self.assertEqual(self.handle.stats['hits'],1)
 def test_cold_signed_original_admission_and_warm_hit(self):
  self.assertEqual(self.invoke(),old.baseline(self.q,self.records,self.context,self.x.root,30));self.invoke();self.assertEqual(self.handle.stats['baseline_calls'],1);self.assertEqual(self.handle.stats['hits'],1)
 def test_request_signature_tampering_same_metadata_refused(self):
  self.invoke();q=copy.deepcopy(self.q);v=json.loads(q['report_request']);v['signature']='A'*88;q['report_request']=cache.canonical(v).decode();self.write(q)
  with self.assertRaises(ValueError):self.invoke()
  self.assertEqual(self.handle.stats['hits'],0)
 def test_report_body_tamper_with_stored_digest_unchanged_refused(self):
  self.invoke();q=copy.deepcopy(self.q);v=json.loads(q['report']);v['audits'][0]['outcomes'][0]['valid']=False;q['report']=cache.canonical(v).decode();self.write(q)
  with self.assertRaises(ValueError):self.invoke()
  self.assertEqual(self.handle.stats['hits'],0)
 def test_every_metadata_and_lease_field_revalidated(self):
  self.invoke()
  for k,v in [('worker','0'*64),('token','other'),('digest','0'*64),('report_digest','0'*64),('role','train'),('status','queued'),('attempt',2),('expires',900.)]:
   with self.subTest(field=k):
    q=dict(self.q,**{k:v});self.write(q);actual=old.output(self.invoke);expected=old.output(lambda:old.baseline(q,self.records,self.context,self.x.root,30));self.assertEqual(actual,expected)
  self.assertEqual(self.handle.stats['hits'],0)
 def test_raw_whitespace_change_invalidates_without_inventing_fraud(self):
  self.invoke()
  for field in ('envelope','report','report_request'):
   q=dict(self.q);q[field]=' '+q[field];self.write(q);self.assertEqual(self.invoke(),old.baseline(q,self.records,self.context,self.x.root,30))
  self.assertEqual(self.handle.stats['hits'],0);self.assertEqual(self.handle.stats['baseline_calls'],4)
 def test_removed_row_is_absent_and_original_miss_refuses(self):
  v=self.scan()
  with sqlite3.connect(self.path)as db:db.execute('DELETE FROM jobs')
  self.assertEqual(self.scan().bindings,{})
  with self.assertRaisesRegex(ValueError,'removed'):v.fetch(self.q['id'])
 def test_rewrite_between_snapshot_and_cache_miss_refuses(self):
  v=self.scan();q=dict(self.q,token='changed');self.write(q)
  with self.assertRaisesRegex(ValueError,'bytes changed'):self.invoke(v)
 def test_queue_file_replacement_refuses(self):
  v=self.scan();self.path.rename(self.root/'old.sqlite');self.write(self.q)
  with self.assertRaisesRegex(ValueError,'inode'):v.fetch(self.q['id'])
 def test_racing_new_column_or_scalar_to_large_string_refused_before_full_read(self):
  v=self.scan()
  with sqlite3.connect(self.path)as db:db.execute('ALTER TABLE jobs ADD COLUMN injected TEXT');db.execute('UPDATE jobs SET injected=?',('x'*1024**2,))
  with self.assertRaisesRegex(ValueError,'schema changed'):v.fetch(self.q['id'])
  self.write(dict(self.q,number=5));v=self.scan()
  with sqlite3.connect(self.path)as db:db.execute('UPDATE jobs SET number=?',('x'*1024**2,))
  with self.assertRaisesRegex(ValueError,'bytes changed'):v.fetch(self.q['id'])
 def test_fresh_read_context_cannot_reuse_old_admission(self):
  self.invoke();self.handle.close();self.context['workers']={};self.handle=cache.AdmissionCache(self.policy,authority=self.x.root,context=self.context,signing_key=self.key);self.addCleanup(self.handle.close)
  with self.assertRaises(ValueError):self.invoke()
  self.assertEqual(self.handle.stats['hits'],0)
 def test_corrupt_cache_cannot_promote_and_reuses_original_check(self):
  self.invoke();p=next(self.directory.glob('*.json'));v=json.loads(p.read_text());v['payload']['result']['admitted']={};p.write_bytes(cache.canonical(v))
  self.assertEqual(self.invoke(),old.baseline(self.q,self.records,self.context,self.x.root,30));self.assertEqual(self.handle.stats['rejected_entries'],1)
 def test_malformed_report_keeps_small_binding_for_individual_refusal(self):
  self.write(dict(self.q,report='{malformed'));v=self.scan();self.assertEqual(v.bindings[self.q['id']]['completion_type'],'invalid');self.assertIsNone(v.bindings[self.q['id']]['completed_at'])
 def test_nonfinite_or_boolean_report_completion_is_individually_invalid(self):
  for text in ('{"completed_at":1e999}','{"completed_at":true}','{"completed_at":{}}'):
   self.write(dict(self.q,report=text));v=self.scan();self.assertEqual(v.bindings[self.q['id']]['completion_type'],'invalid');self.assertIsNone(v.bindings[self.q['id']]['completed_at'])
 def test_string_types_unicode_null_real_and_extra_column_are_exact(self):
  self.write(dict(self.q,extra='ユニコード\x00quote"',null=None,real=1.25,integer=5));v=self.scan();q=v.fetch(self.q['id']);self.assertEqual(v.bindings[self.q['id']]['fingerprint'],cache.queue_fingerprint(q))
 def test_no_sqlite_write_or_original_body_transfer_on_scan(self):
  before=self.path.read_bytes();v=self.scan();self.assertEqual(self.path.read_bytes(),before);serialized=cache.canonical(v.bindings);self.assertNotIn(self.q['report_request'].encode(),serialized);self.assertNotIn(self.q['envelope'].encode(),serialized)

 def test_oversized_selector_strings_do_not_leave_sqlite(self):
  q=dict(self.q,status='x'*1000000,digest='y'*1000000);self.write(q);v=self.scan()
  self.assertIsNone(v.bindings[q['id']]['status']);self.assertIsNone(v.bindings[q['id']]['digest'])
  self.assertLess(len(cache.canonical(v.bindings)),10000)
 def test_giant_non_numeric_completion_does_not_leave_sqlite(self):
  self.write(dict(self.q,report=json.dumps({'completed_at':'x'*1000000})))
  v=self.scan();self.assertIsNone(v.bindings[self.q['id']]['completed_at']);self.assertEqual(v.bindings[self.q['id']]['completion_type'],'invalid');self.assertLess(len(cache.canonical(v.bindings)),10000)
 def test_oversized_report_and_worker_refused_without_parsing_or_transfer(self):
  q=dict(self.q,report='{"completed_at":1,"pad":"'+'x'*10000+'"}',worker='w'*10000);self.write(q)
  # Smaller test resource budget exercises the same SQL byte predicate.
  with patch.dict(m.POLICY,maximum_string_bytes=512):
   v=self.scan();self.assertIsNone(v.bindings[q['id']]['fingerprint']);self.assertEqual(v.bindings[q['id']]['completion_type'],'invalid')
   with self.assertRaisesRegex(ValueError,'oversized'):v.fetch(q['id'])
 def test_oversized_selected_identifier_refused_before_sql(self):
  with self.assertRaisesRegex(ValueError,'bounded distinct'):m.scan(SimpleNamespace(path=self.path),['i'*129])

 def test_arbitrary_identifier_iterator_refused_without_consumption(self):
  class Unbounded:
   def __iter__(self):raise AssertionError('must not consume arbitrary iterator')
  with self.assertRaisesRegex(ValueError,'bounded distinct'):m.scan(SimpleNamespace(path=self.path),Unbounded())

class LoaderParity(fx.EvidenceControls):
 def setUp(self):
  super().setUp();self.cache_dir=self.root/'cache';self.cache_dir.mkdir(mode=0o700);self.cache_key=SigningKey.generate();self.cache_policy=old.make_policy(self.cache_dir,self.key,self.cache_key,self.authority);self.handles=[]
 def factory(self,*,authority,context):
  h=cache.AdmissionCache(self.cache_policy,authority=authority,context=context,signing_key=self.cache_key);self.handles.append(h);return h
 def normalize(self,v):
  v=copy.deepcopy(v);v['evidence_hashes'].pop('original_queue_byte_binding_sha256',None);v['evidence_hashes'].pop('original_queue_byte_binding_version',None);v['evidence_hashes']['original_queue_view_sha256']=None;return v
 def load(self,**kw):
  args=dict(authority=self.authority,cutoff=30,verifiers=[self.worker],expected_source_admission_sha256=fx.digest(self.config['continuous_audit_service']['source_admission']),**kw)
  expected=baseline.load_evidence(self.cfg,**args)
  actual=candidate.load_evidence(self.cfg,admission_cache_factory=self.factory,compact_queue_evidence=m.POLICY,**args)
  self.assertEqual(cache.canonical(self.normalize(actual)),cache.canonical(self.normalize(expected)));return actual
 def test_queue_outage_propagates(self):
  with patch.object(m,'scan',side_effect=TimeoutError('busy')):
   with self.assertRaises(TimeoutError):self.load()
 def test_complete_output_cold_and_warm_matches_rewards_and_original_refusals(self):
  first=self.load();self.assertEqual(first,self.load());self.assertEqual(self.handles[-1].stats['hits'],1)
 def test_last_duplicate_completion_preserves_original_signed_cutoff_admission(self):
  text=json.dumps(self.queue['report']);self.queue['report']='{"completed_at":999999,'+text[1:];self.save();r=self.load();self.assertEqual(r['refused'],[]);self.assertGreater(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5)
 def test_last_duplicate_escaped_completion_matches_original(self):
  text=json.dumps(self.queue['report']).replace('"completed_at"','"completed\\u005fat"');self.queue['report']='{"completed_at":999999,'+text[1:];self.save();r=self.load();self.assertEqual(r['refused'],[])
 def test_reverse_duplicate_cutoff_order_excluded_identically(self):
  text=json.dumps(self.queue['report']);self.queue['report']=text[:-1]+',"completed_at":999999}';self.save();r=self.load();self.assertEqual(r['refused'],[]);self.assertEqual(r['excluded'][0]['reason'],'original report completed after cutoff')

 def test_default_off_remains_exact_full_queue_evidence(self):
  args=dict(authority=self.authority,cutoff=30,verifiers=[self.worker],expected_source_admission_sha256=fx.digest(self.config['continuous_audit_service']['source_admission']))
  self.assertEqual(candidate.load_evidence(self.cfg,**args),baseline.load_evidence(self.cfg,**args))

if __name__=='__main__':unittest.main(verbosity=2)
