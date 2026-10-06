"""Retired identities may authenticate only ROOT-pinned original reports."""
import copy,unittest
from nacl.signing import SigningKey
import test_continuous_audit_policy as fixture
signed=fixture.signed
from subnet.continuous_audit_policy import admit_queue_reports,historical_report_workers,digest
class HistoricalReportControls(unittest.TestCase):
 def setUp(self):
  self.f=fixture.PolicyControls();self.f.setUp();self.q,self.pins=self.f.queue_fixture();self.policy={'version':'exact-retired-verifier-reports-v1','reports':{self.f.worker:{self.q['digest']:self.q['report_digest']}}}
 def admit(self,q=None,p=None):return admit_queue_reports([q or self.q],[self.f.row],self.f.root,{},self.pins,historical_report_admission=self.policy if p is None else p)
 def test_genuine_retired_terminal_signature_admitted_without_live_credentials(self):
  self.assertEqual(len(self.admit()),1)
  with self.assertRaisesRegex(ValueError,'actual worker'):admit_queue_reports([self.q],[self.f.row],self.f.root,{},self.pins)
 def test_new_job_or_report_cannot_reuse_retired_identity(self):
  for field in ('digest','report_digest'):
   q=copy.deepcopy(self.q);q[field]='0'*64
   with self.assertRaisesRegex(ValueError,'actual worker'):self.admit(q)
 def test_same_digests_do_not_waive_worker_signature_or_lease_token(self):
  for field in ('token','report_request'):
   q=copy.deepcopy(self.q);q[field]='foreign'if field=='token'else signed(SigningKey.generate(),q['report_request']['payload'])
   with self.assertRaises(Exception):self.admit(q)
 def test_unknown_worker_and_nonterminal_never_admitted(self):
  for field,value in [('worker','0'*64),('status','expired')]:
   q=copy.deepcopy(self.q);q[field]=value
   with self.assertRaises(ValueError):self.admit(q)
 def test_malformed_or_wildcard_policy_fails_closed(self):
  for p in [{'version':'wrong','reports':{}},{'version':'exact-retired-verifier-reports-v1','reports':{self.f.worker:{'*':self.q['report_digest']}}},{'version':'exact-retired-verifier-reports-v1','reports':{self.f.worker:[]}}]:
   with self.assertRaises(ValueError):historical_report_workers(p)
 def test_retired_report_still_requires_unchanged_science_source(self):
  self.pins=copy.deepcopy(self.pins);self.pins['9'*64]['model.py']='0'*64
  with self.assertRaisesRegex(ValueError,'source pins'):self.admit()
 def test_real_hourly_snapshot_uses_retired_map_but_preserves_claim_roster(self):
  import tempfile
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from subnet.continuous_audit_service import ContinuousAuditor
  with tempfile.TemporaryDirectory()as root:
   service=ContinuousAuditor.__new__(ContinuousAuditor);service.directory=Path(root);service.queue=SimpleNamespace(workers={});service.controller=SimpleNamespace(authority=SimpleNamespace(id=self.f.root),signed=lambda x:signed(self.f.authority,x));row=self.f.row;identity=digest(row)
   service.state={'jobs':{'job-1':{'row_sha256':identity}},'draws':{identity:{'row':row}},'capture_failures':{},'populations':{'e1':signed(self.f.authority,{'eligible_evidence_ids':[identity]})}}
   service.records=lambda:[row];service.sources=self.pins;service.policy=self.f.p;service.execution_evidence_policy=None;service.backend_evidence_deferral_policy=None;service.historical_report_admission=self.policy;service.publish_immutable=lambda *a:None
   with patch('subnet.continuous_audit_service.queue_rows',return_value={'job-1':self.q}):result=service.hourly_snapshot('e1',1,'a'*64,30)
   self.assertGreater(result['payload']['points']['b'*64],.5);self.assertEqual(service.queue.workers,{})
