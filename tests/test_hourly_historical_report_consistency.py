"""Same ROOT exact-report admission in the independent writer's selected joins."""
import copy,json,unittest
import test_current_assessment_evidence as fixtures
from subnet.continuous_audit_policy import digest
from ops import current_assessment_evidence as evidence

class HistoricalWriterControls(unittest.TestCase):
 setUp=fixtures.EvidenceControls.setUp
 make_queue=fixtures.EvidenceControls.make_queue
 save=fixtures.EvidenceControls.save
 load=fixtures.EvidenceControls.load
 def historical(self):
  source=self.config['continuous_audit_service']['source_admission']['payload']
  source['historical_report_admission']=dict(version='exact-retired-verifier-reports-v1',reports={self.worker:{self.queue['digest']:self.queue['report_digest']}})
  self.config['continuous_audit_service']['source_admission']=fixtures.signed(self.key,source)
 def retired_load(self):
  return evidence.load_evidence(self.cfg,authority=self.authority,cutoff=30,
    verifiers=['0'*64],expected_source_admission_sha256=digest(self.config['continuous_audit_service']['source_admission']))
 def test_exact_retired_original_reaches_admission_observations_and_snapshot(self):
  self.historical();self.save();before=self.cfg.read_bytes();r=self.retired_load()
  self.assertEqual(r['refused'],[]);self.assertGreater(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5)
  self.assertEqual(self.cfg.read_bytes(),before)
 def test_unlisted_retired_job_has_no_credit(self):
  self.historical();self.queue=self.make_queue('verified_valid','job-2',21)
  self.state['jobs']={'job-2':{'row_sha256':digest(self.row)}};self.save();r=self.retired_load()
  self.assertEqual(r['refused'][0]['reason'],'admitted actual worker');self.assertEqual(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5)
 def test_report_allowlist_does_not_bypass_original_signature(self):
  self.historical();self.queue['report_request']['signature']='A'*88;self.save();r=self.retired_load()
  self.assertTrue(r['refused']);self.assertEqual(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5)
 def test_selected_group_scope_not_broadened_by_retired_map(self):
  job=self.queue['envelope']['payload'];extra=copy.deepcopy(job['submissions'][0]);extra['sha256']='b'*64;job['submissions'].append(extra)
  self.queue['envelope']=fixtures.signed(self.key,job);self.queue['digest']=digest(job)
  report=self.queue['report'];report['job_sha256']=digest(job);audit=copy.deepcopy(report['audits'][0]);audit['submission_sha256']='b'*64;report['audits'].append(audit)
  self.queue['report_digest']=digest(report);self.queue['report_request']=fixtures.signed(self.workerkey,dict(action='report',job_id='job-1',token=self.queue['token'],report=report))
  self.historical();self.save();r=self.retired_load();self.assertTrue(r['refused']);self.assertEqual(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5)
 def test_default_no_historical_permission_still_refuses(self):
  self.save();r=self.retired_load();self.assertEqual(r['refused'][0]['reason'],'admitted actual worker')

if __name__=='__main__':unittest.main()
