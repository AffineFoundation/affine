import copy,json
from pathlib import Path
from test_current_assessment_writer import WriterControls
from ops import current_assessment_writer as w
from ops.live_reward_exporter import sign
class SameWindow(WriterControls):
 def prepare_adoption(self):
  self.invoke();self.target=self.path/w.ASSESSMENT_DIRECTORY/'assessment-7200.json';self.raw=self.target.read_bytes();self.snapshot=self.path/'pinned-prior.json';self.snapshot.write_bytes(self.raw)
  self.row=dict(path=str(self.snapshot),sha256=w.file_hash(self.snapshot),writer_policy_sha256=w.sha(self.policy));self.policy=sign(dict(self.policy['payload'],fallback_assessments=[self.row]),self.key)
 def test_explicit_same_hour_preserves_original_bytes_without_recompute(self):
  self.prepare_adoption();before=self.calls[0];self.invoke(AssertionError('evidence must not run'));self.assertEqual(self.calls[1],before);self.assertEqual(self.target.read_bytes(),self.raw)
 def test_wrong_raw_or_writer_pin_refuses(self):
  for field in ('sha256','writer_policy_sha256'):
   self.setUp();self.prepare_adoption();body=copy.deepcopy(self.policy['payload']);body['fallback_assessments'][0][field]='f'*64;self.policy=sign(body,self.key)
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'adoption'):self.invoke()
   self.assertEqual(len(self.calls),1);self.assertEqual(self.target.read_bytes(),self.raw)
 def test_modified_or_symlink_snapshot_refuses(self):
  self.prepare_adoption();self.snapshot.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'snapshot'):self.invoke()
  self.snapshot.unlink();self.snapshot.symlink_to(self.target)
  with self.assertRaisesRegex(ValueError,'snapshot'):self.invoke()
 def test_unrelated_window_refuses(self):
  self.prepare_adoption();other=self.target.with_name('assessment-10800.json');other.write_bytes(self.raw)
  with self.assertRaisesRegex(ValueError,'hourly assessment binding'):self.invoke(now=11000)
 def test_changed_source_or_numerical_policy_refuses(self):
  for field in ('source_admission_sha256','numerical_resolution_policy_sha256'):
   self.setUp();self.prepare_adoption();body=copy.deepcopy(self.policy['payload']);body[field]='f'*64;self.policy=sign(body,self.key)
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'evidence policy'):self.invoke()
 def test_bad_signature_refuses_even_if_bytes_pinned(self):
  self.prepare_adoption();value=json.loads(self.raw);value['signature']='A'*88;self.raw=json.dumps(value).encode();self.target.write_bytes(self.raw);self.snapshot.write_bytes(self.raw);body=copy.deepcopy(self.policy['payload']);body['fallback_assessments'][0]['sha256']=w.file_hash(self.target);self.policy=sign(body,self.key)
  with self.assertRaises(Exception):self.invoke()
 def test_no_replay_of_already_submitted_hour_under_new_policy(self):
  self.prepare_adoption();(self.path/'weights.json').write_text(json.dumps({'last_submitted_window':7200}));self.assertEqual(self.invoke()['status'],'already_submitted');self.assertEqual(len(self.calls),1)
