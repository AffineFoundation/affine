import copy,json,unittest
from unittest.mock import patch
from pathlib import Path
from test_continuous_audit_service import ServiceControls
from test_bounded_audit_grouping import BoundedGrouping
from subnet.continuous_audit_policy import digest
from subnet.continuous_audit_service import atomic
from subnet.distributed_roles import authenticate
from test_continuous_audit_policy import signed

class SimpleExpiry(unittest.TestCase):
 def setUp(self):
  self.f=ServiceControls();self.f.setUp();self.addCleanup(self.f.tearDown);self.s=self.f.service
 def original(self,expires=25):
  identity=digest(self.f.row);jid='continuous-audit-'+identity[:32];self.s.state['draws'][identity]=dict(row=self.f.row,seed='1'*64,selected_at=20)
  envelope=signed(self.f.key,dict(job_id=jid,created_at=20,expires_at=expires));path=self.s.directory/(jid+'-job.json');atomic(path,envelope);return jid,path
 def test_expired_simple_is_immutable_no_credit_no_recapture_idempotent(self):
  jid,path=self.original();raw=path.read_bytes();draw=copy.deepcopy(self.s.state['draws'])
  with patch.object(self.s,'_capture',side_effect=AssertionError('no redraw')),patch.object(self.s.queue,'enqueue',side_effect=AssertionError('no expired enqueue')):
   self.s.tick(now=25);self.s.tick(now=26)
  self.assertEqual(raw,path.read_bytes());self.assertEqual(draw,self.s.state['draws']);self.assertEqual(len(self.s.state['expired_requests']),1);self.assertFalse(self.s.state['jobs'])
  record=authenticate(self.s.state['expired_requests'][jid]['document'],self.f.root)
  self.assertEqual(record['outcome'],'infrastructure_expired');self.assertFalse(record['accepted_proof']);self.assertFalse(record['confirmed_fraud'])
 def test_request_expiring_during_capture_is_terminal_not_enqueued(self):
  jid,path=self.original(expires=26);raw=path.read_bytes()
  with patch('subnet.continuous_audit_service.time.time',side_effect=[25,27]):
   self.s.tick()
  self.assertFalse(self.s.queue.envelopes);self.assertEqual(path.read_bytes(),raw);self.assertIn(jid,self.s.state['expired_requests'])
 def test_unexpired_original_retries_exact_bytes(self):
  _,path=self.original(expires=26);original=json.loads(path.read_bytes());self.s.tick(now=25);self.assertEqual(self.s.queue.envelopes,[original]);self.assertFalse(self.s.state['expired_requests'])
 def test_bad_signature_lifetime_or_identity_is_not_disposed(self):
  for kind in('signature','identity','nan','boolean'):
   self.s.state['expired_requests']={};jid,path=self.original();e=json.loads(path.read_bytes())
   if kind=='signature':e['signature']='0'*128
   else:
    p=e['payload'];p['job_id']='foreign'if kind=='identity'else jid;p['expires_at']='nan'if kind=='nan'else True if kind=='boolean'else 25;e=signed(self.f.key,p)
   atomic(path,e)
   with self.assertRaises(Exception):self.s.tick(now=25)
   self.assertFalse(self.s.state['expired_requests'])

class GroupExpiry(unittest.TestCase):
 def setUp(self):
  self.f=BoundedGrouping();self.f.setUp();self.addCleanup(self.f.tearDown);self.s=self.f.service
 def test_all_stale_groups_disposed_then_new_work_same_tick_without_old_reissue(self):
  self.s.tick();paths=list(self.s.directory.glob('continuous-audit-group-*-job.json'));raw={p:p.read_bytes()for p in paths};draw=copy.deepcopy(self.s.state['draws']);self.s.state['jobs']={}
  for p in self.s.state['group_plans'].values():p['resolved']=False
  future=max(json.loads(b)['payload']['expires_at']for b in raw.values())+1
  # Genuine new row is distinct; the original eight draws remain immutable.
  row=dict(self.f.rows[0],index=999,batch_sha256='8'*64,proof_sha256='7'*64);self.f.rows.append(row);self.f.capture_data[digest(row)]=self.f.capture_data[digest(self.f.rows[0])]
  with patch.object(self.s.queue,'enqueue')as enqueue:
   # Fresh capture intentionally fails, proving scheduler advances past all stale plans.
   with patch.object(self.s,'_capture',side_effect=TimeoutError('fresh-only')):result=self.s.tick(now=future)
   enqueue.assert_not_called()
  self.assertEqual(len(self.s.state['expired_requests']),2);self.assertEqual(result['selected'],1)
  for p,b in raw.items():self.assertEqual(p.read_bytes(),b)
  for i,d in draw.items():self.assertEqual(self.s.state['draws'][i],d)
  self.assertTrue(all(p['resolved']for p in self.s.state['group_plans'].values()if set(p['row_sha256s'])<=set(draw)))
 def test_expired_partial_group_does_not_discard_uncaptured_members(self):
  self.s.tick();self.s.state['jobs']={};key,plan=next(iter(self.s.state['group_plans'].items()));plan['resolved']=False
  path=self.s.directory/('continuous-audit-group-'+key[:32]+'-job.json');e=json.loads(path.read_bytes());original_ids=e['payload']['audit_group']['row_sha256s'];e['payload']['audit_group']['row_sha256s']=original_ids[:2]
  from test_commitment_child_queue import sign
  atomic(path,sign(self.f.root,e['payload']));at=e['payload']['expires_at']+1
  self.s.retire_expired_requests(at);self.assertEqual(self.s.expired_rows(),set(original_ids[:2]));record=next(iter(self.s.state['expired_requests'].values()))['document']['payload'];self.assertEqual(record['plan_row_sha256s'],plan['row_sha256s']);self.assertTrue(plan['resolved'])
 def test_mismatched_signed_group_members_refused(self):
  self.s.tick();self.s.state['jobs']={};key,plan=next(iter(self.s.state['group_plans'].items()));plan['resolved']=False
  path=self.s.directory/('continuous-audit-group-'+key[:32]+'-job.json');e=json.loads(path.read_bytes());e['payload']['audit_group']['row_sha256s']=[]
  from test_commitment_child_queue import sign
  atomic(path,sign(self.f.root,e['payload']))
  with self.assertRaisesRegex(ValueError,'original audit group request'):self.s.tick(now=10**12)
if __name__=='__main__':unittest.main()
