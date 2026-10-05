import copy,threading,unittest
from unittest.mock import patch
from test_commitment_freeze_prefetch import BoundedCommitmentFreeze
from subnet import commitment_transport as c
from subnet.selected_proof_copy import validate_policy,copy_selected
POLICY={'version':'selected-proof-copy-v1','workers':4}
class Controls(unittest.TestCase):
 def fixture(self,n=6):
  g,ids,_=BoundedCommitmentFreeze().fixture(n);g.epochs['e']['commitment_binding']['proof_copy_policy']=POLICY
  return g,ids
 def test_exact_signed_policy_no_unbounded_workers(self):
  self.assertEqual(validate_policy(POLICY),POLICY)
  for v in ({**POLICY,'workers':5},{**POLICY,'workers':True},{**POLICY,'other':1},None):
   with self.assertRaises(ValueError):validate_policy(v)
 def test_population_freezes_all_metadata_without_heavy_copy_or_read(self):
  g,ids=self.fixture();receipts=c.freeze(g,'e');self.assertEqual(len(receipts),6);self.assertEqual(g.bucket.copies,[]);self.assertEqual(g.bucket.heavy_reads,0)
  for r in receipts.values():
   self.assertNotIn(r['artifacts'][0]['frozen_key'],g.bucket.objects);self.assertEqual(r['artifact_public_availability'],'only-successfully-copied-selected-proofs')
 def test_exact_draw_only_conditional_copy_and_recovery_no_repeat(self):
  g,ids=self.fixture();r=c.freeze(g,'e');m={'epoch':'e','proof_copy_policy':POLICY};slots={i.id:([0]if i in ids[:2]else[])for i in ids}
  self.assertEqual(copy_selected(g,m,r,slots,None),{});self.assertEqual(len(g.bucket.copies),2)
  with patch.object(g.bucket,'copy',side_effect=AssertionError('no recopies')):self.assertEqual(copy_selected(g,m,r,slots,None),{})
  self.assertEqual(len(g.epochs['e']['selected_proof_copies']),2)
 def test_postfreeze_etag_mutation_is_infrastructure_no_redraw_or_fraud(self):
  g,ids=self.fixture();r=c.freeze(g,'e');original=copy.deepcopy(r);miner=ids[0].id;b=r[miner]['artifacts'][0];g.bucket.put(b['key'],b'changed')
  errors=copy_selected(g,{'epoch':'e','proof_copy_policy':POLICY},r,{miner:[0]},None)
  self.assertIn(miner,errors);self.assertEqual(r,original);self.assertEqual(g.epochs['e']['rejections'],{});self.assertNotIn(miner,g.epochs['e']['selected_proof_copies'])
 def test_expired_budget_never_copies(self):
  g,ids=self.fixture();r=c.freeze(g,'e')
  with patch.object(g.bucket,'copy',side_effect=AssertionError('no late launch')):errors=copy_selected(g,{'epoch':'e','proof_copy_policy':POLICY},r,{ids[0].id:[0]},0)
  self.assertEqual(errors,{ids[0].id:'TimeoutError'})
 def test_four_actual_copies_overlap_no_more_than_four(self):
  g,ids=self.fixture(8);r=c.freeze(g,'e');old=g.bucket.copy;barrier=threading.Barrier(4);lock=threading.Lock();count={'active':0,'peak':0,'started':0}
  def copied(*a,**kw):
   with lock:count['active']+=1;count['started']+=1;first=count['started']<=4;count['peak']=max(count['peak'],count['active'])
   try:
    if first:barrier.wait(timeout=5)
    return old(*a,**kw)
   finally:
    with lock:count['active']-=1
  g.bucket.copy=copied;self.assertEqual(copy_selected(g,{'epoch':'e','proof_copy_policy':POLICY},r,{i.id:[0]for i in ids},None),{});self.assertEqual(count['peak'],4)
 def test_incomplete_head_population_cannot_finalize_or_redraw(self):
  g,ids=self.fixture();old=g.bucket.head_object
  def head(**kw):
   if ids[0].id in kw['Key']:raise RuntimeError('HEAD unavailable')
   return old(**kw)
  g.bucket.head_object=head
  with self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(g,'e')
  self.assertNotIn('frozen_receipts',g.epochs['e']);self.assertEqual(g.epochs['e']['rejections'],{})
  g.bucket.head_object=old
  with patch.object(g.bucket,'get_object',side_effect=AssertionError('no original commitment reread')):self.assertEqual(len(c.freeze(g,'e')),6)
 def test_actual_controller_draw_copies_only_selected_then_original_jobs_score(self):
  from test_commitment_production import Tests
  case=Tests();case.setUp();self.addCleanup(case.doCleanups)
  case.m['proof_copy_policy']=POLICY;case.g.epochs['e']['commitment_binding']['proof_copy_policy']=POLICY
  # This runs the production finalize method, real authenticated commitment
  # parsing, postfreeze draw and conditional object-copy boundary. Fake GPU
  # reports represent only execution; no real model claim is made here.
  case.test_zero_allocations_no_job_and_no_heavy_download()
  self.assertEqual(len(case.b.copies),1)
  self.assertEqual(len(case.g.epochs['e']['frozen_receipts']),3)
 def test_public_history_never_advertises_unselected_proof_urls(self):
  import tempfile
  from pathlib import Path
  from types import SimpleNamespace
  from subnet.publication import frozen_submission
  g,ids=self.fixture(2);r=c.freeze(g,'e');m=dict(epoch='e',checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64},submission_transport_policy=c.VERSION,proof_copy_policy=POLICY)
  with tempfile.TemporaryDirectory()as folder:
   controller=SimpleNamespace(gateway=g,bucket=g.bucket,state=Path(folder));miner=ids[0].id
   before=frozen_submission(controller,m,miner,r[miner]);self.assertNotIn('url',before['artifacts'][0]);self.assertEqual(before['artifacts'][0]['availability'],'not-publicly-copied')
   self.assertEqual(copy_selected(g,m,r,{miner:[0]},None),{})
   after=frozen_submission(controller,m,miner,r[miner]);self.assertIn('url',after['artifacts'][0]);self.assertEqual(after['artifacts'][0]['availability'],'copied-selected-proof')
 def test_tampered_original_population_or_copy_journal_refused(self):
  g,ids=self.fixture();r=c.freeze(g,'e');miner=ids[0].id;m=dict(epoch='e',proof_copy_policy=POLICY)
  fake=copy.deepcopy(r);fake[miner]['artifacts'][0]['etag']='forged'
  with self.assertRaisesRegex(ValueError,'population'):copy_selected(g,m,fake,{miner:[0]},None)
  copy_selected(g,m,r,{miner:[0]},None);g.epochs['e']['selected_proof_copies'][miner]['0']['sha256']='bad'
  with self.assertRaisesRegex(ValueError,'journal'):copy_selected(g,m,r,{miner:[0]},None)
 def test_copy_failure_persists_original_draw_and_rejects_substitute(self):
  g,ids=self.fixture();r=c.freeze(g,'e');m=dict(epoch='e',proof_copy_policy=POLICY);g.bucket.fail_copy=lambda key:True
  self.assertIn(ids[0].id,copy_selected(g,m,r,{ids[0].id:[0]},None))
  with self.assertRaisesRegex(ValueError,'redraw'):copy_selected(g,m,r,{ids[1].id:[0]},None)
 def test_queue_requires_signed_matching_copy_receipt_for_opt_in_children(self):
  from test_commitment_child_queue import ChildQueueTests,sign
  case=ChildQueueTests();case.setUp();self.addCleanup(case.doCleanups)
  manifest=copy.deepcopy(case.manifest);manifest['proof_copy_policy']=POLICY;miner=case.identity.id;artifact=manifest['audit_frozen_receipts'][miner]['artifacts'][0]
  artifact.update(key='private/staging/original',etag='original-completed-etag');url=artifact.pop('read_url')
  binding={k:artifact[k]for k in ('sha256','size','etag','key','frozen_key')};manifest['proof_copy_receipts']={miner:{'0':dict(binding,read_url=url)}}
  job=copy.deepcopy(case.job);job['manifest']=sign(case.root,manifest);self.assertTrue(case.enqueue(job))
  for mutation in ('missing','etag','sha256','read_url'):
   forged=copy.deepcopy(manifest)
   if mutation=='missing':del forged['proof_copy_receipts']
   else:forged['proof_copy_receipts'][miner]['0'][mutation]='wrong'
   bad=copy.deepcopy(case.job);bad['job_id']='bad-'+mutation;bad['manifest']=sign(case.root,forged)
   with self.assertRaises(ValueError):case.enqueue(bad)
 def test_capabilities_created_only_after_actual_copies_and_reused(self):
  from subnet.selected_proof_copy import signed_copy_inventory
  g,ids=self.fixture();r=c.freeze(g,'e');m=dict(epoch='e',proof_copy_policy=POLICY)
  with patch.object(g.bucket,'presign',side_effect=AssertionError('no unavailable capability')):self.assertEqual(signed_copy_inventory(g,'e'),{})
  copy_selected(g,m,r,{ids[0].id:[0]},None);first=signed_copy_inventory(g,'e')
  with patch.object(g.bucket,'presign',side_effect=AssertionError('no renewable signed job mutation')):self.assertEqual(signed_copy_inventory(g,'e'),first)
 def test_four_heads_overlap_with_serial_owner_journals(self):
  g,ids,journals=BoundedCommitmentFreeze().fixture(12);g.epochs['e']['commitment_binding']['proof_copy_policy']=POLICY
  old=g.bucket.head_object;barrier=threading.Barrier(4);lock=threading.Lock();count={'active':0,'peak':0,'started':0}
  def head(**kw):
   with lock:count['active']+=1;count['started']+=1;first=count['started']<=4;count['peak']=max(count['peak'],count['active'])
   try:
    if first:barrier.wait(timeout=5)
    return old(**kw)
   finally:
    with lock:count['active']-=1
  g.bucket.head_object=head;self.assertEqual(len(c.freeze(g,'e')),12);self.assertEqual(count['peak'],4);self.assertEqual({thread for thread,_ in journals},{threading.get_ident()})
 def test_confirmed_missing_child_rejects_only_incomplete_miner_not_others(self):
  from botocore.exceptions import ClientError
  for code in ('NoSuchKey','NotFound','404'):
   with self.subTest(code=code):
    g,ids=self.fixture();miner=ids[0].id;old=g.bucket.head_object
    def head(**kw):
     if miner in kw['Key']:raise ClientError({'Error':{'Code':code},'ResponseMetadata':{'HTTPStatusCode':404}},'HeadObject')
     return old(**kw)
    g.bucket.head_object=head;receipts=c.freeze(g,'e');self.assertEqual(set(receipts),{i.id for i in ids[1:]});self.assertEqual(g.epochs['e']['rejections'],{miner:'missing completed declared artifact'});self.assertNotIn('commitment_metadata_incomplete',g.epochs['e']);self.assertEqual(g.bucket.copies,[])
 def test_actual_missing_bucket_access_and_503_remain_global_infrastructure(self):
  from botocore.exceptions import ClientError
  for code,status in (('NoSuchBucket',404),('AccessDenied',403),('ServiceUnavailable',503),('503',503)):
   with self.subTest(code=code):
    g,ids=self.fixture();miner=ids[0].id;old=g.bucket.head_object
    def head(**kw):
     if miner in kw['Key']:raise ClientError({'Error':{'Code':code},'ResponseMetadata':{'HTTPStatusCode':status}},'HeadObject')
     return old(**kw)
    g.bucket.head_object=head
    with self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(g,'e')
    self.assertNotIn('frozen_receipts',g.epochs['e']);self.assertEqual(g.epochs['e']['rejections'],{});self.assertEqual(g.bucket.copies,[]);self.assertEqual(g.epochs['e']['commitment_metadata_incomplete']['reason'],'child_HEAD_infrastructure_incomplete')
 def test_successful_head_returning_after_cutoff_is_global_budget_not_fraud(self):
  g,ids=self.fixture();clock=[0];g.epochs['e']['commitment_binding']['freeze_until']=25;old=g.bucket.head_object
  def late(**kw):
   value=old(**kw);clock[0]=30;return value
  g.bucket.head_object=late
  with patch('subnet.selected_proof_copy.time.time',side_effect=lambda:clock[0]),self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(g,'e')
  self.assertNotIn('frozen_receipts',g.epochs['e']);self.assertEqual(g.epochs['e']['rejections'],{});self.assertEqual(g.epochs['e']['commitment_metadata_incomplete']['reason'],'child_HEAD_budget_incomplete');self.assertEqual(g.bucket.copies,[])
