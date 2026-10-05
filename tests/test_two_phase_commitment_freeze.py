import copy,json,threading,unittest
from types import MethodType
from unittest.mock import patch
from test_commitment_freeze_prefetch import BoundedCommitmentFreeze
from subnet import commitment_transport as c
from subnet.storage import Bucket as ProductionBucket

class Controls(unittest.TestCase):
 def fixture(self,count):return BoundedCommitmentFreeze().fixture(count)
 def test_same_valid_signature_with_pretty_outer_json_is_structural_not_queueable(self):
  g,ids,j=self.fixture(4);miner=ids[-1].id;key='private/e/commitments/'+miner+'.json'
  original=g.bucket.objects[key][0];signed=json.loads(original);pretty=json.dumps(signed,indent=2).encode()
  self.assertNotEqual(pretty,original)
  with self.assertRaisesRegex(ValueError,'canonical commitment envelope'):c.validate(pretty,'e',miner)
  self.assertEqual(c.validate(original,'e',miner),signed)
  g.bucket.put(key,pretty);result=c.freeze(g,'e')
  self.assertNotIn(miner,result);self.assertEqual(g.epochs['e']['rejections'][miner],'malformed commitment');self.assertNotIn(miner,g.epochs['e']['commitment_pending'])
 def listing(self,gateway,pages):
  b=gateway.bucket;b.close=lambda:None;b.commitment_read_client=lambda:b;b.calls=[];it=iter(pages)
  def listed(**kw):
   b.calls.append(kw)
   item=next(it)
   if isinstance(item,Exception):raise item
   return item
  b.list_objects_v2=listed;b.complete_commitment_listing=MethodType(ProductionBucket.complete_commitment_listing,b)
 def test_paginated_exact_namespace_avoids_missing_uid_gets_without_trusting_content(self):
  g,ids,j=self.fixture(246);present=ids[-1].id
  for key in list(g.bucket.objects):
   if '/commitments/'in key and present not in key:del g.bucket.objects[key]
  self.listing(g,[dict(Contents=[dict(Key='private/e/commitments/ignored-extra.json'),dict(Key='private/other/commitments/'+ids[0].id+'.json')],IsTruncated=True,NextContinuationToken='actual-next'),dict(Contents=[dict(Key='private/e/commitments/'+present+'.json')],IsTruncated=False)])
  actual=g.bucket.get_object;reads=[]
  g.bucket.get_object=lambda **kw:(reads.append(kw['Key'])or actual(**kw))
  result=c.freeze(g,'e');self.assertEqual(set(result),{present});self.assertEqual(reads,['private/e/commitments/'+present+'.json']);self.assertEqual(g.bucket.calls[1]['ContinuationToken'],'actual-next');self.assertTrue(g.epochs['e']['commitment_capture_complete'])
 def test_list_error_partial_pages_never_becomes_empty_finalized_roster(self):
  g,ids,j=self.fixture(4);self.listing(g,[dict(Contents=[],IsTruncated=True,NextContinuationToken='next'),RuntimeError('actual listing outage')])
  with patch.object(g.bucket,'get_object',side_effect=AssertionError('no incomplete discovery GET')),self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(g,'e')
  self.assertNotIn('frozen_receipts',g.epochs['e']);self.assertEqual(g.bucket.copies,[]);self.assertNotIn('commitment_discovery',g.epochs['e'])
 def test_truncated_page_requires_fresh_continuation_and_failure_not_no_submission(self):
  g,ids,j=self.fixture(4);self.listing(g,[dict(Contents=[],IsTruncated=True)])
  with self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(g,'e')
  self.assertEqual(g.bucket.copies,[])
 def test_missing_pagination_flag_and_malformed_listing_do_not_finalize(self):
  for page in (dict(Contents=[]),dict(Contents='bad',IsTruncated=False),dict(Contents=[dict(Key=7)],IsTruncated=False)):
   g,ids,j=self.fixture(4);self.listing(g,[page])
   with self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(g,'e')
   self.assertNotIn('frozen_receipts',g.epochs['e']);self.assertEqual(g.bucket.copies,[])
 def test_conditional_etag_failure_is_infrastructure_not_fraud_and_keeps_original(self):
  from botocore.exceptions import ClientError
  g,ids,j=self.fixture(4);miner=ids[-1].id;old=g.bucket.copy
  def copied(key,destination,expected_etag=None):
   if miner in key:raise ClientError({'Error':{'Code':'PreconditionFailed'}},'CopyObject')
   return old(key,destination,expected_etag=expected_etag)
  g.bucket.copy=copied
  with self.assertRaises(ClientError):c.freeze(g,'e')
  original=copy.deepcopy(g.epochs['e']['commitment_pending'][miner]);self.assertNotIn(miner,g.epochs['e']['rejections']);self.assertNotIn('frozen_receipts',g.epochs['e'])
  seen=[]
  def recovered(key,destination,expected_etag=None):seen.append((key,expected_etag));return old(key,destination,expected_etag=expected_etag)
  g.bucket.copy=recovered;result=c.freeze(g,'e')
  self.assertEqual(result[miner]['commitment_document'],original['document']);self.assertIn((original['key'],original['etag']),seen)
 def test_all_tiny_captured_before_four_actual_concurrent_copies_serial_journals(self):
  g,ids,j=self.fixture(12);b=g.bucket;original=b.copy;barrier=threading.Barrier(4);lock=threading.Lock();counts=dict(active=0,peak=0,first=0)
  def copied(key,destination,expected_etag=None):
   self.assertTrue(g.epochs['e']['commitment_capture_complete']);self.assertEqual(len(g.epochs['e']['commitment_pending']),12)
   with lock:counts['active']+=1;counts['first']+=1;first=counts['first']<=4;counts['peak']=max(counts['peak'],counts['active'])
   try:
    if first:barrier.wait(timeout=5)
    return original(key,destination,expected_etag)
   finally:
    with lock:counts['active']-=1
  b.copy=copied;result=c.freeze(g,'e');self.assertEqual(len(result),12);self.assertEqual(counts['peak'],4);self.assertEqual({thread for thread,_ in j},{threading.get_ident()})
 def test_expensive_first_copy_cannot_prevent_later_tiny_capture(self):
  g,ids,j=self.fixture(12);clock=[0];g.epochs['e']['commitment_binding']['freeze_until']=25;old=g.bucket.copy
  def slow(key,dest,expected_etag=None):
   self.assertEqual(len(g.epochs['e']['commitment_pending']),12);clock[0]=30;return old(key,dest,expected_etag)
  g.bucket.copy=slow
  with patch('subnet.commitment_transport.time.time',side_effect=lambda:clock[0]):result=c.freeze(g,'e')
  self.assertEqual(result,{});self.assertEqual(len(g.epochs['e']['commitment_pending']),12);self.assertTrue(g.epochs['e']['commitment_capture_complete']);self.assertEqual(len(g.epochs['e']['commitment_deferred']),12);self.assertEqual(g.epochs['e']['rejections'],{})
 def test_persisted_fair_order_and_original_ETags_survive_partial_copy_failure(self):
  g,ids,j=self.fixture(8);b=g.bucket;miner=ids[-1].id;b.fail_copy=lambda key:miner in key
  with self.assertRaises(RuntimeError):c.freeze(g,'e')
  order=copy.deepcopy(g.epochs['e']['commitment_copy_order']);original=copy.deepcopy(g.epochs['e']['commitment_pending'][miner]);b.fail_copy=None
  with patch.object(b,'get_object',side_effect=AssertionError('no original commitment reread')),patch('secrets.token_hex',side_effect=AssertionError('no redraw')):
   result=c.freeze(g,'e')
  self.assertEqual(order,g.epochs['e']['commitment_copy_order']);self.assertEqual(result[miner]['commitment_document'],original['document']);self.assertEqual(len(result),8)
