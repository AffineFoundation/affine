import copy
import threading
import unittest
from unittest.mock import patch
from test_commitment_production import Bucket
from subnet import commitment_transport as c
from subnet.storage import Identity,canonical
from types import SimpleNamespace

class BoundedCommitmentFreeze(unittest.TestCase):
 def fixture(self,count):
  ids=[Identity()for _ in range(count)];bucket=Bucket();snapshots=[]
  state=dict(start=0,deadline=20,miners=set(i.id for i in ids),max_batches=3,commitment_binding=dict(checkpoint='a'*64,source='b'*64))
  gateway=SimpleNamespace(bucket=bucket,epochs={'e':state},persist=lambda:snapshots.append((threading.get_ident(),copy.deepcopy(state))))
  manifest=dict(epoch='e',checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64})
  for i,key in enumerate(ids):
   data=('opaque pair '+str(i)).encode();doc=c.make(key,manifest,[(dict(env_id='math',index=i),data)])
   bucket.put('private/e/commitments/'+key.id+'.json',canonical(doc));bucket.put('private/e/staging/'+key.id+'/0.zip',data)
  return gateway,ids,snapshots
 def test_four_actual_gets_overlap_but_all_journals_serial_before_copy(self):
  gateway,ids,journals=self.fixture(12);bucket=gateway.bucket;original=bucket.get_object;lock=threading.Lock();barrier=threading.Barrier(4);counts=dict(active=0,peak=0,started=0)
  def get(**kwargs):
   with lock:counts['active']+=1;counts['started']+=1;first=counts['started']<=4;counts['peak']=max(counts['peak'],counts['active'])
   try:
    if first:barrier.wait(timeout=5)
    return original(**kwargs)
   finally:
    with lock:counts['active']-=1
  oldcopy=bucket.copy
  def copied(key,destination,expected_etag=None):
   if key.endswith('json'):
    miner=key.split('/')[-1][:-5]
    self.assertTrue(any(miner in s.get('commitment_pending',{})for _,s in journals))
   return oldcopy(key,destination,expected_etag)
  bucket.get_object=get;bucket.copy=copied
  result=c.freeze(gateway,'e')
  self.assertEqual(len(result),12);self.assertEqual(counts['peak'],4);self.assertEqual(counts['started'],12)
  self.assertEqual({thread for thread,_ in journals},{threading.get_ident()});self.assertEqual(bucket.heavy_reads,0)
 def test_recovery_uses_exact_pending_bytes_without_rereading_mutable_commitment(self):
  gateway,ids,journals=self.fixture(5);bucket=gateway.bucket;miner=ids[0].id;bucket.fail_copy=lambda key:miner in key
  with self.assertRaises(RuntimeError):c.freeze(gateway,'e')
  pending=copy.deepcopy(gateway.epochs['e']['commitment_pending'][miner]);original=bucket.get_object
  def get(**kwargs):
   self.assertNotIn(miner,kwargs['Key'],'journaled pending commitment must not be read again')
   return original(**kwargs)
  bucket.get_object=get;bucket.fail_copy=None
  result=c.freeze(gateway,'e');self.assertEqual(result[miner]['commitment_document'],pending['document']);self.assertEqual(len(result),5)
 def test_cutoff_launches_no_speculative_get_and_has_no_fraud_penalty(self):
  gateway,ids,_=self.fixture(12);gateway.epochs['e']['commitment_binding']['freeze_until']=25
  with patch('subnet.commitment_transport.time.time',return_value=30),patch.object(gateway.bucket,'get_object',side_effect=AssertionError('no expired GET')):
   self.assertEqual(c.freeze(gateway,'e'),{})
  self.assertEqual(len(gateway.epochs['e']['commitment_deferred']),12);self.assertEqual(gateway.epochs['e']['rejections'],{})
 def test_malformed_prefetched_sibling_does_not_block_authenticated_other_miners(self):
  gateway,ids,_=self.fixture(12);miner=ids[0].id;gateway.bucket.put('private/e/commitments/'+miner+'.json',b'x'*(c.MAX_BYTES+100))
  result=c.freeze(gateway,'e');self.assertEqual(len(result),11);self.assertIn(miner,gateway.epochs['e']['rejections']);self.assertNotIn(miner,result);self.assertEqual(gateway.bucket.heavy_reads,0)

 def test_public_receipt_put_retry_reuses_exact_frozen_journal_without_fetch_copy(self):
  gateway,ids,_=self.fixture(5);bucket=gateway.bucket;original=bucket.json;attempts=[]
  def publish(key,value):
   attempts.append(key)
   if len(attempts)==1:raise RuntimeError('temporary public PUT outage')
   return original(key,value)
  bucket.json=publish
  with self.assertRaisesRegex(RuntimeError,'PUT outage'):c.freeze(gateway,'e')
  frozen=copy.deepcopy(gateway.epochs['e']['frozen_receipts'])
  with patch.object(bucket,'get_object',side_effect=AssertionError('no repeat fetch')),patch.object(bucket,'copy',side_effect=AssertionError('no repeat copy')):
   self.assertEqual(c.freeze(gateway,'e'),frozen)
  self.assertEqual(attempts,['public/e/receipts.json']*2)
  self.assertEqual(bucket.objects['public/e/receipts.json'][0],canonical(frozen))
