import asyncio,hashlib,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock,patch
from subnet import weight_submission_transaction as t
import bittensor.intents.weights as sdk

class FakeSubstrate:
 def __init__(self,rows=None):
  self.rows=rows or {};self.reads=[];self.composed=[];self.number=9250509
  self.schedule=dict(Tempo=360,RevealPeriodEpochs=1,LastEpochBlock=9250499,PendingEpochAt=0,SubnetEpochIndex=25630,BlocksSinceLastStep=10)
 async def block_number(self):return self.number
 async def block_hash(self,n):return '0x'+'a'*64
 async def query(self,module,name,params,block_hash):
  self.reads.append((module,name,params,block_hash));return None if module=='Drand' else self.schedule[name]
 async def query_map(self,module,name,params,block_hash):
  self.reads.append((module,name,params,block_hash));value=self.rows.get(name,[])
  if isinstance(value,Exception):raise value
  return value
 async def compose(self,call):
  self.composed.append(call)
  return NS(data=repr(call).encode())

class PendingRoundControls(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  self.clock=patch.object(t.time,'time',return_value=1692803367);self.clock.start();self.addCleanup(self.clock.stop)
 async def test_all_three_queues_owner_only_same_mechanism_and_block(self):
  s=FakeSubstrate({'TimelockedWeightCommits':[(25630,[('other',1,b'x',999),('owner',2,b'y',32935282)])], 'CRV3WeightCommits':[(25629,[('owner',b'z',32935280)])],'CRV3WeightCommitsV2':[(25630,[('owner',3,b'a',32935281)])]})
  result=await t._own_pending_rounds(s,'owner',120,0)
  self.assertEqual([r['reveal_round']for r in result['pending']],[32935282,32935280,32935281])
  self.assertTrue(all(x[0]=='SubtensorModule'and x[2]==[120]and x[3]=='0x'+'a'*64 for x in s.reads))
 async def test_unknown_or_unreadable_queue_refuses_before_build(self):
  for value in (TimeoutError(),[(1,[('owner',2,b'x')])],[(1,[('owner',2,b'x',-1)])],[(1,None)]):
   s=FakeSubstrate({'TimelockedWeightCommits':value})
   with patch.object(sdk,'_build_timelocked',new=AsyncMock())as build:
    with self.assertRaises(Exception):await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
    build.assert_not_called();self.assertEqual(s.composed,[])
 async def test_mechanism_refuses_without_queue_or_cipher(self):
  s=FakeSubstrate()
  with self.assertRaises(ValueError):await t._own_pending_rounds(s,'owner',120,1)
  self.assertEqual(s.reads,[])
 async def test_larger_SDK_round_keeps_original_exact_call(self):
  s=FakeSubstrate({'TimelockedWeightCommits':[(1,[('owner',2,b'x',20)])]});original=NS(call=NS(data=b'original'),extras={'reveal_round':21})
  with patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)),patch.object(sdk._core,'encrypt_at_round')as encrypt:
   built,guard=await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
  self.assertIs(built,original);encrypt.assert_not_called();self.assertEqual(guard['chosen_encryption_round'],21);self.assertFalse(guard['ciphertext_reencrypted'])
 async def test_equal_round_retains_SDK_cipher_and_queue_order(self):
  s=FakeSubstrate({'TimelockedWeightCommits':[(1,[('owner',2,b'x',20)])]});original=NS(call=NS(data=b'original'),extras={'reveal_round':20})
  with patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)),patch.object(sdk._core,'encrypt_at_round')as encrypt:
   built,guard=await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
  self.assertIs(built,original);encrypt.assert_not_called();self.assertEqual(guard['chosen_encryption_round'],20)
 async def test_schedule_and_pending_read_use_same_actual_block(self):
  s=FakeSubstrate()
  async def build(proxy,*args):
   self.assertEqual(await proxy.block_number(),9250509);self.assertEqual(await proxy.block_hash(9250509),'0x'+'a'*64)
   with self.assertRaises(ValueError):await proxy.block_hash(9250510)
   return NS(call=NS(data=b'actual'),extras={'reveal_round':20})
  with patch.object(sdk,'_build_timelocked',new=build):await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)

class ActualNativeCipherControls(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  self.clock=patch.object(t.time,'time',return_value=1692803367);self.clock.start();self.addCleanup(self.clock.stop)
 @classmethod
 def setUpClass(cls):
  p=Path(__file__).resolve().parent/'fixtures/weight-reveal-public-drand.json'
  assert hashlib.sha256(p.read_bytes()).hexdigest()=='1f17a5c009fd91dbeb734d2752f20890e05dab3a6c20a2f0d2c066ee1eb0c2b3'
  cls.pulses=json.loads(p.read_bytes())['pulses']
 def decrypt(self,inner,round_number):
  raw=t._compact(len(inner))+inner+round_number.to_bytes(8,'little')
  return sdk._core.decrypt_with_signature(raw,self.pulses[str(round_number)]['signature'].removeprefix('0x'))
 async def test_actual_cipher_bound_to_larger_pending_round_not_header_only(self):
  s=FakeSubstrate({'TimelockedWeightCommits':[(25630,[('owner',9250207,b'historical',32935282)])]})
  original=NS(call=NS(data=b'unused computed ciphertext'),extras={'reveal_round':32935279})
  with patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)):
   built,guard=await t._build_epoch_timelocked(s,'owner',bytes(range(32)),120,0,[2,255],[65535,12345],7)
  call=s.composed[0];self.assertEqual(call.function,'commit_timelocked_mechanism_weights');self.assertEqual(call.params['reveal_round'],32935282)
  self.assertEqual(built.extras['reveal_round'],32935282);self.assertTrue(guard['ciphertext_reencrypted'])
  plain=self.decrypt(call.params['commit'],32935282)
  expected=bytes.fromhex('80')+bytes(range(32))+bytes.fromhex('080200ff0008ffff39300700000000000000')
  self.assertEqual(plain,expected)
  with self.assertRaises(Exception):self.decrypt(call.params['commit'],32935279)
  self.assertEqual(guard['call_sha256'],hashlib.sha256(built.call.data).hexdigest())
 async def test_native_returned_round_mismatch_refuses_compose(self):
  s=FakeSubstrate({'TimelockedWeightCommits':[(25630,[('owner',2,b'x',32935282)])]});original=NS(call=NS(data=b'old'),extras={'reveal_round':32935279})
  with patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)),patch.object(sdk._core,'encrypt_at_round',return_value=(b'wrong',32935279)):
   with self.assertRaisesRegex(ValueError,'native encryption'):await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
  self.assertEqual(s.composed,[])
 def test_payload_integer_and_length_rejection(self):
  for hk,uids,values,key in [(b'x',[1],[2],0),(b'x'*32,[1],[2,3],0),(b'x'*32,[1,1],[2,3],0),(b'x'*32,[1],[65536],0),(b'x'*32,[1],[2],-1)]:
   with self.assertRaises(ValueError):t._weights_payload(hk,uids,values,key)
 def test_cipher_envelope_round_and_trailing_bytes_refused(self):
  wrapped,number=sdk._core.encrypt_at_round(b'public test',32935282)
  self.assertTrue(t._inner_ciphertext(wrapped,number))
  with self.assertRaises(ValueError):t._inner_ciphertext(wrapped,32935279)
  with self.assertRaises(ValueError):t._inner_ciphertext(wrapped+b'x',number)
 def test_SDK_core_extension_and_python_seams_are_both_pinned(self):
  paths=t.sdk_seam_paths();self.assertTrue(any(Path(p).name.startswith('bittensor_core.')and p.endswith('.so')for p in paths))
  self.assertTrue(any(p.endswith('bittensor_core/__init__.py')for p in paths))

class ExactEpochControls(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  self.clock=patch.object(t.time,'time',return_value=1692803367);self.clock.start();self.addCleanup(self.clock.stop)
 def substrate(self,rounds=(32935282,)):
  return FakeSubstrate({'TimelockedWeightCommits':[(25630,[('owner',9250207,b'old',r)for r in rounds])]})
 async def test_same_epoch_uses_exact_lower_round_not_monotonic_max(self):
  s=self.substrate();original=NS(call=NS(data=b'computed higher'),extras={'reveal_round':32935290})
  with patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)):
   result,guard=await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
  self.assertEqual(guard['chosen_encryption_round'],32935282)
  self.assertEqual(s.composed[0].params['reveal_round'],32935282)
  self.assertEqual(guard['selection_reason'],'same_epoch_exact_pending_round')
 async def test_mixed_or_legacy_same_epoch_refuses_before_encryption(self):
  for s in (self.substrate((32935282,32935279)), FakeSubstrate({'CRV3WeightCommits':[(25630,[('owner',b'x',32935282)])]})):
   with patch.object(sdk,'_build_timelocked',new=AsyncMock())as build:
    with self.assertRaises(ValueError):await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
    build.assert_not_called()
 async def test_past_different_epoch_round_does_not_delay_SDK(self):
  s=self.substrate();s.schedule['SubnetEpochIndex']=25631
  original=NS(call=NS(data=b'new epoch'),extras={'reveal_round':32940000})
  with patch.object(t.time,'time',return_value=1692803367+32935282*3),patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)),patch.object(sdk._core,'encrypt_at_round')as enc:
   result,guard=await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
  self.assertIs(result,original);enc.assert_not_called();self.assertEqual(guard['selection_reason'],'SDK_no_same_epoch_pending')
 async def test_auto_manual_safety_and_stale_round_refuse(self):
  changes=[{'LastEpochBlock':9250210}, {'PendingEpochAt':9250520},{'BlocksSinceLastStep':359}]
  for change in changes:
   s=self.substrate();s.schedule.update(change)
   with patch.object(sdk,'_build_timelocked',new=AsyncMock())as build:
    with self.assertRaisesRegex(ValueError,'epoch transition'):await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
    build.assert_not_called()
  with patch.object(t.time,'time',return_value=1692803367+(32935282-1)*3):
   with self.assertRaisesRegex(ValueError,'past-round'):await t._build_epoch_timelocked(self.substrate(),'owner',b'h'*32,120,0,[2],[65535],0)
 async def test_actual_available_pulse_refuses_even_if_local_clock_is_behind(self):
  s=self.substrate();original_query=s.query
  async def query(module,name,params,block_hash):
   if module=='Drand':return {'signature':'already-public'}
   return await original_query(module,name,params,block_hash)
  s.query=query
  with self.assertRaisesRegex(ValueError,'already available'):await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
 async def test_same_epoch_equal_preserves_cipher_and_refresh_checks(self):
  s=self.substrate();original=NS(call=NS(data=b'unchanged'),extras={'reveal_round':32935282})
  with patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)),patch.object(sdk._core,'encrypt_at_round')as enc:
   result,guard=await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
  self.assertIs(result,original);enc.assert_not_called()
  era=dict(period=64,birth=9250509,death=9250573,block_hash='0x'+'a'*64)
  checked=await t._check_epoch_before_sign(s,guard,era);self.assertEqual(checked['era'],era)
  s.rows['TimelockedWeightCommits'][0][1].append(('owner',9250510,b'new',32935282))
  with self.assertRaisesRegex(ValueError,'queue or epoch'):await t._check_epoch_before_sign(s,guard,era)
 async def test_changed_epoch_reorg_era_or_future_queue_refuse(self):
  s=self.substrate();original=NS(call=NS(data=b'unchanged'),extras={'reveal_round':32935282})
  with patch.object(sdk,'_build_timelocked',new=AsyncMock(return_value=original)):
   _,guard=await t._build_epoch_timelocked(s,'owner',b'h'*32,120,0,[2],[65535],0)
  era=dict(period=64,birth=9250509,death=9250573,block_hash='0x'+'a'*64)
  s.schedule['RevealPeriodEpochs']=2
  with self.assertRaisesRegex(ValueError,'queue or epoch'):await t._check_epoch_before_sign(s,guard,era)
  s.schedule['RevealPeriodEpochs']=1
  with patch.object(s,'block_hash',new=AsyncMock(return_value='0x'+'b'*64)):
   with self.assertRaisesRegex(ValueError,'planning block'):await t._check_epoch_before_sign(s,guard,era)
  with self.assertRaisesRegex(ValueError,'era'):await t._check_epoch_before_sign(s,guard,dict(era,birth=9250510))
  s.schedule['SubnetEpochIndex']=25629
  with self.assertRaisesRegex(ValueError,'future epoch'):t._epoch_round(await t._epoch_snapshot(s,'owner',120,0),32935282,era_death=9250573,now=1692803367)

class PulseArrivalOrdering(unittest.TestCase):
 @staticmethod
 def apply(queue, available, current):
  pending=[]
  for label,round_number in queue:
   if round_number in available:current=label
   else:pending.append((label,round_number))
  return pending,current
 def test_monotonic_round_counterexample_with_reordered_pulses(self):
  pending,current=self.apply([('old',20),('new',21)],{21},None)
  self.assertEqual(current,'new')
  _,current=self.apply(pending,{20,21},current)
  self.assertEqual(current,'old')
 def test_older_epoch_missing_pulse_is_expired_before_newer_epoch_applies(self):
  queues={10:[('old',20)],11:[('new',21)]};current=None
  for reveal_epoch,pulses in ((10,{21}),(11,{21}),(11,{20,21})):
   queues={e:q for e,q in queues.items()if e>=reveal_epoch}
   pending,current=self.apply(queues.pop(reveal_epoch,[]),pulses,current)
   if pending:queues[reveal_epoch]=pending
  self.assertEqual(current,'new');self.assertNotIn(10,queues)
 def test_exact_round_keeps_newest_for_every_unrelated_pulse_order(self):
  import itertools
  for order in itertools.permutations((18,20,21)):
   pending=[('old',20),('new',20)];available=set();current=None
   for pulse in order:
    available.add(pulse);pending,current=self.apply(pending,available,current)
    self.assertIn(current,(None,'new'))
   self.assertEqual((pending,current),([],'new'))

if __name__=='__main__':unittest.main()
