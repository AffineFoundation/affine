import copy,unittest
from unittest.mock import AsyncMock,patch
from subnet import weight_submission_transaction as t
class FinalizedEra(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  self.guard=dict(block=103,block_hash='0x'+format(103,'064x'),owner='public',netuid=120,mecid=0,pending=[],schedule=dict(Tempo=360,RevealPeriodEpochs=1,LastEpochBlock=90,PendingEpochAt=0,SubnetEpochIndex=10,BlocksSinceLastStep=13),SDK_computed_round=500,chosen_encryption_round=500,selection_reason='SDK_no_same_epoch_pending')
  self.fresh=copy.deepcopy(self.guard);self.fresh['block']=105;self.fresh['block_hash']='0x'+format(105,'064x');self.fresh['schedule']['BlocksSinceLastStep']=15;self.fresh['already_available_same_epoch_rounds']=[]
  self.era=dict(period=64,birth=100,death=164,block_hash='0x'+format(100,'064x'))
  class S:
   async def block_hash(self,n):return '0x'+format(n,'064x')
  self.s=S()
 async def check(self):
  with patch.object(t,'_epoch_snapshot',new=AsyncMock(return_value=self.fresh)):return await t._check_epoch_before_sign(self.s,self.guard,self.era)
 async def test_finalized_birth_before_planning_best_head_passes(self):
  self.assertLess(self.era['birth'],self.guard['block']);r=await self.check();self.assertEqual(r['era'],self.era)
 async def test_birth_at_planning_still_passes(self):
  self.era=dict(self.era,birth=103,death=167,block_hash='0x'+format(103,'064x'));await self.check()
 async def test_future_birth_and_expiry_refuse(self):
  for birth in (106,41):
   self.era=dict(self.era,birth=birth,death=birth+64,block_hash='0x'+format(birth,'064x'))
   with self.subTest(birth=birth),self.assertRaisesRegex(ValueError,'era'):await self.check()
 async def test_different_finalized_anchor_refuses(self):
  self.era['block_hash']='0x'+'f'*64
  with self.assertRaisesRegex(ValueError,'birth block'):await self.check()
 async def test_different_planning_ancestor_refuses(self):
  self.guard['block_hash']='0x'+'f'*64
  with self.assertRaisesRegex(ValueError,'planning block'):await self.check()
 async def test_fresh_snapshot_reorg_refuses(self):
  self.fresh['block_hash']='0x'+'f'*64
  with self.assertRaisesRegex(ValueError,'fresh planning'):await self.check()
 async def test_regressing_head_refuses(self):
  self.fresh['block']=102;self.fresh['block_hash']='0x'+format(102,'064x')
  with self.assertRaisesRegex(ValueError,'era'):await self.check()
 async def test_period_or_era_corruption_refuses(self):
  for changed in ({'period':128},{'birth':True},{'death':165}):
   old=self.era;self.era=dict(old,**changed)
   with self.subTest(changed=changed),self.assertRaises(ValueError):await self.check()
   self.era=old
 async def test_older_anchor_does_not_waive_queue_or_epoch_drift(self):
  self.fresh['pending']=[dict(storage='TimelockedWeightCommits',epoch=10,reveal_round=500)]
  with self.assertRaisesRegex(ValueError,'queue or epoch'):await self.check()
if __name__=='__main__':unittest.main()
