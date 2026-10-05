import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from subnet.chain import ChainAdapter,activation_message

class Key:
 def __init__(self,ss58_address,crypto_type):self.hotkey=ss58_address;self.public_key=ss58_address.encode()
 def verify(self,message,signature):return signature==b'x'*64 and message==activation_message(self.hotkey)
def commitment(hotkey,signature=b'x'*64):
 text='affine2|activate|'+hotkey+'|'+base64.urlsafe_b64encode(signature).decode().rstrip('=');raw=text.encode();return ((len(raw)<<2)|1).to_bytes(2,'little')+raw
class FakeChain:
 block=123
 def __init__(self,rows,uids,keys):self.rows=rows;self.uids=uids;self.keys=keys;self.calls=[]
 def query_map(self,name,params,block):
  self.calls.append((name,params,block));return {'revealed':self.rows,'uids':self.uids,'keys':self.keys}[name]
 def query(self,*args,**kwargs):raise AssertionError('per-identity query forbidden')
class Tests(unittest.TestCase):
 def adapter(self,chain):
  a=ChainAdapter.__new__(ChainAdapter);a.chain=chain;a.bt=SimpleNamespace(storage=SimpleNamespace(Commitments=SimpleNamespace(RevealedCommitments='revealed'),SubtensorModule=SimpleNamespace(Uids='uids',Keys='keys')));a.netuid=120;a.owner='owner';a.keypair_type=Key;return a
 def test_many_identities_exact_same_block_three_maps_no_scalar_queries(self):
  c=FakeChain([(f'h{i}',[(commitment(f'h{i}'),100)])for i in range(245)],[(f'h{i}',i)for i in range(245)],[(i,f'h{i}')for i in range(245)])
  r=self.adapter(c).registrations();self.assertEqual(len(r),245);self.assertEqual(len(c.calls),3);self.assertTrue(all(row[2]==123 for row in c.calls));self.assertEqual(r['h131'],dict(uid=131,public_key=b'h131'.hex(),activate_block=100,snapshot_block=123))
 def test_signature_owner_stale_reverse_and_missing_uid_still_rejected(self):
  rows=[(h,[(commitment(h,b'y'*64 if h=='bad'else b'x'*64),100)])for h in ['owner','good','bad','missing','stale']]
  c=FakeChain(rows,[('owner',0),('good',85),('bad',1),('stale',2)],[(0,'owner'),(85,'good'),(1,'bad'),(2,'other')]);self.assertEqual(list(self.adapter(c).registrations()),['good'])
 def test_ambiguous_maps_fail_closed(self):
  for uids,keys in [([('h',1),('h',1)],[(1,'h')]),([('h',1)],[(1,'h'),(1,'h')]),([('h',True)],[(1,'h')]),([('h',1),('other',1)],[(1,'h')]),([('h',1)],[(1,'h'),(2,'h')]),([('h',1)],[('1','h')])]:
   with self.subTest(uids=uids,keys=keys),self.assertRaisesRegex(ValueError,'ambiguous'):self.adapter(FakeChain([],uids,keys)).registrations()
 def test_malformed_commitment_ignored_without_affecting_valid_identity(self):
  c=FakeChain([('bad',[(b'bad',100)]),('good',[(commitment('good'),101)])],[('good',0)],[(0,'good')]);self.assertEqual(list(self.adapter(c).registrations()),['good'])
 def test_snapshot_block_read_once_even_chain_advances(self):
  class Advancing(FakeChain):
   reads=0
   @property
   def block(self):self.reads+=1;return 200+self.reads
  c=Advancing([('h',[(commitment('h'),5)])],[('h',1)],[(1,'h')]);r=self.adapter(c).registrations();self.assertEqual(c.reads,1);self.assertEqual(r['h']['snapshot_block'],201);self.assertTrue(all(v[2]==201 for v in c.calls))

 def test_repeated_activation_entries_preserve_original_latest_entry(self):
  c=FakeChain([('h',[(commitment('h'),90),(commitment('h'),100)])],[('h',85)],[(85,'h')]);r=self.adapter(c).registrations();self.assertEqual(r['h']['activate_block'],100);self.assertEqual(len(c.calls),3)
 def test_map_network_failure_never_fabricates_empty_roster(self):
  c=FakeChain([],[],[])
  def failed(*args,**kwargs):raise TimeoutError('RPC observation loss')
  c.query_map=failed
  with self.assertRaises(TimeoutError):self.adapter(c).registrations()
