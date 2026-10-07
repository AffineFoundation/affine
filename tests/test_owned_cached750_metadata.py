import hashlib,tempfile,unittest
from pathlib import Path
from ops.owned_cached750_metadata import fetch_metadata,compare_all750
class Bucket:
 def __init__(self):self.seen=[]
 def get(self,key):self.seen.append(key);return b'{}'
class Tokenizer:
 special_tokens_map={'eos_token':'EOS'};eos_token_id=2;chat_template='CPU template'
 def get_vocab(self):return {'word':1,'EOS':2}
class Session:
 def __init__(self):self.seen=[]
 def reset(self,index,seed):self.seen.append((index,seed));return dict(messages=[index],task_hash=hashlib.sha256(str(index).encode()).hexdigest())
class Tests(unittest.TestCase):
 def test_only_metadata_GET_even_with_weight_and_optimizer_members(self):
  bucket=Bucket();cp=dict(id='a'*64,files={n:hashlib.sha256(b'{}').hexdigest()for n in ['config.json','tokenizer.json','model.safetensors','optimizer.pt','model.safetensors.index.json']})
  with tempfile.TemporaryDirectory()as d:receipts=fetch_metadata(bucket,cp,Path(d)/'metadata')
  self.assertEqual(set(receipts),{'config.json','tokenizer.json'});self.assertEqual(len(bucket.seen),2)
 def test_all750_native_resets_no_conditional_drop_on_different_prompt(self):
  groups=[dict(indices=list(range(g*32,min(750,(g+1)*32))),seeds=[20261002+i*1000 for i in range(g*32,min(750,(g+1)*32))],harness={})for g in range(24)]
  session=Session();a=Tokenizer();b=Tokenizer();b.different=True
  def render(tokenizer,messages,tools,harness):return messages+[1 if getattr(tokenizer,'different',False)else 0]
  result=compare_all750(a,b,{}, {},session,groups,render)
  self.assertEqual(len(session.seen),750);self.assertEqual(len(result['rows']),750);self.assertFalse(result['passed'])
if __name__=='__main__':unittest.main()
