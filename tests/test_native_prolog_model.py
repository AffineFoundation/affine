import unittest,copy,json
from pathlib import Path
from ops.probe_native_prolog_model import validate_contract,validated_batch,candidates
class Tests(unittest.TestCase):
 def plan(self):
  import hashlib
  from subnet.native_prolog_actor import REVISION,BASE,SHIM_SHA
  starter='board_size(12).\nsolve(_) :- fail.  %% TODO: replace this\n'
  public={'revision':REVISION,'task_name':'prolog-nqueens-0005','original_index':5,'kind':'nqueens','messages':[{'role':'user','content':'Solve public NQueens'}],'starter_file':starter,'starter_sha256':hashlib.sha256(starter.encode()).hexdigest(),'source_files':{},'tools':[]}
  environment={'id':'affine_prolog','adapter':'prime_v1','version':'prime-v1-1','num_samples':3,'max_turns':2,'max_output_tokens':512,'success_reward':1.,'config':{'taskset':{'tasks':['prolog-nqueens-0005','prolog-nqueens-0014','prolog-nqueens-0023'],'num_examples':32,'difficulty':'medium'},'prolog_session_revision':'original-nqueens-common-session-v1','prolog_runtime':{'revision':REVISION,'base_image':BASE,'shim_sha256':SHIM_SHA,'image':'sha256:'+'a'*64}}}
  return {'schema':1,'experiment':'original-nqueens-common-model-search-v1','payable':False,'chain_transactions':False,'search_budget':16,'indices':[0],'environment':environment,'public_task':public,'artifact_policy':{'max_compressed_bytes':100000000,'max_uncompressed_bytes':500000000,'array_dtype':'float32','max_turn_tokens':512,'max_vocab_size':200000},'checkpoint':{'id':'a'*64},'harness':{'version':'text-tools-v1','policy':'candidates','max_output_tokens':512,'temperature':4.,'top_p':1.,'candidates':candidates(public),'turn_overrides':{'1':{'policy':'candidates','candidates':['Done']}}}}
 def fixture(self):
  p=self.plan();row={'index':0,'positive':1,'negative':1,'qualifying_K1L1':True};rollouts=[{'sample_index':0,'index':0,'env_id':'affine_prolog','environment_version':p['environment']['version'],'classification':label}for label in ['positive','negative']];batch={'index':0,'env_id':'affine_prolog','checkpoint':p['checkpoint']['id'],'rollouts':rollouts};return p,row,[(batch,[[],[]])]
 def test_exact_contract_and_real_metadata(self):
  p,row,batches=self.fixture();validate_contract(p);self.assertEqual(validated_batch(p,row,batches)[2:],(1,1))
 def test_incomplete_turn_or_boolean_success_rejected(self):
  for key,value in [('max_turns',1),('success_reward',True),('max_output_tokens',511)]:
   p=self.plan();p['environment'][key]=value
   with self.assertRaises(ValueError):validate_contract(p)
 def test_changed_candidate_or_terminal_harness_rejected(self):
  p=self.plan();p['harness']['turn_overrides']['1']['candidates']=['foreign']
  with self.assertRaises(ValueError):validate_contract(p)
 def test_frozen_checkpoint_and_index_rejected(self):
  for key,value in [('checkpoint','b'*64),('index',1),('env_id','affine_numina')]:
   p,row,batches=self.fixture();batches[0][0][key]=value
   with self.assertRaises(ValueError):validated_batch(p,row,batches)
 def test_actual_classes_cannot_be_relabelled(self):
  p,row,batches=self.fixture();batches[0][0]['rollouts'][1]['classification']='positive'
  with self.assertRaises(ValueError):validated_batch(p,row,batches)
 def test_metadata_counts_and_rollout_task_binding_rejected(self):
  p,row,batches=self.fixture();row['negative']=0
  with self.assertRaises(ValueError):validated_batch(p,row,batches)
  p,row,batches=self.fixture();batches[0][0]['rollouts'][1]['sample_index']=True
  with self.assertRaises(ValueError):validated_batch(p,row,batches)
if __name__=='__main__':unittest.main()
