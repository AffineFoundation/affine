import copy,unittest
from ops.probe_rcore_model_search import validate_contract,validated_batch
from subnet.public_rcore import arithmetic_candidates

class ContractTests(unittest.TestCase):
    def plan(self):
        messages=[{'role':'user','content':'Evaluate (-5.80 * -5 * -4 % 3 / 2).\nThe answer is a number.'}]
        return dict(schema=1,experiment='original-rcore-public-arithmetic-model-search-v1',payable=False,chain_transactions=False,search_budget=16,indices=[0],environment=dict(id='affine_rcore',adapter='prime_v1',num_samples=64,max_turns=1,max_output_tokens=512,success_reward=1.,version='prime-v1-1'),public_messages=messages,original_task_snapshot_sha256='aed8f5dce6689a831cbec75d6d6e39835c119bcf8cd963d1c353d8e38846dd38',artifact_policy=dict(max_compressed_bytes=100000000,max_uncompressed_bytes=500000000,array_dtype='float32',max_turn_tokens=512,max_vocab_size=200000),harness=dict(version='text-tools-v1',policy='candidates',candidates=arithmetic_candidates(messages),max_output_tokens=512,temperature=4.,top_p=1.))
    def test_public_policy_contract(self):self.assertEqual(validate_contract(self.plan())['indices'],[0])
    def test_hidden_candidate_override_rejected(self):
        p=self.plan();p['harness']['candidates']=['private answer','wrong']
        with self.assertRaises(ValueError):validate_contract(p)
    def test_scope_and_budget_refused(self):
        for field,value in [('indices',[32]),('search_budget',True),('search_budget',33),('chain_transactions',True)]:
            p=self.plan();p[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_contract(p)
        for field,value in [('success_reward',True),('num_samples',True),('max_turns',2),('max_output_tokens',128)]:
            p=self.plan();p['environment'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_contract(p)
    def test_snapshot_and_artifact_geometry_bound(self):
        p=self.plan();p['original_task_snapshot_sha256']='0'*64
        with self.assertRaises(ValueError):validate_contract(p)
        p=self.plan();p['artifact_policy']['max_vocab_size']=1
        with self.assertRaises(ValueError):validate_contract(p)
    def test_frozen_actual_classes_and_checkpoint_bound(self):
        p=self.plan();p['checkpoint']={'id':'approved'};row=dict(index=0,positive=1,negative=1,qualifying_K1L1=True)
        rollouts=[dict(sample_index=0,index=0,env_id='affine_rcore',environment_version='prime-v1-1',classification=c)for c in ['positive','negative']];batch=dict(index=0,env_id='affine_rcore',checkpoint='approved',rollouts=rollouts)
        self.assertEqual(validated_batch(p,row,[(batch,[[],[]])])[2:],(1,1))
        for field,value in [('checkpoint','different'),('index',1)]:
            b=copy.deepcopy(batch);b[field]=value
            with self.assertRaises(ValueError):validated_batch(p,row,[(b,[[],[]])])
        b=copy.deepcopy(batch);b['rollouts'][1]['classification']='positive'
        with self.assertRaises(ValueError):validated_batch(p,row,[(b,[[],[]])])
if __name__=='__main__':unittest.main()
