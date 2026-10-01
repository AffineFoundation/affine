import unittest
from subnet.native_role_batch import describe_batch

def sample(label,identity):
    return {'schema':'native-role-sample-v1','checkpoint':'a'*64,'environment_index':0,'task_hash':'b'*64,'classification':label,'files':{'receipts.json':{'sha256':identity,'size':10}},'payable':False,'production_admitted':False}
class NativeBatchTests(unittest.TestCase):
    def test_distinct_verified_positive_negative_contract(self):
        batch=describe_batch([sample('positive','c'*64),sample('negative','d'*64)],1,1)
        self.assertEqual((batch['K'],batch['L']),(1,1));self.assertFalse(batch['production_admitted'])
    def test_negative_only_never_claims_KL(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):describe_batch([sample('negative','c'*64)],1,1)
    def test_same_artifact_never_becomes_multiple_experience(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):describe_batch([sample('positive','c'*64),sample('negative','c'*64)],1,1)
    def test_different_task_cannot_enter_same_batch(self):
        b=sample('negative','d'*64);b['task_hash']='e'*64
        with self.assertRaisesRegex(ValueError,'environment'):describe_batch([sample('positive','c'*64),b],1,1)

class NativePreferenceTests(unittest.TestCase):
    def records(self):
        p=sample('positive','c'*64);n=sample('negative','d'*64)
        for x,out in ((p,[2]),(n,[3])):x['training_view']=[{'role':'user','training_eligible':False,'prompt':[0],'output':[9]},{'role':'agent','training_eligible':True,'prompt':[1],'output':out}]
        return p,n
    def test_preference_targets_only_agent_tokens(self):
        from subnet.native_role_batch import preference_pair
        pair=preference_pair(*self.records());self.assertEqual(pair['chosen'],[2]);self.assertEqual(pair['rejected'],[3]);self.assertFalse(pair['auxiliary_tokens_in_loss'])
    def test_different_agent_prompt_rejected(self):
        from subnet.native_role_batch import preference_pair
        p,n=self.records();n['training_view'][1]['prompt']=[4]
        with self.assertRaisesRegex(ValueError,'prompt'):preference_pair(p,n)

class NativeRoleProvenanceTests(unittest.TestCase):
    def test_contract_policy_and_source_equal_actual_plan(self):
        from subnet.native_role_batch import require_declared_roles
        plan={'checkpoint':{'id':'a'*64},'curated_sources':{'subnet/native_tau2_curated.py':'b'*64},'runtime_profile':{'dtype':'float32'}}
        contract={'roles':{r:{'checkpoint':'a'*64,'source_hash':'b'*64,'numerical_policy':{'dtype':'float32'}} for r in ('agent','user')}}
        require_declared_roles(contract,plan)
        contract['roles']['user']['numerical_policy']={'dtype':'float16'}
        with self.assertRaisesRegex(ValueError,'provenance'):require_declared_roles(contract,plan)
    def test_role_source_cannot_be_mislabelled(self):
        from subnet.native_role_batch import require_declared_roles
        plan={'checkpoint':{'id':'a'*64},'curated_sources':{'subnet/native_tau2_curated.py':'b'*64},'runtime_profile':{}}
        contract={'roles':{r:{'checkpoint':'a'*64,'source_hash':'c'*64,'numerical_policy':{}} for r in ('agent','user')}}
        with self.assertRaisesRegex(ValueError,'provenance'):require_declared_roles(contract,plan)

class NativeSemanticDuplicatesTests(unittest.TestCase):
    def test_changed_receipt_timestamps_do_not_make_new_experience(self):
        p=sample('positive','c'*64);n=sample('negative','d'*64)
        p['trajectory_hash']=n['trajectory_hash']='e'*64
        with self.assertRaisesRegex(ValueError,'duplicate'):describe_batch([p,n],1,1)
