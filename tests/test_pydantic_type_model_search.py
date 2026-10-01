import copy
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.storage import canonical
from ops.probe_pydantic_type_model_search import approved, task_harness, REVISION
import base64

class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate()
        self.plan={'revision':REVISION,'payable':False,'chain_transactions':False,'search_budget':8,'indices':[0,1],'environment':{'id':'affine_pydantic','adapter':'prime_v1'},'harness':{'version':'text-tools-v1','policy':'candidates','temperature':4.0,'top_p':1.0,'max_output_tokens':512}}
    def reject(self, plan):
        d={'payload':plan,'signer':self.key.verify_key.encode().hex(),'signature':base64.b64encode(self.key.sign(canonical(plan)).signature).decode()}
        with self.assertRaises(ValueError): approved(d,d['signer'])
    def test_heldout_and_boolean_indices_rejected(self):
        for indices in ([16],[True],[0,0]):
            p=copy.deepcopy(self.plan);p['indices']=indices;self.reject(p)
    def test_policy_override_rejected(self):
        for change in ({'turn_overrides':{}},{'temperature':0.7},{'candidates':['gold']}):
            p=copy.deepcopy(self.plan);p['harness'].update(change);self.reject(p)
    def test_public_messages_only(self):
        class Spec: config={'seed':42}
        class Session:
            closed=False
            def reset(self,index,seed):
                return {'messages':[{'role':'user','content':'public'}],'gold':'SECRET','task_hash':'hash'}
            def close(self): self.closed=True
        session=Session()
        with patch('subnet.environments.create_session',return_value=session), patch('subnet.public_pydantic_type_mutation.proposals',return_value=['one','two']) as proposals:
            harness,initial=task_harness(Spec(),self.plan['harness'],0)
        proposals.assert_called_once_with([{'role':'user','content':'public'}])
        self.assertTrue(session.closed);self.assertEqual(harness['candidates'],['one','two']);self.assertNotIn('gold',harness)
    def test_session_closed_on_unsupported_schema(self):
        class Spec: config={}
        class Session:
            closed=False
            def reset(self,index,seed): return {'messages':[]}
            def close(self): self.closed=True
        session=Session()
        with patch('subnet.environments.create_session',return_value=session),patch('subnet.public_pydantic_type_mutation.proposals',side_effect=ValueError('unsupported')):
            with self.assertRaises(ValueError): task_harness(Spec(),self.plan['harness'],0)
        self.assertTrue(session.closed)

if __name__=='__main__':unittest.main()
