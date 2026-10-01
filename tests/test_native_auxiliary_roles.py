import copy,unittest
from nacl.signing import SigningKey
from subnet.native_tau2_model import envelope
from subnet.native_tau2_probe import digest
from subnet.native_auxiliary_roles import VERSION,validate_contract,admit_records

def fixture():
    key=SigningKey.generate();cp='a'*64
    contract={'version':VERSION,'objective':'agent-only-curated-supervised-v1','payable':False,'roles':{r:{'kind':'agent' if r=='agent' else 'auxiliary','training_eligible':r=='agent','checkpoint':cp,'source_hash':'b'*64,'numerical_policy':{'dtype':'float32'}} for r in ('agent','user')}}
    records=[envelope({'role':r,'checkpoint':cp,'request':{'model':r},'request_hash':digest({'model':r}),'prompt':[1],'output':[2,3],'proofs':['proof'],'probabilities_sha256':'c'*64},key) for r in ('agent','user')]
    audit={'contract_hash':digest(contract),'receipts_hash':digest(records),'full_native_trajectory_verified':True,'all_model_roles_verified':True}
    return key,contract,records,audit

class AuxiliaryRolesTests(unittest.TestCase):
    def test_auxiliary_observation_tokens_never_training_targets(self):
        k,c,r,a=fixture();views=admit_records(envelope(c,k),r,envelope(a,k),k.verify_key.encode().hex())
        self.assertEqual(views[0]['loss_mask'],[True,True]);self.assertEqual(views[1]['loss_mask'],[False,False]);self.assertFalse(views[1]['training_eligible'])
    def test_signed_malformed_auxiliary_contract_rejected(self):
        for field,value in [('training_eligible',True),('kind','unknown'),('checkpoint','unapproved')]:
            k,c,_,_=fixture();c['roles']['user'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_contract(envelope(c,k),k.verify_key.encode().hex())
    def test_signed_audit_cannot_be_reused_for_mutated_receipts(self):
        k,c,r,a=fixture();mutated=copy.deepcopy(r[1]['payload']);mutated['output']=[9];r[1]=envelope(mutated,k)
        with self.assertRaisesRegex(ValueError,'audit admission'):admit_records(envelope(c,k),r,envelope(a,k),k.verify_key.encode().hex())
    def test_wrong_authority_fails_before_payload_admission(self):
        k,c,r,a=fixture()
        with self.assertRaisesRegex(ValueError,'authority'):admit_records(envelope(c,k),r,envelope(a,k),SigningKey.generate().verify_key.encode().hex())
    def test_signed_wrong_role_checkpoint_rejected_even_when_audit_binds(self):
        k,c,r,a=fixture();p=r[1]['payload'];p['checkpoint']='d'*64;r[1]=envelope(p,k);a['receipts_hash']=digest(r)
        with self.assertRaisesRegex(ValueError,'computation binding'):admit_records(envelope(c,k),r,envelope(a,k),k.verify_key.encode().hex())
