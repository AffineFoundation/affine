import copy
import hashlib
import json
import unittest
from pathlib import Path
from nacl.signing import SigningKey
import base64
from subnet.native_eog_split import validate_public, PUBLIC_FIELDS
from subnet.native_eog_isolation import canonical, sha
from subnet.native_eog_admission import validate_model_audit
from subnet.harness import observations
from subnet.backend_jobs import file_map

class SplitTests(unittest.TestCase):
    def test_private_descriptor_fields_fail_closed(self):
        value={name:None for name in PUBLIC_FIELDS};value['verifiers']=[{'query':'PRIVATE'}]
        with self.assertRaises(ValueError):validate_public(value)

    def test_public_descriptor_never_has_actor_operator_credentials(self):
        self.assertFalse({'capability','operator_capability','seed_file','sql_content','verifiers'}&PUBLIC_FIELDS)

    def fixture(self):
        # Synthetic receipt for validator unit controls, never native/model evidence.
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        files={'model.safetensors':'a'*64,'config.json':'c'*64};model={'checkpoint':file_map(files),'checkpoint_files':files,
               'runtime_profile':'synthetic-validator-control','harness_source_sha256':'b'*64}
        artifact={'public':{'messages':[{'role':'user','content':'test'}]},
                  'events':[{'name':'test','arguments':{},'observation':'native'}]}
        body=canonical(artifact)
        payload={'kind':'controlled-original-eog-model-audit-v1',**model,'public_trace_sha256':hashlib.sha256(body).hexdigest(),
          'numerical_tolerances':{'TOPLOC_errors':0,'logprobs_atol':1e-5,'logprobs_rtol':0},
          'full_model_recompute':True,'curated_target_model_computation':True,'originally_sampled':False,
          'records':[{'turn_index':0,'messages_sha256':sha(artifact['public']['messages']),
             'action_sha256':sha({'name':'test','arguments':{}}),'observation_sha256':hashlib.sha256(b'native').hexdigest(),
             'full_proof_verified':True}]}
        def signed(value):return {'signer':authority,'payload':value,'signature':base64.b64encode(key.sign(canonical(value)).signature).decode()}
        return body,payload,model,authority,signed

    def test_valid_synthetic_binding_only(self):
        body,payload,model,authority,sign=self.fixture()
        value,_=validate_model_audit(sign(payload),authority,body,model)
        self.assertEqual(len(value['events']),1)

    def test_resigned_falsifications_rejected(self):
        body,payload,model,authority,sign=self.fixture()
        variants=[]
        value=copy.deepcopy(payload);value['records']=[];variants.append(value)
        value=copy.deepcopy(payload);value['checkpoint']='f'*64;variants.append(value)
        value=copy.deepcopy(payload);value['records'][0]['observation_sha256']='0'*64;variants.append(value)
        value=copy.deepcopy(payload);value['records'][0]['messages_sha256']='0'*64;variants.append(value)
        value=copy.deepcopy(payload);value['numerical_tolerances']['TOPLOC_errors']=1;variants.append(value)
        value=copy.deepcopy(payload);value['public_trace_sha256']='0'*64;variants.append(value)
        for changed in variants:
            with self.assertRaises(ValueError):validate_model_audit(sign(changed),authority,body,model)

    def test_wrong_authority(self):
        body,payload,model,authority,sign=self.fixture()
        with self.assertRaises(ValueError):validate_model_audit(sign(payload),'a'*64,body,model)

if __name__=='__main__':unittest.main()
