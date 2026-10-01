import copy,unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.long_context_runtime import POLICY,validate_job,validate_tokens,prediction_rows,canonical

def envelope(payload,key):
    import base64
    return {'payload':payload,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(payload)).signature).decode()}

class LongContextTests(unittest.TestCase):
    def test_authority_before_source_or_checkpoint_reads(self):
        key=SigningKey.generate()
        with patch('subnet.long_context_runtime.file_sha',side_effect=AssertionError('unexpected file read')):
            with self.assertRaisesRegex(ValueError,'authority'):validate_job(envelope({},key),'wrong')
    def test_signed_relaxed_context_policy_rejected_before_files(self):
        key=SigningKey.generate();authority=key.verify_key.encode().hex();policy=copy.deepcopy(POLICY);policy['max_context']=65536
        with patch('subnet.long_context_runtime.AUTHORITY',authority),patch('subnet.long_context_runtime.file_sha',side_effect=AssertionError('unexpected file read')):
            with self.assertRaisesRegex(ValueError,'role/profile'):validate_job(envelope({'role':'long-context-proof-probe','policy':policy,'payable':False,'chain_transactions':False},key),authority)
    def test_exact_model_revision_and_vram_guard_required(self):
        key=SigningKey.generate();authority=key.verify_key.encode().hex();job={'role':'long-context-proof-probe','policy':POLICY,'payable':False,'chain_transactions':False,'runtime_source_sha256':'approved','min_free_vram_mib':12288,'wait_seconds':1800,'checkpoint':{'model_id':'Qwen/Qwen2.5-0.5B-Instruct','revision':'wrong','files':{'model.safetensors':'a'}}}
        with patch('subnet.long_context_runtime.AUTHORITY',authority),patch('subnet.long_context_runtime.file_sha',return_value='approved'):
            with self.assertRaisesRegex(ValueError,'model revision'):validate_job(envelope(job,key),authority)
            job['min_free_vram_mib']=0
            with self.assertRaisesRegex(ValueError,'VRAM'):validate_job(envelope(job,key),authority)
    def test_context_and_token_ids_cannot_be_truncated_or_coerced(self):
        validate_tokens([1]*17485,[2,3],10)
        for prompt,output in (([1]*32768,[2]),([True],[1]),([1],[]),([1],[-1])):
            with self.subTest(prompt_length=len(prompt),output=output),self.assertRaises(ValueError):validate_tokens(prompt,output,10)
    def test_selective_prediction_rows_equal_full_projection(self):
        import torch
        hidden=torch.arange(18,dtype=torch.float32).reshape(6,3);head=torch.arange(21,dtype=torch.float32).reshape(7,3)
        actual=prediction_rows(hidden,4,2)@head.T
        expected=(hidden@head.T)[3:5]
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(tuple(actual.shape),(2,7))
        self.assertEqual(prediction_rows(hidden,4,2)[0].tolist(),hidden[3].tolist())
