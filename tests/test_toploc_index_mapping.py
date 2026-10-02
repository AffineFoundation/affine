import base64
import unittest
import torch
from toploc import build_proofs_base64, verify_proofs_base64
from subnet.proofs import verify_mapped_proofs


def strict(results):
    return all(not (r.exp_mismatches or r.mant_err_mean or r.mant_err_median) for r in results)


class IndexMappingRegression(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        from toploc.C.csrc.utils import get_fp_parts
        import toploc.poly as poly
        poly.get_fp_parts=lambda tensor:get_fp_parts(tensor,num_threads=2)

    def tearDown(self):
        torch.set_num_threads(2)

    def collision(self):
        tensor=torch.full((65500,),.125,dtype=torch.bfloat16)
        tensor[1:127]=1
        tensor[0]=4
        tensor[65497]=8
        proofs=build_proofs_base64([tensor],decode_batching_size=16,topk=128)
        return tensor,proofs

    def test_honest_collision_matches_native_verifier_without_tolerance(self):
        tensor,proofs=self.collision()
        self.assertLess(int.from_bytes(base64.b64decode(proofs[0])[:2],'big'),65497)
        self.assertFalse(strict(verify_proofs_base64([tensor],proofs,16,128)))
        corrected=verify_mapped_proofs([tensor],proofs,16,128)
        native=verify_proofs_base64(tensor.view(1,-1),proofs,1,128,skip_prefill=True)
        self.assertTrue(strict(corrected))
        self.assertTrue(strict(native))
        self.assertEqual([(r.exp_mismatches,r.mant_err_mean,r.mant_err_median) for r in corrected],
                         [(r.exp_mismatches,r.mant_err_mean,r.mant_err_median) for r in native])

    def test_changed_activation_and_changed_proof_still_fail(self):
        tensor,proofs=self.collision()
        mutated=tensor.clone();mutated[65497]=16
        self.assertFalse(strict(verify_mapped_proofs([mutated],proofs,16,128)))
        raw=bytearray(base64.b64decode(proofs[0]));raw[3]^=1
        forged=[base64.b64encode(raw).decode()]
        self.assertFalse(strict(verify_mapped_proofs([tensor],forged,16,128)))

    def test_normal_prefill_and_partial_decode_keep_original_errors(self):
        torch.manual_seed(17)
        acts=[torch.randn(4,128).to(torch.bfloat16)]+[torch.randn(1,128).to(torch.bfloat16) for _ in range(19)]
        proofs=build_proofs_base64(acts,16,128)
        old=verify_proofs_base64(acts,proofs,16,128)
        corrected=verify_mapped_proofs(acts,proofs,16,128)
        self.assertEqual(len(corrected),3)
        self.assertTrue(strict(old))
        self.assertTrue(strict(corrected))

    def test_zero_modulus_rejected_before_native_evaluation(self):
        tensor,proofs=self.collision()
        raw=bytearray(base64.b64decode(proofs[0]));raw[:2]=b'\x00\x00'
        with self.assertRaisesRegex(ValueError,'framing'):
            verify_mapped_proofs([tensor],[base64.b64encode(raw).decode()],16,128)
