import tempfile,unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from ops.checkpoint_update_diagnostics import compare_checkpoints

class CheckpointChanges(unittest.TestCase):
    def test_records_actual_bf16_change_without_claiming_learning(self):
        with tempfile.TemporaryDirectory()as directory:
            a=Path(directory)/'parent';b=Path(directory)/'candidate';a.mkdir();b.mkdir()
            save_file({'w':torch.tensor([1,2,3,4],dtype=torch.bfloat16)},a/'model.safetensors')
            save_file({'w':torch.tensor([1,2,3,5],dtype=torch.bfloat16)},b/'model.safetensors')
            r=compare_checkpoints(a,b,chunk_elements=2)
            self.assertEqual(r['changed_elements'],1);self.assertEqual(r['changed_fraction'],.25)
            self.assertAlmostEqual(r['relative_l2'],(1/30)**.5)
            self.assertFalse(r['learning_gain_claimed']);self.assertFalse(r['model_mutations'])
    def test_refuses_changed_tensor_schema_or_nonfinite_model(self):
        with tempfile.TemporaryDirectory()as directory:
            a=Path(directory)/'a';b=Path(directory)/'b';a.mkdir();b.mkdir()
            save_file({'w':torch.tensor([1.])},a/'model.safetensors')
            for tensors in ({'other':torch.tensor([1.])},{'w':torch.tensor([float('nan')])}):
                save_file(tensors,b/'model.safetensors')
                with self.assertRaises(ValueError):compare_checkpoints(a,b)
