import unittest
import torch
from subnet.epoch_optimizer import preference_loss

class FixedReferenceTests(unittest.TestCase):
    def test_fixed_reference_changes_repeated_loss_and_keeps_adam_steps(self):
        margin=torch.nn.Parameter(torch.tensor(.5));reference=float(margin.detach())
        optimizer=torch.optim.AdamW([margin],lr=.1,weight_decay=0);losses=[]
        for step in range(3):
            optimizer.zero_grad();loss=preference_loss(torch,margin,reference,beta=1);losses.append(float(loss.detach()));loss.backward();optimizer.step()
            self.assertEqual(int(optimizer.state[margin]['step']),step+1)
        self.assertGreater(float(margin.detach()),reference)
        self.assertGreater(losses[0],losses[1]);self.assertGreater(losses[1],losses[2])
    def test_reset_reference_hides_preference_progress(self):
        reference=.5
        fixed=float(preference_loss(torch,torch.tensor(1.5),reference))
        reset=float(preference_loss(torch,torch.tensor(1.5),1.5))
        self.assertLess(fixed,reset)
    def test_nonfinite_reference_rejected(self):
        for reference in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):preference_loss(torch,torch.tensor(0.),reference)

if __name__=='__main__':unittest.main()
