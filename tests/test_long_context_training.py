import unittest
class SelectiveGradientTests(unittest.TestCase):
    def test_selective_output_head_preserves_full_prefix_gradient(self):
        import torch
        from types import SimpleNamespace
        from subnet.long_context_training import selective_logprob
        class Base(torch.nn.Module):
            def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(9,3)
            def forward(self,ids,use_cache=False):return SimpleNamespace(last_hidden_state=self.embedding(ids).cumsum(1))
        torch.manual_seed(19);base=Base();head=torch.nn.Linear(3,9,bias=False);ids=torch.tensor([[1,2,3,4,5]])
        score=selective_logprob(base,head,ids,3,[4,5]);full=head(base(ids).last_hidden_state)[0,2:4];expected=torch.log_softmax(full,-1).gather(1,torch.tensor([[4],[5]])).mean()
        self.assertTrue(torch.equal(score,expected));score.backward()
        self.assertGreater(float(base.embedding.weight.grad[1].abs().sum()),0)
        self.assertGreater(float(head.weight.grad.abs().sum()),0)
