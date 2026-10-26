"""Small real-autograd integration controls, no network/GPU/model downloads."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

SOURCE = Path(__file__).resolve().parents[2]/'state/root-audits/20261012-sequence-component-logging-candidate-v1/production-insertion-v1/full-source'
if SOURCE.exists(): sys.path.insert(0,str(SOURCE))

from .core import Policy, Stamp, Rollout, Group
from .objectives import Objective
from .scoring import score
from .training import train_update


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([[.1,.2,.3,.4]]*4,dtype=torch.float32))
        self.config=SimpleNamespace(attention_dropout=0.)
    def gradient_checkpointing_enable(self,**kwargs): pass
    def forward(self,input_ids,attention_mask,use_cache):
        return SimpleNamespace(logits=self.weight[input_ids])


class FakeOptimizer:
    def __init__(self,model): self.model=model;self.global_step=0
    def zero_grad(self):
        for p in self.model.parameters(): p.grad=None
    def step(self,gradients):
        with torch.no_grad():
            for name,p in self.model.named_parameters(): p.add_(gradients[name],alpha=-.01)
        self.global_step+=1
        return {'optimizer_step':self.global_step}


class Integration(unittest.TestCase):
    def build(self):
        model=Tiny();reference=copy.deepcopy(model)
        for p in reference.parameters():p.requires_grad_(False)
        policy=Policy(.7,4,3,3,8)
        stamp=Stamp('run','a'*64,0,'b'*64,policy)
        rows=[]
        for i in range(8):
            tokens=(0,3) if i<4 else (1,3)
            old=score(model,(0,1),tokens,policy,gradients=True).detach().tolist()
            rows.append(Rollout(i,tokens,tuple(old),'eos','correct' if i<4 else 'wrong'))
        return model,reference,Group(stamp,'task',(0,1),tuple(rows))

    def test_actual_backwards_one_step_and_frozen_reference(self):
        for mode in ('grpo','group_preference'):
            model,reference,group=self.build();optimizer=FakeOptimizer(model);logs=[]
            before=model.weight.detach().clone();frozen=reference.weight.detach().clone()
            def step(torch,optimizer,gradients,policy):return optimizer.step(gradients),{}
            with patch('subnet.optimizer_threading.step',side_effect=step):
                result=train_update(model,reference,optimizer,[group],objective=Objective(mode=mode),
                    group_size=8,guard=lambda:None,record=logs.append)
            self.assertEqual(result['optimizer_step'],1)
            self.assertEqual(result['trained_rollouts'],8)
            self.assertFalse(torch.equal(before,model.weight))
            self.assertTrue(torch.equal(frozen,reference.weight))
            self.assertTrue(all(p.grad is None for p in reference.parameters()))
            self.assertTrue(any(r['stage']=='all_initial_ratios_admitted' for r in logs))

    def test_drift_failure_recorded_before_any_update(self):
        model,reference,group=self.build();optimizer=FakeOptimizer(model);logs=[]
        row=replace(group.rollouts[0],behavior_logprobs=(-100.,-100.))
        group=replace(group,rollouts=(row,)+group.rollouts[1:])
        with self.assertRaisesRegex(ValueError,'usability guard'):
            train_update(model,reference,optimizer,[group],objective=Objective(),group_size=8,
                         guard=lambda:None,record=logs.append)
        self.assertEqual(optimizer.global_step,0)
        self.assertEqual(logs[-1]['stage'],'initial_ratio_assessment')
        self.assertFalse(logs[-1]['passed'])
        self.assertEqual(logs[-1]['active_trajectories'],8)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_unresolved_group_member_recorded_not_trained(self):
        model,reference,group=self.build();optimizer=FakeOptimizer(model);logs=[]
        cap=Rollout(7,(0,1,2),(-1.,-1.,-1.),'budget','unresolved')
        group=replace(group,rollouts=group.rollouts[:-1]+(cap,))
        def step(torch,optimizer,gradients,policy):return optimizer.step(gradients),{}
        with patch('subnet.optimizer_threading.step',side_effect=step):
            result=train_update(model,reference,optimizer,[group],objective=Objective(),group_size=8,
                                guard=lambda:None,record=logs.append)
        self.assertEqual(result['rollout_count'],8)
        self.assertEqual(result['trained_rollouts'],7)
        self.assertEqual(result['masked_rollouts'],1)
        self.assertTrue(any(r['stage']=='masked_unresolved' and r['nonce']==7 for r in logs))

    def test_sparse_tail_real_backward_keeps_actual_behavior_denominator(self):
        model,reference,group=self.build();optimizer=FakeOptimizer(model);logs=[]
        policy=replace(group.stamp.policy,max_output_tokens=200,max_context_tokens=208)
        stamp=replace(group.stamp,policy=policy)
        rows=[]
        for i in range(8):
            tokens=((0 if i<4 else 1),)*199+(3,)
            behavior=score(model,group.prompt,tokens,policy,gradients=True).detach().tolist()
            # One ordinary finite tail event outside PPO's upper band; 1/200
            # in this trajectory and 1/1600 in the whole intended update.
            if i==0: behavior[30]-=.26
            rows.append(Rollout(i,tokens,tuple(behavior),'eos','correct' if i<4 else 'wrong'))
        group=replace(group,stamp=stamp,rollouts=tuple(rows))
        before=model.weight.detach().clone();frozen=reference.weight.detach().clone()
        def step(torch,optimizer,gradients,policy):return optimizer.step(gradients),{}
        with patch('subnet.optimizer_threading.step',side_effect=step):
            result=train_update(model,reference,optimizer,[group],objective=Objective(),
                group_size=8,guard=lambda:None,record=logs.append)
        assessment=next(r for r in logs if r['stage']=='initial_ratio_assessment')
        backwards=[r for r in logs if r['stage']=='trajectory_backward']
        self.assertTrue(assessment['passed'])
        self.assertEqual(assessment['active_trajectories'],8)
        self.assertEqual(assessment['aggregate']['outside_ppo_band_tokens'],1)
        self.assertEqual(assessment['aggregate']['sign_aware_clipped_tokens'],1)
        self.assertGreater(assessment['aggregate']['max_abs_logratio'],.25)
        self.assertEqual(len(backwards),8)
        self.assertLess(logs.index(assessment),logs.index(backwards[0]))
        self.assertEqual(backwards[0]['behavior_logprobs'],group.rollouts[0].behavior_logprobs)
        self.assertAlmostEqual(backwards[0]['clipped_fraction'],1/200,places=6)
        self.assertEqual(result['optimizer_step'],1)
        self.assertFalse(torch.equal(before,model.weight))
        self.assertTrue(torch.equal(frozen,reference.weight))


if __name__=='__main__': unittest.main()
