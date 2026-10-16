"""v8.2.1: restore an optimizer generation recorded under a predecessor objective."""
from copy import deepcopy
from dataclasses import replace
import json
import pytest
import torch

from subnet.mimo_grpo import TrainerConfig
from subnet.mimo_grpo.config import TRAIN_UNRESOLVED, ALL_DECAY
from subnet.mimo_grpo.data import digest
from subnet.mimo_grpo.optimizer import CPUAdamW, restorable_config, ADAM_SETTINGS
from subnet.mimo_grpo.recovery import restore_transaction_state
from subnet.mimo_grpo.transaction import Job
from subnet.mimo_grpo.service import execute_transaction
from test_mimo_grpo import Runtime, binding, groups, publication

V82 = replace(TrainerConfig(), max_tasks=256, cap_policy=TRAIN_UNRESOLVED, loss_agg_mode='seq-mean-token-sum-norm',
              loss_scale_factor=8192, derail_mask=True, kl_coefficient=.02, kl_reference_checkpoint='8'*64,
              overlong_soft_start=7000, overlong_max_tokens=8192, overlong_penalty=.5)
V7 = replace(TrainerConfig(), max_tasks=32)


def _published_v7(root):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), V7, input_checkpoint=runtime.checkpoint_id)
    execute_transaction(root, runtime, optimizer, groups(), binding(), manifest_sha256='c'*64,
        source_closure_sha256='d'*64, assert_trainer_lease=lambda: None, publish_model=publication,
        shard_bytes=4096, config=V7)
    original = json.loads((root/'TRANSACTION.json').read_bytes())
    job = dict(original['job']); job['prepared_documents_sha256'] = tuple(job['prepared_documents_sha256'])
    masters = {n: v.detach().clone() for n, v in optimizer.masters.items()}
    moments = {n: optimizer.optimizer.state[v]['exp_avg'].clone() for n, v in optimizer.masters.items()}
    return original, Job(**job), masters, moments


def _load(directory, descriptor):
    from safetensors.torch import load_file
    result = Runtime(); result.model.load_state_dict(load_file(str(directory/'model.safetensors')))
    result.checkpoint_id = descriptor['id']
    return result


def test_documents_really_differ_and_adam_matches():
    assert V7.document()['version'].endswith('v1') and V82.document()['version'].endswith('v2')
    assert V7.document() != V82.document()
    assert restorable_config(V7.document(), V82) and restorable_config(V82.document(), V82)


def test_v7_generation_restores_under_v82_config_and_keeps_exact_state(tmp_path):
    original, job, masters, moments = _published_v7(tmp_path/'one')
    assert original['report']['config'] == V7.document()
    runtime, optimizer, journal = restore_transaction_state(tmp_path/'one', expected_job=job,
        load_runtime=_load, config=V82)
    assert journal == original and optimizer.global_step == 1 and optimizer.config == V82
    for name, value in optimizer.masters.items():
        assert torch.equal(value, masters[name])
        assert torch.equal(optimizer.optimizer.state[value]['exp_avg'], moments[name])
    # The next generation is recorded under the adopted (v8.2) objective.
    descriptor = optimizer.save_local(tmp_path/'next', output_checkpoint='c'*64, shard_bytes=4096)
    assert descriptor['config'] == V82.document()


@pytest.mark.parametrize('name,value', [('learning_rate', 2e-6), ('beta1', .8), ('beta2', .99),
    ('adam_epsilon', 1e-7), ('weight_decay', .02), ('weight_decay_policy', ALL_DECAY)])
def test_any_adam_difference_fails_closed(tmp_path, name, value):
    original, job, _, _ = _published_v7(tmp_path/'one')
    changed = replace(V82, **{name: value})
    assert not restorable_config(V7.document(), changed)
    with pytest.raises(ValueError, match='optimizer/objective settings'):
        restore_transaction_state(tmp_path/'one', expected_job=job, load_runtime=_load, config=changed)


def test_optimizer_descriptor_check_uses_same_rule(tmp_path):
    first = Runtime()
    optimizer = CPUAdamW(first.model.named_parameters(), V7, input_checkpoint=first.checkpoint_id)
    descriptor = optimizer.save_local(tmp_path/'state', output_checkpoint='c'*64, shard_bytes=4096)
    ok = CPUAdamW(Runtime(deepcopy(first.model)).model.named_parameters(), V82, input_checkpoint='c'*64)
    ok.restore_local(tmp_path/'state', descriptor_sha256=digest(descriptor), checkpoint='c'*64)
    bad = CPUAdamW(Runtime(deepcopy(first.model)).model.named_parameters(), replace(V82, beta2=.99), input_checkpoint='c'*64)
    with pytest.raises(ValueError, match='config/state inventory mismatch'):
        bad.restore_local(tmp_path/'state', descriptor_sha256=digest(descriptor), checkpoint='c'*64)
    assert bad.failed


@pytest.mark.parametrize('mutate', [lambda d: d.update(optimizer='other'), lambda d: d.update(version='x'),
    lambda d: d['settings'].pop('beta1'), lambda d: d['settings'].update(beta1=0.9000001),
    lambda d: d['settings'].update(weight_decay=0), lambda d: d['settings'].update(max_tasks=0)])
def test_malformed_or_different_recorded_documents_fail_closed(mutate):
    document = json.loads(json.dumps(V7.document())); mutate(document)
    assert not restorable_config(document, V82)
