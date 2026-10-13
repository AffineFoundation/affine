"""Reference equation/gradient and actual trainer/optimizer integration tests."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
import torch

from subnet.mimo_grpo import TrainerConfig, InputBinding, TaskGroup, Trajectory, train_update
from subnet.mimo_grpo.config import MASK_UNRESOLVED, TRAIN_UNRESOLVED
from subnet.mimo_grpo.data import digest
from subnet.mimo_grpo.optimizer import CPUAdamW
from subnet.mimo_grpo.training import _layout
from subnet.mimo_grpo.upstream import policy_loss, reference
from subnet.mimo_grpo.lifecycle import stage_for_publication
from subnet.mimo_grpo.service import execute_transaction
from subnet.mimo_grpo.transaction import RecoveryRequired
from subnet.mimo_grpo.projection import project_document
from subnet.mimo_grpo.recovery import restore_transaction_state
from subnet.mimo_grpo.transaction import Job


class Runtime:
    checkpoint_id = 'a' * 64
    vocab_size = 17
    eos_token_ids = (15, 16)
    max_context_tokens = 64

    def __init__(self, model=None):
        torch.manual_seed(743)
        self.model = model if model is not None else torch.nn.Sequential(torch.nn.Embedding(17, 5), torch.nn.Linear(5, 17))
        self.events = []

    def score(self, prompt, output, *, gradients):
        self.events.append(('grad' if gradients else 'old', tuple(output)))
        tokens = torch.tensor([*prompt, *output[:-1]])
        with torch.set_grad_enabled(gradients):
            logits = self.model(tokens)[len(prompt)-1:]
            return logits.log_softmax(-1).gather(-1, torch.tensor(output).unsqueeze(-1)).squeeze(-1)

    def save_model(self, directory):
        from safetensors.torch import save_file
        directory.mkdir()
        save_file(self.model.state_dict(), str(directory / 'model.safetensors'))
        (directory / 'config.json').write_text('{"fixture":true}\n')

    def acknowledge_model_publication(self, checkpoint_id, checkpoint_files):
        assert checkpoint_id == digest(checkpoint_files)
        self.checkpoint_id = checkpoint_id


def groups(count=2, capped=False):
    result = []
    for task in range(count):
        rows = []
        for nonce in range(8):
            length = 1 + nonce % 3
            cap = capped and nonce == 7
            rows.append(Trajectory(nonce, tuple([1 + nonce] * length + ([] if cap else [15 + nonce % 2])),
                'unresolved' if cap else ('correct' if nonce < 3 else 'wrong'), 'max_tokens' if cap else 'eos'))
        result.append(TaskGroup(str(task), (1, 2 + task), tuple(rows), str(task + 1) * 64))
    return result


def binding(step=0, checkpoint='a' * 64):
    return InputBinding('test', 'epoch1', checkpoint, 'b' * 64, step)


def test_vendored_files_are_exact():
    root = Path(__file__).resolve().parents[1] / 'vendor/mimo_verl_pinned'
    provenance = json.loads((root / 'PROVENANCE.json').read_text())
    assert provenance['revision'] == 'a2ad9f6160b03ff2d47e59832bfb6b289f37c917'
    for row in provenance['files']:
        assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']


def test_exact_upstream_advantages_without_std_normalization():
    rewards = torch.tensor([[1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.]])
    result, returns = reference().compute_grpo_outcome_advantage(rewards, torch.ones_like(rewards),
        np.array(['task'] * 8), norm_adv_by_std_in_grpo=False)
    assert torch.equal(result[:, 0], torch.tensor([.75, .75, -.25, -.25, -.25, -.25, -.25, -.25]))
    assert torch.equal(result, returns)


def test_exact_upstream_ppo_dualclip_loss_and_gradient_against_equations():
    config = TrainerConfig()
    old = torch.zeros((2, 5), dtype=torch.float64)
    new = torch.tensor([[-2., -.1, .05, .4, 2.], [2., .4, -.3, -.1, .2]], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[.5] * 5, [-.5] * 5], dtype=torch.float64)
    mask = torch.tensor([[1., 1., 1., 1., 0.], [1., 1., 1., 0., 0.]], dtype=torch.float64)
    weights = reference().compute_prompt_loss_weights(mask, ['a', 'b'])
    actual, metrics = policy_loss(old, new, advantages, mask, weights, config)
    ratio = new.exp()
    clipped = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(.8, 1.2))
    dual = torch.where(advantages < 0, torch.minimum(-advantages * 3, clipped), clipped)
    expected = (dual[0, :4].sum()/4 + dual[1, :3].sum()/3)/2
    assert torch.allclose(actual, expected, atol=1e-15, rtol=1e-15)
    assert torch.allclose(torch.autograd.grad(actual, new, retain_graph=True)[0], torch.autograd.grad(expected, new)[0], atol=1e-15, rtol=1e-15)
    assert metrics['actor/pg_clipfrac_lower'] > 0


@pytest.mark.parametrize('partition', [1, 2, 3, 7])
def test_prompt_mean_loss_and_gradient_invariant_to_microbatch_partition(partition):
    torch.manual_seed(119)
    x = torch.randn(16, 7, dtype=torch.float64, requires_grad=True)
    old = x.detach() + torch.randn_like(x) * .4
    mask = torch.zeros_like(x)
    for row in range(16):
        mask[row, :1 + row % 7] = 1
    adv = torch.tensor([.75, -.25] * 8, dtype=torch.float64).unsqueeze(-1).expand_as(x)
    weights = reference().compute_prompt_loss_weights(mask, ['a'] * 8 + ['b'] * 8)
    whole, _ = policy_loss(old, x, adv, mask, weights, TrainerConfig())
    parts = [policy_loss(old[i:i+partition], x[i:i+partition], adv[i:i+partition], mask[i:i+partition],
             weights[i:i+partition], TrainerConfig())[0] for i in range(0, 16, partition)]
    split = sum(parts)
    assert torch.allclose(whole, split, atol=1e-15, rtol=1e-15)
    assert torch.allclose(torch.autograd.grad(whole, x, retain_graph=True)[0], torch.autograd.grad(split, x)[0], atol=1e-15, rtol=1e-15)


def test_unresolved_in_group_mean_but_masked_from_policy():
    rows = _layout(groups(1, capped=True), TrainerConfig())
    assert rows[-1]['advantage'] == -.375
    assert rows[-1]['active'] is False and rows[-1]['prompt_weight'] == 0
    assert rows[0]['advantage'] == .625
    unmasked = _layout(groups(1, capped=True), replace(TrainerConfig(), cap_policy=TRAIN_UNRESOLVED))
    assert unmasked[-1]['active'] and unmasked[-1]['prompt_weight'] > 0


def test_old_prefill_all_before_backward_and_one_step():
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    report = train_update(runtime, optimizer, groups(capped=True), binding())
    assert report['group_count'] == 2 and report['rollout_count'] == 16
    assert report['trained_rollouts'] == 14 and report['masked_rollouts'] == 2
    assert report['optimizer_step'] == 1
    assert [e[0] for e in runtime.events] == ['old'] * 14 + ['grad'] * 14
    assert report['preclip_gradient_norm'] > 0
    assert report['optimizer']['gradient_accumulation_dtype'] == 'float32'


def test_actual_update_partition_invariance():
    left, right = Runtime(), Runtime()
    a = CPUAdamW(left.model.named_parameters(), TrainerConfig(), input_checkpoint=left.checkpoint_id)
    b = CPUAdamW(right.model.named_parameters(), TrainerConfig(), input_checkpoint=right.checkpoint_id)
    first = train_update(left, a, groups(), binding(), microbatch_rollouts=1)
    second = train_update(right, b, groups(), binding(), microbatch_rollouts=3)
    assert first['optimizer_step'] == second['optimizer_step'] == 1
    for name in a.masters:
        assert torch.allclose(a.masters[name], b.masters[name], atol=1e-8, rtol=1e-7)
        assert torch.allclose(a.optimizer.state[a.masters[name]]['exp_avg'], b.optimizer.state[b.masters[name]]['exp_avg'], atol=1e-8, rtol=1e-6)


def test_cpu_offload_matches_single_pytorch_adamw_step_and_moments():
    torch.manual_seed(17)
    parameters = [('one', torch.nn.Parameter(torch.randn(4, 5, dtype=torch.bfloat16))),
                  ('two', torch.nn.Parameter(torch.randn(5, dtype=torch.bfloat16)))]
    config = TrainerConfig()
    actual = CPUAdamW(parameters, config, input_checkpoint='a' * 64)
    controls = {name: torch.nn.Parameter(p.detach().float().clone()) for name, p in parameters}
    expected = torch.optim.AdamW([dict(params=[controls['one']],weight_decay=.01),
        dict(params=[controls['two']],weight_decay=0.)], lr=1e-6, betas=(.9, .999), eps=1e-8,
        foreach=False, fused=False)
    for step in range(3):
        gradients = {name: torch.randn_like(p, dtype=torch.float32) for name, p in parameters}
        actual.step(gradients)
        for name, p in controls.items():
            p.grad = gradients[name].clone()
        expected.step()
        expected.zero_grad(set_to_none=True)
        for name in controls:
            assert torch.equal(actual.masters[name], controls[name])
            for slot in ('step', 'exp_avg', 'exp_avg_sq'):
                assert torch.equal(actual.optimizer.state[actual.masters[name]][slot], expected.state[controls[name]][slot])


def test_mcore_bias_1d_exemption_embedding_decay_and_explicit_legacy_policy():
    from subnet.mimo_grpo.config import MCORE_DECAY, ALL_DECAY
    parameters=[('layer.weight',torch.nn.Parameter(torch.ones(2,2))),
                ('layer.bias',torch.nn.Parameter(torch.ones(2,2))),
                ('norm.weight',torch.nn.Parameter(torch.ones(2))),
                ('embedding.weight',torch.nn.Parameter(torch.ones(2,2)))]
    config=TrainerConfig(learning_rate=.01,weight_decay=.1)
    optimizer=CPUAdamW(parameters,config,input_checkpoint='a'*64)
    report=optimizer.step({n:torch.zeros_like(p)for n,p in parameters})
    assert report['weight_decay_policy']==MCORE_DECAY
    assert (report['decay_parameter_count'],report['no_decay_parameter_count'])==(2,2)
    assert dict((r['name'],r['weight_decay'])for r in optimizer.decay_inventory)=={
        'layer.weight':.1,'layer.bias':0.,'norm.weight':0.,'embedding.weight':.1}
    assert torch.equal(optimizer.masters['layer.bias'],torch.ones(2,2))
    assert torch.equal(optimizer.masters['norm.weight'],torch.ones(2))
    assert torch.equal(optimizer.masters['embedding.weight'],torch.full((2,2),.999))
    legacy=CPUAdamW(parameters,replace(config,weight_decay_policy=ALL_DECAY),input_checkpoint='a'*64)
    assert all(r['weight_decay']==.1 for r in legacy.decay_inventory)


def test_local_state_roundtrip_preserves_next_update(tmp_path):
    first = Runtime()
    optimizer = CPUAdamW(first.model.named_parameters(), TrainerConfig(), input_checkpoint=first.checkpoint_id)
    train_update(first, optimizer, groups(), binding())
    descriptor = optimizer.save_local(tmp_path/'state', output_checkpoint='c'*64, shard_bytes=4096)
    restored_runtime = Runtime(deepcopy(first.model))
    restored_runtime.checkpoint_id = 'c'*64
    restored = CPUAdamW(restored_runtime.model.named_parameters(), TrainerConfig(), input_checkpoint='c'*64)
    restored.restore_local(tmp_path/'state', descriptor_sha256=digest(descriptor), checkpoint='c'*64)
    first.checkpoint_id = optimizer.input_checkpoint = 'c'*64
    a = train_update(first, optimizer, groups(), binding(1, 'c'*64))
    b = train_update(restored_runtime, restored, groups(), binding(1, 'c'*64))
    assert a['optimizer_step'] == b['optimizer_step'] == 2
    for name in optimizer.masters:
        assert torch.equal(optimizer.masters[name], restored.masters[name])


@pytest.mark.parametrize('fault', ['late-shard', 'model-projection'])
def test_failed_partial_restore_poisons_optimizer_until_recreated(tmp_path, fault):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    train_update(runtime, optimizer, groups(), binding())
    descriptor = optimizer.save_local(tmp_path/'state', output_checkpoint='c' * 64, shard_bytes=4096)
    other = Runtime(deepcopy(runtime.model))
    if fault == 'late-shard':
        last = tmp_path/'state'/descriptor['chunks'][-1]['file']
        data = bytearray(last.read_bytes())
        data[-1] ^= 1
        last.write_bytes(data)
    else:
        with torch.no_grad():
            next(other.model.parameters()).add_(1.)
    restored = CPUAdamW(other.model.named_parameters(), TrainerConfig(), input_checkpoint='c' * 64)
    with pytest.raises(ValueError, match='shard content|projection differs'):
        restored.restore_local(tmp_path/'state', descriptor_sha256=digest(descriptor), checkpoint='c' * 64)
    assert restored.failed is True
    with pytest.raises(ValueError, match='healthy optimizer'):
        restored.step({n: torch.zeros_like(p, dtype=torch.float32) for n, p in restored.parameters})


def test_staging_only_publishes_model_and_keeps_optimizer_local(tmp_path):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    report = train_update(runtime, optimizer, groups(), binding())
    staged = stage_for_publication(runtime, optimizer, report, tmp_path/'stage', shard_bytes=4096)
    assert all(name.startswith('model/') for name in staged['publish_files'])
    assert all(name.startswith('optimizer/') for name in staged['local_state_files'])
    assert set(staged['publish_files']).isdisjoint(staged['local_state_files'])
    assert staged['optimizer_step'] == 1
    with pytest.raises(ValueError, match='reconciliation'):
        stage_for_publication(runtime, optimizer, report, tmp_path/'stage')
    assert optimizer.global_step == 1


def test_missing_or_repeated_nonce_and_all_correct_rejected():
    runtime = Runtime()
    group = groups(1)[0]
    for bad in (replace(group, trajectories=group.trajectories[:-1]),
                replace(group, trajectories=(*group.trajectories[:-1], group.trajectories[0])),
                replace(group, trajectories=tuple(replace(r, outcome='correct') for r in group.trajectories))):
        optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
        with pytest.raises(ValueError):
            train_update(runtime, optimizer, [bad], binding())
        assert optimizer.global_step == 0


def test_cap_is_not_completed_wrong_and_second_eos_is_valid():
    group = groups(1)[0]
    group.validate(8, vocab_size=17, eos_token_ids={15, 16})
    bad = replace(group, trajectories=tuple(replace(r, outcome='wrong') if r.nonce == 7 else r for r in groups(1, capped=True)[0].trajectories))
    with pytest.raises(ValueError, match='completed outcome'):
        bad.validate(8, vocab_size=17, eos_token_ids={15, 16})


def test_unresolved_eos_without_final_answer_is_retained_and_masked():
    group = groups(1)[0]
    group = replace(group, trajectories=(*group.trajectories[:-1], replace(group.trajectories[-1], outcome='unresolved')))
    group.validate(8, vocab_size=17, eos_token_ids={15, 16})
    rows = _layout([group], TrainerConfig())
    assert rows[-1]['advantage'] == -.375
    assert not rows[-1]['active']


def test_empty_update_does_not_advance_or_prepare_model():
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    report = train_update(runtime, optimizer, [], binding())
    assert report['status'] == 'no-update' and optimizer.global_step == 0 and not runtime.events


def test_stale_checkpoint_and_duplicate_tasks_rejected():
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    with pytest.raises(ValueError, match='input checkpoint'):
        train_update(runtime, optimizer, groups(), binding(checkpoint='f'*64))
    with pytest.raises(ValueError, match='distinct complete tasks'):
        train_update(runtime, optimizer, [groups()[0]]*2, binding())


def publication(request):
    assert request['optimizer_state_uploaded'] is False
    assert all(not name.startswith('optimizer/') for name in request['model_files'])
    return dict(job_id=request['job_id'], output_checkpoint=request['output_checkpoint'],
        model_full_get_verified=True, optimizer_state_uploaded=False, receipt_sha256='f'*64)


def execute(path, runtime, optimizer, current, data=None, publish=publication):
    return execute_transaction(path, runtime, optimizer, groups() if data is None else data, current,
        manifest_sha256='c'*64, source_closure_sha256='d'*64,
        assert_trainer_lease=lambda: None, publish_model=publish, shard_bytes=4096)


def test_two_epochs_empty_interlude_and_normal_optimizer_retention(tmp_path):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    first = execute(tmp_path/'one', runtime, optimizer, binding())
    assert first['state'] == 'published' and optimizer.global_step == 1
    checkpoint = first['output_checkpoint']
    assert runtime.checkpoint_id == optimizer.input_checkpoint == checkpoint
    interlude = execute(tmp_path/'empty', runtime, optimizer, replace(binding(1, checkpoint), epoch='empty'), data=[])
    assert interlude['state'] == 'no-update' and optimizer.global_step == 1
    second = execute(tmp_path/'two', runtime, optimizer, replace(binding(1, checkpoint), epoch='epoch2'))
    assert second['state'] == 'published' and optimizer.global_step == 2
    assert not (tmp_path/'one/optimizer').exists()
    assert (tmp_path/'one/TRANSACTION.json').exists()
    assert (tmp_path/'two/optimizer/descriptor.json').exists()
    assert second == json.loads((tmp_path/'two/TRANSACTION.json').read_text())


def test_lost_publication_ack_resumes_without_optimizer_step(tmp_path):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    calls = []
    def lost_ack(request):
        calls.append(request['output_checkpoint'])
        raise TimeoutError('upload succeeded but acknowledgement lost')
    with pytest.raises(TimeoutError):
        execute(tmp_path/'one', runtime, optimizer, binding(), publish=lost_ack)
    assert optimizer.global_step == 1
    checkpoint = json.loads((tmp_path/'one/TRANSACTION.json').read_text())
    assert checkpoint['state'] == 'staged'
    events = list(runtime.events)
    result = execute(tmp_path/'one', runtime, optimizer, binding())
    assert result['state'] == 'published' and optimizer.global_step == 1
    assert calls == [result['output_checkpoint']] and events == runtime.events
    # Even an acknowledged retry performs no second step.
    again = execute(tmp_path/'one', runtime, optimizer, binding())
    assert again['output_checkpoint'] == result['output_checkpoint'] and optimizer.global_step == 1


def test_interrupted_optimizer_or_stage_never_automatically_retrains(tmp_path):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    runtime.save_model = lambda path: (_ for _ in ()).throw(RuntimeError('injected export failure'))
    with pytest.raises(RuntimeError, match='injected'):
        execute(tmp_path/'one', runtime, optimizer, binding())
    assert optimizer.global_step == 1
    with pytest.raises(RecoveryRequired):
        execute(tmp_path/'one', runtime, optimizer, binding())
    assert optimizer.global_step == 1


@pytest.mark.parametrize('lost_ack', [False, True])
def test_cold_process_restores_actual_model_and_original_optimizer_without_retraining(tmp_path, lost_ack):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    root = tmp_path/'one'
    if lost_ack:
        with pytest.raises(TimeoutError):
            execute(root, runtime, optimizer, binding(),
                    publish=lambda request: (_ for _ in ()).throw(TimeoutError('lost ACK')))
    else:
        execute(root, runtime, optimizer, binding())
    original = json.loads((root/'TRANSACTION.json').read_bytes())
    job = dict(original['job'])
    job['prepared_documents_sha256'] = tuple(job['prepared_documents_sha256'])
    masters = {name: value.detach().clone() for name, value in optimizer.masters.items()}
    first_moments = {name: optimizer.optimizer.state[value]['exp_avg'].clone()
                     for name, value in optimizer.masters.items()}
    # Neither the old actor nor its optimizer is reused after this point.
    del runtime, optimizer
    def load_model(directory, descriptor):
        from safetensors.torch import load_file
        result = Runtime()
        result.model.load_state_dict(load_file(str(directory/'model.safetensors')))
        result.checkpoint_id = descriptor['id']
        return result
    restored, state, journal = restore_transaction_state(root,
        expected_job=Job(**job), load_runtime=load_model)
    assert journal == original and state.global_step == 1
    for name, value in state.masters.items():
        assert torch.equal(value, masters[name])
        assert torch.equal(state.optimizer.state[value]['exp_avg'], first_moments[name])
    completed = execute(root, restored, state, binding())
    assert completed['state'] == 'published'
    assert state.global_step == 1 and restored.events == []
    assert restored.checkpoint_id == state.input_checkpoint == completed['output_checkpoint']


def test_project_document_uses_supplied_native_outcomes_and_keeps_all_nonces():
    group = groups(1, capped=True)[0]
    native, rollouts = [], []
    for row in group.trajectories:
        classification = dict(correct='positive', wrong='negative', unresolved='neutral')[row.outcome]
        turn = dict(prompt=list(group.prompt), output=list(row.tokens), classification=classification,
            reward=float(row.outcome == 'correct'), stop_reason='budget' if row.stop == 'max_tokens' else 'eos',
            done=True, observations=[])
        native.append(dict(turn, task_hash='e'*64, nonce=row.nonce, output_sha256=digest(turn['output'])))
        rollouts.append(dict(seed=row.nonce, turns=[turn], classification=classification,
            reward=turn['reward'], task_hash='e'*64))
    document = dict(version='complete-group-training-document-v1', epoch='epoch1', checkpoint='a'*64,
        batch=dict(env_id='math', environment_version='1', index=8, task_hash='e'*64, rollouts=rollouts))
    projected, receipt = project_document(document, expected_document_sha256=digest(document), binding=binding(),
        trusted_prompt=group.prompt, native_results=native, vocab_size=17, eos_token_ids=(15, 16))
    assert projected.trajectories == group.trajectories
    assert projected.admission_sha256 == digest(receipt)
    forged = deepcopy(document)
    forged['batch']['rollouts'][0]['classification'] = 'negative'
    with pytest.raises(ValueError, match='native outcome claim'):
        project_document(forged, expected_document_sha256=digest(forged), binding=binding(),
            trusted_prompt=group.prompt, native_results=native, vocab_size=17, eos_token_ids=(15, 16))
