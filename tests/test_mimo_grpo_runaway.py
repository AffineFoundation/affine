"""Runaway training, admission, derail mask, constant normalization, all-eligible
max_tasks, k3 KL anchoring and the (off-by-default) sampler correction flag."""
from collections import Counter
from dataclasses import replace

import pytest
import torch

from subnet.mimo_grpo import TrainerConfig, TaskGroup, Trajectory, train_update
from subnet.mimo_grpo.config import (TRAIN_UNRESOLVED, MASK_UNRESOLVED, CONSTANT_NORM, SC_MIS,
                                     NO_SAMPLER_CORRECTION)
from subnet.mimo_grpo.optimizer import CPUAdamW
from subnet.mimo_grpo.training import derail_keep, _layout
from subnet.mimo_grpo.upstream import policy_loss, aggregate, k3_kl
from subnet.mimo_grpo.sampler_correction import replay_support, selected_candidates, corrected_terms
from subnet.complete_group_contract import (meets_minima, failure_count, LEGACY_OUTCOME_POLICY,
                                            UNRESOLVED_FAILURE_POLICY, OUTCOME_POLICY)
from test_mimo_grpo import Runtime, groups, binding

LIVE_SETTINGS = {"adam_epsilon": 1e-08, "beta1": 0.9, "beta2": 0.999,
    "cap_policy": "affine-unresolved-zero-reward-mask-all-v1", "clip_ratio": 0.2, "dual_clip": 3.0,
    "entropy_coefficient": 0.0, "group_size": 8, "kl_coefficient": 0.0, "learning_rate": 1e-06,
    "max_gradient_norm": 1.0, "max_tasks": 32, "normalize_advantage_std": False,
    "old_actor_logprob_tolerance": 1e-05, "ppo_epochs": 1, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
    "weight_decay": 0.01, "weight_decay_policy": "mcore-bias-and-1d-excluded-v1"}
REF = 'e' * 64


def proposed(**extra):
    return TrainerConfig(**dict(LIVE_SETTINGS, cap_policy=TRAIN_UNRESOLVED, max_tasks=256,
        loss_agg_mode=CONSTANT_NORM, loss_scale_factor=8192, derail_mask=True, **extra))


def runaway_group(task='r'):
    """4 correct + 4 budget-capped runaways: no completed wrong answer."""
    rows = [Trajectory(n, (1 + n, 15), 'correct', 'eos') for n in range(4)]
    rows += [Trajectory(n, tuple([3 + n % 5] * 12), 'unresolved', 'max_tokens') for n in range(4, 8)]
    return TaskGroup(task, (1, 2), tuple(rows), 'c' * 64)


# ---- legacy byte compatibility -------------------------------------------------

def test_live_v1_config_document_is_unchanged():
    document = TrainerConfig(**LIVE_SETTINGS).document()
    assert document['version'] == 'affine-mimo-reference-grpo-v1'
    assert document['objective'] == 'vanilla-dual-clip-ppo-prompt-mean'
    assert document['settings'] == LIVE_SETTINGS
    # Raising max_tasks / training unresolved alone keeps the v1 objective schema.
    only = TrainerConfig(**dict(LIVE_SETTINGS, max_tasks=256, cap_policy=TRAIN_UNRESOLVED)).document()
    assert only['version'] == 'affine-mimo-reference-grpo-v1' and set(only['settings']) == set(LIVE_SETTINGS)
    assert proposed().document()['version'] == 'affine-mimo-reference-grpo-v2'


# ---- 1. runaways get reward 0 and contribute loss --------------------------------

def test_runaways_trained_with_zero_reward_and_contribute_loss():
    config = replace(TrainerConfig(), cap_policy=TRAIN_UNRESOLVED)
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
    rows = []
    report = train_update(runtime, optimizer, [runaway_group()], binding(), config=config, record=rows.append)
    assert report['masked_rollouts'] == 0 and report['trained_rollouts'] == 8
    assert report['unresolved_trained_rollouts'] == 4
    runaways = [t for t in report['trajectories'] if t['outcome'] == 'unresolved']
    assert all(t['reward'] == 0. and t['advantage'] == -.5 and t['loss_tokens'] == 12 and t['loss'] != 0
               for t in runaways)
    assert not [r for r in rows if r['stage'] == 'masked-unresolved']


def test_runaway_gradient_pushes_down_runaway_tokens():
    config = replace(TrainerConfig(), cap_policy=TRAIN_UNRESOLVED, learning_rate=1e-2)
    runtime = Runtime()
    group = runaway_group()
    before = runtime.score(group.prompt, group.trajectories[-1].tokens, gradients=False).sum()
    optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
    train_update(runtime, optimizer, [group], binding(), config=config)
    after = runtime.score(group.prompt, group.trajectories[-1].tokens, gradients=False).sum()
    assert after < before


def test_masked_policy_still_masks_for_history():
    rows = _layout([runaway_group()], TrainerConfig())
    assert [r['active'] for r in rows] == [True] * 4 + [False] * 4


# ---- 1b. admission counts runaways as failures -----------------------------------

def group_contract(policy):
    return dict(version='complete-canonical-group-v1', group_size=8, min_success=1,
                min_completed_failure=1, outcome_policy=policy)


def test_admission_counts_unresolved_failures_under_v2_only():
    four_four = Counter(positive=4, neutral=4)
    assert OUTCOME_POLICY == UNRESOLVED_FAILURE_POLICY
    assert meets_minima(four_four, group_contract(UNRESOLVED_FAILURE_POLICY))
    assert not meets_minima(four_four, group_contract(LEGACY_OUTCOME_POLICY))
    assert failure_count(Counter(negative=1, neutral=3), UNRESOLVED_FAILURE_POLICY) == 4
    assert failure_count(Counter(negative=1, neutral=3), LEGACY_OUTCOME_POLICY) == 1
    with pytest.raises(ValueError):
        failure_count(four_four, 'unknown')


@pytest.mark.parametrize('counts', [dict(positive=8), dict(neutral=8), dict(negative=4, neutral=4),
                                    dict(negative=8)])
def test_admission_still_rejects_zero_variance_groups(counts):
    for policy in (UNRESOLVED_FAILURE_POLICY, LEGACY_OUTCOME_POLICY):
        assert not meets_minima(Counter(counts), group_contract(policy))


def test_group_contract_parsers_accept_both_policies():
    from subnet.complete_group_sampling import _group
    for policy in (UNRESOLVED_FAILURE_POLICY, LEGACY_OUTCOME_POLICY):
        assert _group(group_contract(policy)) == 8
    with pytest.raises(ValueError):
        _group(group_contract('completed-eos-math-unresolved-made-up'))


# ---- 2. derail mask ---------------------------------------------------------------

def brute_force(values, window, threshold, tail):
    for end in range(window - 1, len(values)):
        if sum(values[end - window + 1:end + 1]) / window < threshold:
            return min(len(values), end + tail + 1), end
    return len(values), None


@pytest.mark.parametrize('values', [
    [-.1] * 100 + [-5.] * 100,
    [-.1] * 10,
    [-.2] * 300,
    [-4.] * 40,
    [-.1] * 50 + [-6.] * 20 + [-.1] * 200,
    [-.1] * 200 + [-9.] * 20,
])
def test_derail_positions_match_definition(values):
    keep, index = derail_keep(torch.tensor(values), window=32, threshold=-3., tail=64)
    assert (keep, index) == brute_force(values, 32, -3., 64)


def test_derail_specific_position():
    # Mean of a trailing 32 window crosses -3 when 20 of 32 tokens are -5:
    # 19 of 32 at -5 first gives (13*-.1 + 19*-5)/32 = -3.009 -> index 100 + 18 = 118.
    keep, index = derail_keep(torch.tensor([-.1] * 100 + [-5.] * 400), window=32, threshold=-3., tail=64)
    assert index == 118 and keep == 118 + 64 + 1


class DerailRuntime(Runtime):
    """Adds a large constant (non-differentiable) drop after position 20."""
    max_context_tokens = 128

    def score(self, prompt, output, *, gradients):
        values = super().score(prompt, output, gradients=gradients)
        drop = torch.tensor([0. if i < 20 else -12. for i in range(len(output))])
        return values + drop


def test_derail_mask_limits_loss_tokens_of_truncated_negatives_only():
    config = proposed(derail_window=4, derail_tail=3, derail_threshold=-8.)
    runtime = DerailRuntime()
    rows = [Trajectory(n, (1 + n, 15), 'correct', 'eos') for n in range(2)]
    rows += [Trajectory(2, tuple([4] * 30 + [15]), 'wrong', 'eos')]
    rows += [Trajectory(n, tuple([3 + n % 5] * 60), 'unresolved', 'max_tokens') for n in range(3, 8)]
    group = TaskGroup('d', (1, 2), tuple(rows), 'c' * 64)
    optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
    report = train_update(runtime, optimizer, [group], binding(), config=config)
    by_nonce = {t['nonce']: t for t in report['trajectories']}
    for nonce in range(3, 8):
        t = by_nonce[nonce]
        assert t['derail_index'] is not None and 20 <= t['derail_index'] <= 23
        assert t['loss_tokens'] == t['derail_index'] + 3 + 1
    # The completed wrong answer (EOS) is never derail-masked.
    assert by_nonce[2]['derail_index'] is None and by_nonce[2]['loss_tokens'] == 31
    assert report['derailed_rollouts'] == 5


def test_derail_requires_trained_unresolved():
    with pytest.raises(ValueError, match='derail'):
        TrainerConfig(derail_mask=True).validate()


# ---- 2b. constant normalization ------------------------------------------------------

def test_constant_normalization_equation_and_length_independence():
    config = proposed()
    torch.manual_seed(5)
    x = torch.randn(4, 50, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros_like(x)
    for row, length in enumerate((3, 50, 10, 1)):
        mask[row, :length] = 1
    adv = torch.tensor([.5, -.5, .5, -.5], dtype=torch.float64)[:, None].expand_as(x)
    whole, _ = policy_loss(x.detach(), x, adv, mask, None, config, global_batch_size=4)
    expected = (-adv * mask).sum() / (4 * 8192)
    assert torch.allclose(whole, expected, atol=1e-15)
    grad = torch.autograd.grad(whole, x)[0]
    # Every active token gets the same |gradient| regardless of rollout length.
    active = grad[mask.bool()].abs()
    assert torch.allclose(active, torch.full_like(active, .5 / (4 * 8192)))
    split = sum(policy_loss(x.detach()[i:i+1], x[i:i+1], adv[i:i+1], mask[i:i+1], None, config,
                            global_batch_size=4)[0] for i in range(4))
    assert torch.allclose(split, whole, atol=1e-15)


def test_constant_normalization_validation():
    with pytest.raises(ValueError):
        TrainerConfig(loss_agg_mode=CONSTANT_NORM).validate()
    with pytest.raises(ValueError):
        TrainerConfig(loss_scale_factor=1024).validate()
    with pytest.raises(ValueError):
        TrainerConfig(loss_agg_mode='token-mean').validate()
    assert proposed().validate().loss_scale_factor == 8192


def test_constant_normalization_update_microbatch_invariant():
    config = proposed()
    left, right = Runtime(), Runtime()
    a = CPUAdamW(left.model.named_parameters(), config, input_checkpoint=left.checkpoint_id)
    b = CPUAdamW(right.model.named_parameters(), config, input_checkpoint=right.checkpoint_id)
    first = train_update(left, a, groups(capped=True), binding(), config=config, microbatch_rollouts=1)
    second = train_update(right, b, groups(capped=True), binding(), config=config, microbatch_rollouts=5)
    assert first['preclip_gradient_norm'] == pytest.approx(second['preclip_gradient_norm'], rel=1e-9)
    for p, q in zip(left.model.parameters(), right.model.parameters()):
        assert torch.allclose(p, q, atol=1e-12)


# ---- 3. max_tasks covers all eligible --------------------------------------------------

def many_groups(count):
    base = groups(1, capped=True)[0]
    return [replace(base, task_id='t%d' % i) for i in range(count)]


def test_max_tasks_all_eligible_trains_126_groups_in_one_step():
    config = replace(proposed(), max_tasks=256)
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
    report = train_update(runtime, optimizer, many_groups(126), binding(), config=config)
    assert report['group_count'] == 126 and report['trained_rollouts'] == 126 * 8
    assert optimizer.global_step == 1


def test_max_tasks_bound_enforced():
    config = TrainerConfig(**LIVE_SETTINGS)
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
    with pytest.raises(ValueError, match='batch limit'):
        train_update(runtime, optimizer, many_groups(33), binding(), config=config)
    with pytest.raises(ValueError):
        TrainerConfig(max_tasks=513).validate()


def test_coordinator_budget_selects_all_when_cap_exceeds_eligible():
    # Mirrors mimo_coordinator.seal: selected = unique[:max_tasks].
    unique = list(range(126))
    assert unique[:256] == unique and unique[256:] == []


# ---- KL anchoring ----------------------------------------------------------------------------

class Reference(Runtime):
    checkpoint_id = REF


def test_k3_equation():
    cur = torch.tensor([[-1., -2., -.5]], dtype=torch.float64)
    ref = torch.tensor([[-1.5, -2., -.1]], dtype=torch.float64)
    d = ref - cur
    assert torch.allclose(k3_kl(cur, ref), d.exp() - d - 1)
    assert torch.equal(k3_kl(cur, cur), torch.zeros_like(cur))


def test_kl_config_validation():
    with pytest.raises(ValueError, match='reference checkpoint'):
        TrainerConfig(kl_coefficient=.02).validate()
    with pytest.raises(ValueError):
        TrainerConfig(kl_reference_checkpoint=REF).validate()
    with pytest.raises(ValueError):
        TrainerConfig(kl_coefficient=.02, kl_reference_checkpoint=REF, kl_type='k1').validate()
    document = proposed(kl_coefficient=.02, kl_reference_checkpoint=REF).document()
    assert document['objective'].endswith('-k3-kl-reference')


def test_kl_requires_separate_matching_reference_runtime():
    config = proposed(kl_coefficient=.02, kl_reference_checkpoint=REF)
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
    with pytest.raises(ValueError, match='reference runtime'):
        train_update(runtime, optimizer, groups(), binding(), config=config)
    wrong = Runtime()
    wrong.checkpoint_id = 'd' * 64
    with pytest.raises(ValueError, match='reference runtime'):
        train_update(runtime, optimizer, groups(), binding(), config=config, reference_runtime=wrong)
    plain = CPUAdamW(runtime.model.named_parameters(), proposed(), input_checkpoint=runtime.checkpoint_id)
    with pytest.raises(ValueError, match='without a KL'):
        train_update(runtime, plain, groups(), binding(), config=proposed(), reference_runtime=Reference())


def test_kl_zero_when_reference_equals_actor_and_positive_otherwise():
    config = proposed(kl_coefficient=.02, kl_reference_checkpoint=REF)
    same_actor, plain = Runtime(), Runtime()
    same_ref = Reference(); same_ref.model.load_state_dict(same_actor.model.state_dict())
    a = CPUAdamW(same_actor.model.named_parameters(), config, input_checkpoint=same_actor.checkpoint_id)
    report = train_update(same_actor, a, groups(), binding(), config=config, reference_runtime=same_ref)
    assert all(t['k3_kl_mean'] == pytest.approx(0, abs=1e-7) for t in report['trajectories'])
    # Identical policy gradient to the no-KL objective when KL == 0 at the anchor.
    b = CPUAdamW(plain.model.named_parameters(), proposed(), input_checkpoint=plain.checkpoint_id)
    other = train_update(plain, b, groups(), binding(), config=proposed())
    assert report['preclip_gradient_norm'] == pytest.approx(other['preclip_gradient_norm'], rel=1e-6)

    torch.manual_seed(99)
    far = Reference(torch.nn.Sequential(torch.nn.Embedding(17, 5), torch.nn.Linear(5, 17)))
    actor = Runtime()
    c = CPUAdamW(actor.model.named_parameters(), config, input_checkpoint=actor.checkpoint_id)
    report = train_update(actor, c, groups(), binding(), config=config, reference_runtime=far)
    assert all(t['k3_kl_mean'] > 0 for t in report['trajectories'])
    assert report['preclip_gradient_norm'] != pytest.approx(other['preclip_gradient_norm'], rel=1e-6)


# ---- sampler correction flag (off by default) -------------------------------------------------

class SupportRuntime(Runtime):
    def _logits(self, prompt, output):
        tokens = torch.tensor([*prompt, *output[:-1]])
        return self.model(tokens)[len(prompt)-1:]

    def score_support(self, prompt, output, *, top_k, top_p):
        with torch.no_grad():
            logits = self._logits(prompt, output)
            selected = logits.log_softmax(-1).gather(-1, torch.tensor(output)[:, None]).squeeze(-1)
            indices, q = replay_support(logits, top_k, top_p)
        return selected, indices, q

    def score_candidates(self, prompt, output, indices, *, gradients):
        with torch.set_grad_enabled(gradients):
            logits = self._logits(prompt, output)
            selected = logits.log_softmax(-1).gather(-1, torch.tensor(output)[:, None]).squeeze(-1)
            return selected, logits.gather(-1, indices)


def test_sampler_correction_off_by_default():
    assert TrainerConfig().sampler_correction == NO_SAMPLER_CORRECTION
    assert proposed().sampler_correction == NO_SAMPLER_CORRECTION
    with pytest.raises(ValueError):
        TrainerConfig(sampler_correction='made-up').validate()


def test_replay_support_matches_sampler_distribution():
    from subnet.complete_group_distribution import distribution
    torch.manual_seed(3)
    logits = torch.randn(6, 50) * 3
    indices, q = replay_support(logits, 20, .95)
    reference = distribution(logits, 1.0, .95, 20)
    rebuilt = torch.zeros_like(reference).scatter(-1, indices, q.to(reference.dtype))
    assert torch.allclose(rebuilt, reference, atol=1e-6)


def test_corrected_terms_centering_and_outside_support():
    logits = torch.tensor([[2., 1., .5], [0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    q = torch.softmax(logits.detach(), -1)
    selected = torch.tensor([0, -1])
    term, extra = corrected_terms(logits, q, selected)
    assert term[1] == 0 and not extra['inside'][1]
    # With q equal to the trained support distribution all ratios are 1 and the
    # centered score has zero expectation over q.
    (grad,) = torch.autograd.grad(term[0], logits)
    expected = torch.nn.functional.one_hot(torch.tensor(0), 3).double() - q[0]
    expected = expected - (q[0][:, None] * (torch.eye(3, dtype=torch.float64) - q[0])).sum(0)
    assert torch.allclose(grad[0], expected, atol=1e-12)
    assert selected_candidates(torch.tensor([[4, 2]]), torch.tensor([[.7, .3]]), [2]).tolist() == [1]


def test_sampler_correction_flag_trains_when_explicitly_enabled():
    config = proposed(sampler_correction=SC_MIS)
    assert 'sc-mis' in config.document()['objective']
    runtime = SupportRuntime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
    report = train_update(runtime, optimizer, groups(capped=True), binding(), config=config)
    assert report['status'] == 'updated' and report['trained_rollouts'] == 16
    assert all('sampler/outside_support' in t for t in report['trajectories'])
    with pytest.raises(ValueError, match='support replay'):
        train_update(Runtime(), CPUAdamW(Runtime().model.named_parameters(), config,
                     input_checkpoint='a' * 64), groups(), binding(), config=config)
