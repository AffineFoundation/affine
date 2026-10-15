"""Soft overlong penalty (v8.1): reward shaping past the soft threshold."""
from dataclasses import replace
import pytest
import torch

from subnet.mimo_grpo import TrainerConfig, TaskGroup, Trajectory
from subnet.mimo_grpo.config import TRAIN_UNRESOLVED, MASK_UNRESOLVED
from subnet.mimo_grpo.training import _layout
from test_mimo_grpo_runaway import proposed, LIVE_SETTINGS


def cfg(**k):
    return proposed(kl_coefficient=0.02, kl_reference_checkpoint='e' * 64, **k)


def test_off_by_default_and_v1_v2_documents_unchanged():
    assert TrainerConfig(**LIVE_SETTINGS).document()['settings'] == LIVE_SETTINGS
    base = cfg().document()
    assert 'overlong' not in base['objective']
    assert base['settings']['overlong_penalty'] == 0.
    assert TrainerConfig().reward('correct', 10 ** 6) == 1. and TrainerConfig().reward('wrong', 5) == 0.


def test_validation():
    with pytest.raises(ValueError):
        cfg(overlong_soft_start=7000, overlong_max_tokens=8192).validate()  # window without penalty
    for bad in [dict(overlong_penalty=.5), dict(overlong_penalty=.5, overlong_soft_start=8192, overlong_max_tokens=8192),
                dict(overlong_penalty=1.5, overlong_soft_start=7000, overlong_max_tokens=8192),
                dict(overlong_penalty=float('nan'), overlong_soft_start=7000, overlong_max_tokens=8192),
                dict(overlong_penalty=.5, overlong_soft_start=7000.0, overlong_max_tokens=8192)]:
        with pytest.raises(ValueError):
            cfg(**bad).validate()
    good = cfg(overlong_penalty=.5, overlong_soft_start=7000, overlong_max_tokens=8192)
    assert good.document()['objective'].endswith('-overlong-soft-penalty')


def test_reward_equation():
    c = cfg(overlong_penalty=.5, overlong_soft_start=7000, overlong_max_tokens=8192)
    assert c.reward('correct', 6999) == 1. and c.reward('correct', 7000) == 1.
    assert c.reward('correct', 7596) == pytest.approx(1 - .5 * 596 / 1192)
    assert c.reward('correct', 8192) == pytest.approx(.5)
    assert c.reward('wrong', 100) == 0. and c.reward('unresolved', 8192) == pytest.approx(-.5)
    assert c.reward('unresolved', 9000) == pytest.approx(-.5)  # clipped


def test_layout_advantages_use_shaped_rewards():
    c = cfg(overlong_penalty=.5, overlong_soft_start=10, overlong_max_tokens=20)
    rows = [Trajectory(0, tuple([5] * 5), 'correct', 'eos'), Trajectory(1, tuple([5] * 20), 'correct', 'eos')]
    rows += [Trajectory(n, tuple([5] * 6), 'wrong', 'eos') for n in range(2, 6)]
    rows += [Trajectory(n, tuple([3] * 20), 'unresolved', 'max_tokens') for n in range(6, 8)]
    out = _layout([TaskGroup('t', (1, 2), tuple(rows), 'c' * 64)], c)
    rewards = [1., .5, 0., 0., 0., 0., -.5, -.5]
    mean = sum(rewards) / 8
    assert [r['reward'] for r in out] == pytest.approx(rewards)
    assert [r['advantage'] for r in out] == pytest.approx([x - mean for x in rewards], abs=1e-6)
    # short correct beats long correct; truncated runaways are below short wrong answers
    assert out[0]['advantage'] > out[1]['advantage'] > out[2]['advantage'] > out[6]['advantage']


def test_penalty_off_layout_identical_to_v8():
    rows = [Trajectory(n, tuple([5] * (3 + n)), 'correct' if n < 4 else 'wrong', 'eos') for n in range(8)]
    g = TaskGroup('t', (1, 2), tuple(rows), 'c' * 64)
    a = [(r['advantage'], r['active'], r['keep']) for r in _layout([g], cfg())]
    assert a == [(.5, True, 3 + n) for n in range(4)] + [(-.5, True, 3 + n) for n in range(4, 8)]
