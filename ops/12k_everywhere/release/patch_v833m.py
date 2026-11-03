#!/usr/bin/env python3
"""RA/20261102-12k-everywhere: trainer host v8.3.3m = v8.3.3l + the 12,288-token output cap in training (Jacob 2026-11-02 18:48 CR:
"12 k should be everywhere"). Host files only (no source / engine / pin change):
  * v8211_batched.py: TRAIN_MAX_CONTEXT = 16384 (= the gate: 12,288 output + prompt); context_limit(runtime) = max(signed runtime
    context 12,288, 16,384) replaces runtime.max_context_tokens in the host prompt+output checks. Free-file overlong window
    FF_OVERLONG = (10096, 12288): penalty = overlong_penalty (0.5) x clip((len - 10096) / 2192, 0, 1), i.e. the same 2,192-token ramp
    ending at the new cap (v8.3.3l: 6000..8192 from the signed job settings). FF_OVERLONG = None gives v8.3.3l exactly.
  * v83_dp.py, v83_fast_kernels.py: same context_limit in their prompt+output checks.
usage: patch_v833m.py SRC_DIR(host-v833l) OUT_DIR"""
import sys, shutil
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(); shutil.copytree(src, out)
(out / "TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json").unlink()
def rep(name, old, new, count=1):
    p = out / name; t = p.read_text(); n = t.count(old)
    assert n == count, (name, old[:60], n); p.write_text(t.replace(old, new))
OLD_RR = '''def rollout_reward(config, rollout):
    """v8.3.3h: native rollouts -> config.reward (unchanged). Free-file rollouts carry the miner's reward; the same
    soft overlong penalty applies (config.reward of a non-correct outcome is exactly minus that penalty)."""
    reward = getattr(rollout, 'reward', None)
    if reward is None:
        return config.reward(rollout.outcome, len(rollout.tokens))
    return reward + config.reward('wrong', len(rollout.tokens))
'''
NEW_RR = '''# v8.3.3m (RA/20261102-12k-everywhere, Jacob 2026-11-02 18:48 CR "12 k should be everywhere"): the 12,288-token output cap in training.
TRAIN_MAX_CONTEXT = 16384          # prompt + output tokens (12,288 output + prompt; the gate serves 16,384)
FF_OVERLONG = (10096, 12288)       # free-file overlong window (soft_start, max_tokens); None = the signed job settings (v8.3.3l: 6000, 8192)


def context_limit(runtime):
    """v8.3.3m: prompt+output limit for training/scoring = max(signed runtime context, TRAIN_MAX_CONTEXT)."""
    return max(int(runtime.max_context_tokens), TRAIN_MAX_CONTEXT)


def rollout_reward(config, rollout):
    """v8.3.3h: native rollouts -> config.reward (unchanged). Free-file rollouts carry the miner's reward; the same
    soft overlong penalty applies (config.reward of a non-correct outcome is exactly minus that penalty).
    v8.3.3m: free-file rollouts use the FF_OVERLONG window (same penalty and formula, ramp ending at the 12,288 cap)."""
    reward = getattr(rollout, 'reward', None)
    if reward is None:
        return config.reward(rollout.outcome, len(rollout.tokens))
    if FF_OVERLONG is not None and config.overlong_penalty:
        start, cap = FF_OVERLONG
        value = 0.
        value -= config.overlong_penalty * min(1., max(0., (len(rollout.tokens) - start) / (cap - start)))
        return reward + value
    return reward + config.reward('wrong', len(rollout.tokens))
'''
rep("v8211_batched.py", OLD_RR, NEW_RR)
rep("v8211_batched.py", "        if len(prompt) + len(output) > runtime.max_context_tokens:\n", "        if len(prompt) + len(output) > context_limit(runtime):\n")
rep("v8211_batched.py", "        if any(len(group.prompt) + len(r.tokens) > runtime.max_context_tokens for r in group.trajectories):\n",
    "        if any(len(group.prompt) + len(r.tokens) > context_limit(runtime) for r in group.trajectories):\n")
rep("v83_dp.py", "            if any(len(group.prompt) + len(r.tokens) > runtime.max_context_tokens for r in group.trajectories):\n",
    "            if any(len(group.prompt) + len(r.tokens) > max(int(runtime.max_context_tokens), 16384) for r in group.trajectories):   # v8.3.3m = v8211_batched.context_limit\n")
rep("v83_fast_kernels.py", "        if len(prompt) + len(output) > runtime.max_context_tokens:\n",
    "        if len(prompt) + len(output) > max(int(runtime.max_context_tokens), 16384):   # v8.3.3m = v8211_batched.context_limit\n")
print("patched", out)
