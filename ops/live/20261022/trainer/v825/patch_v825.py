from pathlib import Path
P = Path('/home/const/subnet120-rewrite/state/root-audits/20261021-v825-drop-unresolved/stage/trainer-host-v825')
def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)
# ---- v825_batched.py
b = (P/'v825_batched.py').read_text()
b = sub(b, '"""v8.2.4 candidate: micro-batched single-step GRPO update (same objective).\n',
 '"""v8.2.5: v8.2.4 micro-batched single-step GRPO update + config-gated unresolved-rollout drop.\n\n'
 'v8.2.5 delta (only when DROP_UNRESOLVED is True; set by the ROOT-admitted host launcher from\n'
 'engine.drop_unresolved): rollouts with outcome == \'unresolved\' (budget-capped or EOS without a\n'
 'gradeable final answer) are excluded from the loss (inactive rows: no gradient, not counted in\n'
 'global_batch_size / prompt weights). GRPO advantages are recomputed with the same pinned upstream\n'
 'compute_grpo_outcome_advantage over the RESOLVED rollouts of each group only (mean baseline over\n'
 'resolved rewards; no std normalisation, as before). A group with fewer than 2 resolved rollouts is\n'
 'dropped from the loss entirely (all rows inactive). group_count / rollout_count / admitted groups\n'
 'are unchanged (all admitted groups are still reported, so host validate_result holds); dropped rows\n'
 'are counted in masked_rollouts and in report[\'unresolved_filter\']. With DROP_UNRESOLVED False the\n'
 'update is byte-for-byte the v8.2.4 path.\n\n'
 'v8.2.4 notes (unchanged):\n')
b = sub(b, "from subnet.mimo_grpo.training import _layout, derail_keep, _versions\n",
 "from subnet.mimo_grpo.training import _layout, _prompt_weights, derail_keep, _versions\n"
 "from subnet.mimo_grpo.upstream import reference\n"
 "import numpy as np\n\n"
 "DROP_UNRESOLVED = False   # v8.2.5: set only by mimo_host_trainer_v825.install from engine['drop_unresolved']\n"
 "DROP_POLICY = 'affine-drop-unresolved-resolved-only-advantage-min2-v1'\n")
b = sub(b, "ENGINES = ('padded', 'packed')\n", '''ENGINES = ('padded', 'packed')


def layout_drop_unresolved(groups, config):
    """v8.2.5: rows like training._layout, but unresolved rollouts never enter the loss and the
    group baseline is computed over resolved rollouts only; groups with < 2 resolved are dropped."""
    rows = []
    stats = dict(enabled=True, policy=DROP_POLICY, groups_total=len(groups), dropped_groups=0,
                 dropped_unresolved_rollouts=0, dropped_resolved_rollouts_in_dropped_groups=0,
                 dropped_unresolved_by_stop={}, dropped_unresolved_tokens=0, kept_rollouts=0)
    for group in groups:
        resolved = [r for r in group.trajectories if r.outcome != 'unresolved']
        keep_group = len(resolved) >= 2
        advantage = {}
        if keep_group:
            rewards = torch.tensor([[config.reward(r.outcome, len(r.tokens))] for r in resolved], dtype=torch.float32)
            values, _ = reference().compute_grpo_outcome_advantage(rewards,
                torch.ones_like(rewards), np.array([group.task_id] * len(resolved)),
                norm_adv_by_std_in_grpo=False)
            for rollout, value in zip(resolved, values[:, 0], strict=True):
                advantage[id(rollout)] = float(value)
        else:
            stats['dropped_groups'] += 1
            stats['dropped_resolved_rollouts_in_dropped_groups'] += len(resolved)
        for rollout in group.trajectories:
            unresolved = rollout.outcome == 'unresolved'
            if unresolved:
                stats['dropped_unresolved_rollouts'] += 1
                stats['dropped_unresolved_tokens'] += len(rollout.tokens)
                stop = str(rollout.stop)
                stats['dropped_unresolved_by_stop'][stop] = stats['dropped_unresolved_by_stop'].get(stop, 0) + 1
            active = keep_group and not unresolved
            stats['kept_rollouts'] += int(active)
            rows.append(dict(group=group, rollout=rollout, advantage=advantage.get(id(rollout), 0.0), active=active,
                             keep=len(rollout.tokens), derail_index=None,
                             reward=config.reward(rollout.outcome, len(rollout.tokens))))
    if not stats['kept_rollouts']:
        raise ValueError('drop_unresolved left no trainable rollouts in the batch')
    _prompt_weights(rows)
    return rows, stats
''')
b = sub(b, "    rows = _layout(groups, config)\n",
 "    if DROP_UNRESOLVED:\n"
 "        rows, unresolved_filter = layout_drop_unresolved(groups, config)\n"
 "    else:\n"
 "        rows, unresolved_filter = _layout(groups, config), dict(enabled=False)\n"
 "    record(dict(stage='unresolved-filter', **unresolved_filter))\n")
b = sub(b, "            microbatch=dict(engine=engine, token_budget=token_budget, max_rows=max_rows, count=len(plan)),\n",
 "            microbatch=dict(engine=engine, token_budget=token_budget, max_rows=max_rows, count=len(plan)),\n"
 "            unresolved_filter=unresolved_filter,\n")
(P/'v825_batched.py').write_text(b)
# ---- launcher
l = (P/'mimo_host_trainer_v825.py').read_text()
l = sub(l, '"""ROOT-admitted v8.2.4 trainer host launcher over the unchanged worker closure.\n',
 '"""ROOT-admitted v8.2.5 trainer host launcher over the unchanged worker closure.\n\n'
 'v8.2.5 = v8.2.4 + engine.drop_unresolved (bool): when true, unresolved rollouts are dropped from the\n'
 'loss (see v825_batched.py). Rollback: drop_unresolved=false config or the v8.2.4 owned worker.\n')
l = sub(l, "VERSION = 'affine-mimo-trainer-host-v824'\nSCOPE = 'sn120-mimo-trainer-host-overrides-v8.2.4'\n",
 "VERSION = 'affine-mimo-trainer-host-v825'\nSCOPE = 'sn120-mimo-trainer-host-overrides-v8.2.5'\n")
l = sub(l, "HOST_FILES = ('mimo_host_trainer_v824.py', 'v824_batched.py',", "HOST_FILES = ('mimo_host_trainer_v825.py', 'v825_batched.py',")
l = sub(l, "               'part_concurrency', 'shard_concurrency', 'optimizer_step_threads'}\n",
 "               'part_concurrency', 'shard_concurrency', 'optimizer_step_threads', 'drop_unresolved'}\n")
l = sub(l, "         'exact v8.2.4 trainer host configuration')", "         'exact v8.2.5 trainer host configuration')")
l = sub(l, "         and set(config['engine']) == ENGINE_KEYS, 'exact v8.2.4 host inventory')",
 "         and set(config['engine']) == ENGINE_KEYS and type(config['engine']['drop_unresolved']) is bool,\n         'exact v8.2.5 host inventory')")
l = sub(l, "    batched = load('v824_batched', 'v824_batched.py')\n",
 "    # imported by the patched service as `v824_batched`; v8.2.5 module registered under that name\n"
 "    batched = load('v824_batched', 'v825_batched.py')\n"
 "    batched.DROP_UNRESOLVED = engine['drop_unresolved']\n")
l = sub(l, "    return dict(component='trainer-host-v824', engine=engine, sites=sorted(SITES))",
 "    return dict(component='trainer-host-v825', engine=engine, drop_unresolved=batched.DROP_UNRESOLVED,\n"
 "                drop_policy=batched.DROP_POLICY, sites=sorted(SITES))")
l = sub(l, "usage: mimo_host_trainer_v824.py", "usage: mimo_host_trainer_v825.py")
(P/'mimo_host_trainer_v825.py').write_text(l)
print('ok')
