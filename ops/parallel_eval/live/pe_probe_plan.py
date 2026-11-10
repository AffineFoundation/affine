"""Shared podctl start-run plan builder for live pe_lane.run_probe and detached eval.

No host-path / replace_pod imports — safe for PIT unit tests.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional, Sequence


def build_probe_plan(
    cid: str,
    indices: Sequence[Any],
    plan_seed: str,
    k: int = 8,
    max_tokens: int = 65536,
    env_id: str = 'affine_gatepool',
    run_name: Optional[str] = None,
    max_num_seqs: int = 512,
    harness: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build the exact podctl start-run plan used by live ``pe_lane.run_probe``.

    When ``harness`` (public_harness.json) provides a ``harness`` key, include
    ``harness`` and ``manifest_projection`` (mimo_answer_projection,
    sampling_contract) — same as live. Detached eval MUST call this; never
    hand-copy the plan fields.
    """
    name = run_name or (
        'pe-%s-%s' % (cid[:12], hashlib.sha256(str(plan_seed).encode()).hexdigest()[:8])
    )
    pub = dict(harness or {})
    cpr = min(512, int(max_num_seqs))
    plan: dict = dict(
        run_name=name,
        checkpoint_id=cid,
        seed=plan_seed,
        concurrency=4096,
        grade_workers=160,
        resume=False,
        bandit_topk=False,
        concurrency_per_replica=cpr,
        envs=[dict(
            env_id=env_id,
            indices=list(indices),
            k=int(k),
            max_tokens=int(max_tokens),
        )],
    )
    if pub.get('harness'):
        plan['harness'] = pub['harness']
        plan['manifest_projection'] = dict(
            mimo_answer_projection=pub.get('mimo_answer_projection'),
            sampling_contract=pub.get('sampling_contract'),
        )
    return plan
