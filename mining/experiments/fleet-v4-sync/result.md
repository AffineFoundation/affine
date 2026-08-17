# fleet-v4-sync — Reason v4 affine_pkg patch (p3664)

## Why
Live contract forked to **wvk=7** (k=3 teacher refs, τ=0.03 tempered LME, n_turns=1300).
All `mine-*` pods still had `affine_pkg` at **wvk=6 / n_teacher_samples=1 / tau=None / n_turns=2080**,
so n80 margins were not v4-isomorphic (R643 stamped k=1).

## What
Copied live validator sources (read-only) into
`mining/experiments/fleet-v4-sync/affine_pkg/` and deployed to all 6 `mine-*` pods:

- `affine.toml`, `affine/score.py`, `affine/config.py`
- `evalsrv/terms.py`, `evalsrv/dueling.py`, `evalsrv/chat.py`

## Verify (zesty-comet-da)
- cfg.wvk=7 · duel k=3 · τ=0.03 · n_turns=1300
- `turn_reason` k=1 identity OK; LME@τ=0.03 prefers best of 3 refs (0.967 > mean 0.433)

## Artifact
`v4_affine_pkg_patch.tar.gz` + `verify_v4.py`

## Next
R634 SCP (~19G/5sh at p3664) → chall → n80 will pick up patched pkg automatically via `PYTHONPATH=/root/mining_src/affine_pkg`.
