# PIT — per-item train+eval (shadow)

See `docs/PIT.md` for operator steps and DESIGN rules.

**On-demand rent only** (hard rule): empty queue ⇒ 0 boxes; B queued with
`max_boxes` N ⇒ ≤ min(B, N) rentals; release as soon as result is stored or the
item fails. No warm pool.

```bash
PYTHONPATH=. python -m unittest \
  ops.parallel_eval.pit.test_state_machine \
  ops.parallel_eval.pit.test_binding \
  ops.parallel_eval.pit.test_decisions \
  ops.parallel_eval.pit.test_signer \
  ops.parallel_eval.pit.test_expected_master \
  ops.parallel_eval.pit.test_ondemand_rent \
  ops.parallel_eval.pit.test_pe_score \
  ops.parallel_eval.pit.test_shadow_controller \
  ops.parallel_eval.pit.test_box_fake
```
