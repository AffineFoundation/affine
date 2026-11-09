# PIT — per-item train+eval (serverless)

See `docs/PIT.md` for operator cutover / revert steps, ROOT ceremony, and
deploy file placement.

**On-demand rent only** (hard rule): empty queue ⇒ 0 item boxes; B queued with
`max_boxes` N ⇒ ≤ min(B, N) rentals; release as soon as result is stored or the
item fails. No warm pool. Baseline / bench / prior sit outside the item cap.
Live: `mode=live`, box_cap 5, B300 pin, $56/h ceiling, masters after GREEN,
REDs earn nothing. ROOT: **delegated host key only** (ceremony once; host issues
short-lived box certs).

```bash
PYTHONPATH=. python3 -m unittest \
  ops.parallel_eval.pit.test_state_machine \
  ops.parallel_eval.pit.test_binding \
  ops.parallel_eval.pit.test_decisions \
  ops.parallel_eval.pit.test_signer \
  ops.parallel_eval.pit.test_ceremony \
  ops.parallel_eval.pit.test_decision_emit \
  ops.parallel_eval.pit.test_durable \
  ops.parallel_eval.pit.test_expected_master \
  ops.parallel_eval.pit.test_ondemand_rent \
  ops.parallel_eval.pit.test_pe_score \
  ops.parallel_eval.pit.test_shadow_controller \
  ops.parallel_eval.pit.test_box_fake \
  ops.parallel_eval.pit.test_live_controller \
  ops.parallel_eval.pit.test_cutover \
  ops.parallel_eval.pit.test_jobs
```
