# PIT — per-item train+eval (shadow)

Each queued registration gets its own GPU box. The box downloads the bound head
and fp32 master, trains with trainer-v834e, then runs the live-gate probe on the
same box. The host scores the probe. The first GREEN for a head becomes the new
head; later GREENs bound to an older head are paid but `merge_blocked` and never
applied.

## Flag (default off)

```json
{
  "parallel_eval": {
    "pit": {
      "enabled": false,
      "mode": "shadow",
      "box_cap": 2
    }
  }
}
```

- `enabled=false` by default.
- This PR only authorizes `mode=shadow`.
- Shadow never writes `DECISION.json`, weights, `HEAD-CHAIN`, or R2 head pointers.

## DESIGN §12 defaults shipped here

| Knob | Value |
|------|--------|
| Baseline | option A — one secret draw per head epoch via existing baseline pod |
| Admission | per-box ephemeral ed25519 + ROOT-signed short-lived cert (verifier + request only; no real ROOT signatures) |
| M(C) upload | during probe |
| Box cap | configurable, default 2 |
| Trainer pod | kept as fallback |

Gate math is frozen: k_cand 8, head 12, strict z>1, 64K, bf16, no fp8, N=1000.

## Modules

| File | Role |
|------|------|
| `box.py` | Box runner (download → train v834e → probe + M(C) upload) |
| `host.py` | Host controller (queue, cap, bind, score, decision log, shadow guard) |
| `dequeue.py` | Dequeue + head binding + staleness / merge_blocked |
| `decision_log.py` | Hash-chained decision log (green-time order) |
| `admission.py` | Ephemeral key + admission request + cert verify (fixture ROOT only) |
| `adapters.py` | pe_rent / pe_lane / pe_score / pe_live seams |
| `shadow.py` | Fail-closed refusal of live artifact writes |
| `config.py` / `state.py` | Config + state machine |

## Tests

```bash
cd /path/to/affine
PYTHONPATH=. python -m unittest \
  ops.parallel_eval.pit.test_state_machine \
  ops.parallel_eval.pit.test_binding \
  ops.parallel_eval.pit.test_decision_log \
  ops.parallel_eval.pit.test_shadow \
  ops.parallel_eval.pit.test_admission \
  ops.parallel_eval.pit.test_failure_retry \
  ops.parallel_eval.pit.test_host_e2e
```

## Operator: start shadow mode

See `docs/PIT.md`.
