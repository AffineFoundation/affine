# After-GREEN record (Jacob pt919u, 2026-10-31)

After each GREEN the k1 transparency sidecar (`affine-k1-transparency.service`, `services/k1_transparency.py` = the live copy)
publishes `/k1/turns/<epoch>/green.json` and `/k1/turns/<epoch>/tasks.json`. That happens only once the turn's held-out round
is CLOSED and the round reveal, the epoch reveal and the turn record are already public. RED turns and open rounds get nothing.

- `green.json` holds the decision math (z, gain, SE, sigma_run, kept, pass and cap rates, recomputed decision) and the
  baseline summary plus link (`/gate/heldout/<round>.json`: seed, blocks 0-3 per task, kept, settings). It also links the
  probe results (`/gate/reveal/<epoch>.json`: head and candidate cells per task), the tasks file, and the winning file
  with its lr / kl (`batch.json`, `turn.json`).
- `tasks.json` lists the round's 1000 sampled tasks (id, prompt, answer, level, subject, kept), read from the public pool
  (sha256 77a4524a...).
- `/k1/turns/index.json` marks each GREEN with a `green` link.

Change: one new function `publish_green` in the publish loop, plus the index link. Gate decisions, the gate service and the
trainer are untouched. It backfilled e292, e354, e356, e358, e360 and e368 on first start.
Revert: `sh revert.sh` (restores the previous sidecar, removes green.json/tasks.json, rebuilds the index, restarts the sidecar).
