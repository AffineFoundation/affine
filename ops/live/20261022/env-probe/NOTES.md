# SN120 env probe (v10 candidate piece 1)  — notes (CR = America/Costa_Rica, UTC-6)
Task from Jacob 2026-10-22 ~12:48 CR: detached probe that samples every new published checkpoint on all envs,
publishes per-env pass-rate series at affine.io/probe/, stages (does NOT submit) an on-chain commitment, measures scaling.
Scope guard: trainer pod, weights, hyperparameters, v9 writer/scorer (seq7), Tobias mech/machines, R2 writes: NOT touched.

## Discovery (12:48-13:10 CR)
- Checkpoint publication: lifecycle writes ~/subnet120-rewrite/state/mimo-mainnet-v1/CURRENT.json (checkpoint id+file sha256s,
  epoch_index, optimizer_step, signed checkpoint_authorization) when the trainer publishes; evaluator jobs
  (mimo-mainnet-v1/evaluator/mimo-eval-*.job.json) carry 24h presigned R2 GET URLs for those files.
- Evaluator: Lium H200 pod <EVALUATOR_HOST>:<EVALUATOR_SSH_PORT> (143 GB), worker = subnet.distributed_worker (HF transformers, eager attention,
  single stream, its own sampler). Uses ~21 GB GPU. Per checkpoint: fixed32 affine_math, ~66 min, ~28 tok/s, then idle.
  Epoch cadence observed e79-e90: ~70-90 min (not 40). e90 (10:32 CR) is the last; trainer e91 stalled (other worker on it).
- Live mechanism has ONE env in the signed manifest: affine_math (7496-task snapshot, completed-boxed-math outcome,
  final-channel projection mimo-final-channel-math-v1, temp 1.0 top_p 0.95 top_k 20, max_out 8192, ctx 12288, thinking on).
- Registry (subnet/environments.py) has ~45 envs; single-turn text ones are runnable on the pod with the same session code.
- Public site: ~/subnet120/affine/website served by Caddy+cloudflared as https://affine.io/ (seq7 public_scores_v9.py
  writes scores.json/scores/ with atomic 0644 writes). Probe uses the same path: website/probe/.
