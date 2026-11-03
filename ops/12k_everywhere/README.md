# 12K everywhere (2026-11-02, live from epoch 590)

12,288 output tokens now apply to generation (miner kit), batch-file validation, training and the gate.

* Trainer v8.3.3m = v8.3.3l + two changes (`release/patch_v833m.py`, signed release `release-v833m-signed`):
  the host prompt+output limit is max(signed runtime 12288, 16384) in v8211_batched / v83_dp / v83_fast_kernels, and free-file
  rollouts use the overlong window 10096..12288 (same 0.5 penalty, same 2,192-token ramp, moved to the new cap).
  With `FF_OVERLONG = (6000, 8192)` v8.3.3m is identical to v8.3.3l (tests/equiv_pod.py: rows identical on real e585-e587 files).
* ff_format: MAX_OUTPUT 12288, MAX_CONTEXT 16384, LEGACY_CAP 8192 (an output of exactly 8192 tokens is still read as capped,
  so 8K-generated files queued before the switch keep their meaning; nothing valid before became invalid).
* Memory (tests/memtest_pod.py, 1 x H200, packed bins of 32,768 tokens): peak alloc 44.15 GB at 4x8192, 44.04 GB at 2x16384,
  44.16 GB at 2x12688. Bins are bounded by tokens, so 12K rows do not raise peak memory.
* Activation: `MODE=switch activation/activate_12k.sh` (idle decision gap, locks held, backups). Watchdog `activation/watchdog_12k.py`
  auto-reverts on a CUDA OOM in the trainer log, an epoch failure, or a training job open > 90 min.
  Revert: `MODE=revert bash <VALIDATOR_STATE>/root-audits/20261102-12k-everywhere/activation/activate_12k.sh`
