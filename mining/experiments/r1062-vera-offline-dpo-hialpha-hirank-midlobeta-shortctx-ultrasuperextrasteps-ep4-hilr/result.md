# R1062 — pass notes

## p4205 (2026-08-20T20:56Z)
- train.done @20:43Z (884 steps, adapter OK).
- First merge hit **ENOSPC** writing shard 12/16 of `/tmp/r1062_merged`.
- Root cause: ~8 leftover `/tmp/*_merged` (~66G each) on 509G overlay (r938 4×H200).
- Action: deleted stale merges + old `/root/r*` workdirs + train checkpoints; relaunched merge pid **41047**; arm writes `r1062_merge.done` / `r1062_merge_ready` for existing n80 waiter.
- Decision rule unchanged: fresh v4 n80 vs reign36; submit iff margin > max(2·SE, δ=0.002) ∧ thought≥80 ∧ B≥0.30.
