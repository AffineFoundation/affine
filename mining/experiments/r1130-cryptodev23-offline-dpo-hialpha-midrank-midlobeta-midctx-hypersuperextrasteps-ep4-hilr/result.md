# R1130 — result notes

## p4275 (2026-08-21T07:09Z)
- Prior n80 (p4258): chall TP2 util=0.72 OOM on prompt-logprobs (+7.58 GiB) → EngineDead → sim ConnectError → `FATAL missing sim result`.
- Action: exact-PID reap stale chall/sim; re-arm **TP1 util=0.85** on GPU3 port **:8002** from `/tmp/r1130_merged` (16 shards intact).
- Script: `p4275_reap_rearm_tp1_util085.sh` · chall pid **165122** · n80 arms after CHALL_READY+probe.
- Decision rule unchanged: crown iff margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 under wvk7 k=3 τ=0.03 vs reign36.
- Check: `ssh -p 32301 root@93.120.231.186` → `/root/logs/p4275_r1130_rearm_tp1.log`, `/root/affine_data/r1130_decision_reign36_wvk7.json`.
