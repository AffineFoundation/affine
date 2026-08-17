# R634 result

## Status (p3679)
- **SCP_READY** `2026-08-17T15:13:00Z` on zesty — 16/16 shards · 66G · shard4 = 3962207584 (matched source). Mid-SCP hole self-healed as live tar rewrote shard4 then 15–16.
- **v4 lean chall LIVE** GPUs **6,7** / `:8003` pid**772101** (p3673 waiter; Triton seed REUSE; outs `*_wvk7` fail-closed k=3).
- p3678 repair waited correctly through tar; after READY it stuck on **self-matching** `pgrep -af "tar xf" | grep r634_merged` (remote check argv contains both strings) → killed PID **724571** only (SCP already done; chall up).
- **R647** host-relay auto-started brave→golden once R634 uplink freed (~4G at p3679 close).

## Decision rule (pre-registered)
Crown iff paired mean(Reason_c−Reason_k) > max(2·SE, 0.002) **and** median |z|≥80 **and** B pass≥0.30 on fresh v4 (k=3, τ=0.03) slice vs reign34.
