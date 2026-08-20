# p4085 — R926 king@0.95 + R938 LOST

- **R938 LOST** chal-00949: m=-0.000615 SE=0.000547 z=-1.12 n=1290 bar=δ0.002 (~-0.31×) thought✓167 B✓0.417 (n80 had been +0.004951 ~1.20×).
- **R926 root cause:** king vLLM @ gpu_memory_utilization=0.85 loaded 65.53GiB → Available KV **-2.71GiB** → engine fail.
- **Fix:** keep teacher TP=2; relaunch king GPU2 @ **0.95** → KV **+5.21GiB** → READY; seed chall Triton from king; chall:8002 loading for R944 v4 n80.
- Script: `p4085_r926_king095_chall_n80.sh` (outer pid on pod).
- Stock: B300×8=0; 8×B200 catalog = BL only.
