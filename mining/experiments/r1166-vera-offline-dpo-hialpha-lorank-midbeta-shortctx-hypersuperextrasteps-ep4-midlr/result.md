# R1166 — ShortCtx LoRank Midβ Hyper MidLR

**Status (p4300):** **n80 LIVE** on `mine-r339` GPU **6** TP1 util0.85 :8003 vs reign36 (after TP2 cutlass-stall kill).

| field | value |
|---|---|
| parent | R1145 ShortCtx LoRank Midβ Hyper UltraLoLR REFUTE m=−0.001512 SE=0.002307 ~−0.33× thought✓166 B✓0.428 k=3 → MidLR isolate |
| knobs | β=0.1 α=128 r=**16** lr=**1e-6** @6144 Soft Mid Mid Soft→ShortCtx **max_steps=38400** |
| merge | `/tmp/r1166_merged` · merge.done **2026-08-21T10:49:14Z** |
| chall | pid**84308** TP1 util**0.85** GPU**6** :8003 · CHALL_READY+probe **10:56:01Z** |
| n80 | pid**86323** · `r1166_sim_result_reign36_wvk7.json` · SSH `23.153.44.20:40299` |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |

## Axes ≠
UltraLoLR R1145 / HiLR R1127 / MidCtx LoRank Midβ MidLR R1163 / SoftCtx LoRank Midβ UltraLoLR R1149 / Online / GRPO

## Ops note (p4300)
p4286 lean launched **TP2** util0.72 → stuck ~38 GiB cutlass init. Kill-by-pid (lean81654/wait75360/chall81785); relaunch `lean_chall_n80_r339_gpu6_tp1_p4300.sh`.
