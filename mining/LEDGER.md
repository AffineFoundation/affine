# LEDGER — money in / money out

**Cap: 40 lines.** Totals + last 10 movements. Older → archive/.
Floor ≥ **$10,000**. Mining B300 burn floor **>$20,000/day = $833/h**.
Live burn = Σ $/h over `mine-*` pods every pass.

**Funding (operator 2026-08-16):** free to convert **SN120 Alpha → TAO → Lium**
(`lium fund -w miner`) to scale mining. Keep ≥ τ50 free coldkey for regs.
**SN120 α cannot pay Lium directly** — CLI only takes TAO or **SN51** α
(`lium fund --alpha …`). Optional SN51 probe OK; log every Alpha/TAO/Lium move.

## Totals

| item | value | as of |
|---|---|---|
| $UTC | 80724.50194339531 | p4045 |
| Lium balance | **$80724.50194339531** | p4045 |
| cumulative mining spend | ~$135,792 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$70724** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$329.79/h** · vs floor $833/h · gap **−$503.21/h** · B300=0 · +H200 R938 | p4045 |
| miner coldkey free | τ1258.144 | kept ≥τ50 |
| miner stake | r252 **~29.5α/~τ1.63** (below τ5 sweep) | p4041 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T21:27:05Z | 80724.50194339531 | p4045 rent **mine-r938** 8×H200 `$15.96` SoftCtx Hiβ + R927 MERGE fix (burn **~$329.79/h**) |
| 2026-08-19T21:16:47Z | 80797.1264194002 | p4044 R924 MERGE→**host-relay→crown n80** (no rent; B300=0 BL B200; H200×8 `$31.92` noted); burn **~$313.82/h** |
| 2026-08-19T21:10:19Z | 80868.13321982036 | p4043 R337 vera MERGE→n80 + R924 merge fix (no rent; BL-only); burn **~$313.82/h** |
| 2026-08-19T21:03:07Z | 80902.65320280581 | p4042 R338 REFUTE→reap chall→**R937 TRAIN** (no rent; BL `8f34559f`); burn **~$313.82/h** |
| 2026-08-19T20:55:52Z | 80941.58836310239 | p4041 R338 n80 relaunch (king local-id) + crown idle GPUs1,3 → **R936 TRAIN** (no rent; BL-only); burn **~$313.82/h** |
| 2026-08-19T20:49:30Z | 81049.50624060768 | p4040 R338 idle GPUs6–7 → **R935 TRAIN** (no rent; BL `8f34559f`); burn **~$313.82/h** |
| 2026-08-19T20:43:28Z | 81049.50624060768 | p4039 R338 TRAIN_DONE→vera merge→n80 (no rent; BL `8f34559f`+`fbb1135f`); burn **~$313.82/h** |
| 2026-08-19T20:34:56Z | 81120.16587304366 | p4038 R926 idle GPUs4–7 → R933+R934 TRAIN (no rent; BL `8f34559f`); burn **~$313.82/h** |
| 2026-08-19T20:29:45Z | 81157.75142185499 | p4037 R923 REFUTE→R932 + R930/R931 idle fill (no rent; BL `8f34559f`); burn **~$313.82/h** |
| 2026-08-19T20:20:20Z | 81193.74944032084 | p4036 α→τ→Lium: unstake r252 all (236α/~τ13) + `btcli transfer` τ12.5 → Lium ck (lium fund broken); burn **~$313.82/h** |
