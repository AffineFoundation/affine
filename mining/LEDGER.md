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
| $UTC | 78641.06598231226 | p4035 |
| Lium balance | **$78641.06598231226** | p4035 |
| cumulative mining spend | ~$135,321 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68641** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$313.82/h** · vs floor $833/h · gap **−$519.18/h** · stock BL-only | p4035 |
| miner coldkey free | τ1257.620 | kept ≥τ50 |
| miner stake | **236α / τ13.03** (r252 — α→τ→Lium next) | p4035 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T20:10:15Z | 78641.06598231226 | p4035 R913 REFUTE→R929 TRAIN + R923 MERGE fix (no rent; BL `fbb1135f`); burn **~$313.82/h** |
| 2026-08-19T20:02:56Z | 78709.90182762625 | p4034 R912 REFUTE→R928 TRAIN + R913 n80 (no rent; stock 0); burn **~$313.82/h** |
| 2026-08-19T19:47:22Z | 78819.91686205856 | p4033 crown TK READY + R912/R913 n80 chall launch (no rent; only BL 8×B200); burn **~$313.82/h** |
| 2026-08-19T19:38:37Z | 78892.41047967844 | p4032 crown TK DL + R912/R913 MERGE→n80 arm + R926/R927 TRAIN (no rent; only BL 8×B200); burn **~$313.82/h** |
| 2026-08-19T19:27:56Z | 78965.05059548139 | p4031 R927 ARMED on R926 GPUs2,3 MidCtx MidLoβ (no rent; only BL 8×B200); burn **~$313.82/h** |
| 2026-08-19T19:22:56Z | 79000.0507029855 | p4030 rent `mine-r926` 8×H100 $13.76/h SoftCtx MidLoβ + R925 idle-GPU fill; burn **~$313.82/h** |
| 2026-08-19T18:43:46Z | 79280.45844954444 | p4028 rent `mine-r924` 8×H200 $33.81/h MidCtx Hiβ (no B300/non-bl B200); burn **~$300.06/h** |
| 2026-08-19T18:37:04Z | 79312.69880907168 | p4027 R914 REFUTE→R923 + R337/R338 vera-pivot (no rent; bl-only); burn **~$266.26/h** |
| 2026-08-19T18:26:56Z | 79377.14684740201 | p4026 crown cold R912+R913 (no rent; bl-only stock); burn **~$266.26/h** |
| 2026-08-19T18:21:36Z | 79409.0809999166 | p4025 R914 n80 relaunch + R252/R337/R338 bootstrap (no rent; stock 0); burn **~$266.26/h** |
