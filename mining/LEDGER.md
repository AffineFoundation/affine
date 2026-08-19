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
| $UTC | 79000.0507029855 | p4030 |
| Lium balance | **$79000.0507029855** | p4030 |
| cumulative mining spend | ~$134,962 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$69000** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$313.82/h** · vs floor $833/h · gap **−$519.18/h** · +H100 R926 | p4030 |
| miner coldkey free | τ1257.620 | kept ≥τ50 |
| miner stake | **0α / τ0** (p4019 swept r252) | p4030 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T19:22:56Z | 79000.0507029855 | p4030 rent `mine-r926` 8×H100 $13.76/h SoftCtx MidLoβ + R925 idle-GPU fill; burn **~$313.82/h** |
| 2026-08-19T18:43:46Z | 79280.45844954444 | p4028 rent `mine-r924` 8×H200 $33.81/h MidCtx Hiβ (no B300/non-bl B200); burn **~$300.06/h** |
| 2026-08-19T18:37:04Z | 79312.69880907168 | p4027 R914 REFUTE→R923 + R337/R338 vera-pivot (no rent; bl-only); burn **~$266.26/h** |
| 2026-08-19T18:26:56Z | 79377.14684740201 | p4026 crown cold R912+R913 (no rent; bl-only stock); burn **~$266.26/h** |
| 2026-08-19T18:21:36Z | 79409.0809999166 | p4025 R914 n80 relaunch + R252/R337/R338 bootstrap (no rent; stock 0); burn **~$266.26/h** |
| 2026-08-19T18:17:23Z | 79441.59615116383 | p4024 fleet rebuild: +crown B300 $64 +R337 $46.8 +R338 $52.25 +R252 $64; tore bl R339 (~$50.56); burn **~$266.26/h** |
| 2026-08-19T10:36:28Z | 82619.80256348639 | p4022 R903–7 REFUTE→R918–21 TRAIN (no rent; stock 0); burn **~$306.66/h** |
| 2026-08-19T10:22:36Z | 82782.29166695496 | p4021 golden R903+R904 MERGE idle→n80; R910 REFUTE→R917 TRAIN (no rent; stock 0); burn **~$306.66/h** |
| 2026-08-19T10:10:38Z | 82822.43124536946 | p4020 brave TP2 hang→TP1; TK restore; R905–7 n80 LIVE (no rent; stock 0); burn **~$306.66/h** |
| 2026-08-19T09:41:12Z | 83063.67459711197 | p4019 α→τ: unstake r252 all (~206α→τ11.4) + `btcli transfer` τ11.4→Lium ck (lium fund transfer-attr fail); R905–7 n80 arm; burn **~$306.66/h** |
