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
| $UTC | 85111.62493313767 | p3966 |
| Lium balance | **$85111.62493313767** | p3966 |
| cumulative mining spend | ~$126,626 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$75112** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.49/h** · vs floor $833/h · gap **−$466.51/h** · B300×8=0 · B200×8=`fbb1135f` bl · waiters armed | p3966 |
| miner coldkey free | τ1260.446 | kept ≥τ50 |
| miner stake | **0** (p3966 swept r252 284α) | p3966 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T02:02:30Z | 85111.62493313767 | p3966 α→TAO→Lium: unstake 284α r252 (ext 8875682-0010) + transfer τ**15.6288** → Lium key (ext 8875686-0013; `lium fund` broken); +~$3132 |
| 2026-08-19T01:55:00Z | 82070.67046012853 | p3965 R849 corpus-race→n80 relaunch (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-19T01:51:00Z | 82070.67046012853 | p3964 brave R848/R849 TP1 READY + n80 LIVE (no rent; stock 0); burn **~$366.49/h** |
| 2026-08-19T01:45:17Z | 82160.03102885015 | p3963 brave king TP1 READY + R848/R849 n80 armed (no rent; stock 0); burn **~$366.50/h** |
| 2026-08-19T01:05:24Z | 82511.06348158202 | p3961 R848/R849 MERGE_DONE + cold TK LOAD (no rent; stock 0); burn **~$366.50/h** |
| 2026-08-19T00:53:28Z | 82602.48348426091 | p3960 brave NCCL→reboot→R848/R849 remerge (no rent; stock 0); burn **~$366.49/h** |
| 2026-08-19T00:36:25Z | 82777.67182108681 | p3959 vera SIZE_OK→brave cold TK (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-19T00:30:16Z | 82823.50278112969 | p3958 brave vera lunar-APPEND×4 (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-19T00:23:58Z | 82867.33586451829 | p3957 R337 idle→R869–R871 TRAIN (no rent; stock 0); burn **~$366.49/h** |
| 2026-08-19T00:18:30Z | 82956.05656220863 | p3956 brave cold TK+R848/R849 n80 (no rent; stock 0); burn **~$366.49/h** |
