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
| $UTC | 84176.63823396455 | p3980 |
| Lium balance | **$84176.63823396455** | p3980 |
| cumulative mining spend | ~$127,569 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$74177** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$405.70/h** · vs floor $833/h · gap **−$427.30/h** · B300/8×B200 stock=0 | p3980 |
| miner coldkey free | τ1257.618 | kept ≥τ50 |
| miner stake | **0** | p3980 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T03:48:49Z | 84176.63823396455 | p3980 R888 hub-cache fix + teacher HF LIVE (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T03:42:58Z | 84223.41138091912 | p3979 R888 BOOT unstuck (`hf download` LIVE); no rent; burn **~$405.70/h** |
| 2026-08-19T03:37:09Z | 84272.94766383574 | p3978 rent **mine-r888** gentle-orbit-0d 8×B200 **$39.20/h** + BOOT; burn **~$405.70/h** |
| 2026-08-19T03:30:31Z | 84361.58418370338 | p3977 R863 REFUTE→R864 relay + brave R887 TRAIN (no rent; stock 0); burn **~$366.50/h** |
| 2026-08-19T03:22:16Z | 84404.79358358723 | p3976 crown idle→R885/R886 TRAIN + R863 n80 LIVE (no rent; stock 0); burn **~$366.50/h** |
| 2026-08-19T03:14:19Z | 84494.05190643293 | p3975 R337/R338 idle→R879–R884 TRAIN (no rent; stock 0); burn **~$366.50/h** |
| 2026-08-19T03:03:50Z | 84578.4925971328 | p3974 golden R858/R859→R877/R878 TRAIN (no rent; B200 bl); burn **~$366.50/h** |
| 2026-08-19T02:56:40Z | 84627.75041430761 | p3973 R866 REFUTE→R863 host-relay (no rent; stock 0); burn **~$366.50/h** |
| 2026-08-19T02:50:36Z | 84714.56148858926 | p3972 R338 R863–R865 MERGE relaunch (no rent; stock 0); burn **~$366.50/h** |
| 2026-08-19T02:44:45Z | 84759.77225874645 | p3971 R850/R851/R860 REFUTE→next TRAIN; R861 LOST; R866 reload (no rent; stock 0); burn **~$366.50/h** |
