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
| $UTC | 79958.7414582562 | p4075 |
| Lium balance | **$79958.7414582562** | p4075 |
| cumulative mining spend | ~$137,926 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$69959** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$329.79/h** · vs floor $833/h · gap **−$503.21/h** · B300×8=0 | p4075 |
| miner coldkey free | τ~1256 | kept ≥τ50 · −τ2.20 reg burn r938 (p4070) |
| miner stake | **~τ1.65** (1 position) | p4075 |
| registrations / submissions | **12** / **12** (… **r938** submitted / queued) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T02:14:48Z | 79958.7414582562 | p4075 R950 MERGE idle→**v4 n80 LIVE** on R888 (no rent; B300×8=0); burn **~$329.79/h** |
| 2026-08-20T02:07:14Z | 79997.75663291932 | p4074 R942 REFUTE→**R960 TRAIN** + R926 **cold-TK** (no rent; B300×8=0); burn **~$329.79/h** |
| 2026-08-20T01:58:52Z | 80075.0736562743 | p4073 R942 MERGE idle→**v4 n80 LIVE** on R252 (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T01:47:09Z | 80151.37183644611 | p4072 R943/R945/R946/R947 REFUTE→**R956–R959 TRAIN** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T01:40:03Z | 80229.25954109519 | p4071 R938 **SUBMITTED** (HF@`8ef1b06a` reveal 31462190) + R941/R948 REFUTE→R954/R955 TRAIN + R947 n80 (no rent; B300=0); burn **~$329.79/h** |
| 2026-08-20T01:30:43Z | 80265.29896601572 | p4070 R938 CROWN_OK→reg `8882722-0013` (−τ2.20 free) + HF push + R941/R948 n80 (no rent; B300=0); burn **~$329.79/h** |
| 2026-08-20T01:20:14Z | 80381.71756557153 | p4069 crown R943/R945/R946 MERGE→triple n80 (no rent; B300=0 BL-only); burn **~$329.79/h** |
| 2026-08-20T01:11:52Z | 80420.49329828919 | p4068 R930 REFUTE~−0.56×→reap→**R953 TRAIN** (no rent; B300=0 BL-only); burn **~$329.79/h** |
| 2026-08-20T01:03:01Z | 80494.7096402942 | p4067 R931 REFUTE~0.0002×→reap→**R952 TRAIN** + R938 **cold-TK** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T00:56:43Z | 80535.97300565851 | p4066 R930 Triton miss→FORCE seed chall_r931→**R930+R931 n80 LIVE** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
