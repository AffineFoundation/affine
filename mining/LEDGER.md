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
| $UTC | 79309.67372872595 | p4084 |
| Lium balance | **$79309.67372872595** | p4084 |
| cumulative mining spend | ~$138,575 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$69310** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$329.78/h** · vs floor $833/h · gap **−$503.22/h** · B300×8=0 | p4084 |
| miner coldkey free | τ~1256 | kept ≥τ50 · −τ2.20 reg burn r938 (p4070) |
| miner stake | **~τ3.31** (59α r252) | p4084 · below ~τ5 sweep |
| registrations / submissions | **12** / **12** (… **r938** scoring chal-00949) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T03:36:59Z | 79309.67372872595 | p4084 R957 REFUTE→**R967 TRAIN** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T03:28:10Z | 79383.42987896815 | p4083 R957 MERGE idle→**v4 n80 LIVE** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T03:19:03Z | 79459.52424097626 | p4082 R956+R958 REFUTE→**R965+R966 TRAIN** (no rent; B300×8=0 BL `8f34559f`); burn **~$329.78/h** |
| 2026-08-20T03:10:20Z | 79537.00037931421 | p4081 R955 REFUTE→**R964 TRAIN** + R926 FORCE Triton cold-TK (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T03:02:40Z | 79572.35041310577 | p4080 R956+R958+R955 MERGE idle→**v4 n80 LIVE** (no rent; B300×8=0 BL); burn **~$329.78/h** |
| 2026-08-20T02:54:40Z | 79651.93867731644 | p4079 R949 REFUTE→**R963 TRAIN** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T02:47:03Z | 79689.98979538587 | p4078 R949 MERGE idle→**v4 n80 LIVE** on R337 (no rent; B300×8=0); burn **~$329.79/h** |
| 2026-08-20T02:39:40Z | 79766.63374227504 | p4077 R926 H100 teacher OOM→**TP=2 cold-TK** (no rent; B300×8=0); burn **~$329.79/h** |
| 2026-08-20T02:32:22Z | 79803.44474208078 | p4076 BL 8×B200 rent→1×B200 mislabel→**rm** (~$0.1 spent); R938→**R962 TRAIN**; burn **~$329.79/h** |
| 2026-08-20T02:14:48Z | 79958.7414582562 | p4075 R950 MERGE idle→**v4 n80 LIVE** on R888 (no rent; B300×8=0); burn **~$329.79/h** |
