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
| $UTC | 78184.76498907164 | p4099 |
| Lium balance | **$78184.76498907164** | p4099 |
| cumulative mining spend | ~$139,700 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68185** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 · R888 gone | p4099 |
| miner coldkey free | τ~1256 | kept ≥τ50 · −τ2.20 reg burn r938 (p4070) |
| miner stake | **~τ3.31** (59α r252) | p4099 · below ~τ5 sweep |
| registrations / submissions | **12** / **12** (… **r938** LOST chal-00949) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T06:18:42Z | 78184.76498907164 | p4099 R960+R951 REFUTE→**R974+R975 TRAIN** + R338 dual n80 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T06:11:30Z | 78255.8977189032 | p4098 R337 dual n80 + R944 REFUTE→**R973 TRAIN** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T06:02:18Z | 78288.12338375958 | p4097 R252 R960+R951 MERGE idle→**dual v4 n80** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T05:52:32Z | 78360.81475617203 | p4096 R944 teacher OOM→**TP4@0.85**→**v4 n80 LIVE** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T05:41:09Z | 78431.60659591865 | p4095 R944 teacher→**v4 n80 LIVE** (no rent; B300×8=0; R888 gone); burn **~$290.58/h** |
| 2026-08-20T05:28:38Z | 78537.09628205527 | p4094 R967 REFUTE→**R972 TRAIN** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T05:21:59Z | 78571.39708656559 | p4093 R966+R965 REFUTE→**R970+R971 TRAIN** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T05:17:35Z | 78606.60674223464 | p4092 R967 MERGE idle→**v4 n80** + R952+R953 REFUTE→**R968+R969 TRAIN** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T05:09:33Z | 78677.48756843583 | p4091 R965+R966 MERGE idle→**dual v4 n80** + R944 OOM→**0.65 n80** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T04:59:26Z | 78748.0038939759 | p4090 R926 teacher TP2@0.88→**R944 v4 n80 LIVE** (no rent; B300×8=0); burn **~$329.78/h** |
