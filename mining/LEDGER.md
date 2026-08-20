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
| $UTC | 79733.2927284628 | p4103 |
| Lium balance | **$79733.2927284628** | p4103 |
| cumulative mining spend | ~$139,841 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$69733** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 · R888 gone | p4103 |
| miner coldkey free | τ~1254 | kept ≥τ50 · −τ8 fund Lium (p4103) |
| miner stake | **τ0** | p4103 · swept r252 α147.6 |
| registrations / submissions | **13** / **13** (… **r959** **chal-00957** queued) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T06:45:14Z | 79733.2927284628 | p4103 α147.6/~τ8.27→**τ8**→Lium (`lium fund` Subtensor.transfer fail → `btcli`→`5FqACMt…zsThe`); +~$1724; burn **~$290.58/h** |
| 2026-08-20T06:40:33Z | 78044.53793871804 | p4102 R962 MERGE idle→**v4 n80 LIVE** R938 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T06:33:15Z | 78115.43403423327 | p4101 R959 **SUBMITTED** + R964→**R978 TRAIN** (no rent; B300×8=0); reg −τ2.61; burn **~$290.58/h** |
| 2026-08-20T06:24:30Z | 78149.85908530184 | p4100 R963+R954 REFUTE→**R976+R977 TRAIN** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T06:18:42Z | 78184.76498907164 | p4099 R960+R951 REFUTE→**R974+R975 TRAIN** + R338 dual n80 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T06:11:30Z | 78255.8977189032 | p4098 R337 dual n80 + R944 REFUTE→**R973 TRAIN** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T06:02:18Z | 78288.12338375958 | p4097 R252 R960+R951 MERGE idle→**dual v4 n80** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T05:52:32Z | 78360.81475617203 | p4096 R944 teacher OOM→**TP4@0.85**→**v4 n80 LIVE** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T05:41:09Z | 78431.60659591865 | p4095 R944 teacher→**v4 n80 LIVE** (no rent; B300×8=0; R888 gone); burn **~$290.58/h** |
| 2026-08-20T05:28:38Z | 78537.09628205527 | p4094 R967 REFUTE→**R972 TRAIN** (no rent; B300×8=0); burn **~$329.78/h** |
