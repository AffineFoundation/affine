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
| $UTC | 84151.55329158228 | p3935 |
| Lium balance | **$84151.55329158228** | p3935 |
| cumulative mining spend | ~$124,631 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$74152** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.49/h** · vs floor $833/h · gap **−$466.51/h** · B300×8=0 · B200×8=bl | p3935 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~47α ≈ τ2.60** (r252; under τ5 sweep) | p3917 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T21:56:32Z | 84151.55329158228 | p3935 **R839/R840 REFUTE→R846+R847 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T21:48:33Z | 84240.23376161132 | p3934 **R833/R834 REFUTE→R844+R845 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T21:41:24Z | 84284.569116522 | p3933 **R837 REFUTE→R843 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T21:36:42Z | 84330.24554851363 | p3932 crown ENOSPC→reap merges→**R839+R840 n80 LOAD** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T21:30:46Z | 84371.87754707636 | p3931 golden **vera SWAP_OK** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T20:17:40Z | 85038.21538459286 | p3928 **R836 REFUTE→R842 TRAIN** + golden vera DIRECT (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T20:06:15Z | 85126.43956326392 | p3927 **R828 REFUTE→R841 TRAIN** + R836 n80 LIVE (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T19:56:07Z | 85214.70160046287 | p3926 **R829+R827 REFUTE→R838/R839/R840** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T19:32:58Z | 85437.26310785863 | p3925 **R824 REFUTE→R829 n80** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T19:27:18Z | 85480.71660609606 | p3924 **R826 REFUTE→R837** + vera DIRECT×4 (no rent; B300×8=0); burn **~$366.49/h** |
