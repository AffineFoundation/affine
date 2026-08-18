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
| $UTC | 86270.85847279444 | p3843 |
| Lium balance | **$86270.85847279444** | p3843 |
| cumulative mining spend | ~$120,871 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76271** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3843 |
| miner coldkey free | τ1260.384 | kept ≥τ50 |
| miner stake | **~29.5α ≈ τ1.68** (r252; below ~τ5) | p3841 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T09:35:13Z | 86270.85847279444 | p3843 **R761 REFUTE→R762 parallel relay** (no rent; bl B200 only); burn **~$331.45/h** |
| 2026-08-18T09:25:48Z | 86312.25161547569 | p3842 **R761 visual graft→N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T09:18:25Z | 86393.6582669515 | p3841 **R777+R776 REFUTE→R787+R788**; **R761 N80** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T09:06:53Z | 86475.17678012644 | p3840 **R775 REFUTE→R786 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:56:14Z | 86556.26578071532 | p3839 **R772 REFUTE→R785 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:48:30Z | 86627.13950871985 | p3838 **R761 parallel×4 resume** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:42:00Z | 86678.35728693128 | p3837 **R768 ARMED** + **R784 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:36:20Z | 86720.33945337946 | p3836 **R767 ARMED** + **R783 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:31:01Z | 86759.05271647852 | p3835 **R762 hardened** resume (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T08:24:51Z | 86841.09060717696 | p3834 **R762 ARMED** resume-after-R761 (no rent; bl_skip=1); burn **~$331.45/h** |
