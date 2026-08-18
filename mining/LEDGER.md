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
| $UTC | 86556.26578071532 | p3839 |
| Lium balance | **$86556.26578071532** | p3839 |
| cumulative mining spend | ~$120,588 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76556** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3839 |
| miner coldkey free | τ1260.384 | kept ≥τ50 |
| miner stake | **29.52α ≈ τ1.68** (r252; below ~τ5) | p3839 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T08:56:14Z | 86556.26578071532 | p3839 **R772 REFUTE→R785 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:48:30Z | 86627.13950871985 | p3838 **R761 parallel×4 resume** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:42:00Z | 86678.35728693128 | p3837 **R768 ARMED** + **R784 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:36:20Z | 86720.33945337946 | p3836 **R767 ARMED** + **R783 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T08:31:01Z | 86759.05271647852 | p3835 **R762 hardened** resume (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T08:24:51Z | 86841.09060717696 | p3834 **R762 ARMED** resume-after-R761 (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T08:20:32Z | 86841.09060717696 | p3833 **R770 REFUTE→R782** + **R781 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T08:10:37Z | 86922.14503131087 | p3832 **R769 REFUTE→R780 TRAIN** brave 0,1 (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T08:04:57Z | 87001.08546687457 | p3831 **α→TAO→Lium** r252 88.56α→τ5.0231 (lium fund fail→btcli `--dest`); burn **~$331.45/h** |
| 2026-08-18T08:00:59Z | 85986.16057722457 | p3830 **R761 RELAY** brave→R252 + R769 N80 LIVE (no rent; bl_skip=1); burn **~$331.45/h** |
