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
| $UTC | 85660.89122843964 | p3852 |
| Lium balance | **$85660.89122843964** | p3852 |
| cumulative mining spend | ~$121,480 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$75661** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3852 |
| miner coldkey free | τ1260.384 | kept ≥τ50 |
| miner stake | **~59α ≈ τ3.33** (r252; below ~τ5) | p3852 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T10:46:30Z | 85660.89122843964 | p3852 **R767 SCP→N80 LIVE** (meta accel; no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:39:18Z | 85742.41338370687 | p3851 **R785 REFUTE→R796 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:33:13Z | 85782.54632478474 | p3850 **R794+R795 TRAIN** brave idle 0–3 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:26:53Z | 85823.99575186541 | p3849 **R785 MERGE→N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:20:20Z | 85865.37038654275 | p3848 **R780+R781 MERGE→WAIT_RELAY** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:13:32Z | 85946.06458805887 | p3847 **R762 REFUTE→R767 parallel×4** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:05:05Z | 86024.67792603628 | p3846 **R782 REFUTE→R793** + **R762 N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T09:56:33Z | 86068.05958493476 | p3845 **R779/R773/R774 REFUTE→R790/R791/R792** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T09:48:14Z | 86150.17300654396 | p3844 **R773/R774 port-split N80** + **R778→R789** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T09:35:13Z | 86270.85847279444 | p3843 **R761 REFUTE→R762 parallel relay** (no rent; bl B200 only); burn **~$331.45/h** |
