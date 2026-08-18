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
| $UTC | 86652.1763535039 | p3877 |
| Lium balance | **$86652.1763535039** | p3877 |
| cumulative mining spend | ~$120,484 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76652** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3877 |
| miner coldkey free | τ1260.384 | kept ≥τ50 |
| miner stake | **~59α ≈ τ3.33** (r252; below ~τ5) | p3870 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T13:45:00Z | 86652.1763535039 | p3877 **R795 REFUTE→R808 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T13:37:42Z | 86687.77787836925 | p3876 **R802/R794/R784 REFUTE→R806+R807** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T13:28:57Z | 86758.32564159477 | p3875 **R784 SIZE_OK+stamp+lean** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T13:21:06Z | 86793.49405850221 | p3874 **R784 SIGSTOP+tail+meta** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T13:15:37Z | 86829.38349027254 | p3873 **stampfix** R784/R783 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T13:10:45Z | 86864.85681612037 | p3872 **R790/R789 free→R794/R795 lunar relay** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T13:05:25Z | 86932.70654936299 | p3871 **R784/R783 parallel accel** (kill tar; no rent); bal ↑ vs p3870 (shared-acct credit); burn **~$331.45/h** |
| 2026-08-18T12:55:32Z | 84602.9651961965 | p3870 **R798+R799 REFUTE→R784+R783 relay** + brave ENOSPC rematch R794/R795 (no rent); burn **~$331.45/h** |
| 2026-08-18T12:47:25Z | 84723.99904669935 | p3869 **R792+R781 REFUTE→R804+R805 TRAIN** (no rent; B300×8=0 bl B200); burn **~$331.45/h** |
| 2026-08-18T12:39:05Z | 84765.45268969834 | p3868 **R781** SIZE_OK+stamp + N80 chall loading (no rent; B300×8=0 bl B200); burn **~$331.45/h** |
