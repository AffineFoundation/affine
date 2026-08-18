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
| $UTC | 85658.12817005384 | p3921 |
| Lium balance | **$85658.12817005384** | p3921 |
| cumulative mining spend | ~$123,123 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$75658** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.49/h** · vs floor $833/h · gap **−$466.51/h** · B300×8=0 · B200×8=bl | p3921 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~47α ≈ τ2.60** (r252; under τ5 sweep) | p3917 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T19:05:42Z | 85658.12817005384 | p3921 **R823 REFUTE** reap + crown vera retarget (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T18:58:46Z | 85746.67638694175 | p3920 **R825** merge path-fix+relaunch (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T18:53:48Z | 85790.16640140212 | p3919 **R826** MERGE abort→wait_vera→n80 armed (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T18:27:53Z | 86011.87682857578 | p3918 **R818 REFUTE→R836 TRAIN** + vera king-swap lunar (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T18:18:40Z | 86099.75606291299 | p3917 **R835 TRAIN** R337 idle 6,7 fill (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T18:13:06Z | 86144.5788437599 | p3916 **R820 REFUTE→R834 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T18:07:32Z | 86189.1810329072 | p3915 **R818 merge path-fix+relaunch** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T18:03:49Z | 86229.33608798779 | p3914 **R820 n80 pathfix+relaunch** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T17:57:06Z | 86279.02973445617 | p3913 **R819 REFUTE→R833 TRAIN** + **R820 merge kick** (no rent; B300×8=0); burn **~$366.50/h** |
| 2026-08-18T17:49:12Z | 86365.49292071548 | p3912 **R821–R824** arm merge→relay→crown n80 (no rent; B300×8=0); burn **~$366.50/h** |
