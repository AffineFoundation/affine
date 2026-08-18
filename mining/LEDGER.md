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
| $UTC | 86810.93653610324 | p3905 |
| Lium balance | **$86810.93653610324** | p3905 |
| cumulative mining spend | ~$121,998 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76811** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.49/h** · vs floor $833/h · gap **−$466.51/h** · B300×8=0 · B200×8=bl | p3905 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~47α ≈ τ2.60** (r252; under τ5 sweep) | p3905 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T17:00:56Z | 86810.93653610324 | p3905 **R814 REFUTE→R825 TRAIN** + **R810 wave2 ACCEL** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:53:58Z | 86853.04195928285 | p3904 **R809 REFUTE** + **R810 ACCEL** + brave **R821–R824 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:46:11Z | 86897.72577170336 | p3903 **R815 REFUTE→R820 TRAIN** + R809 chall + R810 defer RELAY (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:39:54Z | 86986.12678699766 | p3902 **R816 REFUTE→R819 TRAIN** + R815 n80 LIVE (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:34:45Z | 87028.79109626895 | p3901 **R811+R812 REFUTE** free crown + **R809** ×6 accel (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:28:51Z | 87074.8193336013 | p3900 **R809 host-relay LIVE** + crown wait-slot (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:21:56Z | 87163.28237809529 | p3899 **R811+R812 dual n80 LIVE** crown (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:15:26Z | 87208.00088174417 | p3898 **R808 REFUTE→R818 TRAIN**; R337+R338 TRAIN (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:09:06Z | 87252.3682908381 | p3897 **teacher skip+R337 BOOT**; R808 n80 ~26/80 (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:01:00Z | 87292.315750923 | p3896 **R808 RELOAD** + **teacher host-relay** R337/R338 (no rent; B300×8=0); burn **~$366.49/h** |
