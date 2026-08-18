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
| $UTC | 86279.02973445617 | p3913 |
| Lium balance | **$86279.02973445617** | p3913 |
| cumulative mining spend | ~$122,500 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76279** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.50/h** · vs floor $833/h · gap **−$466.50/h** · B300×8=0 · B200×8=bl | p3913 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~47α ≈ τ2.60** (r252; under τ5 sweep) | p3913 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T17:57:06Z | 86279.02973445617 | p3913 **R819 REFUTE→R833 TRAIN** + **R820 merge kick** (no rent; B300×8=0); burn **~$366.50/h** |
| 2026-08-18T17:49:12Z | 86365.49292071548 | p3912 **R821–R824** arm merge→relay→crown n80 (no rent; B300×8=0); burn **~$366.50/h** |
| 2026-08-18T17:43:57Z | 86409.67848032674 | p3911 **R337+R338** kill local post_train → merge-only + host relay→lunar; burn **~$366.50/h** |
| 2026-08-18T17:36:15Z | 86454.34685036512 | p3910 **TTL+24h** R337+R338 →17:35:23Z + wait→merge R796/R830/R831/R832; burn **~$366.50/h** |
| 2026-08-18T17:30:42Z | 86498.0179769548 | p3909 **idle GPU fill** R337 R796+R830 + R338 R831+R832 (no rent; B300×8=0); burn **~$366.50/h** |
| 2026-08-18T17:23:24Z | 86587.65566702014 | p3908 **TTL+24h ×5** (crown/lunar/golden/brave/R252; was 19:04Z); burn **~$366.50/h** |
| 2026-08-18T17:17:54Z | 86631.68526722309 | p3907 **R810+R813 REFUTE→R829+R827+R828 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T17:07:28Z | 86720.48934197395 | p3906 **R817 REFUTE→R826 TRAIN** + **R810 n80 LIVE** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T17:00:56Z | 86810.93653610324 | p3905 **R814 REFUTE→R825 TRAIN** + **R810 wave2 ACCEL** (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T16:53:58Z | 86853.04195928285 | p3904 **R809 REFUTE** + **R810 ACCEL** + brave **R821–R824 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
