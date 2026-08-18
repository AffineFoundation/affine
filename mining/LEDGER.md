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
| $UTC | 87340.40925428995 | p3895 |
| Lium balance | **$87340.40925428995** | p3895 |
| cumulative mining spend | ~$121,468 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77340** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.49/h** · vs floor $833/h · gap **−$466.51/h** · B300×8=0 · B200×8=bl | p3895 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~17α ≈ τ0.96** (r252; after 160α sweep) | p3885 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T15:55:49Z | 87340.40925428995 | p3895 **R337+R338 SIZE_OK+BOOT** (p3893 DONE; no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T15:51:37Z | 87385.05519315449 | p3894 **meta prestage** R337/R338 (no rent; B300×8=0; B200×8 bl); burn **~$366.49/h** |
| 2026-08-18T15:46:07Z | 87429.0587525648 | p3893 **continuous×8 fill** R337/R338 (no rent; B300×8=0; B200×8 bl); burn **~$366.49/h** |
| 2026-08-18T15:40:18Z | 87517.23900568223 | p3892 **dual accel** R337/R338 (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T15:35:34Z | 87517.23900568223 | p3891 **parallel accel** R337/R338 (no rent; B300×8=0; bl B200×8 only); burn **~$366.50/h** |
| 2026-08-18T15:29:38Z | 87606.27218338831 | p3890 **reap zesty** SSH-dead (−$64/h); burn **~$366.49/h**; waiter HEAD R339 |
| 2026-08-18T15:24:26Z | 87651.04616224977 | p3889 **R807 REFUTE→R817 TRAIN** lunar 4,5; burn **~$430.49/h** |
| 2026-08-18T15:17:57Z | 87694.75838897185 | p3888 **R806 REFUTE→R816 TRAIN** + parallel R338 relay; burn **~$430.49/h** |
| 2026-08-18T15:09:25Z | 87783.41160172987 | p3887 **R337+R338** marsplan cache relay + **R806 N80**; invent R338 ($52); burn **~$430.49/h** |
| 2026-08-18T15:03:30Z | 87867.77158460244 | p3886 **R803/R804/R805 REFUTE→R813+R814+R815 TRAIN** (no rent; B300×8=0); burn **~$378.50/h** |
