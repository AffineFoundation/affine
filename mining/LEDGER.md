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
| $UTC | 87651.04616224977 | p3889 |
| Lium balance | **$87651.04616224977** | p3889 |
| cumulative mining spend | ~$121,157 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77651** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$430.49/h** · vs floor $833/h · gap **−$402.51/h** · B300×8=0 | p3889 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~17α ≈ τ0.96** (r252; after 160α sweep) | p3885 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T15:24:26Z | 87651.04616224977 | p3889 **R807 REFUTE→R817 TRAIN** lunar 4,5; burn **~$430.49/h** |
| 2026-08-18T15:17:57Z | 87694.75838897185 | p3888 **R806 REFUTE→R816 TRAIN** + parallel R338 relay; burn **~$430.49/h** |
| 2026-08-18T15:09:25Z | 87783.41160172987 | p3887 **R337+R338** marsplan cache relay + **R806 N80**; invent R338 ($52); burn **~$430.49/h** |
| 2026-08-18T15:03:30Z | 87867.77158460244 | p3886 **R803/R804/R805 REFUTE→R813+R814+R815 TRAIN** (no rent; B300×8=0); burn **~$378.50/h** |
| 2026-08-18T14:54:30Z | 87867.77158460244 | p3885 **α→TAO→Lium**: unstake **160α** r252 → +τ~8.94 free; `lium fund` Metadata bug → `btcli transfer` **τ8.9** to Lium ck; bal **+$1791**; burn **~$378.50/h** |
| 2026-08-18T14:44:06Z | 86196.43516098657 | p3884 **R801 SIZE_OK→lean :8003** (no rent; B300×8=0); burn **~$378.50/h** |
| 2026-08-18T14:34:39Z | 86274.96614972026 | p3883 **R800 REFUTE→R809+R810 TRAIN** (no rent; B300×8=0); burn **~$378.50/h** |
| 2026-08-18T14:26:40Z | 86316.3625179578 | p3882 **R800 SIZE_OK→n80** + **R801 SIGSTOP+tail×6**; burn **~$378.50/h** |
| 2026-08-18T14:10:12Z | 86475.77108451827 | p3881 **R800 SIGSTOP+tail×6** + rediscover **R337** B300 ($47.04); burn **~$378.50/h** |
| 2026-08-18T14:04:46Z | 86509.29770981436 | p3880 **R783 REFUTE→R801 armed** (no rent; B300×8=0); burn **~$331.45/h** |
