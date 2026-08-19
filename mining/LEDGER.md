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
| $UTC | 82956.05656220863 | p3956 |
| Lium balance | **$82956.05656220863** | p3956 |
| cumulative mining spend | ~$125,741 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$72956** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.49/h** · vs floor $833/h · gap **−$466.51/h** · B300×8=0 · B200×8=0 · waiters armed | p3956 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~47α ≈ τ2.60** (r252; under τ5 sweep) | p3917 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T00:18:30Z | 82956.05656220863 | p3956 brave cold TK+R848/R849 n80 (no rent; stock 0); burn **~$366.49/h** |
| 2026-08-19T00:12:39Z | 82956.05656220863 | p3955 R843/R844/R845 REFUTE→R866/R867/R868 TRAIN (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-19T00:03:18Z | 83040.86065910391 | p3954 R854 RELAY + R863–R865 TRAIN (no rent; stock 0); burn **~$366.49/h** |
| 2026-08-18T23:56:33Z | 83088.8705157734 | p3953 R843 triton-seed RELOAD + R252 vera SWAP_OK (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-18T23:49:27Z | 83178.07519292047 | p3952 R252 named-shard HUB_OK + vera SWAP LOADING (no rent; stock 0); burn **~$366.49/h** |
| 2026-08-18T23:42:46Z | 83222.2182449106 | p3951 vera SIZE_OK+SWAP; R846/R847 REFUTE→R861/R862 TRAIN (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-18T23:35:32Z | 83266.35464915654 | p3951 vera part0/1 resume mid-pass (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-18T23:23:30Z | 83399.21902244067 | p3950 **R830 REFUTE→R860 TRAIN** + vera part1 resume (no rent; B200 bl); burn **~$366.49/h** |
| 2026-08-18T23:12:10Z | 83487.08773814619 | p3949 **R830 EngineDead→RELOAD** :8003 (no rent; B200×8 listed); burn **~$366.49/h** |
| 2026-08-18T23:08:14Z | 83531.53502103833 | p3948 R252 vera **part1 resume** + SIZE_OK concat (no rent; B300=0 B200=0); burn **~$366.49/h** |
