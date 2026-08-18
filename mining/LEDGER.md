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
| $UTC | 83664.4306795756 | p3946 |
| Lium balance | **$83664.4306795756** | p3946 |
| cumulative mining spend | ~$125,077 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$73664** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$366.49/h** · vs floor $833/h · gap **−$466.51/h** · B300×8=0 · B200 JSON=[] | p3946 |
| miner coldkey free | τ1260.426 | kept ≥τ50 |
| miner stake | **~47α ≈ τ2.60** (r252; under τ5 sweep) | p3917 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T22:54:10Z | 83664.4306795756 | p3946 **R838 REFUTE→R859 TRAIN** golden 4,5 (no rent; B300=0 B200 JSON=[]); burn **~$366.49/h** |
| 2026-08-18T22:46:30Z | 83708.43420900856 | p3945 **R842 EngineDead→RELOAD** :8003 (no rent; B300=0 B200=0); burn **~$366.49/h** |
| 2026-08-18T22:42:09Z | 83755.23600551371 | p3944 **R841 REFUTE→R838 RELOAD+R858 TRAIN** (no rent; B300=0 bl B200); burn **~$366.49/h** |
| 2026-08-18T22:36:07Z | 83797.3662421743 | p3943 **R842 EngineDead→RELOAD** :8003 (no rent; B300=0); burn **~$366.49/h** |
| 2026-08-18T22:31:39Z | 83840.39110268108 | p3942 **R852–R857 TRAIN** fill R337/R338 idle 2–7 (no rent; B300=0 B200=0); burn **~$366.49/h** |
| 2026-08-18T22:25:41Z | 83885.69899273105 | p3941 **R830 host-relay×4** R337→lunar + n80 claimer (no rent; B300=0 bl B200); burn **~$366.49/h** |
| 2026-08-18T22:19:51Z | 83974.57584080889 | p3940 R252 vera **parallel×4** stuck-chunk resume (no rent; B300=0 bl B200); burn **~$366.49/h** |
| 2026-08-18T22:15:54Z | 83974.57584080889 | p3939 R252 vera DIRECT resume+lium swap (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T22:07:47Z | 84063.36209805681 | p3937 **lunar→R252 vera DIRECT×6** + swap waiter (no rent; B300×8=0); burn **~$366.49/h** |
| 2026-08-18T22:02:18Z | 84105.04586341498 | p3936 **brave R848–R851 TRAIN** (no rent; B300×8=0); burn **~$366.49/h** |
