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
| $UTC | 88972.87103187863 | p3693 |
| Lium balance | **$88972.87103187863** | p3693 |
| cumulative mining spend | ~$114,302 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$78973** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T16:43:34Z | 88972.87103187863 | p3693 **R652 REFUTE** + **R653 SCP** brave→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:36:51Z | 89014.5473246602 | p3692 **R673 TRAIN** crown 6,7 + API waiter p3692 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:31:56Z | 89053.83944436345 | p3691 **R652 SCP_READY→CHALL** + API waiter p3691 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:17:35Z | 89176.17995128914 | p3690 **R633 REFUTE** + **R652 SCP** brave→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:05:31Z | 89257.27615641385 | p3689 **R670/R671/R672 TRAIN** brave 2–7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:58:49Z | 89338.21618472386 | p3688 **R651 REFUTE** + **R655 SCP** R252→lunar (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:49:45Z | 89418.86732571699 | p3687 **R633 SCP** brave→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:45:08Z | 89458.69272416751 | p3686 **R651 REPAIR** shard4 trunc (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:41:51Z | 89458.69272416751 | p3685 **R669 TRAIN** brave 0,1 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:39:37Z | 89499.30125277331 | p3684 **R647 REFUTE** + **R668 TRAIN** zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
