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
| $UTC | 87399.91547054957 | p3782 |
| Lium balance | **$87399.91547054957** | p3782 |
| cumulative mining spend | ~$119,180 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77484** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | $UTC |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **~60.1α / τ3.41** r252 (below ~τ5 sweep) | p3781 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T03:01:00Z | 87399.91547054957 | p3782 **R737+R729 N80 ARMING** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:54:57Z | 87483.78874678482 | p3781 **R734+R735+R731 N80 LIVE** (no rent; B300×8=0); α~τ3.41 skip sweep; burn **~$331.45/h** |
| 2026-08-18T02:50:20Z | 87483.78874678482 | p3780 **R732 REFUTE** + **R740 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:43:06Z | 87564.47293212486 | p3779 **R733 REFUTE** + **R739 TRAIN** + **R732 N80** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:35:57Z | 87605.39797360613 | p3778 **R728 REFUTE** + **R738 TRAIN** + **R733 N80** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:26:50Z | 87686.41647949364 | p3777 **R728 N80 LIVE** lunar 6,7 + R736 wait reaped (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:20:06Z | 87768.97715059611 | p3776 **brave NCCL abort** + **R715 relay→crown** (freed R736; no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:09:00Z | 87849.86205615434 | p3775 **R727 REFUTE** + **R737 TRAIN** crown 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:01:52Z | 87887.57828641804 | p3774 **R726 REFUTE** + **R736 TRAIN** crown 4,5 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:56:59Z | 87931.00583067711 | p3773 **brave cold TK+R715/R716** armed (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:50:09Z | 88012.08748004797 | p3772 **R726+R727 MERGE→CHALL→N80 LIVE** crown 4–7 (no rent; B300×8=0); burn **~$331.45/h** |
