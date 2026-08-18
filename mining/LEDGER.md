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
| $UTC | 86355.30645199752 | p3822 |
| Lium balance | **$86355.30645199752** | p3822 |
| cumulative mining spend | ~$119,774 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76355** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3822 |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **~59.04α ≈τ3.35** on r252 (below τ5 sweep) | p3815 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T07:16:38Z | 86355.30645199752 | p3822 **R755–R758 REFUTE→R771–R774 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T07:00:31Z | 86518.22213129186 | p3819 **R760 REFUTE→R770 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T06:52:19Z | 86558.6740067903 | p3818 **R760 N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:48:02Z | 86599.55524644416 | p3817 **R759 REFUTE→R769 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:40:53Z | 86639.43180152056 | p3816 **R759 N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:35:20Z | 86721.17406911239 | p3815 **API BL fix** fbb1135f (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:29:54Z | 86762.78364134047 | p3814 **brave idle→R767+R768 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:24:30Z | 86802.95725626276 | p3813 **R337 REAPED** SSH-dead (−$45.60/h); burn **~$331.45/h**; B300×8=0 |
| 2026-08-18T06:18:22Z | 86844.13102679935 | p3812 **R754 REFUTE→R766 TRAIN** (no rent; B300×8=0); burn **~$377.05/h** |
| 2026-08-18T06:12:42Z | 86871.91920548776 | p3811 **R752 REFUTE→R765 TRAIN** (no rent; B300×8=0); burn **~$377.05/h** |
