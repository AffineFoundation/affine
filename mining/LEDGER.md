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
| $UTC | 85090.58611741604 | p3862 |
| Lium balance | **$85090.58611741604** | p3862 |
| cumulative mining spend | ~$122,050 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$75091** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3862 |
| miner coldkey free | τ1260.384 | kept ≥τ50 |
| miner stake | **~59α ≈ τ3.33** (r252; below ~τ5) | p3862 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T11:59:30Z | 85090.58611741604 | p3862 **R780 dedupe dual-13 + 08fix + 13vis** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T11:54:21Z | 85131.92243691323 | p3861 **R780 stall-11 kill + tail accel 13–16+vis** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T11:49:32Z | 85172.9144649741 | p3860 **R791+R793 REFUTE→R803+R802 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T11:42:56Z | 85213.47208655027 | p3859 **R800+R801 TRAIN** brave idle 4–7 (no rent; B300×8=0 bl B200); burn **~$331.45/h** |
| 2026-08-18T11:35:10Z | 85294.83534427937 | p3858 **R768 REFUTE + R780 meta** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T11:26:52Z | 85335.90082259085 | p3857 **R768 SCP→N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T11:10:15Z | 85498.52382658802 | p3856 **R768 meta accel** cfg✓ (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T11:06:31Z | 85498.52382658802 | p3855 **R786/R788 REFUTE→R798+R799 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:57:04Z | 85580.77155684584 | p3854 **R768 SCP→parallel×4** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T10:52:56Z | 85620.30197593632 | p3853 **R787 REFUTE→R797 TRAIN** + R767 REFUTE (no rent; B300×8=0); burn **~$331.45/h** |
