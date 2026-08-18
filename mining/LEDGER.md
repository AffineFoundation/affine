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
| $UTC | 86864.85681612037 | p3872 |
| Lium balance | **$86864.85681612037** | p3872 |
| cumulative mining spend | ~$120,272 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76865** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3872 |
| miner coldkey free | τ1260.384 | kept ≥τ50 |
| miner stake | **~59α ≈ τ3.33** (r252; below ~τ5) | p3870 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T13:10:45Z | 86864.85681612037 | p3872 **R790/R789 free→R794/R795 lunar relay** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T13:05:25Z | 86932.70654936299 | p3871 **R784/R783 parallel accel** (kill tar; no rent; B300×8=0); bal ↑ vs p3870 (shared-acct credit); burn **~$331.45/h** |
| 2026-08-18T12:55:32Z | 84602.9651961965 | p3870 **R798+R799 REFUTE→R784+R783 relay** + brave ENOSPC rematch R794/R795 (no rent); burn **~$331.45/h** |
| 2026-08-18T12:47:25Z | 84723.99904669935 | p3869 **R792+R781 REFUTE→R804+R805 TRAIN** (no rent; B300×8=0 bl B200); burn **~$331.45/h** |
| 2026-08-18T12:39:05Z | 84765.45268969834 | p3868 **R781** SIZE_OK+stamp + N80 chall loading (no rent; B300×8=0 bl B200); burn **~$331.45/h** |
| 2026-08-18T12:34:55Z | 84804.69856410786 | p3867 **R781** finish size-verify armed (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T12:28:32Z | 84847.00665073033 | p3866 **R781** mid accel 11–12 + STOP parent (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T12:24:03Z | 84886.91239544186 | p3865 **R781** tail+meta accel (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T12:19:10Z | 84928.00730343281 | p3864 **R780 REFUTE** ~0.17× + free :8002 + R781 SCP (no rent; B300×8=0 bl B200); burn **~$331.45/h** |
| 2026-08-18T12:14:41Z | 84967.65197827487 | p3863 **R780 size-verify→lean relaunch N80 LIVE** + R781 relay (no rent; B300×8=0); burn **~$331.45/h** |
