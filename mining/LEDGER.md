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
| $UTC | 88582.25965479956 | p3763 |
| Lium balance | **$88582.25965479956** | p3763 |
| cumulative mining spend | ~$118,084 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$78582** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **0** (r252 swept p3763) | p3763 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T00:37:29Z | 88582.25965479956 | p3763 **α→TAO→Lium**: unstake r252 **295.20α→τ16.73** (ext 8868051-0005) + `btcli transfer` τ16.73→Lium ck (ext 8868054-0014; `lium fund` transfer-attr bug) · bal **+$3447** · free τ**1260.38** |
| 2026-08-18T00:32:21Z | 85217.1590028673 | p3762 **reign35** retarget (no rent; B300×8=0); burn **~$331.45/h**; α~295≈τ16.7 pending sweep |
| 2026-08-18T00:26:04Z | 85257.2117560615 | p3761 **R711 REFUTE** + **R725 TRAIN** R252 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:21:19Z | 85257.2117560615 | p3760 **R710 REFUTE** + **R711 N80** + **R724 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:06:23Z | 85378.97363748537 | p3758 **R708/R709 REFUTE** + **R722+R723 TRAIN** zesty 4–7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T23:55:44Z | 85461.04332011218 | p3757 **R721 TRAIN** lunar 6,7 after r537 reap (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T23:48:24Z | 85541.97685530674 | p3756 **R709+R708 N80 LIVE** zesty 4–7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T23:41:57Z | 85582.36221824915 | p3755 **R719+R720 TRAIN** golden 4–7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T23:34:37Z | 85663.33179500113 | p3754 **R696 REFUTE** + **R718 TRAIN** crown 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T23:27:07Z | 85745.25562401353 | p3753 **R693 REFUTE** + **R717 TRAIN** crown 4,5 (no rent; B300×8=0); burn **~$331.45/h** |
