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
| $UTC | 88215.69490882389 | p3769 |
| Lium balance | **$88215.69490882389** | p3769 |
| cumulative mining spend | ~$118,450 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$78216** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | $UTC |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **0** (r252 swept p3763) | p3763 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T01:25:30Z | 88215.69490882389 | p3769 **R719 REFUTE** + **R722–R725 N80** + **R729 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:15:45Z | 88296.85158235152 | p3768 **R712 REFUTE** + **R719/R720/R721 N80** + **R728 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:00:50Z | 88376.44652397608 | p3767 **R718 REFUTE** + **R727 TRAIN** + **R712 N80** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:52:21Z | 88459.51878187543 | p3766 **R717 REFUTE** + **R726 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:48:07Z | 88500.47766640561 | p3765 **R718 N80 LIVE** vs reign35 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:43:14Z | 88582.25965479956 | p3764 **R717 N80 LIVE** vs reign35 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:37:29Z | 88582.25965479956 | p3763 **α→TAO→Lium**: unstake r252 **295.20α→τ16.73** (ext 8868051-0005) + `btcli transfer` τ16.73→Lium ck (ext 8868054-0014; `lium fund` transfer-attr bug) · bal **+$3447** · free τ**1260.38** |
| 2026-08-18T00:32:21Z | 85217.1590028673 | p3762 **reign35** retarget (no rent; B300×8=0); burn **~$331.45/h**; α~295≈τ16.7 pending sweep |
| 2026-08-18T00:26:04Z | 85257.2117560615 | p3761 **R711 REFUTE** + **R725 TRAIN** R252 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:21:19Z | 85257.2117560615 | p3760 **R710 REFUTE** + **R711 N80** + **R724 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
