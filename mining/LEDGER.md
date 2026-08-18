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
| $UTC | 86922.14503131087 | p3832 |
| Lium balance | **$86922.14503131087** | p3832 |
| cumulative mining spend | ~$120,222 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76922** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3832 |
| miner coldkey free | τ1260.384 | kept ≥τ50 |
| miner stake | **0** | p3832 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T08:10:37Z | 86922.14503131087 | p3832 **R769 REFUTE→R780 TRAIN** brave 0,1 (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T08:04:57Z | 87001.08546687457 | p3831 **α→TAO→Lium** r252 88.56α→τ5.0231 (lium fund fail→btcli `--dest`; extr 8870291-0005 / 8870294-0008); burn **~$331.45/h** |
| 2026-08-18T08:00:59Z | 85986.16057722457 | p3830 **R761 RELAY** brave→R252 + R769 N80 LIVE (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T07:52:10Z | 86070.38051473569 | p3829 **R765 REFUTE→R779 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T07:42:05Z | 86151.45157002847 | p3828 **R764 REFUTE→R778 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T07:36:58Z | 86192.12473401024 | p3827 **R763+R766 REFUTE→R776+R777 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T07:26:52Z | 86273.9742137993 | p3823 **R771→R775 TRAIN** + R763/R766 N80 (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T07:16:38Z | 86355.30645199752 | p3822 **R755–R758 REFUTE→R771–R774 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T07:00:31Z | 86518.22213129186 | p3819 **R760 REFUTE→R770 TRAIN** (no rent; bl_skip=1); burn **~$331.45/h** |
| 2026-08-18T06:52:19Z | 86558.6740067903 | p3818 **R760 N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
