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
| $UTC | 90388.50267150775 | p3661 |
| Lium balance | **$90388.50** | p3661 |
| cumulative mining spend | ~$112,928 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$80389** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T13:47:08Z | 90388.50267150775 | p3661 **R646 REFUTE** ~−0.49×; **R647 ARMED** (gated R634); R633 re-gate; burn **~$331.45/h** · B300 empty |
| 2026-08-17T13:39:26Z | 90469.60579157784 | p3660 **R641 REFUTE** ~0.31×; **R634 SCP** (bypass R631 DEFER); R646 n80; burn **~$331.45/h** · B300 empty |
| 2026-08-17T13:35:02Z | 90508.97689182592 | p3659 **R641 n80 LIVE** (SCP+Triton fix); R646 CHALL; burn **~$331.45/h** · B300 empty |
| 2026-08-17T13:20:00Z | 90632.78429121959 | p3658 **R648 REFUTE** ~0.68×; **R646 ARMED**; R631 DEFER; burn **~$331.45/h** · B300 empty · stake ~29.5α |
| 2026-08-17T13:11:14Z | 90673.45085823236 | p3657 **R648 CHALL+n80** golden 4,5/:8003 (lean pipefail fix); burn **~$331.45/h** · B300 stock empty · stake none |
| 2026-08-17T13:02:09Z | 90753.10862355558 | p3656 **R652 TRAIN** Soft HiRank LoBeta ep3×LoLR brave 2,3 (parallel R651 0,1); burn **~$331.45/h** · B300 stock empty |
| 2026-08-17T12:55:53Z | 90796.3147131677 | p3655 **R631 STALL→DEFER** (kill flat@12G; wait R648; host pid428475); burn **~$331.45/h** |
| 2026-08-17T12:55Z | **90836.83** | **FUND** SN120 α on r252 → TAO τ**31.545** → Lium (`btcli transfer` to funding coldkey; `lium fund` CLI MetadataVersioned fail). Unstake extrinsic **8864533-0009**; transfer **8864540-0013**. ΔLium **+$6476** |
| 2026-08-17T12:51:23Z | 84360.76560890493 | p3654 **R645 REFUTE** m=−0.001633~−0.29×; **R648 ARMED**; R651~step130; burn **~$331.45/h** |
| 2026-08-17T12:45:34Z | 84442.04132890662 | p3653 **R645 n80 LIVE**; R651~step95; burn **~$331.45/h** |
