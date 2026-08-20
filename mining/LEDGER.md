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
| $UTC | 78070.45781489636 | p4207 |
| Lium balance | **$78070.45781489636** | p4207 |
| cumulative mining spend | ~$145,092 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68070** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$354.58/h** · vs floor $833/h · gap **−$478.42/h** · B300×8=0 | p4207 |
| miner coldkey free | τ~1249 | kept ≥τ50 |
| miner stake | **0** (swept p4207) | p4207 |
| registrations / submissions | **15** / **15** (… **r1032 LOST chal-00967**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T21:07:30Z | 78103.54768824481 | p4207 **α→TAO→Lium**: unstake r252 **59.04α** (ext **8888601-0013**) → free +τ3.364; `lium fund` bug → `btcli transfer` τ**3.36** to Lium ck (ext **8888607-0009**) · bal **77358→78103** (+~$745) · free τ~1249 |
| 2026-08-20T21:13:38Z | 78070.45781489636 | p4207 post-burn snapshot (fleet still ~$354.58/h; no rent; B300×8=0) |
| 2026-08-20T21:02:22Z | 77392.16780964343 | p4206 **R1032 LOST** + **R1065 REFUTE→R1077 TRAIN** + R1062 n80 LIVE (no rent; burn **~$354.58/h**; B300×8=0 bl cosmic-raven-04) |
| 2026-08-20T20:56:05Z | 77429.71489411636 | p4205 **R1062 ENOSPC→clean+merge retry** r938 (no rent; burn **~$354.58/h**; B300×8=0) |
| 2026-08-20T20:51:22Z | 77463.66005192383 | p4204 **R1058+R1055 REFUTE→R1075+R1076 TRAIN** r252 (no rent; burn **~$354.58/h**; B300×8=0 B200×8=0) |
| 2026-08-20T20:42:15Z | 77533.2656349733 | p4203 **R1045+R1054 REFUTE→R1073+R1074** + R1055 chall relaunch (no rent; burn **~$354.58/h**; B300×8=0; bl cosmic-raven-04) |
| 2026-08-20T20:32:45Z | 77601.74605555866 | p4202 **R1061 REFUTE→R1072 TRAIN** r338 (no rent; burn **~$354.58/h**; B300×8=0; bl cosmic-raven-04 only) |
| 2026-08-20T20:26:42Z | 77638.20394121204 | p4201 **R1052 REFUTE→R1071 TRAIN** r339 (no rent; burn **~$354.58/h**; B300×8=0 B200×8=0) |
| 2026-08-20T20:20:42Z | 77671.96871299978 | p4200 **R1052 Triton-reseed n80 LIVE** r339 (no rent; burn **~$354.58/h**; B300×8=0) |
| 2026-08-20T20:10:54Z | 77776.96455726952 | p4199 **R1059+R1053 REFUTE→R1069+R1070 TRAIN** (no rent; burn **~$354.58/h**; B300×8=0) |
