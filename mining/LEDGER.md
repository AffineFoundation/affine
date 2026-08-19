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
| $UTC | 79664.99313959652 | p4058 |
| Lium balance | **$79664.99313959652** | p4058 |
| cumulative mining spend | ~$136,852 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$69665** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$329.79/h** · vs floor $833/h · gap **−$503.21/h** · B300=0 · BL-only B200 | p4058 |
| miner coldkey free | τ1258.144 | kept ≥τ50 |
| miner stake | r252 **~29.5α/~τ1.63** (below τ5 sweep) | p4041 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T23:50:13Z | 79664.99313959652 | p4058 R934 REFUTE~-0.34×→reap→**R946 TRAIN**+**R937 n80** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T23:38:36Z | 79742.08325474679 | p4057 R926 REFUTE~-0.80×→reap→**R944+R945 TRAIN** + R934 lean→n80 (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T23:21:14Z | 79857.0274108562 | p4056 R940 REFUTE~0.77×→reap→**R943 TRAIN** + R926 SIZE_OK→lean (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T23:16:18Z | 79895.46633318713 | p4055 R3 REFUTE~-0.03×→reap→**R942 TRAIN** (no rent; B300/H200/H100=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T22:56:20Z | 80049.52448572023 | p4052 R936 REFUTE→R934 relay + R252 Triton recover (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T22:46:48Z | 80163.02716236464 | p4051 R936 n80 LIVE + R926→crown relay (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T22:29:07Z | 80275.38971973065 | p4050 R924+R927 REFUTE→R928 n80+R933 relay+R926 rematch (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T22:03:11Z | 80459.59433719006 | p4049 R927 n80 + R924 false-fail→chall + R252 recover (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T21:53:48Z | 80536.29922668624 | p4048 R929 REFUTE→**R940 TRAIN** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T21:48:21Z | 80611.5782672317 | p4047 R929 chall:8003+n80 + R337 REFUTE→**R939 TRAIN** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
