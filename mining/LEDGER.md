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
| $UTC | 80535.97300565851 | p4066 |
| Lium balance | **$80535.97300565851** | p4066 |
| cumulative mining spend | ~$137,349 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$70536** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$329.79/h** · vs floor $833/h · gap **−$503.21/h** · B300=0 · BL-only B200 | p4066 |
| miner coldkey free | τ1258.232 | kept ≥τ50 |
| miner stake | **0** (p4061 swept r252 118α) | p4061 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T00:56:43Z | 80535.97300565851 | p4066 R930 Triton miss→FORCE seed chall_r931→**R930+R931 n80 LIVE** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T00:44:18Z | 80650.74632501046 | p4065 R925 REFUTE~−0.001×→reap→**R951 TRAIN** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T00:36:46Z | 80689.05473551793 | p4064 R939/R932 REFUTE→reap→**R949+R950 TRAIN** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T00:29:44Z | 80766.09024527614 | p4063 R924 idle→**cold-TK+R930/R931 dual n80** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T00:19:08Z | 80842.40593092538 | p4062 R939 MERGE→chall:8002+n80 on R337 (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-20T00:15:11Z | 80880.8878608843 | p4061 **α→τ→Lium**: r252 unstake ALL 118α→+τ6.589 free; `lium fund` fail; `btcli transfer` τ6.5→Lium ck (+~$1407.6); R932 rematch; burn **~$329.79/h** |
| 2026-08-20T00:07:14Z | 79511.85967453076 | p4060 R935 REFUTE~0.065×→reap→**R948 TRAIN** + R925 host-relay→R252 (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T23:59:43Z | 79589.11288047479 | p4059 R937 REFUTE~0.27×→reap→**R947 TRAIN**+**R935 n80** + R924 R925/R930 MERGE (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T23:50:13Z | 79664.99313959652 | p4058 R934 REFUTE~-0.34×→reap→**R946 TRAIN**+**R937 n80** (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
| 2026-08-19T23:38:36Z | 79742.08325474679 | p4057 R926 REFUTE~-0.80×→reap→**R944+R945 TRAIN** + R934 lean→n80 (no rent; B300=0 BL `8f34559f`); burn **~$329.79/h** |
