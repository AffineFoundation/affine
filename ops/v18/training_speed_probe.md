# Training-speed probe — 30 wvk-22/23 verdicts (chal-00644 … chal-00676), 59730 side-turns

## 1. Reward density under min(z_R, typ_c, z_A)

- binding leg (valid side-turns, n=58530): R 0.33 / Gc 0.16 / A 0.51
- margin of the binding leg over the next leg: p25 0.15 / p50 0.46 / p75 1.06 / p90 2.01 sd; **gap ≥ 1 sd on 0.27 of turns** (the other two legs give zero gradient there); gap < 0.25 sd on 0.35
- soft-min τ = 0.5: mean weight on the binding leg 0.70 (min = 1.00, mean = 0.33)
- forfeits: 0.0201 of side-turns carry 0.48 of the total per-turn score variance (floor −12)

| combiner | sd of valid per-turn score | sd of paired diff | verdict flips vs live (of 30) | crowns |
|---|---|---|---|---|
| min | see below | 1.48 | 0 | 1 |
| softmin | see below | 1.44 | 0 | 1 |
| mean | see below | 1.03 | 1 | 0 |

Per-verdict decisions (margin / z under each combiner; live rule = min):

| duel | uid | live | min | soft-min τ0.5 | mean |
|---|---|---|---|---|---|
| chal-00644 | 253 | -0.144 z -2.3 | -0.113 z -1.8 | -0.118 z -1.9 | -0.085 z -1.8 |
| chal-00645 | 101 | -0.026 z -0.6 | -0.013 z -0.3 | -0.015 z -0.4 | -0.031 z -1.1 |
| chal-00647 | 233 | -0.321 z -4.2 | -0.263 z -3.6 | -0.264 z -3.7 | -0.238 z -5.1 |
| chal-00648 | 10 | -0.190 z -3.0 | -0.159 z -2.6 | -0.157 z -2.6 | -0.106 z -2.5 |
| chal-00649 | 224 | +0.135 z +3.2 | +0.122 z +3.0 | +0.115 z +2.9 | +0.061 z +2.1 |
| chal-00650 | 34 | +0.061 z +1.2 | +0.081 z +1.7 | +0.083 z +1.8 | +0.115 z +3.4 |
| chal-00651 | 37 | +0.166 z +3.4 | +0.152 z +3.2 | +0.151 z +3.3 | +0.090 z +3.0 |
| chal-00652 | 38 | +0.174 z +4.1 | +0.159 z +3.8 | +0.155 z +3.9 | +0.111 z +3.8 |
| chal-00653 | 40 | +0.186 z +3.5 | +0.188 z +4.1 | +0.185 z +4.2 | +0.152 z +4.6 |
| chal-00654 | 22 | -0.023 z -0.3 | -0.001 z -0.0 | -0.022 z -0.3 | -0.052 z -1.2 |
| chal-00655 | 47 | +0.112 z +2.5 | +0.112 z +2.5 | +0.108 z +2.5 | +0.096 z +3.0 |
| chal-00656 | 191 | +0.050 z +1.1 | +0.059 z +1.4 | +0.057 z +1.4 | +0.026 z +0.9 |
| chal-00657 | 201 | -0.394 z -5.5 | -0.411 z -5.8 | -0.413 z -5.9 | -0.363 z -6.8 |
| chal-00658 | 148 | +0.032 z +0.6 | +0.003 z +0.1 | +0.009 z +0.2 | +0.029 z +0.8 |
| chal-00659 | 213 | -0.307 z -5.0 | -0.288 z -5.0 | -0.278 z -4.9 | -0.259 z -5.3 |
| chal-00660 | 230 | -0.037 z -0.7 | +0.004 z +0.1 | +0.008 z +0.2 | +0.015 z +0.6 |
| chal-00661 | 4 | -0.023 z -0.3 | -0.015 z -0.2 | -0.021 z -0.3 | -0.036 z -0.8 |
| chal-00662 | 36 | +0.229 z +4.9 **crown** | +0.224 z +5.1 **crown** | +0.213 z +5.0 **crown** | +0.127 z +5.0 |
| chal-00663 | 11 | -0.051 z -1.2 | -0.054 z -1.5 | -0.045 z -1.3 | -0.023 z -1.0 |
| chal-00664 | 73 | -0.237 z -4.7 | -0.221 z -4.5 | -0.217 z -4.6 | -0.177 z -5.2 |
| chal-00665 | 235 | +0.000 z +0.0 | -0.028 z -0.7 | -0.031 z -0.8 | -0.020 z -0.8 |
| chal-00667 | 237 | +0.013 z +0.4 | +0.013 z +0.4 | +0.012 z +0.4 | -0.011 z -0.5 |
| chal-00668 | 234 | -0.025 z -0.6 | +0.014 z +0.4 | +0.015 z +0.5 | +0.020 z +0.9 |
| chal-00669 | 155 | +0.008 z +0.3 | -0.005 z -0.2 | -0.002 z -0.1 | -0.018 z -1.0 |
| chal-00670 | 25 | -0.107 z -1.8 | -0.085 z -1.7 | -0.097 z -1.9 | -0.108 z -2.7 |
| chal-00671 | 45 | -0.001 z -0.0 | +0.016 z +0.5 | +0.016 z +0.5 | +0.020 z +0.9 |
| chal-00672 | 33 | +0.024 z +0.7 | +0.020 z +0.6 | +0.018 z +0.5 | +0.008 z +0.4 |
| chal-00673 | 218 | -0.139 z -3.6 | -0.130 z -3.5 | -0.118 z -3.3 | -0.076 z -2.5 |
| chal-00675 | 46 | -0.038 z -1.3 | -0.022 z -0.8 | -0.020 z -0.8 | +0.002 z +0.1 |
| chal-00676 | 39 | -0.290 z -4.2 | -0.237 z -3.7 | -0.253 z -3.9 | -0.177 z -3.4 |

## 2. Sequential stopping (looks every 100 turns, slice order)

| k per look | turns-to-decision p50 / p90 | agreement with full-1000 | crowns seq / full | false-crown rate under null (sign-flip permutations) | duel min p50 / p90 (37 min per 1000 + 8 load) | verdicts/day |
|---|---|---|---|---|---|---|
| 2.0 | 150 / 810 | 29/30 | 2 / 1 | 0.026 per duel | 14 / 38 | 76 (full 1000: 32) |
| 2.6 | 250 / 979 | 29/30 | 2 / 1 | 0.005 per duel | 17 / 44 | 64 (full 1000: 32) |
| 2.83 | 350 / 982 | 29/30 | 2 / 1 | 0.005 per duel | 21 / 44 | 60 (full 1000: 32) |

null false-crown for the full-1000 rule (same permutations): 0.001
