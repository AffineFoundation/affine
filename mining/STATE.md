# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 5 · wvk=7 Reason v4 · KING reign36 · R1064 LOST chal-00974**
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$424.19/h** · gap **−$408.81/h** · B300×8=0 · B200×8=0 · H200×8=0 (just rented) |
| Lium | **~$72868** · free τ**~1247** · stake **r252 ≈τ1.69** (<τ5 paygo) |
| fleet | **10** mine-* · TTL fleet **→2026-08-21T13:26Z** · r340 **→21:24Z** · r339 **→15:50Z** · r1158 **→22T11:23Z** |
| **p4303** | rented **R1158** 8×H200 `eager-matrix-57`@$32 (node `golden-orbit-7b` / `e350ebc9…`) · BOOTSTRAP LIVE |
| eval | phase **duel** · chal-00984 |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · **R1175+76+79 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | TK · **R1168 + R1186 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | TK · **R1171+R1172 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | TK · **R1177+R1180 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | $64.00 | TK · **R1184+R1185 TRAIN** · SSH `23.153.44.20:40299` |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | $37.60 | TK · **R1181+82+83 TRAIN** · SSH `18.118.83.97:40127` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | TK · **R1169+R1173+R1178 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | TK · **R1170 n80** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | TK · **R1174 TRAIN** · SSH `38.255.28.21:20100` |
| mine-r1158-vera-reason-grpo-1 | eager-matrix-57 | $32.00 | **R1158 BOOTSTRAP** (vera×Reason-GRPO) · SSH `204.12.163.23:20301` |

## Blocked
No rentable B300×8 / B200×8 (BL-only when present). **R1158** now on H200×8 — keep fleet waiters for more 8×. Never blind `lium up --gpu`.

## Next action
1. Poll **R1158** BOOT→teacher→train (`/root/logs/bootstrap_r1158.log`). 2. Harvest **R1186** / **R1170** / MERGE→n80 when ready. 3. Fleet TTL **13:26Z** — extend before kill. 4. Rent next non-BL 8× if stock appears (still −$409/h vs floor).
