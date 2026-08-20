# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R938 LOST · R952+R953 REFUTE · R967+R965+R966 n80 LIVE · R968+R969 TRAIN · R944 DEAD**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$329.78/h** · gap **−$503.22/h** · B300×8=0 · catalog 8×B200 + 1×B300 only |
| Lium | **~$78607** · free τ**~1256** · stake **~τ3.31** (below ~τ5 sweep) |
| fleet | 8 mine-* · crown **triple n80** R965+R966+R967 · R924 R968+R969 TRAIN · R944 sim **DEAD** |
| **R952** | **REFUTE** m=**−0.001046** SE=0.001843 z=−0.57 n=79 bar≈0.00369 (~**−0.28×**) thought✓168 B✓0.418 k=3 → **R968** |
| **R953** | **REFUTE** m=**+0.002543** SE=0.002680 z=0.95 n=79 bar≈0.00536 (~**0.47×**) thought✓170 B✓0.367 k=3 → **R969** |
| **p4092** | R967 MERGE idle→**v4 n80 LIVE** :8004 pid**121844**; R952+R953 REFUTE→reap→**R968+R969 TRAIN** |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · **R965+R966+R967 v4 n80 LIVE** :8002/:8003/:8004 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | **R960+R951 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | **R963+R954 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | **R959+R964 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R961 TRAIN** · SSH `192.9.163.79:20500` **refused** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | **R969+R968 TRAIN** 4,5/6,7 · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | TKC · **R944 n80 DEAD** (sim pid52208 gone) · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | T+K · **R962 TRAIN** · SSH `38.255.28.21:20100` |

## Blocked
No rentable B300×8; sole catalog 8×B200s historically BL. Never `pkill -f`.

## Next action
1. Poll R965/R966/R967 **n80** → CROWN_OK/REFUTE. 2. On REFUTE: exact-PID reap → next axis TRAIN. 3. Repair **R944** n80 (sim dead; chall may still be up@0.65). 4. Poll R968/R969→MERGE→n80. 5. Fix R888 SSH / rent B300×8 when non-BL stock. 6. α→τ→Lium when stake ≥~τ5.
