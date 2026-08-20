# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R938 scoring · R965+R966 TRAIN · R957 n80 LIVE · R926 cold-TK king loading**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$329.78/h** · gap **−$503.22/h** · B300×8=0 · catalog 8×B200 only (no non-BL 8×) |
| Lium | **~$79383** · free τ**~1256** · stake **~τ3.31** (below ~τ5 sweep) |
| fleet | 8 mine-* · crown R965+R966 **TRAIN** + R957 **n80 LIVE** · R252 R960+R951 · R337 R963+R954 · R338 R959+R964 · R888 R961 · R924 R953+R952 · R926 **cold-TK** T READY K loading · R938 R962 + **chal-00949** |
| **R938** | **SCORING chal-00949** · challenger **~999/1300** · HF@`8ef1b06a` · reveal **31462190** |
| **R957** | **n80 LIVE** crown :8002 GPUs1,3 (Triton seed chall_r956) · MERGE_DONE idle→lean · sim pid**110249** |
| **p4083** | R957 MERGE_DONE sat idle → chall+**v4 n80 LIVE**; R938 ~999/1300; R926 K still loading |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · **R965+R966 TRAIN** 4,5/6,7 · **R957 n80** :8002 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | **R960+R951 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | **R963+R954 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | **R959+R964 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R961 TRAIN** · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | R953+R952 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | **cold-TK** T:8000 READY · K:8001 loading · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | T+K · **R962 TRAIN** · SSH `38.255.28.21:20100` |

## Blocked
No rentable B300×8; no non-BL 8×B200 in catalog. Never `pkill -f`.

## Next action
1. Poll R957 n80 → CROWN_OK/REFUTE (reap→next axis if REFUTE). 2. Watch R938 **chal-00949** (~999/1300). 3. R926 king READY→chall:8002 R944 n80. 4. Poll R965/R966→MERGE→n80. 5. Rent B300×8 when non-BL stock. 6. α→τ→Lium when stake ≥~τ5.
