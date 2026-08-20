# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 5 DONE · wvk=7 Reason v4 · KING reign36 · R959 QUEUED · R973 MERGE→n80 · R990+R991 TRAIN · R989+R987 TRAIN · R988+R975 TRAIN · R986+R984+R985 TRAIN · R981+R982+R983 TRAIN · R980 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$290.58/h** · gap **−$542.42/h** · B300×8=0 · 1×B300 only · skip BL B200 `8f34559f` |
| Lium | **~$79281** · free τ**~1254** · stake **α29.5/~τ1.66** r252 (under τ5 sweep) |
| fleet | 7 mine-* · crown R981–R983 TRAIN · R924 R986+R984+R985 · R337 R987+R989 · R252 R988+R975 · R338 R990+R991 · R926 **R973 MERGE→chall n80** · R938 R980 |
| **p4118** | R973 TRAIN_DONE@08:05 merge FATAL (CUDA 0,1 teacher offload) → rematch MERGE GPUs**3,4** `--no-save-original-format` → chall:8002+v4 n80 arm outer pid**61774** |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · **R981+R982+R983 TRAIN** 6,7/4,5/1,3 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | TK · **R988 TRAIN** 4,5 + **R975 TRAIN** 6,7 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | TK · **R987 TRAIN** 4,5 + **R989 TRAIN** 6,7 · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | TK · **R991 TRAIN** 4,5 + **R990 TRAIN** 6,7 · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | **R986 TRAIN** 6,7 + **R984 TRAIN** 1,3 + **R985 TRAIN** 4,5 · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | T+K · **R973 MERGE→n80** GPUs3,4 :8002 · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | TK · **R980 TRAIN** GPUs2,3 · SSH `38.255.28.21:20100` |

## Blocked
No rentable B300×8 / H200×8 / H100×8; skip BL B200. 1×B300 only. Never `pkill -f`.

## Next action
1. Poll R973 MERGE→chall:8002→v4 n80 → decision. 2. Poll trains R990+R991 / R989+R987 / R988+R975 / R986+R984+R985 / crown R981–R983 / R980. 3. Watch R959 **chal-00957**. 4. Rent B300×8 when stock.
