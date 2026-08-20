# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R938 LOST · R965+R966+R967 REFUTE · R970+R971+R972+R968+R969 TRAIN · R944 n80 LIVE**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$290.58/h** · gap **−$542.42/h** · B300×8=0 · catalog 8×B200=`fbb1135f`/`8f34559f` (hist BL) + 1×B300 only |
| Lium | **~$78361** · free τ**~1256** · stake **~τ3.31** (below ~τ5 sweep) |
| fleet | 7 mine-* · crown **R972+R971+R970 TRAIN** · R924 R968+R969 TRAIN · **R944 v4 n80 LIVE** |
| **R967** | **REFUTE** m=**−0.002955** SE=0.002859 z=−1.03 n=79 bar≈0.00572 (~**−0.52×**) → **R972** |
| **R966** | **REFUTE** m=**−0.006979** SE=0.004807 z=−1.45 n=80 → **R970** |
| **R965** | **REFUTE** m=**−0.001530** SE=0.002237 z=−0.68 n=79 → **R971** |
| **p4096** | R944 teacher **EngineDead** CUDA OOM@TP2/0.88 → FORCE seed n_so=7 + **TP=4 GPUs0,1,5,6 @0.85** batchtok4096 → probe_ok → **v4 n80 LIVE** pid**58462** |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · **R972+R971+R970 TRAIN** 1,3/4,5/6,7 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | **R960+R951 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | **R963+R954 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | **R959+R964 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | **R969+R968 TRAIN** 4,5/6,7 · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | TKC · **R944 v4 n80 LIVE** pid58462 · T TP4@0.85 · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | T+K · **R962 TRAIN** · SSH `38.255.28.21:20100` |

## Blocked
No rentable B300×8; sole catalog 8×B200s hist-BL. **mine-r888** absent. Never `pkill -f`.

## Next action
1. Poll R944 n80→decision; on CROWN_OK → Stage5; on REFUTE → exact-PID reap → next axis. 2. Poll R972/R970/R971/R968/R969→MERGE→n80. 3. Rent B300×8 when non-BL stock; replace lost R888 slot. 4. α→τ→Lium when stake ≥~τ5.
