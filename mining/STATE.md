# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4 · wvk=7 Reason v4 · KING reign36 · R932 MERGE→chall:8002 LIVE · R925 RELAY · R948/R947/R946/R945/R943/R944/R942/R941 TRAIN · R930/R931 MERGE idle · R938/R939 TRAIN_DONE idle**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$329.79/h** · gap **−$503.21/h** · B300×8=0 · BL B200 only `8f34559f`/`fbb1135f` |
| Lium | **~$80881** · free τ**1258.2** · stake **0** (p4061 α→τ→Lium) |
| fleet | 8 mine-* · crown TK+trains · R338 R947+R948 · R252 R942+R925relay · R888 R932 chall |
| **R932** | wait FATAL flat-adapter → **MERGE_DONE** rematch → chall:8002 GPUs5,6 pid**45099** lean**45001** |
| **R925** | host-relay mid (~8/16 PIPE) → R252 :8002 lean armed (outer alive) |
| **R930/R931** | MERGE_DONE idle on R924 (after R925 n80 / free TKC) |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · R943+R945+R946 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | R942 TRAIN + **R925 relay→n80** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | R939 DONE + **R941 TRAIN** · free GPUs4,5 · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | **R947+R948 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R932 chall:8002** · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | R925/R930/R931 MERGE_DONE · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | **R944 TRAIN** · free GPUs2–7 · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | R938 TRAIN_DONE idle · SSH `38.255.28.21:20100` |

## Blocked
No rentable B300×8 / non-bl 8×B200 / 8×H200 / 8×H100. Marsplan queen **gated**. Never `pkill -f`.

## Next action
1. Await R932 CHALL_READY→n80 then verdict. 2. Await R925 SIZE_OK→lean:8002 n80. 3. Host-relay R930→R337 GPUs4,5 (or R931) when free; merge R938/R939→relay. 4. Rent B300 when stock.
