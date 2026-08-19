# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4 · wvk=7 Reason v4 · KING reign36 · R924 RELAY→n80 · R337 CHALL→n80 · R927 MERGE_DONE · R938 BOOT · R935/R936/R937 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$329.79/h** · gap **−$503.21/h** · B300×8=0 · 8×B200 only BL · H200×8 rented |
| Lium | **~$80724** · free τ**1258.1** · stake r252 **~29.5α/~τ1.63** |
| fleet | 8 mine-* · crown TK+R929/R936 · R252 GRPO · R924→crown relay · R926–R934 · R337 chall · R338 R935+R937 · R888 R932 · **R938** SoftCtx Hiβ |
| **R924** | host-relay mid (shards 1/3/4 done; more piping) → crown GPUs**6,7** :8002 v4 n80 · outer **627137** |
| **R337** | KING_READY · chall:8002 loading (pid**19488**) → v4 n80 · outer **15785** |
| **R927** | **MERGE_DONE** 16shards `/tmp/r927_merged` @21:25Z · need host-relay→n80 |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · R929/R936 TRAIN · **R924 relay→chall** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | R3 GRPO MERGE→n80 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | **R337 chall→n80** · idle 6,7 · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | R935+R937 TRAIN · T+king · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · R932 TRAIN · GRPO 2,3 |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | MERGE_DONE · R925/R930/R931 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | R926/R933/R934 TRAIN · **R927 MERGE_DONE** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | **R938** SoftCtx MidRank Hiβ BOOT→TRAIN · SSH `38.255.28.21:20100` |

## Blocked
No rentable B300×8 / non-bl 8×B200. Marsplan queen **gated**. Never `pkill -f`. `lium fund` broken → `btcli wallet transfer`.

## Next action
1. Poll R924 relay→chall:8002→v4 n80; clear→Stage5. 2. Poll R337 chall→n80; REFUTE→fill idle 6,7. 3. Host-relay R927→R337:8003 (or crown)→v4 n80. 4. Confirm R938 TRAIN + fill idle H200 GPUs; rent B300 when stock.
