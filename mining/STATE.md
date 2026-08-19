# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R882 REFUTE · R899 n80 ~71/80 · R900 chall LOADING · R337/R338 SSH-dead**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$405.70/h** · gap **−$427.30/h** · B300×8 stock=0 · 8×B200=0 |
| Lium | **~$81190** · free τ**1257.6** · stake under τ5 sweep |
| **R899** | **n80 ~71/80** R252 :8002 pid**610406** · progress `@08:52Z` |
| **R900** | MERGE idle→**chall LOADING** :8003 GPUs6,7 pid**611324** (p4013 cleared bad LAUNCHED; wrong path `…hibeta-midctx…`) |
| **R882** | **REFUTE v4** m=**−0.007812** ~−1.03× · slot was→R884 (R338 dead) |
| **R337/R338** | both **TCP fail + lium exec timeout** (`86.38.182.{67,55}`) — still billed |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R901+R902 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R896+R897 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R903+R904 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** · **R905+R906+R907 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | **R899 n80** + **R900 chall** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | **SSH-DEAD** · R908+R909 stranded |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | **SSH-DEAD** · R883/R884 stranded |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | **R898 MERGE** |

## Blocked
No B300×8 / 8×B200 stock. **R337+R338 unreachable** (same /24). Next: tear+replace those `mine-*` only when stock appears (or tear now if still dead next pass). Never `pkill -f`.

## Next action
1. Poll R899 result (near done) + R900 CHALL_READY→n80. 2. Tear R337+R338 if still dead; re-rent axes when B300/B200×8 appears. 3. Rent if stock appears.
