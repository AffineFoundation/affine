# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R905–R907 n80 LIVE · R903+R904 n80 LIVE · R917 TRAIN · R914–R916/R912–R913/R911 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$306.66/h** · gap **−$526.34/h** · B300×8 / 8×B200 stock=0 |
| Lium | **~$82782** · free τ**1257.6** · stake **0** |
| **R905** | **n80 LIVE** brave :8002 GPU**6** TP1 |
| **R906** | **n80 LIVE** brave :8003 GPU**2** TP1 |
| **R907** | **n80 LIVE** brave :8004 GPU**4** TP1 |
| **R903** | **n80 LIVE** golden :8002 GPUs**4,5** TP2 sim**764116** |
| **R904** | **n80 LIVE** golden :8003 GPUs**6,7** TP2 sim**764449** |
| **R910** | **REFUTE** m=+0.000448~**0.13×** thought✓(159) B✓(0.46) → slot→**R917** |
| **R917** | **TRAIN** R252 GPUs**4,5** ShortCtx Loβ=0.02 |
| **R914–R916** | **TRAIN** · **R912–R913** **TRAIN** · **R911** **TRAIN** |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R912+R913 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R915+R916 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R903+R904 n80 LIVE** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK vera TP1 · **R905+R906+R907 n80 LIVE** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | **R917 TRAIN** + **R911 TRAIN** |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | **R888 GRPO** + **R914 TRAIN** |

## Blocked
No B300×8 / 8×B200 stock. Brave chall must be **TP=1** (TP=2 hangs post-pynccl). Never `pkill -f`; never reap bare `VLLM::EngineCore` (kills TK).

## Next action
1. Poll R905–R907 / R903–R904 n80 → REFUTE/LOST→next TRAIN or WIN→Stage5. 2. Poll R917/R914–R916/R912–R913/R911 TRAIN→merge→n80. 3. Rent if 8×B300/B200 stock appears.
