# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign35 · R769 N80 LIVE · R761 RELAY**.
King=**reign35** tammy · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `tammyfritz/Affine-5hmwhnfbix-tammy2`@`7e5fd5f8…` **reign35** |
| burn | **~$331.45/h** · gap **−$501.55/h** · B300×8 stock **0** (bl_skip=1; B200 ghost fbb1135f) |
| Lium | **~$86030** · free τ**1260.38** · r252 stake ~τ3.35 (below τ5 sweep) |
| **p3830** | **R761 MERGE idle** → **host-relay brave→R252** queue after **R769 N80**; R762 MERGE idle (next) |
| **R769** | **N80 LIVE** R252 :8002 GPUs 4,5 pid**371678** |
| **R761** | **RELAY** tar pipe + waiter pid**371826** → n80 after R769 decision |
| **R778+R779** | **TRAIN** lunar 6,7 / 4,5 |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK · **R775+R776 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK · **R779+R778 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK · **R774+R773 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da | $64.00 | TK · **R772+R777 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | **R767+R768 TRAIN**; R761/R762 MERGE |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK · **R769 N80** + **R770 TRAIN** + **R761 RELAY** |

## Blocked
No rentable 8×B300/B200 (only bl ghost fbb1135f). Never `pkill -f`. Brave NCCL — host-relay n80. Hard-pin BASE after `mine.env`.

## Next action
1. Await R769 n80 → R761 auto-chall; harvest decision. 2. Host-relay R762 after R761 slot frees. 3. Rent when non-blacklisted B300/B200×8 appears. 4. Sweep α≥~τ5.
