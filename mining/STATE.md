# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 5 · FORK wvk=7 Reason v4 · R653 REFUTE · R663/R655 SCP · R681/R680/R679/R678/R677/R676/R675/R674/R673 TRAIN · R670/R668/R671/R664/R665/R666 MERGE_DONE**.
King=reign34 · r252 reign33 earning. Burn floor **≥$833/h** on mine-* 8×B300.

## Live facts

| item | value |
|---|---|
| contract | wvk=**7** · Reason v4 tempered LME · **k=3** · **τ=0.03** · n_turns=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| crown rule | margin > max(2·SE, 0.002) **and** median `|z|≥80` **and** B pass ≥0.30 |
| Reason | per turn `τ·log(mean_i exp(a_i/τ))` with `a_i=lpC(y_i\|z_A)−lpC(y_i|∅)` |
| king | **`cryptoDev23/Affine-5Dku3dYp9j-hk8161`** @ `55b7ffe0…` **reign 34** |
| our reign | **r252** reign33 / uid90 / earning |
| miner burn | **~$331.45/h** · floor $833/h · **gap -$501.55/h** |
| B300 stock | **empty** (API+CLI waiter live; 8×B200 empty too) |
| Lium bal | **~$88635** · floor $10k OK |
| submissions | **9** · r252 CROWNED r33 · r596 chal-00822 · r637 chal-00829 |
| **p3703** | **R681 TRAIN** brave **4,5** UltraExtra after R671 MERGE |

## Running

| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | **R676** 0,1 · **R677** 2,3 · **R678** 4,5 · **R673** 6,7 · **R663 uplink** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK · R537 :8002 · **R655 SCP+REPAIR→chall** 4,5/:8003 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK · R637 :8004 · **R663 SCP→chall** 4,5/:8003 |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da | $64.00 | TK · **R680 TRAIN** 6,7 · **R674 TRAIN** 4,5 · R668 keep |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | **R669/R672 TRAIN** · **R679** 2,3 · **R681 TRAIN** 4,5 · R670/R671 keep |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK · R596 :8002 · **R675 TRAIN** 6,7 · **R655 uplink** |

SSH: crown `…90:40099` · R165 `…118:20299` · R262 `…127:40299` · R260 `…95:20299` · brave `18.118.83.97:40127` · R252 `95.133.252.28:40299`

## Blocked
Under $833/h — **no 8×B300/B200**. Pre-p3664 n80s stamped k=1 — ignore for submit. **Never `pkill -f`**. Stake ~29.5α (~τ1.68) below τ5 sweep. **Do not** dual-pipe R252 while R655 live.

## Next action
1. Watch **R663 SCP** (~17G/4sh) → `r663_scp_ready.done` (+repair) → lean 4,5/:8003 v4 n80; fail if stamp k≠3.
2. Watch **R655 REPAIR** (~54G/14sh) → `r655_scp_ready.done` → lunar lean 4,5/:8003 v4 n80.
3. After R663 resolve: queue **R666** / **R664** / **R665** / **R670** / **R671** → golden; R676–R681 own trains.
4. Blind rent **R337** when 8×B300/B200 appears (waiter live).
