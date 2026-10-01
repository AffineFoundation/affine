# Three completed wider evaluations

Three GPU epochs completed frozen miner uploads, full audits, three full-model
optimizer updates each, immutable checkpoint publication and paired evaluation.
The third epoch added original i3math; the previous nine environment definitions,
16 held-out tasks per environment, seeds and 256-token evaluation budget stayed
fixed. Each preceding post-update value exactly matches the next pre-update value.
Rewards come from the original graders, including their fractional rewards.

| Environment | Baseline | After epoch 1 | After epoch 2 | After epoch 3 | Held-out tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `affine_ifeval` | 0.125000 | 0.187500 | 0.125000 | 0.125000 | 16 |
| `affine_logic` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_math` | 0.437500 | 0.375000 | 0.500000 | 0.437500 | 16 |
| `affine_oolong` | 0.188614 | 0.251114 | 0.188614 | 0.251114 | 16 |
| `affine_rgym` | 0.003343 | 0.001364 | 0.007853 | 0.008953 | 16 |
| `affine_scitext` | 0.000000 | 0.000000 | 0.000000 | 0.062500 | 16 |
| `affine_unscramble` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_verbatim` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_when2call` | 0.187500 | 0.312500 | 0.312500 | 0.250000 | 16 |

i3math has its own first paired measurement in epoch 3: 0 before and 0 after
on 16 fixed held-out tasks. It was not measured at the earlier checkpoints.

Epoch 1 optimized Logic, SciText and Unscramble pairs; epoch 2 optimized
Verbatim, Math and Reasoning Gym; epoch 3 optimized an i3math pair. Other
measured changes reflect the shared model, rather than training contributions
from those environments. Epoch 3 had three increasing, two decreasing and five
unchanged metrics. Relative to the original nine-environment baseline, four
metrics increased and five were unchanged. These small tasksets do not establish
broad improvement or improvement on every environment.

The continuing pilot has moved to a separately pinned extension adding original
Trivia tasks. Its pending epochs are not counted here. The older two-task GPU
series uses a different taskset and output budget; the dashboard separates those
measurement groups. The first failed wider epoch remains aborted and untrained.

Private evidence in `state/gpu-wide/root-continuous-independent-evidence.json`
binds source archives, frozen submissions, full audits, optimizer attribution,
proposed weights, checkpoint descriptors and paired measurements. All three new
checkpoints' six R2 files were independently streamed and hashed: 3,426,302,727
bytes per checkpoint, without another local weight copy.

Reproduce evidence inspection in the approved operator workspace:

```sh
.venv/bin/python -m ops.check_gpu_continuous_evidence --state state/gpu-wide
```

This inspects operator-authenticated reports and actual published artifacts; it
does not itself rerun inference. Epochs remain nonpayable and submit no chain
weights. Broader native integration and coverage remain unfinished.
