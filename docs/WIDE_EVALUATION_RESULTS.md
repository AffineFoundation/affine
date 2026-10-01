# Two completed wider evaluations

Two corrected wider GPU epochs completed miner uploads, full audits, three
full-model optimizer updates each, checkpoint publication and paired evaluation.
The same 16 held-out tasks per environment and 256-token output budget were used
at all three checkpoints. The first epoch's post-training values match the
second epoch's pre-training values exactly. Rewards are from the original
environment graders, including fractional rewards where those graders return them.

| Environment | Baseline | After epoch 1 | After epoch 2 | Held-out tasks |
| --- | ---: | ---: | ---: | ---: |
| `affine_ifeval` | 0.125000 | 0.187500 | 0.125000 | 16 |
| `affine_logic` | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_math` | 0.437500 | 0.375000 | 0.500000 | 16 |
| `affine_oolong` | 0.188614 | 0.251114 | 0.188614 | 16 |
| `affine_rgym` | 0.003343 | 0.001364 | 0.007853 | 16 |
| `affine_scitext` | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_unscramble` | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_verbatim` | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_when2call` | 0.187500 | 0.312500 | 0.312500 | 16 |

Epoch 1 optimized audited Logic, SciText and Unscramble pairs. Epoch 2 optimized
Verbatim, Math and Reasoning Gym pairs. Other measured changes reflect the shared
model; they do not establish training contributions from those environments.
Epoch 1 had three increasing, two decreasing and four unchanged metrics.
Epoch 2 had two increasing, two decreasing and five unchanged metrics.
Relative to the original baseline, three metrics increased and six stayed
unchanged. These small fixed tasksets do not establish broad improvement or
improvement on every environment.

The continuing pilot has moved to a separately pinned extension adding original
i3math tasks. Its generation, audit and training results remain separate from
this completed nine-environment series. The older two-task GPU series also uses
a different taskset and output budget; the dashboard separates those measurement
groups. The first failed wider epoch remains recorded as aborted and untrained.

Private operator evidence is retained in `state/gpu-wide/`:
`root-continuous-independent-evidence.json` binds source archives, frozen
submissions, full audits, pair attribution, proposed weights, checkpoints and
paired measurements. Both new checkpoints' six R2 files were independently
streamed and hashed, 3,426,302,727 bytes per checkpoint, without another local
weight copy.

Reproduce the evidence inspection in the approved operator workspace:

```sh
.venv/bin/python -m ops.check_gpu_continuous_evidence --state state/gpu-wide
```

This inspects operator-authenticated reports and published bytes; it does not
itself rerun model inference. Test epochs remain nonpayable and submit no chain
weights. Broader native-adapter integration and coverage remain unfinished.
