# Seven completed wider evaluations

Five GPU epochs completed frozen miner uploads, full original-environment audits,
three full-model optimizer updates each, immutable checkpoint publication and
paired evaluation. The later extensions added i3math and Trivia while preserving
the prior environment definitions, fixed task IDs, seeds and 256-token budget.
Each previous post-update measurement matches its next pre-update measurement.
The following table preserves the first four completed measurements.

| Environment | Baseline | After 1 | After 2 | After 3 | After 4 | Held-out tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `affine_ifeval` | 0.125000 | 0.187500 | 0.125000 | 0.125000 | 0.187500 | 16 |
| `affine_logic` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_math` | 0.437500 | 0.375000 | 0.500000 | 0.437500 | 0.312500 | 16 |
| `affine_oolong` | 0.188614 | 0.251114 | 0.188614 | 0.251114 | 0.188614 | 16 |
| `affine_rgym` | 0.003343 | 0.001364 | 0.007853 | 0.008953 | 0.001776 | 16 |
| `affine_scitext` | 0.000000 | 0.000000 | 0.000000 | 0.062500 | 0.000000 | 16 |
| `affine_unscramble` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_verbatim` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 16 |
| `affine_when2call` | 0.187500 | 0.312500 | 0.312500 | 0.250000 | 0.250000 | 16 |

i3math was first measured in epoch 3 and remained 0 before and after both
epochs 3 and 4. Trivia was first measured in epoch 4 and remained 0.4375
before and after that update. Each uses sixteen fixed original held-out tasks.

Epochs 1–4 optimized, respectively: Logic/SciText/Unscramble;
Verbatim/Math/Reasoning Gym; i3math; and Trivia. Epoch 4 had one improving,
four declining and six unchanged environment metrics. The full twelve updates
do not demonstrate improvement across all environments. Small fixed tasksets
and differing fractional reward scales also limit broad conclusions.

The fifth epoch added PopQA under a separate pinned source and completed three
more full-model updates. Its evaluation used twelve environments with sixteen
fixed held-out tasks each. Four metrics improved and eight were unchanged:

| Environment | Before epoch 5 | After epoch 5 |
| --- | ---: | ---: |
| Math | 0.312500 | 0.375000 |
| Reasoning Gym | 0.001776 | 0.001878 |
| When2Call | 0.250000 | 0.312500 |
| PopQA | 0.187500 | 0.562500 |

Math remains below its first baseline. The other eight metrics match their
epoch-four values. Fifteen total updates and these small held-out sets do not
establish sustained improvement across all environments.

The older two-task series remains a separate
measurement group, and the failed initial wider epoch remains aborted and untrained.

Private evidence in `state/gpu-wide/root-continuous-independent-evidence.json`
binds worker source bytes, frozen submissions, full audits, optimizer attribution,
proposed weights, checkpoint descriptors, and paired measurements. All seven
successor checkpoints were independently streamed and hashed: six R2 files and
3,426,302,727 bytes each, without another local copy of the weights. Independent
HTTPS checks confirmed the corresponding epoch/checkpoint/UID-grid records and
all 150 evaluation records on affine.io.

```sh
.venv/bin/python -m ops.check_gpu_continuous_evidence --state state/gpu-wide
```

This checks authenticated operator reports and actual artifacts; it does not
rerun inference itself. Epochs are nonpayable and submit no chain weights.
Broader native integration and coverage remain unfinished.

## Sixth epoch: fixed-reference update and mixed held-out results

The sixth completed epoch made three full-model updates on one verified original
i3math preference pair. The reference margin from the frozen batch probabilities
was 0.5493861153; the worker used 0.5493862629, within the pinned 1e-5 tolerance.
It held that reference fixed while persistent Adam counters advanced through
1, 2 and 3. The training loss fell from 0.693147 to 0.678692. This confirms the
optimizer operation; it does not establish generalization.

| Environment | Before epoch 6 | After epoch 6 |
| --- | ---: | ---: |
| Math | 0.375000 | 0.437500 |
| Reasoning Gym | 0.001878 | 0.001509 |
| When2Call | 0.312500 | 0.250000 |
| IFEval | 0.187500 | 0.125000 |
| Trivia | 0.437500 | 0.375000 |

The other seven metrics remained unchanged, including i3math at zero and PopQA
at 0.5625. Each of the twelve environments used the same sixteen held-out tasks
and 256-token budget before and after training. There are now eighteen verified
updates across six completed wider epochs. One improving metric, four declines
and seven unchanged metrics do not support improvement across all environments.
Balanced multi-family training remains work to qualify in a prospective source
version; this result must not be replaced with an assumption of future recovery.

The sixth successor checkpoint is
`088fdf925b4c0063e9113b1a99aab49ba9f6b749fd77939d1a0cfba3fd474b89`.
Its six R2 objects were independently streamed and hashed, totaling
3,426,302,727 bytes. The published dashboard was checked against all paired
reports, exact checkpoint IDs and the 256-cell UID grid.


## Seventh epoch: twelve fixed held-out families

The seventh completed epoch used verified preference pairs from Logic, SciText
and Unscramble for three full-model updates. All twelve families retained their
sixteen fixed held-out tasks, seeds and 256-token autoregressive budget.

| Environment | Before epoch 7 | After epoch 7 |
| --- | ---: | ---: |
| Trivia | 0.375000 | 0.500000 |
| Math | 0.437500 | 0.375000 |
| Reasoning Gym | 0.001509 | 0.001489 |
| When2Call | 0.250000 | 0.187500 |

The other eight metrics stayed unchanged, including all three directly trained
families at zero. The loop has twenty-one verified optimizer updates across
seven completed epochs; these results remain mixed. They do not establish
improvement across all environments or transfer from the curated mining policy
to unrestricted held-out sampling.

The seventh successor checkpoint is
`d8e047f13278692e0baa0df213b4d5566318e582bb74e47754a63a91d6125383`.
All six R2 objects were independently streamed and hashed, totaling
3,426,302,727 bytes. Independent HTTPS checks matched all 150 wide-series
public evaluation records, checkpoint bindings and 256-UID grids. Native
coverage and qualified balanced multi-family replay remain unfinished work.
