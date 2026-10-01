# First completed wider evaluation

The first corrected wider GPU epoch completed its miner uploads, full audits,
three full-model optimizer updates, checkpoint publication and paired evaluation.
The same 16 held-out tasks per environment and 256-token output budget were used
before and after training. These measurements use the original environment
rewards, including fractional rewards where the original grader returns them.

| Environment | Before | After | Held-out tasks |
| --- | ---: | ---: | ---: |
| `affine_ifeval` | 0.125000 | 0.187500 | 16 |
| `affine_logic` | 0.000000 | 0.000000 | 16 |
| `affine_math` | 0.437500 | 0.375000 | 16 |
| `affine_oolong` | 0.188614 | 0.251114 | 16 |
| `affine_rgym` | 0.003343 | 0.001364 | 16 |
| `affine_scitext` | 0.000000 | 0.000000 | 16 |
| `affine_unscramble` | 0.000000 | 0.000000 | 16 |
| `affine_verbatim` | 0.000000 | 0.000000 | 16 |
| `affine_when2call` | 0.187500 | 0.312500 | 16 |

This epoch optimized audited positive/negative pairs from Logic, SciText and
Unscramble. Other measured changes are effects on the shared model, rather
than evidence that those environments contributed training pairs in this epoch.
Three metrics increased, two decreased, and four stayed unchanged. A single
16-task evaluation does not establish broad performance improvement.

This wider series has a different taskset and output budget from the earlier
two-task GPU series. The dashboard keeps the measurement groups separate.
The first failed wider epoch remains recorded as aborted and untrained.

Operator evidence (ignored by Git) is retained in `state/gpu-wide/`:
`root-continuous-independent-evidence.json` binds the source archive, frozen
submissions, full audit reports, pair attribution, proposed weights, checkpoints
and paired measurements. The checkpoint publication check streamed all six
files from R2 and verified 3,426,302,727 bytes without storing another weight copy.

Reproduce the evidence inspection from the approved operator workspace:

```sh
.venv/bin/python -m ops.check_gpu_continuous_evidence --state state/gpu-wide
```

This inspects operator-authenticated execution reports and published bytes; it
does not itself rerun model inference. Test epochs remain nonpayable and submit
no chain weights. The continuing service opens its next epoch from the newly
published checkpoint; broader native-adapter integration remains unfinished.
