# Learning-method investigation

This is research status, not a change to the active mining contract. The signed
epoch manifest and `llms.txt` continue to define the requirements for miners.

## Positive-only SFT and the role of failures

The proposed learner trains on completed, correct responses using supervised
negative log likelihood. Incorrect responses remain required for a valid mining
batch, but contribute no training gradient. Requiring both outcomes directs
miners toward tasks on which the checkpoint exhibits mixed performance. It does
not establish that either the reasoning or the sampling distribution is ideal
for learning.

This is close to rejection-sampling fine-tuning, often called RAFT. Published
math experiments find it a useful baseline, while also observing reduced
exploration with prolonged positive-only training. Collecting negative responses
without using them in the objective does not itself prevent that effect. See
[the RAFT, RAFT++, and Reinforce-Rej comparison](https://arxiv.org/html/2504.11343v2).

For Affine, a correct final answer is not a guarantee of correct intermediate
reasoning. Small development-score gains and lower training loss are also not
enough to demonstrate improved generalization.

## What has actually been measured

The first isolated, eight-update positive-only SFT candidate solved 405 of 512
independent evaluation tasks; the genuine upstream model solved 404. There were
ten paired gains and nine losses. The predeclared conservative 95% interval for
the difference was −2.72 to +3.10 percentage points. This is an inconclusive
result, and the candidate has not replaced production on that evidence.

A larger study fixes 1,024 training tasks and 32 updates before evaluating either
endpoint. One arm uses successful finalized network responses; another uses
official training solutions for the same tasks. Both start from the genuine
upstream checkpoint with fresh optimizer state and are compared with an upstream
baseline on the same fresh 512-task evaluation. This study is now complete:
base 405/512, network 268/512 and reference 389/512. Network-positive SFT regressed
by 26.76 percentage points, with a paired 95% interval of −33.33 to −19.77 points.
It has not been promoted. See the [complete results and stopping analysis](POSITIVE_SFT32_RESULTS.md).
The network arm contains more responses and substantially more tokens, so this
is a comparison of training recipes, not an isolated causal test of reasoning
quality.

## The next distinct hypothesis: fresh-group policy gradients

A private RLOO prototype collects exactly eight fresh attempts per task at the
current model, retaining the whole group instead of searching until fixed
success and failure quotas are filled. Each attempt's baseline is the mean
reward of the other seven. A single optimizer update follows the full collection.
This tests whether a direct estimate of expected task reward works better than
learning from outcome-conditioned submissions. See the
[RLOO formulation](https://arxiv.org/html/2402.14740v2).

The prototype uses a constant token divisor rather than dividing each policy
gradient by its realized response length. This choice preserves the sequence
score estimator up to a fixed scale; it does not make every trajectory's
gradient contribution independent of length. This distinction concerns policy
gradients and does not mean supervised token-mean loss is inherently wrong.
See [Dr. GRPO](https://arxiv.org/html/2503.20783v2).

Existing miner batches cannot simply be treated as these fresh groups: their
outcomes were deliberately selected. The private prototype also uses a
different sampling temperature and BF16 numerical path. It is a broader method
comparison, not proof that outcome selection caused the historical decline.

The prototype passes 38 CPU tests, including tiny-model generation, actual
gradient-checkpointed scoring, gradient-estimator controls, and once-only update
accounting. The subsequent isolated H200 study is now running a frozen schedule
of 16 updates on 256 training tasks, collecting eight attempts per task. It
starts from the original base with fresh optimizer state. Its original process
and fixed endpoint are preserved; partial progress is not a learning result.
No production RLOO change is active.

In parallel, a separate 32-update SFT study tests per-task token-weighted
normalization on the same frozen 1,024-task network corpus as the failed SFT32
arm. The data, task grouping, training horizon and learning rate remain fixed;
only the within-task response weighting changes. This directly tests one
plausible contributor to the short-answer collapse. Original endpoints will be
compared under a fixed evaluation profile, with independent confirmation still
required before promotion. The live network is not using this experimental
objective.

Original signed study-scope SHA256 values:

- Fresh-group RLOO16: `e82c7587425d03ea9410258ad52a5172c6e2d350c414c03018bbc6a00f5b05d9`
- Pooled-token SFT32: `ed283a6c6e91638998e6c37e58367dc8159461440b7a82b60cff7134bfec39ce`

## Further methods under review

[Dynamic Fine-Tuning (DFT)](https://arxiv.org/html/2508.05629v3) weights each
positive token's cross-entropy by its detached model probability. It reduces the
relative influence of unlikely target tokens and needs no negative examples or
additional reward model. It is a candidate if positive SFT fits the submitted
text without improving held-out accuracy. Its published gains use different
data and training settings, and its confidence weighting is not a test of
reasoning correctness. We have not tested DFT on Affine.

[SED-SFT](https://arxiv.org/html/2602.07464v1) selectively regularizes token
confidence to preserve response diversity. Its study includes
Qwen2.5-Math-7B-Instruct, but the reported advantage is principally after a
subsequent RL stage; its SFT-only results do not establish a universal accuracy
improvement. We would prioritize this only if our own measurements show lost
exploration, rather than assuming every disappointing SFT result is entropy
collapse. Neither method changes the running experiment or the live contract.

## How a change earns promotion

Preserve original experiment failures and all endpoint results. Compare fixed
endpoints against the genuine upstream checkpoint with an unchanged evaluation
profile, independent held-out tasks, paired uncertainty, and explicit cap and
infrastructure-error accounting. Do not select a checkpoint using its confirmation
score. A promising result must then survive continued online training; a lower
training loss alone is insufficient.

Other justified follow-ups include regularized positive-only updates and
reasoning-quality controls. Their priority depends on the completed comparisons,
not on which method has the most appealing name. Live mining and hourly
current-registration rewards continue independently of these research jobs.
