# Pinned MiMo trainer source

Unmodified selected files from XiaomiMiMo/verl commit
`a2ad9f6160b03ff2d47e59832bfb6b289f37c917`. `PROVENANCE.json` records each file's
SHA256. The upstream Apache-2.0 license and copyright notices are retained.

`subnet.mimo_grpo.upstream` loads the exact advantage, prompt-normalization,
aggregation and PPO loss function bodies from this source. It strips only
registration decorators and supplies small import/config adapters. It does not
install a second `verl` package or load unrelated agent, Ray and Megatron code.

Affine uses eight complete trajectories per task and up to 512 task groups;
the released reference recipe uses sixteen and 64. Local CPU FP32 AdamW,
termination policy and checkpoint lifecycle are explicit integration adaptations.
