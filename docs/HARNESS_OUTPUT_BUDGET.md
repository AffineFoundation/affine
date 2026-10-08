# Longer math rollouts

The October 8 controller update increases the math mining output budget from
1,024 to 2,048 tokens at a prospective epoch boundary. Existing manifests remain
immutable. The first longer epoch requires actual successor calibration for its
checkpoint, runtime, and normalized harness; a shorter-budget calibration cannot
be reused for the longer harness.

Miners must read `environments[*].harness.max_output_tokens` from each signed
manifest. Do not hard-code either budget. The existing `text-tools-long-v2`
harness, verifier, and calibration request validator support 2,048 tokens. The
model backend separately limits prompt plus generated output to 8,192 tokens or
the model's smaller configured context limit. The active
Qwen2.5-Math-7B-Instruct checkpoint's authenticated config has a 4,096-token total
context, so a 4,096-token output allowance would leave no room for the prompt.

The four-success/four-failure quota, three batches per UID, allowed nonces
0–999, sampling checks, penalties, and optimizer remain unchanged. Held-out
evaluation retains its existing 1,024-token budget so existing measurements stay
comparable; a longer-budget evaluation must be reported as a separate experiment.

The controller's signed output-budget grant pins both complete configs, the
first eligible round, and the exact old/new budgets. Its validator rejects any
unrelated configuration change. A boundary wrapper retains the prior contract
for a current epoch and supplies the longer contract only to later openings.

The recovery launcher can additionally admit an existing signed, bounded
same-epoch capture-recovery document when coordinator downtime exhausts the
original capture window. It authenticates the original public manifest and
leaves miner upload eligibility unchanged. It does not skip submission discovery,
manufacture audit evidence, or discard the current epoch to accelerate cutover.

This change permits longer reasoning but is not evidence of better learning.
The inspected garbled tails can begin before the old cutoff. Measure solve rate,
truncation, malformed responses, and runtime before attributing gains to length.
Budgets above 2,048 require coordinated harness/calibration support and resource
qualification rather than changing a manifest alone.
