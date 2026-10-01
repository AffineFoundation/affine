# Original Calendar balanced terminal controls

This new control preserves the published six-turn v2 controls and the v3
broker source. It uses original Calendar task 54, original selected tools,
original SQL grading, and the disclosed pinned clock/UUID/SQLite runtime.
The operator fixture and seed remain private.

The positive public policy moves the events to the requested Building 1.
The new negative policy changes only the destination to Building 2. Their
first three native tool events are identical; the fourth `patch_event` is the
first divergence under the same complete public context. Destination strings
have equal character length. The pinned Qwen tokenizer measured 83 output tokens for each fourth-turn
candidate. This is a length-balanced curated policy, not unrestricted sampling.

Run `.venv/bin/python -m ops.probe_native_eog_balanced_terminal`. Both policies
execute six actual original tools, seal with the actor's terminal `finish`,
then replay in fresh brokers. The original rewards are 1 and 0 respectively.
Reports are under `state/native-eog-balanced-terminal-v3`; operator reports
are mode 0600. Public model-input exports include only the original public
messages, full selected schemas, tool actions and observations, claimed
reward, and runtime/source/seed hashes.

`subnet/native_eog_terminal_admission.py` requires an independently authorized
model audit with all six complete tool turns and a seventh curated `DONE`
response computed from the complete post-tool history. The terminal record
binds the exact harness action, complete messages and empty observation hash.
Native admission independently replays the tools and original terminal grade.
The `DONE` control is an explicitly curated target-model computation, not a
claim the model originally sampled the response or that a miner epoch ran.

Five unit tests exercise synthetic authenticated receipts, including missing,
forged and incorrect terminal records. These unit controls do not replace real
GPU inference. The real GPU controls now pass all seven full probability/TOPLOC checks
against checkpoint `426c3ddeaacc849d1792b5fb22d8c3fc37f4aefabfa2ef7627dd82e13864ccd7`.
Each artifact carries 182,323,200 bytes of full-vocabulary float32 log probabilities
before compression. Separate fresh native admission replays six original tools
and verifies the sealed terminal reward: positive 1, negative 0. Independent
inspection authenticates signed model receipts and approved source jobs, hashes
all actual artifacts, and checks complete-context bindings. Native replay reports
are operator-collected and unsigned; they are not cryptographic execution proofs.
Common Calendar epoch/training integration and held-out quality improvement
remain unproven. No blockchain transactions are made.
