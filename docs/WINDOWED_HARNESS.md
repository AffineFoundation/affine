# Versioned visible-history window

An original six-hop Wikispeedia task exposed a context-budget failure that the
all-good and all-bad controls missed. Taking a bad link midway through a rollout
can repeat a large available-link observation. The full-history harness reached
8,084 prompt tokens; reserving the signed 256-token output budget exceeds the
pinned model's 8,192-token limit. The original failed audit and partial trace
remain under `state/native-wikispeedia-candidate-cohort20-v1`.

The new `text-tools-window-v1` harness keeps a signed number of initial task
messages and recent complete messages. Its defaults are two prefix messages and
two recent messages. It never truncates an observation's text. An individual
task or observation that still exceeds the model context remains a failure.
The existing full-history harnesses retain their rendering behavior.

Example configuration, normalized before signing a new challenge:

```json
{
  "version": "text-tools-window-v1",
  "history_prefix_messages": 2,
  "history_window_messages": 2,
  "policy": "autoregressive",
  "max_output_tokens": 256,
  "temperature": 0.7,
  "top_p": 1.0
}
```

The full trajectory remains in the artifact. `Runtime.verify` reconstructs the
same visible prompt at every turn, verifies model probabilities and TOPLOC, and
replays every original action and observation, including those no longer visible
to the model. A focused verifier test rejects a forged old observation after it
has fallen outside the history window. Its model/proof calculations are mocked;
it tests native observation binding, not cryptographic inference verification.

The actual original Wikispeedia cohort was audited across all 384 combinations
of its public navigation choices: twenty successful paths and 364 unsuccessful
ones. Both full-history harnesses had eight overflowing turn contexts. The
windowed harness had none, with a maximum 1,895-token prompt. Both final ordinary
reply candidates were checked against the output token budget; native execution
used `Done.`. This audit executes original tools and the original grader and
uses hash-checked tokenizer files from checkpoint
`0081b0698c0ccc103edfca0506a2c60aeaf9d0a521716889e6a51fdef0a6513b`.
It performs no model inference, TOPLOC generation or optimizer step.

The candidate search is explicitly curated from public graph data. These checks
do not establish autonomous model navigation, remote proof acceptance or common
training. A subsequent GPU experiment must approve the new source bundle and
harness version; no historical manifest or active GPU source was changed.

Reproduce with a fresh output directory:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/python -m ops.probe_wikispeedia_mixed_paths \
  --qualified-state state/native-wikispeedia-candidate-cohort20-v1 \
  --tokenizer state/native-wikispeedia-tokenizer-preflight/0081b0698c0ccc103edfca0506a2c60aeaf9d0a521716889e6a51fdef0a6513b \
  --state state/my-wikispeedia-mixed-audit
.venv/bin/python -m unittest discover -s tests -p test_window_harness.py
```

The two original public SNAP resource archives were also uploaded to immutable
content-addressed R2 locations and independently streamed back, checking all
45,745,970 bytes. The resource receipt is
`state/native-wikispeedia-candidate-cohort20-v1/root-direct-r2-resource-publication.json`.
Role-local hydration must independently verify the approved archive and all
extracted file digests before importing the original provider.
