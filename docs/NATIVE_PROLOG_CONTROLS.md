# Original Prolog native controls

`python -m ops.probe_native_prolog --out state/native-prolog-controls/new-run`
uses the cached SWI-Prolog image and exact original setup shim. It creates only
owned, nonroot containers with no network, host mounts or writable root filesystem.
The public actor receives original messages and starter facts. The operator retains
private task metadata and invokes the original grader and answer verifier.

Three original medium NQueens fixtures (indices 5, 14 and 23) passed public
CLPFD positive controls, invalid-column negative controls, and exact fresh native
replay. Indices 5 and 14 both describe the same size-12 board; index 23 describes a
size-11 board. These are **two problem geometries**, with no training/heldout split.
They do not establish independent learning examples or coverage of the other eight
original Prolog CSP types.

Fresh infinite-output and infinite-loop commands were rejected by output and wall
clock bounds, and their exact owned containers were removed. Cleanup releases the
exec client's pipes before removing its container; unsuccessful removal remains an
explicit error rather than claiming a closed actor.

Evidence: `state/native-prolog-controls/v1-qualified-retry/controls.json`.
The earlier failed cleanup run remains in `v1-qualified`. This prerequisite has no
model inference, TOPLOC proof, common epoch, checkpoint update or proposed score.
The shorter negative command also needs a balanced candidate policy before any
model-sampled preference batch qualification.

Root independently reran the same frozen source in fresh actors:
`state/native-prolog-controls/root-independent-v1/controls.json`. All three
original task hashes, public descriptors, positive/negative outcomes and exact
fresh observations match the earlier qualified run. Both hostile controls again
removed their owned actors; six focused runtime/policy controls also pass.

A subsequent prospective public policy keeps the full CLPFD program and changes
only its diagonal constraint, rather than using a shorter invalid program. Root
checked the actual R2 tokenizer bytes of approved shared checkpoint
`0081b0698c0ccc103edfca0506a2c60aeaf9d0a521716889e6a51fdef0a6513b`
against its pinned file digest. All three native fixtures yield 340/340 command
tokens and 379/379 compact tool-call JSON tokens. Exact candidate bytes and public
bindings are in `state/native-prolog-controls/root-wide-tokenizer-candidate-check.json`.
This is a tokenizer prerequisite; model-sampled K/L, TOPLOC proofs and common
training remain unqualified.

A prospective session is available in `subnet.native_prolog_session` and a public
candidate policy in `subnet.native_prolog_public_policy`. Fresh Docker conformance
passed both original grades and exact two-turn observation replay for all three
fixtures (`state/native-prolog-session/native-v2-balanced/controls.json`). The full
programs share public facts and differ in one diagonal constraint, with harmless
whitespace chosen for equal token lengths. Root checked the actual approved
1.7B checkpoint tokenizer SHA: compact JSON actions have 379 tokens each at all
three fixtures. This establishes candidate comparability, not sampled K/L success.
Five session trust/state/terminal-grade controls and two public-policy controls
pass. The session is deliberately absent from the active dispatcher; admitting it
requires a new signed source, qualified deployment image and fresh model proof
search plus independent model/native replay. Active sixteen-family pilot bytes
are unchanged.

Root repeated that frozen session in fresh owned actors. Original public task
bindings, two-turn observations, native grades and replay results exactly match
the qualification report. Fresh Docker builds produced different image IDs and
therefore different environment source hashes; each report preserves its own
runtime pin. This does not establish identical deployment images. Evidence:
`state/native-prolog-session/root-native-v2-balanced/controls.json`.

## Isolated common-model qualification probe

`ops/probe_native_prolog_model.py` runs a bounded candidate search followed by a
separate model reload and verification. It requires an operator-signed plan with
the exact checkpoint, complete module inventory, probe and public-policy hashes,
native runtime image, two-turn harness and artifact limits. Verification binds
the frozen ZIP, environment index, actual retained positive/negative classes and
all probability arrays; it does not trust the search report's K/L counts alone.
Seven controls cover changed candidates, incomplete output budgets,
checkpoint/index changes and altered classification or sample metadata.

Run from the exact approved isolated source directory, using the operator's plan
and trusted public key:

```sh
python -m ops.probe_native_prolog_model --plan /path/to/plan.json \
  --authority TRUSTED_PUBLIC_KEY --out /path/to/new-proof-search
python -m ops.probe_native_prolog_model --plan /path/to/plan.json \
  --authority TRUSTED_PUBLIC_KEY --out /path/to/new-proof-search --verify
```

The reviewed qualification source archive is
`c5390abe76933ebbe319d82459ecbc1c730cde58314dabac4fe373c1eb87d581`
with 51 pinned modules. Its approved native image is
`sha256:05a468b3b41348dd8534532f536475b70b70bae1aa0b876c8a626a7af98cad08`.
Fresh remote CPU controls passed positive/negative original grades and exact
replay for all three selected fixtures. Root checked the signed source plan,
archive bytes, module digests and received control report:
`state/native-prolog-model-qualification/root-signed-source-native-control-check.json`.
This is still a prerequisite: model-proof qualification and a common training
epoch have not completed. The public starter candidate policy is curated;
successful proof verification would not establish autonomous program discovery.
The active sixteen-family source remains unchanged.

The first remote model probe exited before producing a rollout: its terminal
turn supplied only `Done`, while the common candidate harness requires at least
two candidates. That source, signed plan, exit and private failure trace are
preserved. The revised `original-nqueens-common-model-search-v2-terminal` plan
uses `Done` and `Finished` and validates the common harness before loading the
model. Four fresh native controls confirmed that both terminal responses retain
the same original positive/negative outcomes and observations after the tool
action. They are not model-proof results.

The revised archive is
`7b5b74409d6bfb2efdf53d28743f5d03854509a11c43e72a35cab7416033544e`.
All 51 compute modules and the approved checkpoint, native image and artifact
bounds are unchanged; only the qualification probe contract changed. Root's
review is recorded in
`state/native-prolog-model-qualification-v2/root-v2-signed-source-native-control-check.json`.
Actual model generation, independent proof verification and common training
remain separate requirements.
