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
