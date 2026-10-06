# Source-aware independent evaluator

The optional `evaluation_source_routes` config names a ROOT-signed document of
version `independent-evaluator-source-routes-v1`. Historical configuration stays
unchanged when this field is absent. This operator scheduler change adds no
training prerequisite and changes no heldout task, seed, sampler or grader.

The payload contains `physical_id` and `sources`, keyed by the exact SHA256 in
the original signed generation manifest's source bundle. Each row contains
`endpoint`, `local_source_path`, `source_files`, `runtime_versions`, and boolean
`new_dispatch_approved`. Every row must address the same SSH host/port/user and
original evaluator workspace. The approved complete local Python inventory and
actual remote inventory/runtime versions must match exactly. Unknown source
hashes are rejected. CPU-qualified future routes may set dispatch approval false:
they can adopt an already-issued authenticated original request, but cannot
issue another GPU job.

Each source imports its own historical CPU admission package under an isolated
module namespace. Original job signatures, complete manifests, heldout fields,
source maps, runtime versions, role labels and ledgers stay unchanged. The only
shared transport is the reviewed detached `launch_runner`. For older inline
nohup implementations, an exact three-statement launch block is replaced in a
local CPU function copy; unfamiliar launch shapes fail closed. Worker source
archives remain immutable. Historical timeout types are normalized without
changing the original remote job identity.

Before new dispatch, all authenticated evaluator ledgers are checked for
physical liveness. A live job or unknown status blocks a conflicting launch;
SSH errors cannot grant permission. The existing process lock still reserves
one scheduler. Mapping approval is prospective authority, not evidence that a
future source or checkpoint has executed successfully.

Validation includes two distinct isolated CPU package ABIs, actual retained6a
launch-block compatibility, signed-map/job tampering, source/hash and runtime
mismatch, unchanged heldout seeds, unknown source/physical status, source-tree
symlinks, unapproved future dispatch, timeout adoption and conflicting live jobs.
No GPU qualification or deployment is asserted by these CPU controls.

An opt-in `evaluation_dispatch_order: latest-approved-source-first-v1` prioritizes
issued originals before newest unissued checkpoints. Sources with
`new_dispatch_approved: false` retain unissued requests unchanged; a separate
`checkpoint-evaluation-deferrals` record states that dispatch approval is pending.
This does not create a remote-job observation or a completed score. Already issued
requests still require original signature, source, heldout and physical-liveness
checks and are adopted without replacement.

A signed route endpoint may specify positive integer
`evaluation_min_free_disk_bytes`. Before a new job is issued, a bounded CPU-only
filesystem probe must meet this threshold; insufficient capacity defers the same
request. It never blocks adoption of an already issued original. Deployments can
approve only sources with authenticated automatic evaluator retention and defer
older unissued sources until owned-cache retention is qualified. Historical
requests, reports and heldout cohorts remain unchanged. Defaults retain the
original oldest-first behavior.
