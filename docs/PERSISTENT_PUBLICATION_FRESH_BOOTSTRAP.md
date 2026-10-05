# Persistent publication bootstrap admission

The original E13 trainer child exited before GPU/model loading: signed job
validation calls `persistent_training_protocol.validate_job`, which imports the
pure `persistent_publication.export_policy` helper. The fresh-source installer
subsequently rejected this validation-induced import as preloaded execution.
This was bootstrap infrastructure failure, not failed sampling, grading or
optimizer computation; the original failure remains immutable.

The prospective correction records whether publication was loaded before signed
job admission. Only a module newly loaded by that admission can be discarded,
and only after every signed source SHA and runtime pin passes and the publication
execution pin is explicitly present. The helper is then reimported from the
verified source using the bytecode-independent finder. Publication preloaded
before admission and unrelated execution imports still fail closed. Failure
messages name the offending module so infrastructure diagnosis does not require
loosening guards. Historical execution without the signed publication pin does
not gain this bootstrap exception.

The regression uses a complete signed v4 lineage/receipt fixture in genuinely
fresh subprocesses. Honest validation reaches the checkpoint boundary with the
fresh publication loader, while prior publication and optimizer imports are
rejected before checkpoint access. These are CPU controls; no scientific GPU
qualification, original-job restart or live source modification is claimed.
