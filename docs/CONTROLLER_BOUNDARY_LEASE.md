# Expiring completed-epoch deployment hold

`python -B -m ops.controller_boundary_lease` is an operator deployment guard.
It watches one exact active epoch under the original controller PID/start ticks,
private process record, workspace and immutable config hash. It does not stop
mining, verification, training or evaluation in that epoch. If the next epoch
already opened, the guard records a missed boundary and leaves it running.

After the target epoch finishes, the guard stops the same controller only if its
state shows the next round, no active epoch, a consistent checkpoint digest and
nondecreasing completed training steps. It rechecks the state bytes after the
observed stop. A race into another epoch causes an immediate resume.

The hold is a lease, never an indefinite deployment barrier. The guard records
its own PID/start ticks and an expiry, and resumes its original controller after
at most 900 seconds. Interruption also releases its own observed hold. It does
not resume another operator's hold, signal a reused PID, restart a job or create
another controller. An actual completed retirement of the original controller
ends the lease without signalling a replacement.

Supply private `--config`, `--controller-process`, `--epoch` and a new private
`--output` directory. `--wait-seconds` is bounded to 1–7200;
`--hold-seconds` is bounded to 1–900. Before relying on the barrier, a deployment
must independently check the lease process is live under its original ticks,
the controller is still stopped under its original ticks, the expiry has not
passed, and the status/config hashes still match. Original completed jobs,
authenticated reports, empty queue and actual idle GPUs remain separate gates.
If qualification exceeds the lease, keep its evidence and wait for a later
completed boundary instead of cancelling the newly opened epoch.

Tests use actual child processes to exercise stop/expiry/resume, interrupted
holds, original process exit, missed boundaries and another owner's hold. Changed
config, original ticks or target epoch refuse before any signal. These checks do
not constitute source, checkpoint or GPU qualification for a deployment.
