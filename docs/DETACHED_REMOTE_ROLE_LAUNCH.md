# Detached remote role launch

The E15 remote trainer survived while the local controller failed: its SSH
launch call waited 1,800 seconds despite the original trainer running. A
background shell command is insufficient as a reliable transport boundary.

The prospective dispatcher launches a short, foreground Python bootstrap.
It creates an exclusive per-job dispatch-attempt marker, then starts the
unchanged remote runner in a new session, with stdin disconnected, stdout and
stderr redirected to the private runner log, and other descriptors closed.
The SSH call has a 30-second observation limit. A lost reply triggers a probe
of the same original job; it never issues a second launch. Unknown observation
and confirmed terminal failure remain distinct. Existing original jobs still
resume through their signed request and durable process markers.

Five controls cover a real local transport process returning while its detached
child remains live, exclusive duplicate prevention, lost-reply observation,
unknown observation, terminal failure and transport-error preservation. The
existing 24 remote job/report controls also pass. This does not claim GPU,
sampling or model-training qualification. Activation requires the operator
launch scope to pin this controller overlay; running E15 files are unchanged.
