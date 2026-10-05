# Prospective bounded miner search and one-pass upload

This changes owned-miner scheduling and transport preparation, not the signed
sampler, per-task attempt namespace, required success/failure quota, proof
contents, grader, or public artifact format. Existing sealed sources retain
their original executions.

The owned worker searches a bounded frontier of at most eight tasks, taking two
attempts per task per sweep. It retains each task's existing class pools and
continues with that task's next original seed. Exhausted or completed pools are
retired. The root-signed job may specify `owned_search_policy` with version
`bounded-round-robin-v1`, `active_tasks` in 1..8 and `dwell_attempts` in 1..2.
This cannot expand the signed maximum attempts. Runtime selection remains lazy;
no additional model is loaded per pool.

Small-commitment mining prepares each newly completed pair directly from its
original arrays into one stable ZIP pass. Framing is byte-identical to the old
canonical pair ZIP: DEFLATED members, timestamp 1980-01-01, Unix creator, mode
0600. There is no initial cumulative ZIP encode/decode and no canonical ZIP
recompression. Already acknowledged immutable pair bytes are reused.

The historical signed cumulative compressed and raw limits still apply. The
uploader checks both the conservative sum of independent pair archives and the
hypothetical cumulative ZIP's exact framing, array compressed sizes and combined
canonical manifest. This handles two-digit slot prefixes without decoding or
recompressing arrays; exceeding either cap prevents all PUTs. A pair PUT is
journaled only after actual success. The signed small commitment is uploaded
last, so a partial failure preserves the previous authoritative snapshot.

The local diagnostic for this transport is `submission-commitment.json`, the
actual uploaded small signed document; legacy transport retains `submission.zip`.
The miner report hashes and sizes the bytes actually submitted.

`mining-progress.json` and logs expose task index, attempt seed/count,
classification, token counts, completed batches and elapsed phase times:
generation, probability computation, TOPLOC, grading, artifact packing and PUT.
They contain no tokens, prompts, probabilities, proofs, capabilities or signing
material. Phase records are operational telemetry, not scientific proof.

The original V9 control found a pair but failed before its first PUT when
redundant serialization exhausted the unchanged upload deadline. It ran
1192.6007 seconds and produced no R2 artifact or accepted verifier result.
The new source requires its own fresh-nonce genuine GPU qualification; CPU tests
alone do not demonstrate its mining throughput.

## External miner client parity

The public `Miner` client uses the same prepared-pair cumulative-budget helper
as the owned GPU worker. Completing a new pair performs one canonical ZIP
compression, reused across subsequent uploads and restarts. Legacy submission
transport continues to use its original cumulative ZIP path.

For small commitments, the normal local state path becomes a private
`prepared-miner-pairs-v1` JSON index; adjacent `<state-name>.pairs/` stores each
immutable artifact under its full SHA256. Startup checks the original epoch,
checkpoint, source and sampling-contract hash, then fully reads and hashes each
referenced artifact and validates tensor framing before reuse. Corrupt or
changed state fails before network uploads. Old cumulative ZIP state is read
and migrated once on upload. No runtime or proof computation is resumed from
unverified local metadata.

The existing acknowledgement journal stays separate and is updated only after
a successful pair PUT. A restart skips acknowledged slots, retries missing
slots using the exact original bytes, and uploads the signed commitment last.
Without a state path, this cache is memory-only; process restart regenerates
its local candidate state. CLI-managed state paths retain their usual location.
Deadline expiry before commitment publication preserves the prior authoritative
commitment. Local cache deletion remains an operator retention decision after
the epoch no longer needs resume or forensic evidence.
