# Private native-grader startup reuse v2

Research-only, default off. No live worker, contract, numerical experiment or
public documentation is changed by this package. ROOT owns executable scopes.

This copies the prior qualified `job_scoped_native_math_grader` engine, SHA
`1553cf6954d154f9ce73a9d7801dcfc2f3d3aade2272e3f5360faab9fc2cd435`, into a new
private version. The original native `verify.py` is loaded without editing its
prefix or grading tail. Trusted imports and closure authentication are reused;
answers and outcomes are never cached. Each grading call forks a fresh disposable
CPU child from a separately exec'd, single-threaded, CUDA-free parent. The GPU
caller never forks, even when constructing this client after model loading.

## Integration

```python
from research.native_math_reuse_v2.broker import BrokerClient

with BrokerClient(signed_capability, authority=ROOT_AUTHORITY,
                  job_id=job_id, environment_sha256=environment_sha256) as grader:
    ready = grader.preflight(index=task_index, task_hash=task_hash, answer=answer)
    # 0.0 / reference_unparseable skips the entire predeclared task slot.
    # Only 1.0 allows generation. There is no replacement candidate or nonce.
    verdict = grader.grade(index=task_index, task_hash=task_hash, reply=reply)
    # Exactly '1.0' or '0.0', preserving original worker semantics.
```

The caller retains existing completion admission: budget-capped trajectories and
completed answers without a final box are unresolved, are not sent for a negative
grade, and remain in complete-group records. Infrastructure exceptions abort the
original experiment, never becoming a wrong answer or triggering regeneration.
The sole special failure is native `grader_reference_unparseable` during reference
preflight, which consumes that frozen candidate slot as an explicit skip.

`preflight` obtains the gold answer from the signed snapshot; its optional `answer`
argument must match and cannot replace the reference. `grade` binds each request
to the signed task index/hash, job and increasing request sequence.

`last_result` is the unchanged native result including return code, stdout/stderr,
fresh child PID and child time. `observations` contains one startup timer and each
whole-call versus native-child timer. Persist these in the worker's original log.
There are no model calls, optimizer hooks, remote services, automatic restarts,
capability renewal or retries. Use `finally: grader.close()` if not using `with`.

## Capability

Signed envelope: `{payload, signer, signature}`; signature is Ed25519 over sorted,
compact JSON with NaN forbidden, signer must equal the caller's pinned authority.
The new payload version is `private-job-scoped-native-math-grader-v2`. All fields
are required; unknown fields are rejected by the CPU engine:

```text
version, execute_allowed=true, job_id, created_at, expires_at,
source_root, source_files, asset_files, grader_path, grader_sha256,
interpreter, broker_interpreter, broker_interpreter_sha256,
environment_sha256, snapshot_path, snapshot_sha256,
tasks, max_requests, outer_timeout_seconds, memory_bytes,
native_runtime_binding, cpu_soft_seconds, cpu_hard_seconds
```

- Lifetime comes from the experiment scope and cannot exceed seven hours.
- At most 768 fixed TRAIN task entries and 6,912 calls, covering reference plus
  eight completions per candidate across sixteen frozen 48-task blocks.
- CPU limits remain exactly **8:9 seconds**, wall timeout exactly **15 seconds**,
  address space exactly **768 MiB**, and core dumps off. The old prototype tied CPU
  limits to wall time; this new version separates them and preserves the original
  experiment's limits. Trusted CPU parent address space remains bounded to 1 GiB.
- `source_files` is the full `source_root` inventory, relative names -> SHA256,
  excluding `.git`, `__pycache__`, `.pyc`. It includes both this engine and broker,
  `subnet/native_math_grader.py` and unchanged native `verify.py`. Own broker and
  engine files must be covered. Absolute immutable assets live in `asset_files`.
- Snapshot is the original list of task rows. `tasks[str(original_index)]` has
  exactly `{task_hash, row_sha256}`; row SHA is canonical JSON. Selected task
  references must correspond to the frozen TRAIN schedule, not eval tasks.
- Grader path must be byte-identical to
  `source_root/subnet/vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1/verify.py`.
- Runtime binding is SHA256 of canonical native `RUNTIME_LOCK`. The isolated
  `interpreter` authenticates the unchanged grader dependency/interpreter closure.
  It can remain the original math-only UV environment, without PyNaCl.
- `broker_interpreter` is the experiment's prepared main runtime, which includes
  PyNaCl, with its resolved executable SHA pinned in `broker_interpreter_sha256`.
  It runs isolated with site startup off and imports no Torch or CUDA code. Its
  authentication dependencies remain under the experiment's main-runtime trust
  boundary; the mathematical child still uses only the original native closure.

Source, asset, directory and native runtime file identities are rechecked before
and after each child. Source changes invalidate results. Timeout, signal, malformed
output, exhausted scope or mismatched identity closes the service and fails.
Normal close reaps the native parent. Parent-death signals stop broker, native
parent and its current child if an owner dies unexpectedly; this does not claim
that SIGKILL allows an application to write a final receipt.

## Verification

Run `.venv/bin/python -B -m unittest research.native_math_reuse_v2.test_broker -v`.
CPU controls compare actual unchanged native grading on arithmetic, wrong answers,
fraction equivalence and last-box semantics, then exercise limits, mutation,
reference/task substitution, malformed output, timeout and exhaustion. Test keys
are ephemeral and cannot authorize production or GPU work. The local broker fixture
uses symlinks to approved installed dependencies; its native child uses the actual
original math-only interpreter without PyNaCl or runtime modification.
Current-host speed measurements and independent review are required before a
successor launch; these tests establish no GRPO efficacy or H200 speed claim.
