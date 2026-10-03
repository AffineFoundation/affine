# Separate nonpayable GPU roles

The validator/controller remains on the current operator machine (Arbos.life).
A miner, two verifier workers, a trainer and an evaluator run on five separate
GPU endpoints. `gpu_service` retains the signed synchronous state machine:
collect to the original deadline, freeze, verify, score, evaluate before training,
train verified pairs, independently hash and publish the successor, evaluate
after training, then open the next epoch. No role imports a chain weight writer.
The read-only registration lookup still binds the owned miner identity.

`remote.roles` enables `subnet.role_router.RoutedJobs`; the historical single-host
`remote` configuration retains its behavior. `mine`, `train`, and `evaluate` use
individual signed SSH-dispatched `backend_jobs` processes. Each compute role
resolves the signed checkpoint map from R2 on its own host. A miner cache path
is never passed to the trainer or evaluator. Upload is routed to the host that
owns the exact trained output path; the persistent operator ownership record
survives controller recovery. The initial checkpoint owner defaults to `mine`.
The initial source/runtime inventory on each endpoint is checked independently.
All verifier workers must have the same exact reviewed source/package inventory.

Verifiers pull signed jobs from a small coordinator on the operator. SQLite
`BEGIN IMMEDIATE` serializes selection and lease creation across connections and
processes. R2 is an artifact/history store; listing and writing an R2 key is not
used as a lock. Coordinator availability is required for new claims and renewals.
Keep the SQLite database and its journal on local durable disk, within private
operator state; do not place it on object storage or launch distinct databases
for the same queue. The worker roster is operator configuration. Adding another
reviewed worker with its own public identity extends the same queue.

Each worker receives only its own private Ed25519 seed, the operator public
identity, exact reviewed source, its local Python runtime and the coordinator
address. It receives per-object checkpoint and frozen submission GET capabilities
inside the operator-signed job. It receives no chain key, operator signing key,
Cloudflare credential or permanent bucket key. A signed worker request binds its
action, job, lease token and report; request nonces reject replay. The operator
signs replies, preventing a substituted claim through the transport. Use TLS or
SSH forwarding; keep the coordinator bound to loopback. For workers on Lium,
the operator can open a reverse SSH forward to each dedicated host:

```sh
ssh -N -R 127.0.0.1:19081:127.0.0.1:19080 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o UserKnownHostsFile=/private/reviewed-known-hosts \
  -p REVIEWED_PORT root@REVIEWED_HOST
```

Start each verifier in the reviewed source directory, with a distinct private
seed file and workspace:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 /root/miner-venv/bin/python -B \
  -m subnet.distributed_worker --coordinator http://127.0.0.1:19081 \
  --authority OPERATOR_PUBLIC_HEX --seed-file /root/private/verifier.seed \
  --workspace /root/affine-verifier
```

The seed file must be mode 0600. Never place capability URLs in process arguments,
logs or public discovery documents. Coordinator request bodies and signed jobs
are private. Disable access/body logging on any proxy in front of this service.
The coordinator's bound HTTP endpoint is `/request`; it exposes no job enqueue,
status, bucket or operator signing API. Only the local controller enqueues jobs.

Lease expiry allows another registered verifier to retry the unchanged signed
job, with a new token and bounded attempt number. Renewal never extends the
original job expiry. A completion under an old token, an expired lease, a changed
checkpoint/epoch/source/runtime/policy, or a different frozen artifact is rejected.
Exactly identical accepted results are idempotent; conflicting duplicate reports
are rejected. A report's claimed completion must lie inside the original signed
job budget. Observation timeouts retain job identity rather than launching a new
job. Retry workspaces preserve earlier logs and reports; model caches are checked
against the exact file allowlist on every execution.

The operator archives signed jobs, signed worker report requests, exact reports
and lease history under a **private** R2 prefix. Each write uses `If-None-Match: *`;
a preexisting object must match the exact bytes. Objects are immutable and are
never used to claim work. Normal publicly signed audit/score/checkpoint records
remain the existing controller's responsibility. Worker authentication proves
which admitted operator-managed worker returned the report. It does not prove
that a GPU executed the calculation or that a miner originally sampled tokens.
These are trusted numerical runtime reports, not cryptographic sampling proofs.

On Arbos.life, every retained Lium rental must also be registered with the
existing pod reaper **before renting** its unique pod name. The reaper releases
unregistered Affine pods after 90 minutes, regardless of active jobs or provider
TTL settings. Use the existing registry's locked update path; do not stop the
shared reaper or rewrite its registry file directly:

```sh
/home/const/subnet120/.venv/bin/python /home/const/subnet120/ops/pods/registry.py \
  register UNIQUE_POD_NAME --purpose separated_hopper_math_retained \
  --owner manual:authorized-hopper-pilot --hours 0 --price HOURLY_PRICE
```

Rent with that exact name, save the actual provider UUID, and update the same
registry record with the UUID and role. Confirm an explicit manual owner,
`expected_hours = 0`, no release request and no released marker. If rental fails,
retain an accurate failed-attempt receipt; never use an unrelated node's identity.
Provider inventory and exact remote process checks still determine runtime
health; registry ownership establishes retention, not successful computation.

For a bounded qualification, use a fresh nonpayable namespace and new reviewed
source bundle. First run two frozen verification jobs concurrently and establish
that the two worker identities claim different jobs. Preserve their exact reports,
lease records and R2 hash checks. Then run one small complete synchronous epoch,
including genuine positive/negative mining, verification, a full optimizer update,
independent successor hash checks and paired fixed heldout evaluation. Establish
H200 numerical honest/tamper controls for the exact selected 7–8B model, tokenizer,
source and package profile before admitting its real mining. A 1.7B numerical
control alone does not qualify the final larger launch model. Failed or late
attempts remain evidence; issue a separately signed attempt rather than extending
an existing manifest or lease. New qualification must not mutate the retained
3090 pilot, research dispatcher guards or historical immutable worker bundles.

Tests: `python -m unittest discover -s tests -p 'test_distributed_roles.py'` and
`test_role_router.py`. The first exercises actual SQLite claim contention,
independent coordinator connections, two-worker scaling, lease expiry/renewal,
bounded retries, replay rejection, forged bindings, duplicate results, HTTP
signature checks and conditional R2 history creation.

The prospective routing overlay for the actual five rentals is private at
`state/hopper-node-preparation/distributed-routing.prospective.private.json`.
The two verifier seed files are adjacent and never committed. Merge this overlay
with the separately reviewed math checkpoint/environment/source configuration;
it is not a complete approved launch config. `ops/provision_distributed_verifiers.py`
validates roster/key matches without network actions by default:

```sh
.venv/bin/python ops/provision_distributed_verifiers.py \
  --config state/hopper-node-preparation/distributed-routing.prospective.private.json \
  --seed-dir state/hopper-node-preparation
```

After the immutable source and final model/runtime controls pass, the operator
may run the same command with `--provision` to copy only these two narrow seeds,
and with `--start --authority NEW_CONTROLLER_PUBLIC_HEX` to establish reverse
SSH forwards and start verifier workers. The helper preserves operator SSH trust,
records exact process identities, reuses live matching workers/tunnels and never
signals or replaces existing processes. Start the new controller/coordinator
before the workers. These flags do not deploy source or approve numerical controls.
Use the new namespace's controller authority; never use chain signing keys.
Select the actual seed namespace explicitly when using an existing deployment:

```sh
.venv/bin/python -B -m ops.provision_distributed_verifiers \
  --config /private/current-controller.json --seed-dir /private/current-worker-seeds \
  --remote-seed-file /root/current-deployment/private/verifier.seed \
  --start --authority CURRENT_CONTROLLER_PUBLIC_HEX
```

The legacy default seed path is not evidence that a worker has the current
deployment's identity. Before starting any worker, the helper checks that its
actual canonical, private remote seed derives the public identity approved in
the controller roster. A mismatch fails before worker startup. It does not
overwrite an existing seed or relax coordinator authentication. Use `--provision`
only to copy the corresponding narrow key into a reviewed new namespace.
The startup helper waits up to 30 seconds for the operator coordinator to listen
before launching any workers. If it times out, start or inspect the controller
and retry the helper; it does not start workers against an absent coordinator.
