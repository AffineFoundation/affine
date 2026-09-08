# Affine operational observatory

23 self-contained panels under `panels/`, organized on the board by process.
Start with `panels/overview.html`; its subsystem buttons ask the owning chat to focus that panel. Expand **Interpretation & monitoring limits** or **Evidence sources** for provenance. Expansion preferences persist through `arbos.state`.

## Layout

- Control: private submissions → queue → validator → duel progress → weights.
- Serving / score: eval engine → teacher swarm → router → scoring → action dialects.
- Data: datagen → traces → daily fold → corpus D → duel history.
- Independent branches: benchmark engine → benchmark results → post-crown audits → public chat.
- Infrastructure: PM2 services → GPU inventory / estimated rates → host vitals.

`ARCHITECTURE.md` maps the implementation to source modules and explains the critical boundaries. Each panel includes a short description of its process stage.

## Running and verification

From the workspace root:

```bash
python3 -B -m ops.monitoring.build_panels
python3 -B -m ops.monitoring.collector            # continuous, read-only
python3 -B -m ops.monitoring.collector --once     # one pass; don't run concurrently
python3 -B -m unittest ops.monitoring.test_monitoring -v
python3 -B -m ops.monitoring.build_panels --check
```

The collector is already running as an arbos background job for this installation. It survives arbos restarts, but is not installed as a boot service. Stop only its own process with SIGTERM; do not restart production services. A directory lock prevents duplicate collectors targeting the same output directory.

| Source | Collection cadence |
|---|---:|
| Validator / weights, scoring, teacher router | 10s after previous pass |
| Eval / bench / chat health | 15s |
| Queue / registrations | 20s |
| Benchmarks / audits | 60s |
| Corpus, datagen SSH / trace manifest, services / fleet | 120s |
| Overview composition | 5s |
| Host system bridge | 1s while panel is open |

Panels poll the derived JSON through `arbos.subscribe('/api/file?path=…')` every 3 seconds. Host vitals use `arbos.system`. No additional HTTP server, CDN, build toolchain, browser credentials or direct browser network access is required. Python workers are independent, so a slow source cannot freeze all panels. Source-call time adds to the cadence. A stopped/hung worker ages out to **STALE / UNKNOWN**; a failed collector publishes an explicit error instead of retaining a green status. A stopped collector makes all its panels stale.

## Safety and interpretation

- Writes are limited to derived JSON under `panels/data/`, a local collector lock, generated HTML and panel preference documents. Production state and contract are never modified.
- Collector JSON contains allowlisted observations, not raw PM2 environments, cloud inventory, private R2 model references or credentials. Health authentication is read in-process and never emitted. Worker SSH probes read bounded process/log metadata only.
- Collection time is **not** source time. Individual panels expose source freshness; historical verdict/benchmark/audit timestamps are not liveness heartbeats.
- **Observed** means that panel's limited checks passed, not proof of end-to-end correctness. **Review** includes expected transitions, missing evidence and cadence warnings, not just outages.
- Weight timestamps are validator-recorded success, not independently verified chain payments. Cloud costs are inventory estimates, not accrued bills.
- Historical registration `queued` does not mean currently waiting. In-flight entries are separate.
- Corpus turn shares differ from expected slice shares, which depend on strata.
- Benchmarks are advisory and never ranked by the duel. RT-7 is still open: crown/score does not establish coding capability. Audits are a separate possible enforcement path.
- Missing data remains Unknown, not zero. Historical chart gaps remain gaps and parameters are not silently filled from today's TOML.

## Verification performed

Offline tests cover publication atomicity, exception redaction, collector key validation, missing inputs, stale overview summaries, missing history values, loading-chat readiness, safe generated JavaScript strings and reproducible panel generation. Live first-pass telemetry and subsequent refreshes were checked. JavaScript syntax checks passed. All 21 subsystem payloads were rendered in Chromium through the shared renderer; overview and chart layouts were inspected visually. Host vitals use the native system bridge. The actual board was checked for panel errors.
