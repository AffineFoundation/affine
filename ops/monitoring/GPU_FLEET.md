# SN120 GPU fleet panel

`panels/gpu-fleet.html` is a self-contained, theme-aware Arbos canvas. It shows
per-device compute use, VRAM, temperature, power, driver, P-state and optional
fan / volatile uncorrected ECC counters. Per-pod cards include inventory rates,
root disk space, host RAM, uptime, service observations and diagnostic provenance.
Filters, search and expanded diagnostics persist in the panel document; utilization
sparklines are view-local and deliberately start empty.

## Collector

From the repository root, using the workspace Python (provides `websockets`):

```sh
.venv/bin/python -B -m ops.monitoring.gpu_fleet --gateway http://127.0.0.1:24585
.venv/bin/python -B -m unittest ops.monitoring.test_gpu_fleet ops.monitoring.test_monitoring -v
```

Use the actual **local workspace gateway** if its port changes. `--once` performs
one collection and exits. A lock at `panels/data/.gpu-fleet.lock` prevents duplicate
collectors. The continuous collector is a background job, not a boot service.
SIGTERM stops only this collector. It does not touch production services.

Each pass runs `lium ps --format json`, selecting `affine-*` and `swarm-t-*` pods.
Other account rentals are explicitly excluded. At most 32 pods are observed;
truncation is displayed. Running pods are probed with bounded parallel SSH calls
using the existing validated inventory SSH parser. Remote commands only read
`nvidia-smi` and host `/proc` / disk metadata. No inference, model loading, GPU
stress tests, credentials export or production mutations are performed. Optional
ECC is queried separately so unsupported consumer GPUs do not break core readings.

Eval / bench / chat reuse the existing sanitized loopback health checks; teacher
service state comes from live router circuit eligibility. Datagen service state
counts `python -m rollouts.run` supervisors. Process presence is **not** proof of
rollout progress. Health endpoints describe their configured tunnels separately
from the hardware observation. No raw health payloads, connection addresses,
SSH commands, private model names or credentials enter the panel.

The collector sends `{type: "state_set", panel: <absolute HTML path>,
patch: {telemetry: ...}}` through the existing local `/api/board/ws` document
protocol and requires an acknowledgement. This updates only the `telemetry` key;
it does not overwrite user preferences. The canvas renders from
`arbos.state.subscribe`. It never attempts sandboxed filesystem/network access,
and the collector never edits a `.state.json` sidecar directly.

## Interpretation

- Cadence is 30 seconds **after completion**; source timeouts add to the interval.
- Last samples older than 120 seconds become Unknown, including per-card badges
  and the summary. Collection failures publish an explicit unavailable state.
- Unreachable or missing NVIDIA telemetry is Unknown, not a hardware diagnosis.
- Temperature >=80°C warns; >=90°C is critical. Root disk >=90% used warns;
  >=97% is critical. These are monitoring heuristics, not hardware vendor limits.
- Idle GPUs and high reserved VRAM do not trigger failures. Missing core readings,
  inventory-count mismatches and positive volatile uncorrected ECC counters warn.
- Service failures appear independently of hardware readings. Datagen inactivity
  and unknown application state warn without claiming a GPU failure.
- Costs are running-inventory rate estimates in USD, not invoices or spend.
- Host disk/RAM describe the accessible pod namespace; root disk may not be the
  volume used for model caches. No remediations run automatically.

Verified against live telemetry: 8 project pods / 30 GPUs, document revisions
advance on subsequent samples, 25 unit/regression tests pass, browser rendering
shows 30 device rows, attention filtering works, stale samples remove green
health claims, and the 390px layout has no horizontal overflow.
