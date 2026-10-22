# Live SN120 runtime layers, snapshot of 2026-10-22

This is what was running on the Arbos validator host on 2026-10-22 on top of the
committed release line. The base of this branch, `a7bc7f77`
(`release/mimo-v8.2.1-trainer-restore`), is byte-identical to the live lifecycle
deployment source (`release-v822-signed/deployment-final-source-v822`). Each
later release, v8.2.3 to v8.2.10, was signed and activated as stand-alone
launcher or trainer-host files outside git. Those files are copied here verbatim
so the history is reviewable. They are reference copies, not imported modules.

| Folder | Live unit | What it is |
|---|---|---|
| `lifecycle/v822`, `v822-writer-accel` | affine-mimo-lifecycle, affine-live-reward-writer (v8.2.2 drop-ins) | v8.2.2 signing and activation scripts, writer population cache |
| `lifecycle/v823`, `lifecycle-and-trainer/v824`, `lifecycle/v826`, `v827`, `v828` | affine-mimo-lifecycle (the v828 drop-in is the active one) | UID capacity, capture cache and deadline, incremental capture, audit off the critical path |
| `trainer/v825`, `trainer/v8210` | affine-mimo-trainer-supervised (pod) | Host trainer: batching, dropping unresolved groups, async optimizer save (v8.2.10 is live) |
| `trainer/v829-profile` | none | Profiling notes and scripts behind v8.2.9 and v8.2.10 |
| `scorer-writer-v9` | sn120-impact-scorer-v9 timer, live reward writer (`v9-live-weights` drop-in) | v9 measured-impact scorer, hourly writer entry, signer, public scores. `pod-code/v9agg.py` is the seq7 version that runs on the scorer pod |
| `scorer-writer-v9/seq7-window-credit` | none | seq7 window-credit payout fix (live since 2026-10-22, 12:44 PM CR) |
| `scorer-writer-v9/seq6-score-ema` | none | seq6 EMA change, superseded by seq7 |
| `writer/time-budget-v2` | writer (v7e drop-in, superseded by v9) | Writer time budget |
| `mech1-weights` | sn120-mech1-weights timer | Mechanism-1 (Tobias track) weights setter |
| `env-probe` | affine-env-probe | Environment probe daemon and pod code |
| `mining-discovery/observer-v828` | affine-mining-discovery | Public mining.json observer |
| `weights-check` | affine-weight-freshness-monitor | Weight freshness and reveal checks |
| `dashboard/*` | none | Dashboard publication patches (v8.2.4, succession) |

Left out on purpose: every `*.private*` config, signed release JSON, host
compositions, seeds and key files, logs, sqlite state, backups, replay data,
per-miner data and staged duplicates. The authority seed is read at runtime
from a path in a private config and is not in this tree. The 64-hex `AUTH`
constants are the public authority verify key, which is already in `main`.
