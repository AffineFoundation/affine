# Affine public network dashboard

Public site: https://affine.io. Monochrome layout follows the operator mockup: wordmark, full-screen chart, then16×16 UID activity grid. The default chart is environment evaluation reward in Pilot tests; the metric selector also offers batches per epoch. Scope selection applies to both evaluation curves and activity counts, keeping nonpayable trials separate from live network results. Batches are frozen submitted batches, including rejected batches; accepted totals are available in chart tooltips. Only finalized epochs enter the batch chart. Grid counts use recorded UID mappings; mock identities never receive invented UIDs. Grid covers0–255, with unmapped/out-of-range contributions labelled separately.

Backend: Python stdlib HTTP server plus SQLite on the existing Affine host. `affine-network-dashboard.service` refreshes an allowlisted projection every15seconds and atomically exports `network-data.json` into the existing Affine website. No Arbos domain/server or DNS/Namecheap changes were made. No wallet, seed, encrypted upload capability, submission token or private rollout is exported. No chain transactions are performed by this dashboard.

For sampled audits, the public projection distinguishes accepted, rejected, and unchecked batches. A structurally valid batch omitted from the expensive audit is unchecked, not rejected. All three categories contribute to submitted batch counts and the UID grid; only the verifier's accepted batches contribute to accepted totals. `dashboard.test_server` includes a sampled audit regression covering these distinctions.

Source: dashboard/public/ and dashboard/server.py. Runtime database: state/dashboard/network.sqlite. Deployed static files: /home/const/subnet120/affine/website/. Original index and overwritten files backed up at state/dashboard/deployment-20260930T194811Z/. Existing unrelated API routes and legacy asset files were preserved. Fonts: Haffer Regular and DM Mono, matching the current Bittensor website font families.

Checks: `python3 -m unittest dashboard.test_server`; `node --check dashboard/public/network.js`. Desktop/mobile browser checks verified256cells, no JS errors, no390px horizontal overflow, pilot UID131 count1, interactive selection, and successful public HTTPS asset/data retrieval. Public page was independently opened after deployment.

Rollback: stop/disable affine-network-dashboard.service; restore index.html from the private deployment backup. Added network.* files may be removed using files.json if desired. Old unrelated site files remain intact. No validator restart is needed.

Environment performance controls deployed September 30 at 21:11 UTC. Select Evaluation reward and an environment to view actual completed held-out evaluations from state/evaluations/*.json. Curves compare the latest fixed dataset, harness and environment version only; incomplete evaluations do not enter the curve. No improvement is assumed or synthesized. SQLite retains evaluation history if an input file is temporarily removed. Per-epoch training metrics and multi-environment UID mappings are projected from state/multi-environment. The update was checked on public HTTPS in Chrome at 390px: 256 cells, no overflow or JavaScript errors, and an explicit empty evaluation state until real records arrive. Backup for this deployment: state/dashboard/deployment-20260930T211118Z/.

The first completed remote epoch produced two comparable Verbatim evaluation points, both reward1.0 under a curated-copy policy, and UID131 batchcount1. Public mobile checks passed again with real before/after data. The chart labels curated policies explicitly; no autonomous solving or improvement is inferred from this control.

`affine-eval-suite.service` runs independently of mining/training, watching completed controller checkpoint evidence. Its config is state/multi-environment/eval-suite.json. Ten original environments have four pinned original tasks each: training ordinals0,1; held-out ordinals2,3. The worker generates unconstrained16-token samples and verifies their computations/outcomes for each checkpoint. Exact checkpoint filemaps must hash to the controller's immutable commitment; unchanged task/harness/profile definitions use the same comparison group. Failed evaluations persist as error records and never become chart points. Individual run IDs pin checkpoint and suite settings, so old records survive future suite changes. This is a small evaluation budget, not evidence of statistically significant gains.

Current host dedicates cores4–7 to this evaluator while the training runner uses0–3. Both use the same pinned four-thread numerical profile. Adjust service CPUAffinity when moving to a smaller host. Start/stop with `systemctl --user start affine-eval-suite.service` / `systemctl --user stop affine-eval-suite.service`; inspect its journal with `journalctl --user -u affine-eval-suite.service`. Stopping this worker leaves training, mining and production services running. Test checkpoint selection and setting-pinned record IDs with `.venv/bin/python -m unittest discover -s tests -p test_eval_suite.py`.

At22:01UTC, ten original environments each have completed baseline and first-trained-checkpoint evaluation records under the bounded TOPLOC runtime revision. Each checkpoint has two fixed held-out samples per environment. All ten comparisons are currently reward0→0; the dashboard reflects these flat results. Independent checks of three signed epoch score/training reports, complete checkpoint file hashes, matched held-out tasks, and nonpayable payout filtering are recorded in state/multi-environment/three-epoch-independent-check.json. The evaluation worker resumed at22:10UTC after the version-pinned adapter update; historical evaluation groups remain intact and the new source revision gets its own comparison group.

Failed evaluations retry at most three attempts with at least300seconds between attempts. Set `max_evaluation_attempts` (1–10) and `error_retry_interval` (at least30seconds) in the worker config to adjust that budget. Exact attempt records remain under state/evaluations/attempts/; only the latest result for a logical run enters the dashboard projection. Completed evaluations are not repeated. Unapproved checkpoint hashes are rejected for every environment, including environments after an earlier validation failure.

The Oolong 1.7B experiment adds a separate CUDA evaluation group. Import its operator-owned report with `.venv/bin/python -m ops.import_gpu_pilot_evaluations` only after copying the report, raw JSON/NPZ artifacts, and source artifact timestamps into `state/multi-environment/gpu-training-pilot/`. The importer checks artifact hashes, checkpoint identities, disjoint held-out tasks, matching seeds, probability framing and recorded audit results before writing either point. Timestamps are actual remote artifact-write completion times. This imports a completed same-GPU pilot; it does not run inference or establish a full GPU controller epoch. Its output-head training update is explicitly partial, and held-out rewards were zero before and after.

The public projection includes sanitized model identifiers and keeps CUDA and CPU datasets separate. The Evaluation selector exposes each comparable model/runtime/taskset group for an environment, including both CPU and CUDA histories. It defaults to the latest group and preserves the selected group during refreshes. Curves never mix different weights architectures, runtime revisions, harnesses, or fixed tasksets. Verify large published checkpoint file hashes without filling the operator disk using `.venv/bin/python -m ops.check_r2_checkpoint`; this streams R2 bytes and writes only an evidence report, not model weights. The command authenticates publication against the previously accepted operator filemap and does not approve arbitrary model reports or execute a model.

Repeat independent completed-epoch checks with `.venv/bin/python -m ops.check_epoch_evidence`. This reads signed R2 manifests, scores, audit challenges, audit reports and checkpoint descriptors; hashes actual frozen submissions and local trained checkpoint files; matches held-out tasks/profiles; and confirms the test reports are excluded from payout aggregation. It uses the controller report's local authority as its operator trust anchor, reads the existing private bucket config without printing credentials, and performs no chain writes. Results are saved in state/multi-environment/independent-epoch-evidence.json. These checks authenticate the recorded execution evidence; they do not replace inference recomputation by the verifier or establish goal completion.

Public mobile and desktop checks at22:39UTC confirmed15 environment options,256UIDcells, noJavaScript errors or overflow, and realIFEvalbefore/after0.25→0.25. ResponsiveSVGcoordinates keepaxisfont12physicalpixels at390pxand1440px ratherthanshrinking desktopcoordinates to3.4px onphones. Epochselector showsUTCtimes; fullimmutableepochIDs remainoptiontitles/values. Evidence state/dashboard/five-epoch-public-mobile-check.json and responsive-chart-public-check.json; oldJSbackup deployment-20260930T223852Z-responsive-chart.

Tau2's explicit-recovery projection now publishes a matched sixteen-task baseline
and sixteen-task recovered after-checkpoint population, alongside the preserved
original eleven-task partial result. Five original failures and five separate
recoveries are visible as bounded metadata. Both completed means remain zero.
The signed evidence and separate metadata approval are authenticated before
export; no historical failed report is rewritten. See
[TAU2_RECOVERY_DASHBOARD.md](../docs/TAU2_RECOVERY_DASHBOARD.md).

The October 1 independent live-browser check selected all 22 available pilot
environments and all 84 distinct dataset/harness/model/runtime cohorts. Each
curve's point count and latest reward tooltip matched the actual public JSON.
The chart and 256-cell grid passed at 1440px and 390px without horizontal
overflow or JavaScript errors; the batches-per-epoch chart included only
finalized pilot epochs. These observations do not establish improving reward
or admission of environments that have only native or standalone controls.
The separate authenticated wider-epoch check matched all 254 evaluation
records and UID/checkpoint projections for the eleven completed training epochs.

Re-run the public browser check using an installed Chrome and Playwright:

```sh
NODE_PATH=/path/to/node_modules node ops/check_dashboard_browser.cjs \
  https://affine.io state/dashboard
```

Set `AFFINE_CHROME_BIN` if Chrome is outside `/usr/bin/google-chrome`.
The command reads the public site, writes screenshots and a bounded evidence
receipt in the output directory, and performs no uploads or chain operations.
Its coverage reflects the cohorts present at execution time.
