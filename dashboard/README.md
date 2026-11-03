# Affine public dashboard

https://affine.io shows exactly two charts: completed held-out math performance
over time, and frozen submitted batches per finalized epoch. Haffer and DM Mono,
monochrome Affine branding, restrained grid lines and responsive spacing keep the
measurements prominent. The header links to `/llms.txt` and GitHub; the footer
also links to the mechanism. There are no UID grids, metric toggles, environment
selectors, historical-series selectors or network/test-scope switches.

Both charts show the current `live-reward-math` run only. Evaluation points must
belong to its epochs, have `env_id=affine_math`, and be complete with valid numeric
measurements. The first chart automatically selects the latest completed
measurement's comparison cohort. Cohort identity binds dataset/taskset, fixed
task IDs, seed, sample count, requested count, model family, environment version,
harness, output budget, policy and runtime. Different checkpoints are comparable
within that unchanged cohort. Repeated measurements remain separate points at
their actual UTC timestamps; incompatible cohorts are not joined into a trend.
The percentage and solved count describe the latest actual held-out measurement,
not a claim of sustained improvement or a larger benchmark result.

The second chart includes finalized epochs from that same current run, ordered
by epoch start. It excludes pending upload windows and qualification work.
Finalized means the score window closed; it does not assert completed training.
The headline is the latest frozen submission count. Point inspection separates
submitted, fully audited and accepted, rejected, and unchecked counts. All counts
come from the existing allowlisted public projection. Only the chart's actual
measurements are drawn; the selection marker is not an additional data point.

Both charts support pointer/touch inspection and keyboard point navigation:
left/right or up/down arrows, Home/End, Enter/Space, and Escape to dismiss.
Keyboard focus stays on the inspected record across refresh. Empty or initially
unavailable records show explicit empty states without invented values. A failed
refresh retains the last committed measurements with a visible snapshot warning;
resize preserves that warning. Snapshots older than two minutes are labelled as
possibly stale. New corpus qualification is not a performance chart point.

`affine-network-dashboard.service` refreshes SQLite and exports
`network-data.json` every 15 seconds to
`/home/const/subnet120/affine/website/`. Caddy forwards affine.io to the existing
Affine dashboard server on port 8787, which serves that website directory.
Publish reviewed `index.html`, `network.css` and `network.js` separately from the
data export; changing the checkout alone does not publish them. Preserve a
private timestamped backup before replacements. Fonts and favicon assets already
exist in production. The projection export atomically restores canonical
`dashboard/public/llms.txt` and `reward-policy.json` on refresh if a legacy
website generator replaces them. The export does not copy the three frontend
files. No validator restart, DNS change or Arbos site change is involved.

Validation:

```sh
python3 -m unittest dashboard.test_server
node --check dashboard/public/network.js
node --check ops/check_dashboard_browser.cjs
NODE_PATH=/path/to/node_modules AFFINE_CHROME_BIN=/path/to/chrome \
  node ops/check_dashboard_browser.cjs https://affine.io state/dashboard
```

The browser check requires the `playwright` Node package and a Chrome executable.
`NODE_PATH` can point at an existing installation; Chrome defaults to
`/usr/bin/google-chrome` unless `AFFINE_CHROME_BIN` is set. The first argument is
the site URL and the second is the output directory. The command is read-only:
it does not upload, run inference/training, or submit blockchain transactions.

At desktop, mobile and narrow-mobile widths, the check compares rendered point
IDs and values and headline readings with the actual public JSON. It verifies
exactly two simultaneous charts, the current-run/latest-cohort rules, visible
mechanism/GitHub links, no horizontal overflow, truthful audit counts, keyboard
and touch inspection, and absence of browser errors. It also compares the actual
public HTML/CSS/JS and plain-text guide hashes with checkout, detecting publication
drift without hardcoding an obsolete proof contract. Browser HTML may contain
Cloudflare's injected analytics beacon; the receipt records its raw hash and
permits only that known insertion when comparing the source HTML. Other HTML
changes still fail the check. Run it after publishing
those reviewed files and the current guide.

Separate, labelled browser-only response fixtures exercise empty/unavailable
records, changes to each cohort identity, checkpoint-only comparisons, exclusion
of pending/qualification records, preserved keyboard focus, failed refresh and
stale snapshots. They are never published as network measurements. Screenshots
and the `two-chart-browser-check.json` receipt are written to the requested output
directory. If no completed current evaluation exists, cohort/refresh fixtures
are explicitly reported as skipped rather than claimed as tested.
