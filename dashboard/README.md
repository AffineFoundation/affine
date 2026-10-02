# Affine public dashboard

https://affine.io shows exactly two charts: completed held-out evaluation reward
over time, and frozen submitted batches per finalized epoch. The UID grid and
metric toggle have been removed. Haffer and DM Mono preserve the monochrome
Affine branding. The footer links to the current inference-verification guide at
`/llms.txt`.

The initial selection is original MATH and its latest fixed evaluation cohort,
with the newest finalized original MATH pilot batch series. The batch selector
labels SmolLM2-1.7B and Qwen2.5-Math-7B series separately and follows the newest
actual finalized MATH window until the user selects a series. Manual choices
persist through refresh and resize; initial upload and qualification jobs never
become finalized batch points. Environment/cohort and epoch-series
selectors preserve access to historical measurements. Network and nonpayable
pilot scopes remain separate. Cohort identity binds dataset/taskset, fixed task
IDs, seed, sample count, model family, environment version, harness, output
budget, policy and runtime. Different checkpoints can be compared only within
that cohort; repeated measurements remain separate points at their actual UTC
timestamps. Incomplete records are excluded, and missing/unavailable records
show explicit empty states. New corpus qualification is not a chart point.

Batch tooltips separate submitted, accepted, rejected and unchecked counts.
Finalized means the score window closed, rather than asserting completed
training. All counts come from the existing allowlisted public projection.
The historical `affine_numina` label explicitly identifies Lean; it must not be
mistaken for a new NuminaMath-CoT taskset.

`affine-network-dashboard.service` refreshes SQLite and exports
`network-data.json` every 15 seconds to
`/home/const/subnet120/affine/website/`. Caddy forwards affine.io to the existing
Affine dashboard server on port 8787, which serves that website directory.
Static deployment copies only this dashboard's index, CSS, JS and llms.txt.
The projection export also atomically restores canonical llms.txt on refresh if
the legacy website generator replaces it. Only the dashboard projection service
is restarted for that integration. No validator restart, DNS change or Arbos
site change is involved. The local
preview server also serves `/llms.txt` as plain text. Fonts/assets already
exist in production. Keep a private timestamped backup before replacements.

Validation:

```sh
python3 -m unittest dashboard.test_server
node --check dashboard/public/network.js
NODE_PATH=/path/to/node_modules node ops/check_dashboard_browser.cjs \
  https://affine.io state/dashboard
```

The browser check compares every available evaluation cohort and finalized
batch series with the actual public JSON, verifies two simultaneous charts and
absence of the UID grid, checks tooltips and selector persistence, and captures
desktop/mobile screenshots. Separate intercepted empty/unavailable responses
verify absence of fabricated data; these controls are labelled in the receipt.
Set `AFFINE_CHROME_BIN` if Chrome is outside `/usr/bin/google-chrome`.
No upload, inference, training or blockchain transaction occurs in this check.
