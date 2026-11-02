# affine.io chart load sequence (Jacob pt1070u, 2026-11-02)

"charts load badly. I want it to load nicely where the chart shapes load first then the data comes in second."

## What was wrong (measured on the live site, 1440x1000, headless Chrome)
- `<main>` was empty in the HTML; every tile was created by its script only when its JSON arrived, and placed relative to
  whichever tile already existed, so tiles appeared in fetch-completion order and pushed each other around
  (CLS 0.33; 0.36 on a throttled 1.6 Mbps / 150 ms link). The benchmark block was inserted above everything at ~400 ms.
- The price strip was moved into the header by network-desktop.js after load: a 100 px jump of the whole page.
- JSON fetches only started after all 10 stylesheets + 9 deferred scripts loaded (~270 ms).
- k1.js refetched k1/scores/mining/lr.json on every window resize, and tile-details.js fires resize events while
  laying out, so each page load fetched those four files 3 times. site-metrics.json (726 KB raw) was fetched twice
  (top.js + timing.js). 19 JSON requests, 813 KB transferred.
- Charts popped in with no transition; scripts and styles were served `no-store` so nothing was cached between visits.

## What changed
- index.html: tile shells for every chart, table and list at their final size and order (title, reading, legend,
  plot box with faint placeholder gridlines + subtle shimmer). Chart scripts fill these same elements in place
  (k1 tiles via `data-placed=1`, perf by id, timing/regcost adopt their shell, bench replaces its block).
- Inline `AFX` prefetch in <head>: all 10 JSON files start at parse time; each script's first load takes a clone,
  later refreshes fetch exactly as before (timers, cache-busting unchanged).
- Inline price-strip placement before first paint (network-desktop.js reuses its anchor).
- load.js/load.css: placeholder fades out and data draws in once (lines draw left to right, bars grow from the axis,
  points/bands fade, ~300-500 ms; tables fade). Redraws later (refresh, resize, theme) are instant as before.
  prefers-reduced-motion: no shimmer, no animation.
- k1.js: resize redraws from the loaded data instead of refetching. bench.js: fit() guard for the skeleton box.
- /baseline: same skeleton + reveal, prefetch of curves.json.
- Caddy (affine.io block only): `Cache-Control: private, max-age=86400` for `*.js|*.css?v=...`, 7 days for the two
  fonts. private = browsers only, not the Cloudflare edge. **Bump the ?v= of any js/css you edit.**
- No data, gate logic, colours or chart styling changed.

## Revert
`~/subnet120-rewrite/state/bench-results/site-backup/20261102T154146Z/revert.sh` on affinevalidator (restores the 10 files and the
Caddyfile, removes load.js/load.css, validates and reloads Caddy).
