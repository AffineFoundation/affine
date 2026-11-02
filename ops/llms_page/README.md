# llms.txt reading view: centered column (Jacob pt1063u, 2026-11-02)
network.css lays out `#main` as a 4-column tile grid on desktop, which squeezed the llms.txt article (`main#main.text-document`) into
the first ~250px column. `llms-page.html` now sets `#main.text-document{display:block;max-width:760px;margin:0 auto}` (text stays
left-aligned; 20px side margins on phones), and honours the site's light theme (`affine:theme` in localStorage). Header unchanged.
llms.txt: only the "Last updated" line was bumped (2026-11-02 09:19 CR, after the flat z > 1.00 bar, b0d4b78); dashboard/public/llms.txt == live.
Deployed at ~/subnet120/affine/website/llms-page.html on AffineValidator. Revert: ~/subnet120-rewrite/state/bench-results/site-backup/20261102T151952Z/revert.sh
