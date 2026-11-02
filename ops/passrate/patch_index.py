#!/usr/bin/env python3
"""index.html: Performance tile -> Pass rate, all envs (pt1130u). Idempotent; asserts anchors."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'pt1130u-passrate' in s: print('already patched'); sys.exit(0)
assert s.count("'/performance.json',") == 1; s = s.replace("'/performance.json',", "'/passrate.json',")
old = '<script defer src="/perf.js?v=pt1070u-20261102"></script>'; assert s.count(old) == 1
s = s.replace(old, '<script defer src="/perf.js?v=pt1130u-passrate-20261102"></script>')
i = s.index('<section id="panel-performance"'); j = s.index('</section>', i) + len('</section>')
new = ('<section id="panel-performance" class="site-panel k1-tile perf-tile" aria-labelledby="panel-performance-t"><div class="section-heading"><h2 id="panel-performance-t">Pass rate, all envs</h2>'
       '<div class="measurement-reading"><span class="reading" id="panel-performance-r">—</span></div></div><div class="perf-body"><div class="chart-wrap k1-chart perf-chart afx afx-wait" role="img" '
       'aria-label="Pass rate of each king over the whole gate pool, all envs"></div><dl class="perf-latest"></dl></div><div class="legend chart-legend"><span><i class="swatch" style="background:var(--performance)"></i>Prior, full pool</span>'
       '<span><i class="swatch" style="border:1px solid var(--performance)"></i>Gate draw, 1/π</span></div><p class="perf-note"></p></section>')
s = s[:i] + new + s[j:]
assert 'performance.json' not in s and 'Holdout' not in s
open(p, 'w').write(s); print('patched')
