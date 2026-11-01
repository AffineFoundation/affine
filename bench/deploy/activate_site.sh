#!/usr/bin/env bash
# Install the Benchmarks table + tiles on affine.io (host affinevalidator; site root served by affine-dash). Idempotent.
# Backs up index.html first; revert: revert_site.sh. Touches only index.html (2 tags), bench.js, bench.css.
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); W=${BENCH_SITE:-$HOME/subnet120/affine/website}; V=${V:-$(date -u +%Y%m%d%H%M)}
B=${BENCH_STATE:-$HOME/subnet120-rewrite/state/bench-results}/site-backup/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $B
cp -p $W/index.html $B/; for f in bench.js bench.css; do [ -f $W/$f ] && cp -p $W/$f $B/ || true; done; echo $B > $(dirname $B)/LATEST
install -m 644 $H/site/bench.js $W/bench.js.new && mv $W/bench.js.new $W/bench.js
install -m 644 $H/site/bench.css $W/bench.css.new && mv $W/bench.css.new $W/bench.css
python3 - "$W/index.html" "$V" <<'PY'
import os, re, sys
p, v = sys.argv[1], sys.argv[2]; s = open(p).read()
s = re.sub(r'<link rel="stylesheet" href="/bench\.css\?v=[^"]*">\n?', '', s); s = re.sub(r'<script defer src="/bench\.js\?v=[^"]*"></script>\n?', '', s)
a = '<link rel="stylesheet" href="/perf.css'; i = s.index(a); j = s.index('\n', i) + 1
s = s[:j] + '<link rel="stylesheet" href="/bench.css?v=%s">\n' % v + s[j:]
a = '<script defer src="/perf.js'; i = s.index(a); j = s.index('\n', i) + 1
s = s[:j] + '<script defer src="/bench.js?v=%s"></script>\n' % v + s[j:]
open(p + '.new', 'w').write(s); os.chmod(p + '.new', 0o644); os.replace(p + '.new', p)
PY
echo "installed bench.js/bench.css v=$V (backup $B)"
