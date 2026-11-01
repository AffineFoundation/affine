#!/usr/bin/env bash
# Undo activate_site.sh: drop the two tags from index.html and remove bench.js/bench.css (benchmarks.json stays).
set -euo pipefail
W=${BENCH_SITE:-$HOME/subnet120/affine/website}
python3 - "$W/index.html" <<'PY'
import os, re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'<link rel="stylesheet" href="/bench\.css\?v=[^"]*">\n?', '', s); s = re.sub(r'<script defer src="/bench\.js\?v=[^"]*"></script>\n?', '', s)
open(p + '.new', 'w').write(s); os.chmod(p + '.new', 0o644); os.replace(p + '.new', p)
PY
rm -f $W/bench.js $W/bench.css; echo reverted
