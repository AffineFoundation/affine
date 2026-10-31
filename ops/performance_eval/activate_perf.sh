#!/usr/bin/env bash
# Install the SN120 Performance tile + performance.json publisher (Jacob pt911u 2026-10-31). Revert: revert_perf.sh
set -euo pipefail
RA=/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval; C=$RA/code; W=/home/const/subnet120/affine/website
TS=$(date -u +%Y%m%dT%H%M%SZ); B=$RA/backup/$TS; mkdir -p $B
cp -p $W/index.html $W/llms.txt $B/; ls $W/perf.js $W/perf.css $W/performance.json 2>/dev/null | xargs -r -I{} cp -p {} $B/ || true
echo $TS > $RA/backup/LATEST
install -m 644 $C/perf.js $W/perf.js.new && mv $W/perf.js.new $W/perf.js
install -m 644 $C/perf.css $W/perf.css.new && mv $W/perf.css.new $W/perf.css
V=${V:-20261031-perf1}
python3 - "$W/index.html" "$V" <<'PY'
import sys,re
p,v=sys.argv[1],sys.argv[2]; s=open(p).read()
s=re.sub(r'<link rel="stylesheet" href="/perf\.css\?v=[^"]*">\n?','',s); s=re.sub(r'<script defer src="/perf\.js\?v=[^"]*"></script>\n?','',s)
a='<link rel="stylesheet" href="/k1.css'; i=s.index(a); j=s.index('\n',i)+1
s=s[:j]+'<link rel="stylesheet" href="/perf.css?v=%s">\n'%v+s[j:]
a='<script defer src="/k1.js'; i=s.index(a)
s=s[:i]+'<script defer src="/perf.js?v=%s"></script>\n'%v+s[i:]
t=open(p+'.new','w'); t.write(s); t.close()
import os; os.chmod(p+'.new',0o644); os.replace(p+'.new',p)
PY
cp -p $W/llms.txt $W/llms.txt.new && python3 $C/llms_perf.py $W/llms.txt.new && chmod 644 $W/llms.txt.new && mv $W/llms.txt.new $W/llms.txt
mkdir -p ~/.config/systemd/user
for u in affine-perf-eval.service affine-perf-eval-2.service affine-perf-watchdog.service affine-perf-watchdog.timer; do install -m 644 $C/$u ~/.config/systemd/user/$u; done
systemctl --user daemon-reload
systemctl --user enable --now affine-perf-eval.service affine-perf-eval-2.service affine-perf-watchdog.timer
echo "installed (backup $B)"
