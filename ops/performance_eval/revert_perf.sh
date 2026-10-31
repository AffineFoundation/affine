#!/usr/bin/env bash
# Revert the SN120 Performance tile + publisher (pt911u 2026-10-31). Site: remove perf.js/perf.css/performance.json, drop their two
# index.html lines and the llms.txt section (exact markers; other edits made since are kept). Services: stop + disable + remove units.
# The eval pod is only removed with POD=rm (lium rm calm-orbit-ef, exact name).
set -uo pipefail
RA=/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval; C=$RA/code; W=/home/const/subnet120/affine/website
systemctl --user disable --now affine-perf-watchdog.timer affine-perf-eval.service 2>/dev/null
rm -f ~/.config/systemd/user/affine-perf-eval.service ~/.config/systemd/user/affine-perf-watchdog.service ~/.config/systemd/user/affine-perf-watchdog.timer
systemctl --user daemon-reload
python3 - "$W/index.html" <<'PY'
import sys,re,os
p=sys.argv[1]; s=open(p).read()
s=re.sub(r'<link rel="stylesheet" href="/perf\.css\?v=[^"]*">\n?','',s); s=re.sub(r'<script defer src="/perf\.js\?v=[^"]*"></script>\n?','',s)
open(p+'.new','w').write(s); os.chmod(p+'.new',0o644); os.replace(p+'.new',p)
PY
cp -p $W/llms.txt $W/llms.txt.new && python3 $C/llms_perf.py $W/llms.txt.new --remove && mv $W/llms.txt.new $W/llms.txt
rm -f $W/perf.js $W/perf.css $W/performance.json
if [ "${POD:-}" = rm ]; then PATH=$HOME/.lium/bin:$PATH lium rm calm-orbit-ef -y; fi
echo "reverted (results kept in $RA/state/results; pod $( [ "${POD:-}" = rm ] && echo removed || echo kept: POD=rm to remove))"
