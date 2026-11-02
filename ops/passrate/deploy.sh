#!/bin/bash
# pt1130u: affine.io first tile -> pass rate across all envs (passrate.json); performance.json -> king list only (series private).
set -euo pipefail
D=/home/const/subnet120-rewrite/state/root-audits/20261102-passrate; C=$D/code; W=/home/const/subnet120/affine/website
T=/home/const/subnet120/affine/site-tools; PE=/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval/code
U=/home/const/.config/systemd/user; TS=$(date -u +%Y%m%dT%H%M%SZ); B=$D/backups/$TS; mkdir -p $B
cp -p $W/index.html $W/perf.js $W/perf.css $W/llms.txt $B/; cp -p $W/performance.json $B/performance.json.private-backup; chmod 600 $B/performance.json.private-backup
cp -p $PE/perf_service.py $B/; echo $B > $D/LAST-BACKUP
# 1 builder + units, first build
install -m 644 $C/build_passrate.py $T/build_passrate.py
install -m 644 $C/affine-site-passrate.service $C/affine-site-passrate.timer $U/
systemctl --user daemon-reload
/usr/bin/python3 -I -B $T/build_passrate.py
python3 -c "import json;d=json.load(open('$W/passrate.json'));assert len(d['kings'])>=3,d;print('passrate.json',len(d['kings']),d['kings'][-1]['epoch'],d['kings'][-1]['pass_rate'])"
# 2 perf service: series private, public king list only (restart resumes the running eval from its state file)
install -m 644 $C/perf_service.py $PE/perf_service.py
systemctl --user restart affine-perf-eval.service
# 3 site: tile JS, index.html (patched in place), llms.txt (patched in place)
install -m 644 $C/perf.js $W/perf.js.new && mv $W/perf.js.new $W/perf.js
cp -p $W/index.html $W/index.html.new && python3 $C/patch_index.py $W/index.html.new && mv $W/index.html.new $W/index.html
cp -p $W/llms.txt $W/llms.txt.new && python3 $C/patch_llms.py $W/llms.txt.new && mv $W/llms.txt.new $W/llms.txt
sha256sum $W/index.html $W/perf.js $W/llms.txt > $D/DEPLOYED.sha256
systemctl --user enable --now affine-site-passrate.timer
echo "DEPLOYED $TS backup $B"
