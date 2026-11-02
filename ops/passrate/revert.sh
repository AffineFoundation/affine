#!/bin/bash
# Revert pt1130u (pass-rate tile): restore the Performance tile + public performance.json series from the deploy backup.
# Refuses if index.html/llms.txt changed after the deploy (someone else edited them) unless FORCE=1.
set -euo pipefail
D=/home/const/subnet120-rewrite/state/root-audits/20261102-passrate; W=/home/const/subnet120/affine/website
PE=/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval/code; U=/home/const/.config/systemd/user
B=${B:-$(cat $D/LAST-BACKUP)}
if ! sha256sum --quiet -c $D/DEPLOYED.sha256 && [[ ${FORCE:-0} != 1 ]]; then echo "site files changed since deploy; rerun with FORCE=1 to overwrite"; exit 1; fi
systemctl --user disable --now affine-site-passrate.timer || true
cp -p $B/perf_service.py $PE/perf_service.py && systemctl --user restart affine-perf-eval.service
cp -p $B/index.html $B/perf.js $B/perf.css $B/llms.txt $W/
rm -f $U/affine-site-passrate.service $U/affine-site-passrate.timer /home/const/subnet120/affine/site-tools/build_passrate.py $W/passrate.json
systemctl --user daemon-reload
echo "reverted from $B (performance.json is rewritten in full by affine-perf-eval on its next publish)"
