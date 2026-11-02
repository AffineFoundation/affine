#!/usr/bin/bash
# Revert the affine.io/baseline "IRT live" edit (RA/20261102-irt-live): restores index.html / baseline.js / baseline.css / irt-curves.json
# and the site-tools builder + the affine-site-baseline unit from the backup, only where the live file is still exactly our version.
set -u; I=/home/const/subnet120-rewrite/state/root-audits/20261102-irt-live; BK=$(cat $I/state/SITE-BACKUP); W=/home/const/subnet120/affine/website/baseline
ST=/home/const/subnet120/affine/site-tools; U=/home/const/.config/systemd/user/affine-site-baseline.service
for f in index.html baseline.js baseline.css; do
  if cmp -s $W/$f $I/code/site/live/$f; then cp -p $BK/$f $W/$f; echo "restored $f"; else echo "WARN: $W/$f changed since the IRT-live edit: left as is"; fi
done
cp -p $BK/irt-curves.json $W/irt-curves.json; cp -p $BK/build_baseline_irt_curves.py $ST/build_baseline_irt_curves.py
cp -p $BK/affine-site-baseline.service $U && XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user daemon-reload; echo "site reverted"
