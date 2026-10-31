#!/usr/bin/env bash
# wait for e$WAIT_FIN FINALIZED, hold LIFECYCLE-ACTIVATION.lock (skip to the next gap if another job holds it), run install_public.sh MODE=apply
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261031-legacy-removal; E=/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs
n=${WAIT_FIN:?}
while true; do
  while [ ! -e $E/$n/FINALIZED.json ]; do sleep 2; done
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock
  if flock -n 9; then echo "- $(TZ=America/Costa_Rica date +%H:%M) CR: public: installing in the e$n -> e$((n+1)) gap" >> $D/PROGRESS.md
    MODE=apply bash $D/install_public.sh; rc=$?; exec 9>&-; exit $rc; fi
  exec 9>&-; echo "- $(TZ=America/Costa_Rica date +%H:%M) CR: public: lock held after e$n, next gap" >> $D/PROGRESS.md; n=$((n+1))
done
