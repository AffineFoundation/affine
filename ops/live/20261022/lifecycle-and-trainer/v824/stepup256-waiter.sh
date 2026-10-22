#!/bin/bash
# One-shot: flip lifecycle to signed v8.2.4 (256) during the e53 TRAINING job (after seal, before result),
# so e53 keeps its planned 128 and e54 is the first 256 epoch. Never during collection/seal.
set -u
D=/home/const/subnet120-rewrite/state/root-audits/20261017-v824; S=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
L=$D/stepup256-waiter.log; log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
cur(){ python3 -c "import json;print(json.load(open(\"$S/CURRENT.json\"))[\"$1\"])"; }
log "waiter start; waiting for epochs/53/TRAINING-JOB.json"
until [ -e $S/epochs/53/TRAINING-JOB.json ]; do
  [ "$(cur epoch_index)" = 53 ] || { log "ABORT: epoch_index moved to $(cur epoch_index) before e53 job; nothing changed"; exit 1; }
  sleep 15; done
log "e53 TRAINING-JOB seen; waiting 180 s for the trainer to claim/run"
sleep 180
[ "$(cur epoch_index)" = 53 ] && [ "$(cur max_tasks)" = 128 ] || { log "ABORT: state changed (ei=$(cur epoch_index) mt=$(cur max_tasks))"; exit 1; }
[ -e $S/epochs/53/TRAINING-RESULT.json ] && { log "ABORT: e53 already has a result; too close to the boundary; nothing changed"; exit 1; }
cd $D && ./activate_v824.sh --lifecycle v8.2.4 >> $L 2>&1; rc=$?
log "activate rc=$rc; CURRENT max_tasks=$(cur max_tasks) (expect 128 until e53 finalizes)"
journalctl --user -u affine-mimo-lifecycle --since "-3min" --no-pager -o cat | grep -i -E "error|traceback|exception" | cut -c1-250 >> $L
log "waiter done"
