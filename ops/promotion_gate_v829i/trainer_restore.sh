#!/usr/bin/env bash
# Promotion gate RED: restore the pod trainer CURRENT to head H after the lifecycle finalized the epoch no-update onto H.
# Same sequence as gradnorm/rollback2/scripts/stop_and_restore_e141_v832.sh (stop watcher, drain owned worker, install
# generation + CURRENT, start watcher), with the generation from the gate's hardlinked keep and the live worker config.
# usage: trainer_restore.sh H EPOCH
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
H=$1; E=$2
V=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; PY=/home/const/subnet120-rewrite/.venv/bin/python
RSSH=/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh
POD=/workspace/affine-mimo-trainer-prod; LOG=$V/state/restore.log
log(){ echo "$(date -u +%FT%TZ) $(TZ=America/Costa_Rica date '+%H:%M:%S CR') e$E $*" | tee -a "$LOG"; }; die(){ log "ABORT: $*"; exit 1; }
P(){ local i rc; for i in 1 2 3 4 5 6; do "$RSSH" "$@" < /dev/null && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 10; done; return 255; }
EXEC=$(systemctl --user show affine-mimo-trainer-supervised.service -p ExecStart --value)
WCFG=$(grep -o -- '--config [^ ]*' <<<"$EXEC" | tail -1 | cut -d' ' -f2); WSHA=$(grep -o -- '--sha256 [0-9a-f]\{64\}' <<<"$EXEC" | tail -1 | cut -d' ' -f2)
[[ -n "$WCFG" && -n "$WSHA" ]] || die "cannot parse live worker config"
OWK=(P "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $WCFG --sha256 $WSHA")
cur(){ "$PY" -c "import json;d=json.load(open('$STATE/CURRENT.json'));print(d$1)"; }
[[ "$(cur "['checkpoint']['id']")" == "$H" ]] || die "lifecycle CURRENT is not H"
N=$(cur "['epoch_index']")
[[ -e "$STATE/epochs/$N/TRAINING-JOB.json" ]] && die "TRAINING-JOB exists in e$N (guard should have held it)"
P "test -f $POD/private/promotion-gate/keep/$H/KEEP.json" || die "no keep for H on pod"
log "restore start: H=${H:0:12} worker=$(basename $WCFG)"
"${OWK[@]}" --status > "$V/state/worker-status.pre-e$E.json" 2>&1 || true
systemctl --user stop affine-mimo-trainer-supervised.service
"${OWK[@]}" --stop-owned-after-drain 2>&1 | cut -c1-200 | tee -a "$LOG" || log "note: no live worker"
st=True; for i in $(seq 1 120); do st=$("${OWK[@]}" --status | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])") || st=True; [[ "$st" == False ]] && break; sleep 5; done
if [[ "$st" != False ]]; then systemctl --user start affine-mimo-trainer-supervised.service; die "worker still alive after 10 min (watcher restarted, nothing changed)"; fi
for i in $(seq 1 24); do P "pgrep -f '[m]imo_host_trainer_' >/dev/null || pgrep -f '[v]83_dp.py' >/dev/null" || break; sleep 5; done
P "python3 $POD/private/promotion-gate/trainer_keep.py restore $H" | tee -a "$LOG" || { systemctl --user start affine-mimo-trainer-supervised.service; die "pod restore failed (watcher restarted)"; }
systemctl --user start affine-mimo-trainer-supervised.service; sleep 20
ok=0; for i in $(seq 1 90); do "${OWK[@]}" --status | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
[[ $ok == 1 ]] || die "worker not alive after restore"
log "RESTORE DONE: pod trainer CURRENT = ${H:0:12}; watcher active"
