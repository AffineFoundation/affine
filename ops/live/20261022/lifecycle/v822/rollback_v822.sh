#!/usr/bin/env bash
# MiMo v8.2.2 successor -> v8.2.1 rollback, SN120 Arbos.  PREPARED, NOT RUN.
# Usage: rollback_v822.sh [BACKUP_DIR]          (default: path in activation-v822/LAST-ACTIVATION)
#        rollback_v822.sh --check [BACKUP_DIR]  (read-only)
# Reverts: v8.2.2 drop-ins (lifecycle, writer, 5 owned-worker units; the v8.2.1 ones underneath become effective again) and restores
# trainer-role.json, live CURRENT.json, evaluator/state.json, roles.sqlite3, trainer CURRENT.json and a moved unopened epoch.
# Epochs produced under v8.2.2 after activation are moved aside (kept). Host v8.2.2 files and the R2 archive stay inert.
# No chain transactions, never touches the SUBNETOWNER mnemonic.
set -euo pipefail
umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession
A=$D/activation-v822
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user
PY=/home/const/subnet120-rewrite/.venv/bin/python
ROLES=(trainer evaluator verifier-1 verifier-2 verifier-3)
TRAINER_BASE=/workspace/affine-mimo-trainer-prod
TRAINER_STATE=$TRAINER_BASE/trainer-state/affine-mimo9b-grpo-20261013
CHECK=0; [[ "${1:-}" == "--check" ]] && { CHECK=1; shift; }
BK=${1:-$(cat "$A/LAST-ACTIVATION")}
[[ -d "$BK" && -f "$BK/ACTIVATION-RECORD.env" ]] || { echo "no activation backup at $BK"; exit 1; }
source "$BK/ACTIVATION-RECORD.env"
ALOG=$BK/ROLLBACK.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$ALOG"; }
die(){ log "ABORT: $*"; exit 1; }
plan(){ "$PY" -c "import json,sys;print(json.load(open('$A/ACTIVATION-PLAN.json'))['hosts'][sys.argv[1]][sys.argv[2]])" "$1" "$2"; }
rssh(){ local r=$1; shift; ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \
  -o UserKnownHostsFile=/home/const/.lium/known_hosts/$(plan "$r" known_hosts) -p "$(plan "$r" port)" root@"$(plan "$r" host)" "$@"; }
log "rollback from $BK (TS=$TS, EPOCH_MOVED=${EPOCH_MOVED:-none}, check=$CHECK)"
(cd "$BK" && sha256sum --quiet -c BACKUP.sha256) || die "backup hash mismatch"
rssh trainer "test -f $TRAINER_BASE/private/trainer-role.json.bak-v822s-$TS && test -f $TRAINER_STATE/CURRENT.json.bak-v822s-$TS" || die "trainer backups missing"
if [[ $CHECK -eq 1 ]]; then
  ls -la "$BK"; for r in "${ROLES[@]}"; do ls "$UNITS/affine-mimo-$r-supervised.service.d/" 2>/dev/null || true; done
  log "check only; nothing changed"; exit 0
fi
log "stop lifecycle + owned-worker watchers"
systemctl --user stop affine-mimo-lifecycle.service
for r in "${ROLES[@]}"; do systemctl --user stop affine-mimo-$r-supervised.service; done
for r in "${ROLES[@]}"; do
  rssh "$r" "$(plan "$r" owpy) --config $(plan "$r" config_v822s) --sha256 $(plan "$r" config_v822s_sha256) --stop-owned-after-drain" >>"$ALOG" 2>&1 || log "$r: no live v8.2.2 worker"
done
for r in "${ROLES[@]}"; do
  st=True
  for i in $(seq 1 180); do
    st=$(rssh "$r" "test -f $(plan "$r" config_v822s) && $(plan "$r" owpy) --config $(plan "$r" config_v822s) --sha256 $(plan "$r" config_v822s_sha256) --status || echo '{\"alive\": false}'" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])")
    [[ "$st" == "False" ]] && break; sleep 5
  done
  [[ "$st" == "False" ]] || die "$r: v8.2.2 worker still alive after 15 min"
done
log "remove v8.2.2 drop-ins (v8.2.1 drop-ins remain)"
for r in "${ROLES[@]}"; do rm -f "$UNITS/affine-mimo-$r-supervised.service.d/zzzz~~~~v822s-owned-worker.conf"; done
rm -f "$UNITS/affine-mimo-lifecycle.service.d/zzzz~~~~~~~v822s-release.conf"
rm -f "$UNITS/affine-live-reward-writer.service.d/zzzz~~~~~mimo-v822-population-index.conf"
systemctl --user daemon-reload
log "restore state (v8.2.2-era files kept under $BK/v822-era)"
mkdir -p -m 700 "$BK/v822-era/epochs"
cp -a "$STATE/CURRENT.json" "$BK/v822-era/CURRENT.json"
[[ -f "$STATE/evaluator/state.json" ]] && cp -a "$STATE/evaluator/state.json" "$BK/v822-era/evaluator-state.json"
PRE_EI=$("$PY" -c "import json;print(json.load(open('$BK/CURRENT.json.pre-v822'))['epoch_index'])")
for e in "$STATE"/epochs/*; do n=$(basename "$e"); [[ "$n" =~ ^[0-9]+$ ]] && (( n >= PRE_EI )) && mv "$e" "$BK/v822-era/epochs/$n"; done
mv "$STATE/roles.sqlite3" "$BK/v822-era/roles.sqlite3"
[[ -e "$STATE/roles.sqlite3-journal" ]] && mv "$STATE/roles.sqlite3-journal" "$BK/v822-era/roles.sqlite3-journal" || true
cp -a "$BK/roles.sqlite3.pre-v822" "$STATE/roles.sqlite3"
[[ -f "$BK/roles.sqlite3-journal.pre-v822" ]] && cp -a "$BK/roles.sqlite3-journal.pre-v822" "$STATE/roles.sqlite3-journal" || true
cp -a "$BK/CURRENT.json.pre-v822" "$STATE/CURRENT.json"
[[ -f "$BK/evaluator-state.json.pre-v822" ]] && cp -a "$BK/evaluator-state.json.pre-v822" "$STATE/evaluator/state.json"
[[ -n "${EPOCH_MOVED:-}" ]] && cp -a "$BK/epoch-$EPOCH_MOVED.v821-unopened" "$STATE/epochs/$EPOCH_MOVED"
rssh trainer "cp -a $TRAINER_STATE/CURRENT.json $TRAINER_STATE/CURRENT.json.v822s-era-$TS && cp -a $TRAINER_STATE/CURRENT.json.bak-v822s-$TS $TRAINER_STATE/CURRENT.json && cp -a $TRAINER_BASE/private/trainer-role.json.bak-v822s-$TS $TRAINER_BASE/private/trainer-role.json"
log "start v8.2.1 units"
for r in "${ROLES[@]}"; do systemctl --user start affine-mimo-$r-supervised.service; done
systemctl --user start affine-mimo-lifecycle.service
sleep 60
for r in "${ROLES[@]}"; do
  rssh "$r" "$(plan "$r" owpy) --config $(plan "$r" config_now) --sha256 $(plan "$r" config_now_sha256) --status" | tee -a "$ALOG" | grep -q '"alive": true' || die "$r: pre-activation worker not alive"
done
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active"
journalctl --user -u affine-mimo-lifecycle.service --since "-3min" --no-pager | tail -20 | cut -c1-300 | tee -a "$ALOG"
log "ROLLED BACK to v8.2.1."
