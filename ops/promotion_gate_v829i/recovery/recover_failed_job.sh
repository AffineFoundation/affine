#!/usr/bin/env bash
# Generic failed-train-job recovery for the live v8.3.2 owned worker (t694, 2026-10-27). usage: EPOCH=N recover_failed_job.sh
# Idempotent re-issue: stop lifecycle + watcher, drain worker, convert any hardlinked gate keep of the head to copies,
# MOVE THE FAILED JOB'S workspace/<job> ASIDE (the lifecycle re-issues the SAME job id and the host mkdirs attempt-N
# exclusively, so stale attempt dirs crash rank 0 with FileExistsError), back up roles.sqlite3, move TRAINING-JOB /
# RECOVERY-REQUIRED aside, delete ONLY the failed/expired train row, start watcher then lifecycle. Safe to re-run.
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=/run/user/$(id -u)
V=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; PY=/home/const/subnet120-rewrite/.venv/bin/python
RSSH=/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh
POD=/workspace/affine-mimo-trainer-prod
EN=${EPOCH:?EPOCH}; E0=$STATE/epochs/$EN
JOB=$("$PY" -c "import json;print(json.load(open('$E0/RECOVERY-REQUIRED.json'))['job_id'])"); CK=$("$PY" -c "import json;print(json.load(open('$E0/RECOVERY-REQUIRED.json'))['checkpoint'])")
STEP=$("$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['optimizer_step'])")
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$V/recovery/recovery-e$EN-$TS; mkdir -p -m 700 "$BK"; LOG=$BK/RECOVERY.log
log(){ echo "$(date -u +%FT%TZ) $(TZ=America/Costa_Rica date +%H:%M:%S) CR $*" | tee -a "$LOG"; }; die(){ log "ABORT: $*"; exit 1; }
P(){ local i rc; for i in 1 2 3 4 5 6; do "$RSSH" "$@" < /dev/null && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 10; done; return 255; }
EXEC=$(systemctl --user show affine-mimo-trainer-supervised.service -p ExecStart --value)
WCFG=$(grep -o -- '--config [^ ]*' <<<"$EXEC" | tail -1 | cut -d' ' -f2); WSHA=$(grep -o -- '--sha256 [0-9a-f]\{64\}' <<<"$EXEC" | tail -1 | cut -d' ' -f2)
[[ -n "$WCFG" && -n "$WSHA" ]] || die "cannot parse live worker config"
OWK=(P "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $WCFG --sha256 $WSHA")
E=$E0
[[ -f $E/TRAINING-JOB.json && -f $E/RECOVERY-REQUIRED.json && ! -e $E/TRAINING-RESULT.json && ! -e $E/FINALIZED.json ]] || die "e$EN files not as expected"
"$PY" -c "import json;r=json.load(open('$E/RECOVERY-REQUIRED.json'));assert r['queue_status'] in ('failed','expired'),r" || die "recovery-required not failed/expired"
"$PY" -c "import json;c=json.load(open('$STATE/CURRENT.json'));assert c['epoch_index']==$EN and c['checkpoint']['id']=='$CK',c['epoch_index']" || die "lifecycle CURRENT not e$EN/job ckpt"
P "cat $POD/trainer-state/affine-mimo9b-grpo-20261013/CURRENT.json" | "$PY" -c "import json,sys;c=json.load(sys.stdin);assert c['optimizer_step']==$STEP and c['checkpoint']=='$CK'" || die "pod CURRENT not job ckpt/step"
log "checks OK; stopping lifecycle, watcher, draining worker"
systemctl --user stop affine-mimo-lifecycle.service
systemctl --user stop affine-mimo-trainer-supervised.service
"${OWK[@]}" --stop-owned-after-drain 2>&1 | cut -c1-200 | tee -a "$LOG" || log "note: no live worker"
st=True; for i in $(seq 1 120); do st=$("${OWK[@]}" --status | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])") || st=True; [[ "$st" == False ]] && break; sleep 5; done
[[ "$st" == False ]] || { systemctl --user start affine-mimo-trainer-supervised.service; systemctl --user start affine-mimo-lifecycle.service; die "worker still alive after 10 min (units restarted, nothing changed)"; }
for i in $(seq 1 24); do P "pgrep -f '[m]imo_host_trainer_' >/dev/null || pgrep -f '[v]83_dp.py' >/dev/null" || break; sleep 5; done
log "worker drained; converting any hardlinked keep of the head to copies"
P "cd $POD/private/promotion-gate && { [ ! -d keep/$CK ] || python3 trainer_keep.py unlink $CK; }" | tee -a "$LOG"
cp -a "$STATE/roles.sqlite3" "$BK/roles.sqlite3.before-e$EN-reset"
for x in -journal -wal -shm; do [[ -e $STATE/roles.sqlite3$x ]] && cp -a "$STATE/roles.sqlite3$x" "$BK/roles.sqlite3$x.before-e$EN-reset" || true; done
# The re-issued job keeps the same id; the worker mkdirs workspace/<job>/attempt-N exclusively, so stale attempt dirs
# from the failed run crash the host with FileExistsError (found live at 00:22-00:32Z; moved by hand then). Move them aside.
P "cd $POD && mkdir -p -m 700 private/recovery-e$EN && { [ ! -d workspace/$JOB ] || mv workspace/$JOB private/recovery-e$EN/workspace-failed-attempts-$TS; }"
mv "$E/TRAINING-JOB.json" "$BK/TRAINING-JOB.json.moved"; mv "$E/RECOVERY-REQUIRED.json" "$BK/RECOVERY-REQUIRED.json.moved"
"$PY" -B - "$STATE/roles.sqlite3" "$JOB" <<'PY' | tee -a "$LOG"
import sqlite3, sys
db = sqlite3.connect(sys.argv[1], timeout=60, isolation_level=None); JOB = sys.argv[2]
db.execute("BEGIN IMMEDIATE")
rows = db.execute("SELECT status, attempt, role, expires FROM jobs WHERE id=?", (JOB,)).fetchall()
assert len(rows) == 1 and rows[0][2] == "train" and rows[0][0] in ("failed", "expired"), rows
cur = db.execute("DELETE FROM jobs WHERE id=? AND role='train' AND status IN ('failed','expired')", (JOB,))
assert cur.rowcount == 1, cur.rowcount
db.execute("COMMIT")
print("deleted failed train job row", rows)
PY
log "starting trainer watcher then lifecycle"
systemctl --user start affine-mimo-trainer-supervised.service; sleep 20
ok=0; for i in $(seq 1 90); do "${OWK[@]}" --status 2>/dev/null | grep -q '"alive": true' && { ok=1; break; }; sleep 10; done
[[ $ok == 1 ]] && log "trainer worker alive" || log "WARNING: trainer worker not alive after 15 min"
systemctl --user start affine-mimo-lifecycle.service
sleep 60; systemctl --user is-active --quiet affine-mimo-lifecycle.service && log "lifecycle active" || log "WARNING lifecycle not active"
ls "$E" | tr "\n" " " | tee -a "$LOG"; echo
log "RECOVERY STEPS DONE. Undo: stop units, restore $BK/roles.sqlite3.before-e$EN-reset + moved files."
