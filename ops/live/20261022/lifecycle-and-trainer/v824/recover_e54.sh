#!/usr/bin/env bash
# e54 trainer recovery after v8.2.3 activation (warm trainer worker cached the v8.2.2 optimizer config).
# Follows precedent v821-trainer-override/cutover_v821_trainer.sh + reset_e39_job.py (10-16 06:48Z):
#  stop lifecycle + trainer watcher, drain the warm trainer worker (cold restart rebuilds optimizer from signed job config),
#  back up roles.sqlite3, move TRAINING-JOB/RECOVERY-REQUIRED aside, delete ONLY the failed e54 train row,
#  move the failed job's attempt workspace + the never-started 'updating' transaction journal (job/state/version only, no artifacts) aside.
# Nothing is deleted except that one failed queue row (full DB backup kept). No signatures, no chain tx.
set -euo pipefail; umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261017-v824
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; PY=/home/const/subnet120-rewrite/.venv/bin/python
JOB=mimo-train-090278a66863e486d5dab648f6149b0e1e3af0f5b40b909e
B=/workspace/affine-mimo-trainer-prod; T=$B/trainer-state/affine-mimo9b-grpo-20261013
TXN=update-0e90226fd6e279f6cefbf63db76b3440180d8f257ff7ab538b5a3bf8db0ead36
OW="$B/runtime-exact/venv/bin/python -I -B $B/owned-worker-v7.py --config $B/owned-worker-v824s.json --sha256 b8ba9dca9687f7c4f7b3bc83d8be563af63ff1a74e7259f45791418a690357a4"
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/recovery-e54-$TS; mkdir -m 700 "$BK"; LOG=$BK/RECOVERY.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }; die(){ log "ABORT: $*"; exit 1; }
rssh(){ ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o UserKnownHostsFile=/home/const/.lium/known_hosts/0a7aba3d-9443-445b-9601-c78ac0d14b1c -p 40301 root@89.169.120.92 "$@"; }
E=$STATE/epochs/54
[[ -f $E/TRAINING-JOB.json && -f $E/RECOVERY-REQUIRED.json && ! -e $E/TRAINING-RESULT.json ]] || die "e54 recovery files not as expected"
"$PY" -c "import json;r=json.load(open('$E/RECOVERY-REQUIRED.json'));assert r['job_id']=='$JOB' and r['queue_status']=='failed',r" || die "recovery-required mismatch"
rssh "python3 -c \"import json;v=json.load(open('$T/history/$TXN/TRANSACTION.json'));assert v['state']=='updating' and sorted(v)==['job','state','version'] and v['job']['epoch'].endswith('e000054'),v['state']\" && test \$(ls $T/history/$TXN | wc -l) -eq 1" || die "e54 transaction is not a bare never-started updating journal"
rssh "python3 -c \"import json;c=json.load(open('$T/CURRENT.json'));assert c['optimizer_step']==54 and c['checkpoint'].startswith('2cbf44a6'),c\"" || die "trainer CURRENT not at e53 output"
log "checks OK; stopping lifecycle + trainer watcher"
systemctl --user stop affine-mimo-lifecycle.service
systemctl --user stop affine-mimo-trainer-supervised.service
rssh "$OW --stop-owned-after-drain" >>"$LOG" 2>&1 || log "note: no live trainer worker to stop"
st=True; for i in $(seq 1 120); do st=$(rssh "$OW --status" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])"); [[ "$st" == False ]] && break; sleep 5; done
[[ "$st" == False ]] || die "trainer worker still alive after 10 min (lifecycle+watcher left stopped; restart them to undo)"
log "trainer worker drained"
cp -a "$STATE/roles.sqlite3" "$BK/roles.sqlite3.before-e54-reset"
[[ -e $STATE/roles.sqlite3-journal ]] && cp -a "$STATE/roles.sqlite3-journal" "$BK/roles.sqlite3-journal.before-e54-reset" || true
cp -a "$E/TRAINING-JOB.json" "$BK/TRAINING-JOB.json.copy"; mv "$E/TRAINING-JOB.json" "$BK/TRAINING-JOB.json.moved"; mv "$E/RECOVERY-REQUIRED.json" "$BK/RECOVERY-REQUIRED.json.moved"
"$PY" -B "$D/reset_e54_job.py" "$STATE/roles.sqlite3" | tee -a "$LOG"
rssh "mkdir -m 700 -p $B/backup-e54-recovery-$TS && mv $B/workspace/$JOB $B/backup-e54-recovery-$TS/workspace-$JOB && mv $T/history/$TXN $B/backup-e54-recovery-$TS/history-$TXN && ls $B/backup-e54-recovery-$TS" | tee -a "$LOG"
log "starting trainer watcher (cold worker) then lifecycle"
systemctl --user start affine-mimo-trainer-supervised.service
ok=0; for i in $(seq 1 60); do rssh "$OW --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 10; done
[[ $ok == 1 ]] || log "WARNING: trainer worker not alive after 10 min"
systemctl --user start affine-mimo-lifecycle.service
sleep 60; systemctl --user is-active --quiet affine-mimo-lifecycle.service && log "lifecycle active" || log "WARNING lifecycle not active"
ls "$E" | tee -a "$LOG"
log "RECOVERY STEPS DONE. Undo: stop units, restore $BK/roles.sqlite3.before-e54-reset + moved files, mv trainer backup-e54-recovery-$TS items back."
