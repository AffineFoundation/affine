#!/usr/bin/env bash
# Generator-feed one-deploy: old_commit -> new_commit from expect.json.
# Only the operator runs this on prod. It gates on the open round, stops after
# the round, verifies the journal, ingests only the oldest complete epoch,
# starts one loop, and starts ingestd only after the first open-check PASS.
set -u
H=${H:-/root/sn120-run/generator/genfeed-deploy}
L=$H/autodeploy.log
V2DIR=${V2DIR:-/root/sn120/v2}
OP=${OP:-"python3 $H/ol_operator.py"}
RUNLOG=${RUNLOG:-/root/sn120-run/work/logs/run.log}
STOPF=${STOPF:-/root/sn120-run/work/logs/run-loop.stop-after-round}
WD=${WD:-/root/sn120-run/watchdog}
LOOPPAT=${LOOPPAT:-"sn120-v2-run-loo[p]"}
S=${S:-1}
POLL=${POLL:-60}
WAIT=${WAIT:-1200}
GATE_WANT=${GATE_WANT:-r000080}
mkdir -p "$H"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$L"; }
jc(){ python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print(('yes' if s.get('clean') else 'no'),s.get('outcome'),'n_tasks',s.get('n_tasks'),'miner_cuts',[c['uid'] for c in s.get('miner_cuts') or []],'closed',s.get('close_at'),'why',s.get('why'))" "$1" 2>&1; }
cd "$V2DIR" || exit 64
old=$(python3 -c "import json;print(json.load(open('$H/expect.json'))['old_commit'])")
new=$(python3 -c "import json;print(json.load(open('$H/expect.json'))['new_commit'])")
log "pid $$ armed Generator feed ${old:0:8} -> ${new:0:8}; waiting for open $GATE_WANT on old HEAD"
openrid(){ grep -oE "open: round r[0-9]+ is open" "$RUNLOG" | tail -1 | awk '{print $3}'; }
until [ "$(openrid)" = "$GATE_WANT" ]; do
  sleep $((30*S)); [ -e "$H/disarm" ] && { log "disarmed"; exit 0; }
  [ -n "$(openrid)" ] && [ "$(openrid)" \> "$GATE_WANT" ] && { log "REFUSED: gate $GATE_WANT passed (open $(openrid))"; exit 3; }
done
now=$(git rev-parse HEAD)
[ "$now" = "$old" ] || { log "REFUSED: HEAD ${now:0:8} != ${old:0:8}"; exit 3; }
[ ! -e "$STOPF" ] || { log "REFUSED: stop file already exists"; exit 3; }
pgrep -f "$LOOPPAT" >/dev/null || { log "REFUSED: loop not running"; exit 3; }
pgrep -f "genfeed-deploy/(autodeploy|watch).s[h]" >/dev/null && { log "REFUSED: another Generator deploy running"; exit 3; }
GATE=$(openrid)
base=$(grep -c "STOPPED after the round, as asked" "$RUNLOG")
touch "$STOPF"; log "gate $GATE; HEAD ${now:0:8}; stop-after-round touched"
while :; do
  [ "$(grep -c "STOPPED after the round, as asked" "$RUNLOG")" -gt "$base" ] && { how="STOPPED marker"; break; }
  pgrep -f "$LOOPPAT" >/dev/null || { how="loop gone"; break; }
  [ -e "$H/disarm" ] && { rm -f "$STOPF"; log "disarmed before apply; nothing changed"; exit 0; }
  sleep $((30*S))
done
for i in $(seq 1 18); do pgrep -f "$LOOPPAT" >/dev/null || break; sleep $((10*S)); done
rm -f "$STOPF"
pgrep -f "$LOOPPAT" >/dev/null && { log "REFUSED: loop still alive after stop"; exit 3; }
log "loop ended ($how)"
$OP journal-close "$GATE" > "$H/gate-$GATE.json" 2>> "$L"
v=$(jc "$H/gate-$GATE.json"); log "journal gate $GATE: $v"
if [ "${v%% *}" != yes ]; then
  log "GATE FAILED: restarting unchanged code"; $OP start-loop >> "$L" 2>&1; exit 2
fi
touch "$WD/DISARM"; log "watchdog DISARM"
rearm(){ rm -f "$WD/DISARM"; log "watchdog re-armed"; }
if $OP apply >> "$L" 2>&1; then
  log "APPLY OK: HEAD $(git rev-parse --short=8 HEAD)"
else
  log "APPLY FAILED: old code restored, selector removed"; $OP start-loop >> "$L" 2>&1; rearm; exit 1
fi
# Ingest only the oldest complete pending epoch. Later/future epochs are
# skipped until ingestd starts after the first open-check PASS.
if $OP ingest-ready >> "$L" 2>&1; then
  log "INGEST READY OK"
else
  log "INGEST READY FAILED -> stop ingestd, remove selector, rollback"
  $OP stop-ingestd >> "$L" 2>&1
  $OP rollback >> "$L" 2>&1 || log "ROLLBACK FAILED"
  $OP start-loop >> "$L" 2>&1; rearm; exit 1
fi
OPEN0=$(grep -c "open: round r[0-9]* is open" "$RUNLOG")
LN0=$(wc -l < "$RUNLOG")
$OP start-loop >> "$L" 2>&1
log "GO-LIVE one loop started; ingestd waits for first open-check PASS; HEAD $(git rev-parse --short=8 HEAD)"
stop_then_rollback(){
  if pgrep -f "$LOOPPAT" >/dev/null; then
    b=$(grep -c "STOPPED after the round, as asked" "$RUNLOG"); touch "$STOPF"
    while [ "$(grep -c "STOPPED after the round, as asked" "$RUNLOG")" -le "$b" ]; do sleep $((30*S)); pgrep -f "$LOOPPAT" >/dev/null || break; done
    for i in $(seq 1 18); do pgrep -f "$LOOPPAT" >/dev/null || break; sleep $((10*S)); done; rm -f "$STOPF"
  fi
  $OP stop-ingestd >> "$L" 2>&1
  $OP rollback >> "$L" 2>&1 && log "ROLLBACK OK" || log "ROLLBACK FAILED"
  $OP start-loop >> "$L" 2>&1; rearm
}
fail(){ log "FAIL: $*"; stop_then_rollback; exit 1; }
refused(){ tail -n +$((LN0+1)) "$RUNLOG" | grep -q "run-loop: STOPPED: run refused"; }
deadline=$(( $(date +%s) + 3600 ))
while [ "$(grep -c "open: round r[0-9]* is open" "$RUNLOG")" -le "$OPEN0" ]; do
  sleep $((20*S))
  refused && fail "run refused after go-live"
  pgrep -f "$LOOPPAT" >/dev/null || fail "loop not alive before first open"
  [ "$(date +%s)" -gt "$deadline" ] && fail "no first open within 60 min"
done
FIRST=$(tail -n +$((LN0+1)) "$RUNLOG" | grep -oE "open: round r[0-9]+ is open" | head -1 | awk '{print $3}')
$OP open-check "$FIRST" > "$H/open-$FIRST.json" 2>> "$L"
oc=$(python3 -c "import json;s=json.load(open('$H/open-$FIRST.json'));print(('OK' if s['ok'] else 'BAD'),'admitted',s.get('admitted'),'gen',s.get('gen_admitted'),'retired',s.get('retired'),'size',s.get('size'),'params',s.get('params_hash'))")
log "first open $FIRST: $oc"
[ "${oc%% *}" = OK ] || fail "open-check: $oc"
$OP start-ingestd >> "$L" 2>&1
log "first open-check PASS; ingestd started for later epochs"
# Mirror every Generator shard and the first rotation, retrying for 20 minutes.
pubdl=$(( $(date +%s) + WAIT )); attempt=0
while :; do
  attempt=$((attempt+1))
  $OP public-check "$FIRST" > "$H/public-$FIRST.json" 2>> "$L"
  pc=$(python3 -c "import json;s=json.load(open('$H/public-$FIRST.json'));print(('OK' if s['ok'] else 'WAIT'),'generator_shards',s.get('generator_shards'),'manifest_shards',s.get('manifest_shards'),s.get('why') or '')" 2>&1)
  log "public poll $attempt: $pc"
  [ "${pc%% *}" = OK ] && break
  [ "$(date +%s)" -gt "$pubdl" ] && fail "public reserve mirror missing after 20 min: $pc"
  sleep $(( POLL*S ))
done
# Same loop process: first round closes cleanly by the journal, next opens.
deadline=$(( $(date +%s) + 5*3600 ))
while [ "$(grep -c "open: round r[0-9]* is open" "$RUNLOG")" -lt $((OPEN0+2)) ]; do
  sleep $((30*S))
  refused && fail "run refused after first round"
  [ "$(tail -n +$((LN0+1)) "$RUNLOG" | grep -c 'run-loop: loop started')" -gt 1 ] && fail "loop restarted during health watch"
  pgrep -f "$LOOPPAT" >/dev/null || fail "loop not alive"
  [ "$(date +%s)" -gt "$deadline" ] && fail "second open did not occur within 5 h"
  [ -e "$H/disarm" ] && { log "disarmed during health watch; no verdict"; rearm; exit 0; }
done
SECOND=$(tail -n +$((LN0+1)) "$RUNLOG" | grep -oE "open: round r[0-9]+ is open" | sed -n 2p | awk '{print $3}')
$OP journal-close "$FIRST" --next-open-ok > "$H/first-$FIRST.json" 2>> "$L"
fv=$(jc "$H/first-$FIRST.json")
log "first round $FIRST close: $fv; second $SECOND opened in same process"
[ "${fv%% *}" = yes ] || fail "first round close not clean: $fv"
rearm
log "PASS: Generator feed on ${new:0:8}; $FIRST admitted gen:* and closed clean; $SECOND opened in same process; reserve mirrored"
