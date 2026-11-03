#!/bin/bash
# RA/20261103-gain-mult-removal (Jacob t1150u/pt1150u, 2026-11-03: "remove the multiplier, it just adds complexity").
#   MODE=switch (default): in an idle decision gap (activate_12k.sh rule), lifecycle + mining-discovery stopped, both lifecycle locks held:
#     back up, install the pre-multiplier k1_service.py (every payee 1/N, formula k1-last10-green-equal-v1) and prior controller.py,
#     controller.json without parent_hold, training_params.py without training.k1.king_gain_multiplier, llms.txt without the
#     section-8 multiplier bullet, tools/caps without the gain-mult regime step; archive GAIN-MULT.*.json + the unit files into the
#     backup; move RA/20261102-gain-mult to RA/archived/; restart k1 / training-params / prior-table / k1-transparency; verify.
#   MODE=revert: restore every backed-up file, the GAIN-MULT state, the units (timer re-enabled) and the RA dir; restart the same units.
#     One-line revert: MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261103-gain-mult-removal/activate_removal.sh
set -uo pipefail
MODE=${MODE:-switch}
REPO=/home/const/subnet120-rewrite; RA=$REPO/state/root-audits; D=$RA/20261103-gain-mult-removal; STG=$D/stage
STATE=$REPO/state/mimo-mainnet-v1; PY=$REPO/.venv/bin/python; W=/home/const/subnet120/affine/website
G=$RA/20261102-gain-mult; GA=$RA/archived/20261102-gain-mult; UD=/home/const/.config/systemd/user
CTL=$RA/20261101-prior-table/logs/controller.log
LOG=$D/activate.log; cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%Y-%m-%d %H:%M CR'): activate_removal $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
declare -A INST=(
  [$RA/20261027-k1-per-uid/code/k1_service.py]=$STG/k1_service.py
  [$RA/20261026-promotion-gate/code/training_params.py]=$STG/training_params.py
  [$REPO/priors/controller.py]=$STG/controller.py
  [$RA/20261101-prior-table/controller.json]=$STG/controller.json
  [$W/llms.txt]=$STG/llms.txt
  [$REPO/tools/caps/caps_apply.py]=$STG/caps_apply.py
  [$REPO/tools/caps/consumers.py]=$STG/consumers.py
  [$REPO/tools/caps/README.md]=$STG/README.md
)
UNITS="affine-k1 affine-training-params affine-k1-transparency"
idle(){ $PY -B -c "import sys;sys.path.insert(0,'$REPO/tools/caps');import caps_apply as A;w=A.idle();print(w or 'idle');sys.exit(1 if w else 0)"; }
# the prior controller is between head changes (not fetching/serving): last event is not head_change/fetched
ctl_idle(){ tail -1 $CTL | grep -qvE '"event": "(head_change|fetched)"'; }
restart_ctl(){ for i in $(seq 1 120); do ctl_idle && break; sleep 10; done
  systemctl --user restart affine-prior-table.service && log "affine-prior-table restarted ($(tail -1 $CTL | cut -c1-60))"; }

case $MODE in
switch)
  [[ -d $GA ]] && die "already archived ($GA)"
  for live in "${!INST[@]}"; do [[ -f ${INST[$live]} && -f $live ]] || die "missing ${INST[$live]} or $live"; done
  for f in $STG/*.py; do $PY -m py_compile "$f" || die "py_compile $f"; done
  $PY -c "import json;c=json.load(open('$STG/controller.json'));assert 'parent_hold' not in c" || die "controller.json"
  grep -qiE 'multiplier|gain_m' $STG/llms.txt && die "llms.txt still mentions the multiplier"
  systemctl --user is-enabled --quiet affine-gain-mult.timer && die "affine-gain-mult.timer still enabled"
  B=$D/backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p -m 700 $B; echo $B > $D/LAST-BACKUP
  log "preflight ok; waiting for an idle decision gap"
  ok=0; for i in $(seq 1 ${RWAIT:-2160}); do
    w=$(idle) || { [[ $((i % 60)) == 1 ]] && log "not idle: $w"; sleep 5; continue; }
    exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
    if flock -w 30 9 && flock -w 30 8; then
      if idle >/dev/null; then systemctl --user stop affine-mimo-lifecycle.service affine-mining-discovery.service; ok=1; break; fi
      exec 8>&- 9>&-
    fi; sleep 5; done
  [[ $ok == 1 ]] || die "no idle decision gap within the wait"
  EP=$($PY -c "import json;print(json.load(open('$STATE/CURRENT.json'))['epoch_index'])"); echo $EP > $D/ACTIVATION-EPOCH
  log "idle gap at e$EP: lifecycle + mining-discovery stopped, locks held"
  for live in "${!INST[@]}"; do mkdir -p "$B/files$(dirname $live)"; cp -p "$live" "$B/files$live"; done
  mkdir -p -m 700 $B/gain-mult-state $B/units
  for f in GAIN-MULT.private.json GAIN-MULT.public.json; do [[ -f $G/state/$f ]] && mv $G/state/$f $B/gain-mult-state/; done
  for u in affine-gain-mult.service affine-gain-mult.timer; do [[ -f $UD/$u ]] && mv $UD/$u $B/units/; done
  systemctl --user daemon-reload; systemctl --user reset-failed affine-gain-mult.service 2>/dev/null
  log "backed up $(find $B -type f | wc -l) files -> $B (GAIN-MULT state + unit files moved there)"
  for live in "${!INST[@]}"; do install -m "$(stat -c %a $live)" "${INST[$live]}" "$live.gm-new" && mv -f "$live.gm-new" "$live" || die "install $live"; done
  log "installed: ${!INST[*]}"
  mkdir -p $RA/archived && mv $G $GA && chmod a-x $GA/code/activate.sh $GA/revert.sh
  cat > $GA/ARCHIVED.md <<NOTE
# ARCHIVED $(TZ=America/Costa_Rica date '+%Y-%m-%d %H:%M CR') (activation epoch e$EP)
The king gain multiplier was removed (Jacob t1150u/pt1150u: "I think we should remove the multiplier it just adds complexity").
Every payee is back to the equal 1/N split (k1-last10-green-equal-v1); the prior parent_hold is gone; mining.json no longer has
training.k1.king_gain_multiplier. Do not run code/activate.sh or revert.sh here (both made non-executable).
GAIN-MULT.private/public.json and the unit files: $B. Removal + revert: $D (MODE=revert bash $D/activate_removal.sh).
NOTE
  log "moved $G -> $GA (ARCHIVED.md)"
  systemctl --user start affine-mimo-lifecycle.service affine-mining-discovery.service; exec 8>&- 9>&-; log "lifecycle + mining-discovery started, locks released"
  for u in $UNITS; do systemctl --user restart $u.service && log "$u restarted"; done
  restart_ctl
  prog "LIVE from e$EP: multiplier removed (k1 1/N, no parent_hold, no king_gain_multiplier, llms s8 bullet gone); backup $B"
  log done ;;
revert)
  B=$(cat $D/LAST-BACKUP); [[ -d $B/files ]] || die "no backup"
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true
  systemctl --user stop affine-mimo-lifecycle.service affine-mining-discovery.service
  (cd $B/files && find . -type f) | while read -r f; do live=${f#.}; install -m "$(stat -c %a $B/files$live)" "$B/files$live" "$live.gm-old" && mv -f "$live.gm-old" "$live" && log "restored $live"; done
  [[ -d $GA && ! -e $G ]] && mv $GA $G && chmod u+x $G/code/activate.sh $G/revert.sh && rm -f $G/ARCHIVED.md && log "restored $G"
  for f in $B/gain-mult-state/*; do [[ -f $f ]] && cp -p $f $G/state/; done
  for u in $B/units/*; do [[ -f $u ]] && cp -p $u $UD/; done
  systemctl --user daemon-reload; systemctl --user enable --now affine-gain-mult.timer && log "affine-gain-mult.timer re-enabled"
  systemctl --user start affine-mimo-lifecycle.service affine-mining-discovery.service; exec 8>&- 9>&-
  for u in $UNITS; do systemctl --user restart $u.service || true; done
  restart_ctl
  prog "REVERTED (${REASON:-manual}) from $B"; log done ;;
*) die "MODE=switch|revert" ;;
esac
