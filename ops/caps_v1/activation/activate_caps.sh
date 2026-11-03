#!/bin/bash
# RA/20261103-caps-canonical activation (Jacob t1148u/pt1148u: one canonical, easy-to-change token cap; the 8192 prior pod must not recur).
#   MODE=switch (default): in an idle decision gap (activate_12k.sh rule), lifecycle + mining-discovery stopped, both lifecycle locks held:
#     back up, install the staged code (training_params / site_metrics read caps.py; watchdog CR stamps; replace_pod {CAPS:..} + roles.json),
#     render the host consumers from caps.json (caps_apply --locks-held: probe config defaults, prior serve.sh/README), restart the
#     publishers + watchdogs, start the lifecycle, release the locks, verify (caps_check host,pods + published JSON) - all values unchanged.
#   MODE=revert: restore every backed-up file, restart the same units, remove state/caps/ACTIVE.json.
#     One-line revert: MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261103-caps-canonical/activate_caps.sh
set -uo pipefail
MODE=${MODE:-switch}
REPO=/home/const/subnet120-rewrite; RA=$REPO/state/root-audits; D=$RA/20261103-caps-canonical; STG=$D/stage
STATE=$REPO/state/mimo-mainnet-v1; PY=$REPO/.venv/bin/python; CAPSD=$REPO/state/caps
LOG=$D/activate.log; cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%Y-%m-%d %H:%M CR'): activate_caps $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
# live path <- staged file
declare -A INST=(
  [$RA/20261026-promotion-gate/code/training_params.py]=$STG/code/training_params.py
  [$REPO/dashboard/site_metrics.py]=$STG/code/site_metrics.py
  [$RA/20261102-12k-everywhere/activation/watchdog_12k.py]=$STG/code/watchdog_12k.py
  [$RA/20261102-step-mult/activation/watchdog_sm.py]=$STG/code/watchdog_sm.py
  [$REPO/ops/pods/replace_pod.py]=$STG/pods/replace_pod.py
  [$REPO/ops/pods/roles.json]=$STG/pods/roles.json
)
RENDERED="$RA/20261022-env-probe/code/config.json $REPO/priors/pod/serve.sh $REPO/priors/README.md"
UNITS="affine-training-params affine-mining-discovery affine-k1-transparency"
WATCHDOGS="12k-watchdog sm-watchdog"
idle(){ $PY -B -c "import sys;sys.path.insert(0,'$REPO/tools/caps');import caps_apply as A;w=A.idle();print(w or 'idle');sys.exit(1 if w else 0)"; }

case $MODE in
switch)
  [[ -e $CAPSD/ACTIVE.json ]] && die "already active ($CAPSD/ACTIVE.json)"
  for live in "${!INST[@]}"; do [[ -f ${INST[$live]} && -f $live ]] || die "missing ${INST[$live]} or $live"; done
  for f in $STG/code/*.py $STG/pods/replace_pod.py; do $PY -m py_compile "$f" || die "py_compile $f"; done
  $PY -c "import json;json.load(open('$STG/pods/roles.json'))" || die "roles.json"
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
  log "idle gap at e$($PY -c "import json;print(json.load(open('$STATE/CURRENT.json'))['epoch_index'])"): lifecycle + mining-discovery stopped, locks held"
  for live in "${!INST[@]}" $RENDERED; do mkdir -p "$B/$(dirname $live)"; cp -p "$live" "$B$live"; done
  log "backed up $(find $B -type f | wc -l) files -> $B"
  for live in "${!INST[@]}"; do install -m "$(stat -c %a $live)" "${INST[$live]}" "$live.caps-new" && mv -f "$live.caps-new" "$live" || die "install $live"; done
  log "installed: ${!INST[*]}"
  $PY -B $REPO/tools/caps/caps_apply.py --apply --locks-held --no-pods 2>&1 | tail -12 | tee -a $LOG
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started, locks released"
  for u in $UNITS; do systemctl --user restart $u.service && log "$u restarted"; done
  for u in $WATCHDOGS; do systemctl --user is-active --quiet $u.service && systemctl --user restart $u.service && log "$u restarted (CR stamp fix)"; done
  systemctl --user start affine-site-metrics.service && log "site-metrics run"
  for f in $RA/20261102-12k-everywhere/activation/watchdog.log $RA/20261102-step-mult/activation/watchdog.log; do
    echo "{\"event\": \"note\", \"msg\": \"at_cr on earlier lines is UTC (host clock) mislabeled CR; fixed by RA/20261103-caps-canonical\", \"at_cr\": \"$(TZ=America/Costa_Rica date '+%F %T') CR\"}" >> $f; done
  prog "LIVE: caps.json is the token-cap source (training_params + site_metrics read caps.py; roles.json {CAPS:..}; probe/prior rendered); backup $B"
  log done ;;
revert)
  B=$(cat $D/LAST-BACKUP); [[ -d $B ]] || die "no backup"
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true
  systemctl --user stop affine-mimo-lifecycle.service affine-mining-discovery.service
  (cd $B && find . -type f) | while read -r f; do live=${f#.}; install -m "$(stat -c %a $B$live)" "$B$live" "$live.caps-old" && mv -f "$live.caps-old" "$live" && log "restored $live"; done
  rm -f $CAPSD/ACTIVE.json
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-
  for u in $UNITS; do systemctl --user restart $u.service || true; done
  for u in $WATCHDOGS; do systemctl --user is-active --quiet $u.service && systemctl --user restart $u.service || true; done
  systemctl --user start affine-site-metrics.service || true
  prog "REVERTED (${REASON:-manual}) from $B"; log done ;;
*) die "MODE=switch|revert" ;;
esac
