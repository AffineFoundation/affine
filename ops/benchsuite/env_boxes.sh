#!/bin/bash
# env_boxes.sh <reign> — add serving boxes to one king's env-row fill and re-split its remaining
# sources across ALL of its drivers (Jacob 2026-09-22 00:19: "a third box each for 14 and 15 and a
# second for 17, 16 and 18 ... each driver half pointed at its own box, --parallel 4–6").
#
# One driver (tmux session `backfill-<d12>[-<suffix>]` on the driver host) = one serving box. Each
# driver counts only its own data dir, so sources are DISJOINT across drivers (interleaved by the
# board's deficit so every driver gets a similar amount of work). The remaining sources come from the
# kingboard matrix (n < 24), never from a stale launch line. `affine_tau2` is excluded everywhere:
# its harness fails before the first model call (reign 13: infra 300 / graded 0) and renders "errored".
# Boxes rented here are released on their own driver's completion line, as before.
#
#   env_boxes.sh 14        # config block below says which drivers exist / which boxes to add
set -uo pipefail
cd ~/subnet120/ops/benchsuite
HERE=$PWD REPO=$HOME/subnet120 PY=$HOME/subnet120/.venv/bin/python
source env.sh 2>/dev/null; export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
REIGN="$1"
mkdir -p state/logs
LOG=state/logs/env_boxes_$REIGN.log
say() { echo "[env-boxes $REIGN] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# ---- per-reign config: driver host, existing drivers (suffix, "" = primary), new boxes (suffix=plan list),
#      sessions to leave alone (they keep their own sources), parallel per driver
case "$REIGN" in
  14) DRV=root@87.203.233.19:20099; EXIST="_ b"; NEW="d"; KEEP_SESS="c"; KEEP_SRC="affine_uuidctf,affine_prolog"; PAR=6 ;;
  15) DRV=root@87.203.233.19:20099; EXIST="_ b"; NEW="d"; KEEP_SESS="c"; KEEP_SRC="affine_uuidctf,affine_prolog"; PAR=6 ;;
  16) DRV=root@73.139.34.205:20008; EXIST="_";   NEW="b"; KEEP_SESS="";  KEEP_SRC=""; PAR=5 ;;
  17) DRV=root@73.139.34.205:20008; EXIST="_";   NEW="b"; KEEP_SESS="";  KEEP_SRC=""; PAR=5 ;;
  18) DRV=root@73.139.34.205:20008; EXIST="";    NEW="_ b"; KEEP_SESS=""; KEEP_SRC=""; PAR=5 ;;   # no live box, no driver since the 07:45 reboot
  19) DRV=root@73.139.34.205:20008; EXIST="_";   NEW="";  KEEP_SESS="";  KEEP_SRC=""; PAR=4 ;;    # re-split only: stale list lacked kb_synth/tau2_gen/mrcr
  13) DRV=root@73.139.34.205:20008; EXIST="_";   NEW="";  KEEP_SESS="";  KEEP_SRC=""; PAR=3 ;;    # driver completed 18:39 with a stale list; box 67c0 idle since
  12) DRV=root@73.139.34.205:20008; EXIST="_";   NEW="";  KEEP_SESS="";  KEEP_SRC=""; PAR=4 ;;
  *) echo "no config for reign $REIGN"; exit 2 ;;
esac
PLANS="${PLANS:-b200-1x h200-1x pro6000-1x}"
U=${DRV%%@*}; HP=${DRV#*@}; H=${HP%%:*}; P=${HP##*:}
sshd() { ssh -n -o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$P" "$U@$H" "$@" 2>/dev/null | grep -v setlocale; }

rev=$(python3 - "$REIGN" <<'PY'
import json, sys
s = json.load(open("/home/const/subnet120/affine/state/state.json"))["king"]
r = sys.argv[1]
print(s["revision"] if str(s["reign_number"]) == r else next(p["revision"] for p in s["previous"] if str(p["reign_number"]) == r))
PY
)
d12=${rev:0:12}
say "reign $REIGN = $d12; driver host $H; existing drivers [$EXIST]; new boxes [$NEW]"

# ---- 1. rent the new boxes in parallel (weights pull + vLLM ≈ 15–25 min each)
declare -A POD URL KEY
rent_one() {  # suffix -> writes state/logs/env_boxes_<reign>_<suffix>.pod with "pod url key"
  local suf="$1" pod="" plan
  for plan in $PLANS; do
    pod=$($PY kingpod.py rent --plan "$plan" --digest "$rev" --expected-hours 24 2>>"$LOG" | tail -1)
    [ -n "$pod" ] && [[ "$pod" == bench-king-* ]] && break; pod=""
  done
  [ -n "$pod" ] || { say "box $suf: no stock in any of [$PLANS]"; return 3; }
  say "box $suf: rented $pod ($plan)"
  local url; url=$(timeout 2700 $PY kingpod.py wait "$pod" 2>>"$LOG" | tail -1)
  if [ -z "$url" ] || [[ "$url" != http* ]]; then
    say "box $suf: $pod never served; releasing"; $PY kingpod.py release "$pod" --strike "env box never served" >>"$LOG" 2>&1; return 4
  fi
  local key; key=$($PY -c "import json;print(json.load(open('state/pods.json'))['$pod']['key'])")
  echo "$pod $url $key" > "state/logs/env_boxes_${REIGN}_${suf}.pod"
  say "box $suf: $pod READY at $url"
}
pids=()
for suf in $NEW; do rm -f "state/logs/env_boxes_${REIGN}_${suf}.pod"; rent_one "$suf" & pids+=($!); done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done
NEW_OK=""
for suf in $NEW; do
  f="state/logs/env_boxes_${REIGN}_${suf}.pod"
  [ -f "$f" ] || continue
  read -r POD[$suf] URL[$suf] KEY[$suf] < "$f"; NEW_OK="$NEW_OK $suf"
done
DRIVERS="$EXIST$NEW_OK"
[ -n "$(echo $DRIVERS)" ] || { say "no driver has a box; nothing to do"; exit 3; }

# ---- 2. the row's remaining sources from the board (n < 24), most-behind first
GAPS=$(python3 - "$REIGN" "$KEEP_SRC" <<'PY'
import json, sys, urllib.request
mx = json.load(urllib.request.urlopen("http://127.0.0.1:8790/api/matrix.json", timeout=60))
cols = [c.get("key") if isinstance(c, dict) else c for c in mx["columns"]]
envs = [c[4:] for c in cols if c.startswith("env:") and c not in ("env:affine_wiki", "env:affine_tau2")]
keep = {s for s in sys.argv[2].split(",") if s}
row = next(r for r in mx["rows"] if r.get("label") == f"King {sys.argv[1]}")
cells = row.get("cells") or {}
gaps = sorted(((cells.get("env:" + e) or {}).get("n") or 0, e) for e in envs if ((cells.get("env:" + e) or {}).get("n") or 0) < 24 and e not in keep)
print(" ".join(e for _, e in gaps))
PY
)
[ -n "$GAPS" ] || { say "row is full on the board; releasing any box rented here"; for suf in $NEW_OK; do $PY kingpod.py release "${POD[$suf]}" >>"$LOG" 2>&1; done; exit 0; }
say "remaining sources ($(echo $GAPS | wc -w)): $GAPS"

# ---- 3. interleave the sources over the drivers
declare -A SRC
i=0; drv=($DRIVERS); n=${#drv[@]}
for s in $GAPS; do d=${drv[$((i % n))]}; SRC[$d]="${SRC[$d]:+${SRC[$d]},}$s"; i=$((i + 1)); done

# ---- 4. on the driver host: king_env + wrapper for the new drivers, restart every driver with its share
REMOTE="cd /root/rollouts || exit 9; "
for suf in $NEW_OK; do
  if [ "$suf" = "_" ]; then kenv=".king_env_$d12"; else kenv=".king_env_${d12}_$suf"; fi
  REMOTE+="printf '%s\n' '# env box for reign $REIGN, written by ops/benchsuite/env_boxes.sh $(date -u +%FT%TZ) (${POD[$suf]})' 'KING_BASE_URL=${URL[$suf]}' 'KING_MODEL=king-$d12' 'KING_KEY=${KEY[$suf]}' 'KING_DIGEST=$rev' 'KING_REIGN=$REIGN' > $kenv; "
done
for d in $DRIVERS; do
  if [ "$d" = "_" ]; then sess="backfill-$d12"; wrap="run_backfill_$d12.sh"; data="/root/rollouts-data-$d12"; kenv="/root/rollouts/.king_env_$d12"; logf="/root/logs/backfill_$d12.log"
  else sess="backfill-$d12-$d"; wrap="run_backfill_${d12}_$d.sh"; data="/root/rollouts-data-$d12-$d"; kenv="/root/rollouts/.king_env_${d12}_$d"; logf="/root/logs/backfill_${d12}_$d.log"; fi
  REMOTE+="cat > $wrap <<'EOW'
#!/bin/bash
set -uo pipefail
cd /root/rollouts
source /root/affine/.datagen_env; source /root/rollouts/.rollouts_env
export PATH=/root/.local/bin:\$PATH PYTHONPATH=/root/affine:/root/rollouts
export ROLLOUTS_DATA_DIR=$data ROLLOUTS_KING_ENV=$kenv ROLLOUTS_MAX_CONTAINERS=12 ROLLOUTS_BATCH_SIZE=12
[ \"\${ROLLOUTS_R2_PREFIX:-}\" = traces-backfill/ ] || { echo \"refusing: ROLLOUTS_R2_PREFIX=\${ROLLOUTS_R2_PREFIX:-}\"; exit 2; }
exec /root/venv/bin/python -m rollouts.backfill \"\$@\"
EOW
chmod +x $wrap; tmux kill-session -t $sess 2>/dev/null; "
done
REMOTE+="sleep 3; "
for d in $DRIVERS; do
  [ -n "${SRC[$d]:-}" ] || continue
  if [ "$d" = "_" ]; then sess="backfill-$d12"; wrap="run_backfill_$d12.sh"; logf="/root/logs/backfill_$d12.log"
  else sess="backfill-$d12-$d"; wrap="run_backfill_${d12}_$d.sh"; logf="/root/logs/backfill_${d12}_$d.log"; fi
  REMOTE+="tmux new-session -d -s $sess \"/root/rollouts/$wrap --digest12 $d12 --n 50 --parallel $PAR --worker-batch 6 --sources '${SRC[$d]}' >> $logf 2>&1\"; "
  say "driver $sess ← [${SRC[$d]}] parallel $PAR"
done
REMOTE+="sleep 4; tmux ls | grep -c backfill-$d12"
started=$(sshd "$REMOTE")
say "driver sessions for $d12 on $H now: ${started:-?}"

# ---- 5. release each NEW box on its own driver's completion line
for suf in $NEW_OK; do
  if [ "$suf" = "_" ]; then logf="/root/logs/backfill_$d12.log"; else logf="/root/logs/backfill_${d12}_$suf.log"; fi
  (
    while :; do
      sleep 600
      if sshd "grep -q 'backfill $d12 complete' $logf && echo DONE" | grep -q DONE; then break; fi
    done
    say "driver $suf complete; releasing ${POD[$suf]}"; $PY kingpod.py release "${POD[$suf]}" >>"$LOG" 2>&1
  ) &
done
wait
say "done"
