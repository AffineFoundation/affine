#!/usr/bin/env bash
# Re-admit chal-00722 at the next duel boundary (operator directive 2026-09-28 15:23 UTC):
# keepalive off, boundary wait, deadman pause, pm2 stop, inject (requeue_00722.py --apply),
# pm2 start. No code/toml change, no pod redeploy.
set -euo pipefail
HERE=/home/const/subnet120/ops/v21
REPO=/home/const/subnet120
LOG=$HERE/requeue_00722.log
PAUSE=/home/const/.affine/deadman.pause
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === requeue_00722.sh start (HEAD $(git rev-parse --short HEAD))"
VPID=$(pm2 pid affine-validator 2>/dev/null | tr -d '[:space:]' || true)
TOK=$(tr '\0' '\n' < /proc/"$VPID"/environ | grep '^AFFINE_EVAL_TOKEN=' | cut -d= -f2)
[[ -n "$TOK" ]] || { echo "$(ts) no AFFINE_EVAL_TOKEN; abort"; exit 1; }
health() { curl -s -m 8 -H "X-Affine-Token: $TOK" http://127.0.0.1:9000/health 2>/dev/null || true; }
busy_of() { python3 -c 'import json,sys
try: print(json.loads(sys.argv[1]).get("busy"))
except Exception: print("?")' "$1"; }
cur_cid() { python3 -c 'import json;f=json.load(open("affine/state/state.json")).get("in_flight");print((f or {}).get("challenge_id","") if isinstance(f,dict) else (f or ""))'; }
has_verdict() { python3 - "$1" <<'PY'
import json, sys
cid = sys.argv[1]
sys.exit(0 if any(json.loads(l).get("challenge_id") == cid and json.loads(l).get("event") in ("verdict","crowned","failed") for l in open("affine/state/history.jsonl") if l.strip()) else 1)
PY
}
KEEPALIVE_WAS_ON=0; ./ralphs/ralphctl.sh keepalive status 2>/dev/null | head -1 | grep -q '^ON' && KEEPALIVE_WAS_ON=1
./ralphs/ralphctl.sh keepalive off >/dev/null 2>&1 || true
reenable() { [[ "$KEEPALIVE_WAS_ON" == 1 ]] && ./ralphs/ralphctl.sh keepalive on >/dev/null 2>&1 || true; }
trap reenable EXIT
START_CID=$(cur_cid); reached=0
echo "$(ts) boundary wait (in_flight: ${START_CID:-none})"
for i in $(seq 1 4200); do
  cid=$(cur_cid); busy=$(busy_of "$(health)")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi
  if [[ -n "$cid" ]] && has_verdict "$cid"; then reached=1; break; fi
  (( i % 60 == 0 )) && echo "$(ts)   busy=$busy in_flight=${cid:-none}"
  sleep 2
done
[[ "$reached" == 1 ]] || { echo "$(ts) no boundary in 140 min; abort"; exit 1; }
echo "$(ts) boundary reached -> pause deadman, pm2 stop"
mkdir -p "$(dirname "$PAUSE")" && touch "$PAUSE"
unpause() { rm -f "$PAUSE"; echo "$(ts) deadman pause removed"; }
trap 'pm2 start affine-validator >/dev/null 2>&1 || true; unpause; reenable' EXIT
pm2 stop affine-validator >/dev/null; sleep 3
inf=$(cur_cid)
if [[ -n "$inf" ]]; then
  if has_verdict "$inf"; then python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'; echo "$(ts) cleared stale in_flight $inf";
  else echo "$(ts) in_flight $inf has no verdict — abort, nothing changed"; exit 1; fi
fi
python ops/v21/requeue_00722.py --apply
pm2 start affine-validator >/dev/null; sleep 5
echo "$(ts) validator back; queue: $(python3 -c 'import json;print([q["challenge_id"] for q in json.load(open("affine/state/state.json"))["queue"]])')"
echo "$(ts) done"
