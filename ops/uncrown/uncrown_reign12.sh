#!/usr/bin/env bash
# Uncrown reign 12 (chal-00454, king-d76150805915, uid 156) -> reign 11
# (chal-00409, king-0ce59769300c) stands. Operator directive 2026-09-14
# 12:46 UTC ("Remove the last king also, it was a spam model").
# Duel boundary -> deadman paused -> pm2 stop -> in_flight pitfall -> uncrown
# -> dash history index reset (in-place row rewrite is invisible to its
# byte-offset ingest) -> pm2 start -> dash restart -> paid set / weights.
set -euo pipefail
HERE=/home/const/subnet120/ops/uncrown
REPO=/home/const/subnet120
LOG=$HERE/uncrown_reign12.log
PAUSE=/home/const/.affine/deadman.pause
CID=chal-00454
RESTORE=chal-00409
CODE=revoked_operator_spam_model
REASON="operator decision: reign 12 (king-d76150805915) was a spam model"
DIRECTIVE="operator directive 2026-09-14 12:46 UTC (Jacob Steeves: Remove the last king also, it was a spam model)"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === uncrown_reign12.sh start (HEAD $(git rev-parse --short HEAD))"
python ops/uncrown/uncrown_king.py --cid "$CID" --restore-cid "$RESTORE" --code "$CODE" --reason "$REASON" --directive "$DIRECTIVE" --check

set -a; source ops/t0/validator_env.sh >/dev/null 2>&1 || true; set +a
health() { curl -s -m 8 -H "X-Affine-Token: ${AFFINE_EVAL_TOKEN:-}" http://127.0.0.1:9000/health 2>/dev/null || true; }
busy_of() { python3 -c 'import json,sys
try: print(json.loads(sys.argv[1]).get("busy"))
except Exception: print("?")' "$1"; }
cur_cid() { python3 -c 'import json;f=json.load(open("affine/state/state.json")).get("in_flight");print((f or {}).get("challenge_id","") if isinstance(f,dict) else (f or ""))'; }
has_verdict() { python3 - "$1" <<'PY'
import json, sys
cid = sys.argv[1]
ok = any(json.loads(l).get("challenge_id") == cid and json.loads(l).get("event") in ("verdict", "crowned", "failed")
         for l in open("affine/state/history.jsonl") if l.strip())
sys.exit(0 if ok else 1)
PY
}

# keepalive ralph off (restarts a stopped validator)
KEEPALIVE_WAS_ON=0
if ./ralphs/ralphctl.sh keepalive status 2>/dev/null | head -1 | grep -q '^ON'; then KEEPALIVE_WAS_ON=1; fi
./ralphs/ralphctl.sh keepalive off >/dev/null 2>&1 || true
reenable() { if [[ "$KEEPALIVE_WAS_ON" == 1 ]]; then ./ralphs/ralphctl.sh keepalive on >/dev/null 2>&1 || true; fi; }
unpause() { rm -f "$PAUSE"; }
trap 'unpause; reenable' EXIT

START_CID=$(cur_cid)
echo "$(ts) waiting for duel boundary (current in_flight: ${START_CID:-none})"
reached=0
for i in $(seq 1 2400); do
  cid=$(cur_cid); busy=$(busy_of "$(health)")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi
  if [[ -n "$cid" ]] && has_verdict "$cid"; then reached=1; break; fi
  (( i % 60 == 0 )) && echo "$(ts)   busy=$busy in_flight=${cid:-none}"
  sleep 2
done
[[ "$reached" == 1 ]] || { echo "$(ts) no boundary in 80 min; abort (nothing changed)"; exit 1; }
echo "$(ts) boundary reached (in_flight now: $(cur_cid)) -> pause deadman, pm2 stop affine-validator"
mkdir -p "$(dirname "$PAUSE")" && touch "$PAUSE"
pm2 stop affine-validator >/dev/null
sleep 3

inf=$(cur_cid)
if [[ -n "$inf" ]]; then
  if has_verdict "$inf"; then
    cp affine/state/state.json "$HERE/state.before_inflight_clear.$(date -u +%Y%m%dT%H%M%SZ).json"
    python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'
    echo "$(ts) cleared stale in_flight $inf (verdict already in history)"
  else
    echo "$(ts) in_flight $inf has no verdict — left in place; State.load requeues it (it will duel the restored king)"
  fi
fi

python ops/uncrown/uncrown_king.py --cid "$CID" --restore-cid "$RESTORE" --code "$CODE" --reason "$REASON" --directive "$DIRECTIVE" --apply

echo "$(ts) reset dash history index (in-place rewrite of the crowned row)"
pm2 stop affine-dash >/dev/null
cp affine/state/dash.sqlite "$HERE/dash.sqlite.bak-$(date -u +%Y%m%dT%H%M%SZ)"
python3 -c 'import sqlite3;c=sqlite3.connect("affine/state/dash.sqlite");c.execute("DELETE FROM history");c.execute("UPDATE meta SET value=\x270\x27 WHERE key=\x27history_offset\x27");c.commit();print("dash history index reset")'

echo "$(ts) pm2 start affine-validator + affine-dash"
pm2 start affine-validator --update-env >/dev/null
pm2 start affine-dash >/dev/null
sleep 8
unpause; echo "$(ts) deadman pause removed"
python3 -c 'import json;s=json.load(open("affine/state/state.json"));k=s["king"];print("king:", k["challenge_id"], k["revision"][:12], "reign", k["reign_number"], k["hotkey"][:12], "crowned_at", k["crowned_at"])'
curl -s -m 20 "http://127.0.0.1:8787/api/v1/history?q=$CID&limit=5" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("dash rows for the removed king:", [(r["event"], r.get("revoked_code")) for r in d.get("items") or []])'
echo "$(ts) uncrown done. Next: kingctl status, payout sweep (paid set / weights), Discord."
