#!/usr/bin/env bash
# Operator crown of chal-00687 as reign 22 (Jacob Steeves, 2026-09-26 18:51 UTC —
# "crown the last miner model who scored the best against the king and set weights
# to it immediately while we consider the cut over"). No scoring change, no wvk bump.
# Same pitfall handling as the fork deploys: env from /proc of the live validator
# (never pgrep), keepalive off for the window, deadman paused, pm2 stop at a duel
# boundary, stale in_flight cleared only with a verdict in history, NO pod redeploy
# (nothing on the pod changes; the king engine is reloaded by the validator's
# normal king swap on the next duel), validator + dash back, weights on the first
# sweep after the restart.
# Run on the box:  bash ops/v20/operator_crown_00687.sh
set -euo pipefail
HERE=/home/const/subnet120/ops/v20
REPO=/home/const/subnet120
LOG=$HERE/operator_crown_00687.log
PAUSE=/home/const/.affine/deadman.pause
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === operator_crown_00687.sh start (HEAD $(git rev-parse --short HEAD))"

# --- 0. preflight (nothing changes yet)
python ops/v20/operator_crown_00687.py --check
KING_BEFORE=$(python3 -c 'import json;k=json.load(open("affine/state/state.json"))["king"];print(k["challenge_id"], k["reign_number"], k["hotkey"][:12])')
echo "$(ts) king before: $KING_BEFORE"

# --- 1. capture the live validator env (before any stop)
VPID=$(pm2 pid affine-validator 2>/dev/null | tr -d '[:space:]' || true)
ENV_FILE=$HERE/.validator_env.$$
: > "$ENV_FILE"; chmod 600 "$ENV_FILE"
if [[ -n "$VPID" && "$VPID" != "0" && -r /proc/$VPID/environ ]]; then
  tr '\0' '\n' < /proc/"$VPID"/environ | grep -E '^(HF_TOKEN|AFFINE_[A-Z0-9_]+|R2_[A-Z0-9_]+|CLOUDFLARE_[A-Z0-9_]+|HIPPIUS_[A-Z0-9_]+|LIUM_API_KEY|TARGON_API_KEY|DISCORD_[A-Z0-9_]+)=' > "$ENV_FILE" || true
  echo "$(ts) env captured from /proc/$VPID/environ ($(wc -l < "$ENV_FILE") keys)"
fi
set -a; while IFS= read -r line; do export "$line"; done < "$ENV_FILE"; set +a
rm -f "$ENV_FILE"
for k in AFFINE_EVAL_TOKEN R2_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_TOKEN AFFINE_MAILBOX_SIGNING_SEED; do
  [[ -n "${!k:-}" ]] || { echo "$(ts) $k missing in env; abort (promotion needs it)"; exit 1; }
done
echo "$(ts) env ok"
health() { curl -s -m 8 -H "X-Affine-Token: $AFFINE_EVAL_TOKEN" http://127.0.0.1:9000/health 2>/dev/null || true; }
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

# --- 2. keepalive ralph off
KEEPALIVE_WAS_ON=0
if ./ralphs/ralphctl.sh keepalive status 2>/dev/null | head -1 | grep -q '^ON'; then KEEPALIVE_WAS_ON=1; fi
./ralphs/ralphctl.sh keepalive off >/dev/null 2>&1 || true
echo "$(ts) keepalive ralph: was_on=$KEEPALIVE_WAS_ON, now off"
reenable() { if [[ "$KEEPALIVE_WAS_ON" == 1 ]]; then ./ralphs/ralphctl.sh keepalive on >/dev/null 2>&1 && echo "$(ts) keepalive ralph re-enabled"; fi; }
trap reenable EXIT

# --- 3. duel boundary
START_CID=$(cur_cid)
echo "$(ts) boundary check (current in_flight: ${START_CID:-none}, pod busy=$(busy_of "$(health)"))"
reached=0
for i in $(seq 1 4200); do
  cid=$(cur_cid); h=$(health); busy=$(busy_of "$h")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi
  if [[ -n "$cid" ]] && has_verdict "$cid"; then reached=1; break; fi
  (( i % 60 == 0 )) && echo "$(ts)   busy=$busy in_flight=${cid:-none}"
  sleep 2
done
[[ "$reached" == 1 ]] || { echo "$(ts) no boundary in 140 min; abort (nothing changed)"; exit 1; }
echo "$(ts) boundary reached (in_flight now: $(cur_cid)) -> pause deadman, pm2 stop affine-validator"
mkdir -p "$(dirname "$PAUSE")" && touch "$PAUSE"
unpause() { rm -f "$PAUSE"; echo "$(ts) deadman pause removed"; }
trap 'unpause; reenable' EXIT
pm2 stop affine-validator >/dev/null
sleep 3

# --- 4. stale in_flight: clear only if its verdict is already in history
inf=$(cur_cid)
if [[ -n "$inf" ]]; then
  if has_verdict "$inf"; then
    cp affine/state/state.json "$HERE/state.before_inflight_clear.$(date -u +%Y%m%dT%H%M%SZ).json"
    python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'
    echo "$(ts) cleared stale in_flight $inf (verdict already in history)"
  else
    echo "$(ts) in_flight $inf has NO verdict in history — abort, nothing changed; rerun at the next boundary"; pm2 start affine-validator >/dev/null; exit 1
  fi
fi
cp affine/state/state.json "$HERE/state.before_operator_crown.$(date -u +%Y%m%dT%H%M%SZ).json"
cp affine/state/history.jsonl "$HERE/history.before_operator_crown.$(date -u +%Y%m%dT%H%M%SZ).jsonl"

# --- 5. the crown (promotion to models.affine.io + crowned row + king update + bench card)
python ops/v20/operator_crown_00687.py --apply
python3 -c 'import json;s=json.load(open("affine/state/state.json"));k=s["king"];print("king:", k["challenge_id"], "reign", k["reign_number"], k["revision"][:12], k["hotkey"][:12], "crowned_at", k["crowned_at"], "repo", k["repo"])'

# --- 6. llms.txt (reign / king lines are rendered from state)
(cd affine && python scripts/build_llms_txt.py | tail -1)

# --- 7. validator + dash back up; weights on the first sweep
pm2 start affine-validator >/dev/null
sleep 5
pm2 restart affine-dash >/dev/null 2>&1 && echo "$(ts) affine-dash restarted" || true
KING_AFTER=$(python3 -c 'import json;k=json.load(open("affine/state/state.json"))["king"];print(k["challenge_id"], k["reign_number"], k["hotkey"][:12])')
echo "$(ts) king after restart: $KING_AFTER"
W0=$(python3 -c 'import json;print(json.load(open("affine/state/state.json")).get("last_weights_at"))')
echo "$(ts) waiting for the weight sweep (last_weights_at was $W0)"
for i in $(seq 1 120); do
  W=$(python3 -c 'import json;print(json.load(open("affine/state/state.json")).get("last_weights_at"))')
  if [[ "$W" != "$W0" ]]; then echo "$(ts) weights set at $W"; break; fi
  sleep 10
done
pm2 logs affine-validator --nostream --lines 400 2>/dev/null | grep -E "payout sweep|set_weights|weights set|payout set changed" | tail -4 || true
echo "$(ts) done. Next: kingctl status (king seat swap), Discord lines, commit."
