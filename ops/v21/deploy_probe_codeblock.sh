#!/usr/bin/env bash
# Protocol-probe code-fence cases in SHADOW (admission rule, no wvk, no toml contract change):
# ship evalsrv/protocol_probe.py + dueling.py + toml ([protocol_probe].shadow_ids) to the eval pod
# at a duel boundary; validator restarted so its toml copy matches. Same pitfall handling as
# the fork deploys. Run on the box: bash ops/v21/deploy_probe_codeblock.sh
set -euo pipefail
HERE=/home/const/subnet120/ops/v21
REPO=/home/const/subnet120
LOG=$HERE/deploy_probe_codeblock.log
PAUSE=/home/const/.affine/deadman.pause
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_probe_codeblock.sh start (HEAD $(git rev-parse --short HEAD))"
grep -q '^shadow_ids = \["ide_code_block_only"' affine/affine.toml || { echo "$(ts) toml lacks shadow_ids; abort"; exit 1; }
python -c 'import sys; sys.path.insert(0,"affine"); from affine.config import load_config; from evalsrv.protocol_probe import probe_settings; c=load_config(); print("probe settings:", probe_settings(c.raw["protocol_probe"]))'
KING_BEFORE=$(python3 -c 'import json;k=json.load(open("affine/state/state.json"))["king"];print(k["challenge_id"], k["reign_number"])')
POD_SSH_STR=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["eval_machine"]["ssh"])')
read -r POD_USERHOST _ POD_PORT <<<"$POD_SSH_STR"
POD_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$POD_PORT" "$POD_USERHOST")
VPID=$(pm2 pid affine-validator 2>/dev/null | tr -d '[:space:]' || true)
ENV_FILE=$HERE/.validator_env.$$; : > "$ENV_FILE"; chmod 600 "$ENV_FILE"
tr '\0' '\n' < /proc/"$VPID"/environ | grep -E '^(HF_TOKEN|AFFINE_[A-Z0-9_]+|R2_[A-Z0-9_]+|CLOUDFLARE_[A-Z0-9_]+|HIPPIUS_[A-Z0-9_]+|LIUM_API_KEY|TARGON_API_KEY)=' > "$ENV_FILE" || true
set -a; while IFS= read -r line; do export "$line"; done < "$ENV_FILE"; set +a; rm -f "$ENV_FILE"
for k in HF_TOKEN AFFINE_EVAL_TOKEN R2_ENDPOINT AFFINE_EVAL_R2_ACCESS_KEY_ID AFFINE_EVAL_R2_SECRET_ACCESS_KEY; do [[ -n "${!k:-}" ]] || { echo "$(ts) $k missing; abort"; exit 1; }; done
echo "$(ts) env ok; king before: $KING_BEFORE"
health() { curl -s -m 8 -H "X-Affine-Token: $AFFINE_EVAL_TOKEN" http://127.0.0.1:9000/health 2>/dev/null || true; }
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
for i in $(seq 1 4200); do
  cid=$(cur_cid); busy=$(busy_of "$(health)")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi
  if [[ -n "$cid" ]] && has_verdict "$cid"; then reached=1; break; fi
  (( i % 60 == 0 )) && echo "$(ts)   busy=$busy in_flight=${cid:-none}"
  sleep 2
done
[[ "$reached" == 1 ]] || { echo "$(ts) no boundary in 140 min; abort"; exit 1; }
echo "$(ts) boundary reached -> pause deadman, pm2 stop affine-validator"
mkdir -p "$(dirname "$PAUSE")" && touch "$PAUSE"
unpause() { rm -f "$PAUSE"; echo "$(ts) deadman pause removed"; }
trap 'unpause; reenable' EXIT
pm2 stop affine-validator >/dev/null; sleep 3
inf=$(cur_cid)
if [[ -n "$inf" ]]; then
  if has_verdict "$inf"; then python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'; echo "$(ts) cleared stale in_flight $inf";
  else echo "$(ts) in_flight $inf has no verdict — abort"; pm2 start affine-validator >/dev/null; exit 1; fi
fi
(cd affine && python scripts/build_llms_txt.py | tail -1)
cd affine && python scripts/redeploy_pods.py --role eval && cd "$REPO"
"${POD_SSH[@]}" 'grep -E "^(weight_version_key|shadow_ids) " /root/affine/affine.toml; grep -c "assistant_code_only_he" /root/affine/evalsrv/protocol_probe.py' || echo "$(ts) WARNING pod verify ssh failed"
pm2 start affine-validator >/dev/null; sleep 5
pm2 restart affine-dash >/dev/null 2>&1 || true
for i in $(seq 1 60); do h=$(health); [[ -n "$h" ]] && { echo "$(ts) pod health: ${h:0:120}"; break; }; sleep 5; done
echo "$(ts) king after: $(python3 -c 'import json;k=json.load(open("affine/state/state.json"))["king"];print(k["challenge_id"], k["reign_number"])')"
echo "$(ts) done. Next verdicts publish protocol_probe.shadow."
