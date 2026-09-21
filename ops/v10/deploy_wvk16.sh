#!/usr/bin/env bash
# wvk 15 -> 16 (per-duel crown restored) + uncrown reign 13, at the next duel
# boundary. Operator directive 2026-09-13 12:10 UTC. Modeled on
# /tmp/v9flip/deploy.sh with the AGENTS.md pitfalls closed:
#   - env for redeploy_pods.py from the LIVE validator's /proc/<pm2 pid>/environ
#     BEFORE pm2 stop (never a pgrep -f match); ops/t0/validator_env.sh fallback
#   - keepalive ralph off for the window (it restarts a stopped validator)
#   - a stale in_flight is cleared only if its verdict is already in history.jsonl
#   - toml flip + state revert happen while the validator is STOPPED
#   - pod toml verified (wvk 16, crown_mode duel, near_miss off) after redeploy
#   - affine-dash restarted so /api/v1/contract re-reads the toml
# Order inside the stop window: toml flip -> llms.txt build -> uncrown -> pod
# redeploy -> validator start. Discord is NOT posted here (separate step,
# after everything is verified live).
set -euo pipefail
HERE=/home/const/subnet120/ops/v10
REPO=/home/const/subnet120
LOG=$HERE/deploy_wvk16.log
DIRECTIVE_DATE=2026-09-13
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_wvk16.sh start (HEAD $(git rev-parse --short HEAD))"

# --- 0. preflight (nothing changes yet)
grep -q '^weight_version_key = 15$' affine/affine.toml || { echo "$(ts) toml is not at wvk 15; abort"; exit 1; }
grep -q '^crown_mode = "window_best"$' affine/affine.toml || { echo "$(ts) crown_mode is not window_best; abort"; exit 1; }
python ops/v10/wvk16_toml_edits.py --preview >/dev/null
python ops/incident-r13/revert_reign13.py --check
python ops/v10/llms_wvk16_edits.py >/dev/null && git diff --quiet -- affine/scripts/build_llms_txt.py && echo "$(ts) llms builder already patched" || echo "$(ts) llms builder patched (working tree)"
python -c "import ast,sys; ast.parse(open('affine/scripts/build_llms_txt.py').read())"
echo "$(ts) preflight ok"

POD_SSH_STR=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["eval_machine"]["ssh"])')
read -r POD_USERHOST _ POD_PORT <<<"$POD_SSH_STR"
POD_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$POD_PORT" "$POD_USERHOST")
echo "$(ts) eval pod: $POD_SSH_STR"

# --- 1. capture the live validator env (before any stop)
VPID=$(pm2 pid affine-validator 2>/dev/null | tr -d '[:space:]' || true)
ENV_FILE=$HERE/.validator_env.$$
: > "$ENV_FILE"; chmod 600 "$ENV_FILE"
if [[ -n "$VPID" && "$VPID" != "0" && -r /proc/$VPID/environ ]]; then
  tr '\0' '\n' < /proc/"$VPID"/environ | grep -E '^(HF_TOKEN|AFFINE_[A-Z0-9_]+|R2_[A-Z0-9_]+|CLOUDFLARE_[A-Z0-9_]+|HIPPIUS_[A-Z0-9_]+|LIUM_API_KEY|TARGON_API_KEY)=' > "$ENV_FILE" || true
  echo "$(ts) env captured from /proc/$VPID/environ ($(wc -l < "$ENV_FILE") keys)"
fi
if ! grep -q '^AFFINE_EVAL_TOKEN=' "$ENV_FILE" || ! grep -q '^HF_TOKEN=' "$ENV_FILE" || ! grep -q '^AFFINE_EVAL_R2_ACCESS_KEY_ID=' "$ENV_FILE"; then
  echo "$(ts) /proc env incomplete — falling back to ops/t0/validator_env.sh (~/.affine-validator.env)"
  source ops/t0/validator_env.sh
else
  set -a; while IFS= read -r line; do export "$line"; done < "$ENV_FILE"; set +a
fi
rm -f "$ENV_FILE"
for k in HF_TOKEN AFFINE_EVAL_TOKEN R2_ENDPOINT AFFINE_EVAL_R2_ACCESS_KEY_ID AFFINE_EVAL_R2_SECRET_ACCESS_KEY; do
  [[ -n "${!k:-}" ]] || { echo "$(ts) $k missing in env; abort (pod would boot without creds)"; exit 1; }
done
echo "$(ts) env ok"

health() { curl -s -m 8 -H "X-Affine-Token: $AFFINE_EVAL_TOKEN" http://127.0.0.1:9000/health 2>/dev/null || true; }
busy_of() { python3 -c 'import json,sys
try: print(json.loads(sys.argv[1]).get("busy"))
except Exception: print("?")' "$1"; }
inflight() { python3 -c 'import json;print(json.load(open("affine/state/state.json")).get("in_flight") or "")'; }

# --- 2. keepalive ralph off
KEEPALIVE_WAS_ON=0
if ./ralphs/ralphctl.sh keepalive status 2>/dev/null | head -1 | grep -q '^ON'; then KEEPALIVE_WAS_ON=1; fi
if [[ -f ralphs/keepalive/pid ]] && kill -0 "$(cat ralphs/keepalive/pid)" 2>/dev/null; then KEEPALIVE_WAS_ON=1; fi
./ralphs/ralphctl.sh keepalive off >/dev/null 2>&1 || true
echo "$(ts) keepalive ralph: was_on=$KEEPALIVE_WAS_ON, now off"
reenable() { if [[ "$KEEPALIVE_WAS_ON" == 1 ]]; then ./ralphs/ralphctl.sh keepalive on >/dev/null 2>&1 && echo "$(ts) keepalive ralph re-enabled"; fi; }
trap reenable EXIT

# --- 3. wait for a duel boundary. With a non-empty queue the validator pops
# the next entry within a second of recording a verdict, so "pod idle AND
# in_flight empty" may never be observed. Boundary = the verdict of the
# CURRENT in_flight duel is in history.jsonl (or in_flight is empty and the
# pod idle). Stopping right after that costs at most the first minute of
# the next duel's load phase: it has no verdict, State.load requeues it.
has_verdict() { python3 - "$1" <<'PY'
import json, sys
cid = sys.argv[1]
ok = any(json.loads(l).get("challenge_id") == cid and json.loads(l).get("event") in ("verdict", "crowned", "failed")
         for l in open("affine/state/history.jsonl") if l.strip())
sys.exit(0 if ok else 1)
PY
}
cur_cid() { python3 -c 'import json;f=json.load(open("affine/state/state.json")).get("in_flight");print((f or {}).get("challenge_id","") if isinstance(f,dict) else (f or ""))'; }
START_CID=$(cur_cid)
echo "$(ts) waiting for duel boundary (current in_flight: ${START_CID:-none})"
reached=0
for i in $(seq 1 2400); do
  cid=$(cur_cid); h=$(health); busy=$(busy_of "$h")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi   # moved on: START_CID is decided
  if [[ -n "$cid" ]] && has_verdict "$cid"; then reached=1; break; fi
  (( i % 60 == 0 )) && echo "$(ts)   busy=$busy in_flight=${cid:-none}"
  sleep 2
done
[[ "$reached" == 1 ]] || { echo "$(ts) no boundary in 80 min; abort (nothing changed)"; exit 1; }
echo "$(ts) boundary reached (in_flight now: $(cur_cid), pod busy=$(busy_of "$(health)")) -> pm2 stop affine-validator"
pm2 stop affine-validator >/dev/null
sleep 3

# --- 4. stale in_flight: clear only if its verdict is already in history
inf=$(inflight)
if [[ -n "$inf" ]]; then
  cid=$(python3 -c 'import json;f=json.load(open("affine/state/state.json"))["in_flight"];print(f.get("challenge_id") if isinstance(f,dict) else f)')
  if python3 - "$cid" <<'PY'
import json, sys
cid = sys.argv[1]
rows = [json.loads(l) for l in open("affine/state/history.jsonl") if l.strip()]
sys.exit(0 if any(r.get("challenge_id") == cid and r.get("event") in ("verdict", "crowned", "failed") for r in rows) else 1)
PY
  then
    cp affine/state/state.json "$HERE/state.before_inflight_clear.$(date -u +%Y%m%dT%H%M%SZ).json"
    python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'
    echo "$(ts) cleared stale in_flight $cid (verdict already in history)"
  else
    echo "$(ts) in_flight $cid has NO verdict in history — leaving it; State.load requeues it"
  fi
fi

# --- 5. contract flip (toml + website mirror), llms.txt
python ops/v10/wvk16_toml_edits.py --apply "$DIRECTIVE_DATE" --wvk-to 16
grep -E '^(weight_version_key|crown_mode|near_miss_enabled|min_margin|k_sigma|min_margin_mode|min_z) ' affine/affine.toml
python -c 'from affine.config import load_config; c=load_config(); d=c.duel; print("config ok: wvk", c.weight_version_key, "crown_mode", d.crown_mode, "near_miss", d.near_miss_enabled, "k_sigma", d.k_sigma, "min_margin", d.min_margin)'
(cd affine && python scripts/build_llms_txt.py)
grep -c "Fork history: wvk 16" affine/website/llms.txt || { echo "$(ts) llms.txt lacks the wvk 16 section; abort before pod redeploy"; exit 1; }

# --- 6. uncrown reign 13 -> reign 12 (validator is stopped)
bash ops/incident-r13/revert_reign13.sh

# --- 7. redeploy the eval pod (code + toml)
echo "$(ts) redeploy eval pod"
(cd affine && python scripts/redeploy_pods.py)
echo "$(ts) verify pod tree"
"${POD_SSH[@]}" 'grep -E "^(weight_version_key|crown_mode|near_miss_enabled|min_margin|k_sigma) " /root/affine/affine.toml'

# --- 8. validator + dash back up
echo "$(ts) pm2 start affine-validator"
pm2 start affine-validator --update-env >/dev/null
pm2 restart affine-dash >/dev/null && echo "$(ts) affine-dash restarted"
for i in $(seq 1 90); do
  h=$(health)
  ok=$(python3 -c 'import json,sys
try:
  d=json.loads(sys.argv[1]); print(d.get("ok"), "busy=%s" % d.get("busy"), "epoch=%s" % (d.get("corpus") or {}).get("corpus_epoch"), (d.get("versions") or {}).get("vllm"))
except Exception: print("?")' "$h")
  echo "$(ts) pod health: $ok"
  [[ "$ok" == True* ]] && break
  sleep 30
done
sleep 5
curl -s -m 10 http://127.0.0.1:8787/api/v1/contract | python3 -c 'import json,sys
d=json.load(sys.stdin); du=d["duel"]
print("contract: wvk", d["subnet"]["weight_version_key"], {k: du.get(k) for k in ("crown_mode","near_miss_enabled","min_margin","k_sigma","min_margin_mode","min_z")})'
python3 -c 'import json;s=json.load(open("affine/state/state.json"));k=s["king"];print("king:", k["challenge_id"], k["revision"][:12], "reign", k["reign_number"], k["hotkey"][:12], "| crown_window:", s.get("crown_window"))'
echo "$(ts) deploy done. Next: commit, verify first wvk-16 verdict (ranking_formula without the near-miss suffix), then Discord."
