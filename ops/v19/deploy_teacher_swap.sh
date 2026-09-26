#!/usr/bin/env bash
# Teacher swap Qwen3.8-27B -> GLM-5.3-Flash + 262k window, at a duel boundary.
# STAGED. Refuses to run without DIRECTIVE_DATE (the dated operator directive
# that names weight_version_key) and WVK_TO. Modeled on ops/v12/deploy_wvk18.sh
# with the same pitfalls closed (env from the live validator's /proc, keepalive
# ralph off, deadman paused, stale in_flight cleared only with a verdict in
# history, toml flipped while the validator is stopped, Discord NOT posted here).
#
# Order inside the window (T0):
#   0. preflight   toml at wvk N; GLM tokenizer + template load; ToolBaker can
#                  bake under the GLM template (parity gate) — abort otherwise;
#                  the GLM swarm boxes are already HEALTHY on the router (rented
#                  and bootstrapped by the pre-warm step below; zero verdict gap)
#   1. env         capture the live validator env for redeploy_pods.py
#   2. keepalive   off for the window
#   3. boundary    wait until the in-flight duel has its verdict
#   4. stop        pm2 stop affine-validator; clear a stale in_flight
#   5. flip        ops/v19/teacher_swap_toml_edits.py --apply DATE --wvk-to N+1
#                  (affine.toml + swarm.toml + slicer + policies + sources)
#   6. router      swarm manager re-reads swarm.toml; the router must list ONLY
#                  zai-org/GLM-5.3-Flash backends before the pod is redeployed
#   7. llms.txt    rebuild (contract page reads the toml)
#   8. pod         redeploy_pods.py --all (max_model_len 262144, toml, code)
#   9. start       pm2 start affine-validator; affine-dash restart
#  10. fold        ops/corpus_build.py --rederive (tool turns re-baked under the
#                  GLM template; text/bash turns unchanged); datagen pods get the
#                  new policies.toml (teacher seat -> glm-5.3-flash)
#
# PRE-WARM (T-1 day, no contract change): bring the GLM boxes up next to the
# Qwen ones so T0 has no gap —
#   cp ops/teacher-swarm/swarm.toml /tmp/swarm_glm53.toml && python - <<'PY'
#   (apply the [swarm]/[types] part of ops/v19/teacher_swap_toml_edits.py to the copy,
#    set pod_prefix = "swarm-g-", target=1 on the chosen type)
#   PY
#   python ops/teacher-swarm/manager.py --config /tmp/swarm_glm53.toml   # second manager
# The router serves whatever the manager state lists; at T0 step 6 the Qwen
# manager's targets go to 0 and its boxes are released once the last Qwen
# duel is decided.
#
# Directive: Jacob Steeves 2026-09-26 09:09 UTC ("lets do this switch"); T0 2026-09-30 14:00 UTC; wvk 24 -> 25.
# Scoring bundle knobs (fork worker, ops/v20/wvk25_toml_edits.py) are applied right after the
# teacher/window edits below; the wvk integer is bumped ONCE, here.
# Run: DIRECTIVE_DATE=2026-09-26 WVK_TO=25 bash ops/v19/deploy_teacher_swap.sh
set -euo pipefail
HERE=/home/const/subnet120/ops/v19
REPO=/home/const/subnet120
LOG=$HERE/deploy_teacher_swap.log
PAUSE=/home/const/.affine/deadman.pause
: "${DIRECTIVE_DATE:?set DIRECTIVE_DATE=YYYY-MM-DD (the dated operator directive)}"
: "${WVK_TO:=25}"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_teacher_swap.sh start (GLM-5.3-Flash, 262k) HEAD $(git rev-parse --short HEAD) directive $DIRECTIVE_DATE wvk->$WVK_TO"

# --- 0. preflight (nothing changes yet)
WVK_FROM=$((WVK_TO - 1))
grep -q "^weight_version_key = $WVK_FROM\$" affine/affine.toml || { echo "$(ts) toml is not at wvk $WVK_FROM; abort"; exit 1; }
grep -q '^repo = "Qwen/Qwen3.8-27B"$' affine/affine.toml || { echo "$(ts) [teacher].repo is not Qwen3.8-27B; abort"; exit 1; }
grep -q '^max_model_len = 131072$' affine/affine.toml || { echo "$(ts) max_model_len is not 131072; abort"; exit 1; }
python -m py_compile ops/corpus_build.py affine/datagen/slicer.py affine/affine/toolbake.py
python ops/v19/teacher_swap_toml_edits.py --preview >/dev/null
python - <<'PY' || { echo "$(ts) ToolBaker cannot bake under the GLM template — the re-bake port is not done; abort"; exit 1; }
from affine.toolbake import ToolBaker
b = ToolBaker.from_pretrained("zai-org/GLM-5.3-Flash")
tools = [{"name": "bash", "description": "run", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}]
msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "bash", "arguments": "{\"cmd\": \"ls\"}"}]},
        {"role": "tool", "content": "out", "tool_call_id": "1"}, {"role": "user", "content": "next"}]
assert b.parity_ok(msgs, tools, b.bake(msgs, tools)), "parity gate fails under the GLM template"
print("toolbake parity ok under GLM-5.3-Flash")
PY
ROUTER_MODELS=$(curl -s -m 8 http://127.0.0.1:9100/v1/models | python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' || true)
echo "$(ts) router serves: ${ROUTER_MODELS:-<dark>}"
[[ "$ROUTER_MODELS" == *"zai-org/GLM-5.3-Flash"* ]] || { echo "$(ts) GLM swarm not on the router yet (pre-warm first); abort"; exit 1; }
echo "$(ts) preflight ok"

POD_SSH_STR=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["eval_machine"]["ssh"])')
echo "$(ts) eval pod: $POD_SSH_STR"

# --- 1. live validator env (before any stop)
VPID=$(pm2 pid affine-validator 2>/dev/null | tr -d '[:space:]' || true)
ENV_FILE=$HERE/.validator_env.$$
: > "$ENV_FILE"; chmod 600 "$ENV_FILE"
if [[ -n "$VPID" && "$VPID" != "0" && -r /proc/$VPID/environ ]]; then
  tr '\0' '\n' < /proc/"$VPID"/environ | grep -E '^(HF_TOKEN|AFFINE_[A-Z0-9_]+|R2_[A-Z0-9_]+|CLOUDFLARE_[A-Z0-9_]+|HIPPIUS_[A-Z0-9_]+|LIUM_API_KEY|TARGON_API_KEY)=' > "$ENV_FILE" || true
fi
if ! grep -q '^AFFINE_EVAL_TOKEN=' "$ENV_FILE" || ! grep -q '^HF_TOKEN=' "$ENV_FILE" || ! grep -q '^AFFINE_EVAL_R2_ACCESS_KEY_ID=' "$ENV_FILE"; then
  echo "$(ts) /proc env incomplete — falling back to ops/t0/validator_env.sh"
  source ops/t0/validator_env.sh
else
  set -a; while IFS= read -r line; do export "$line"; done < "$ENV_FILE"; set +a
fi
rm -f "$ENV_FILE"
for k in HF_TOKEN AFFINE_EVAL_TOKEN R2_ENDPOINT AFFINE_EVAL_R2_ACCESS_KEY_ID AFFINE_EVAL_R2_SECRET_ACCESS_KEY; do
  [[ -n "${!k:-}" ]] || { echo "$(ts) $k missing; abort"; exit 1; }
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

# --- 2. keepalive off
KEEPALIVE_WAS_ON=0
if ./ralphs/ralphctl.sh keepalive status 2>/dev/null | head -1 | grep -q '^ON'; then KEEPALIVE_WAS_ON=1; fi
./ralphs/ralphctl.sh keepalive off >/dev/null 2>&1 || true
reenable() { if [[ "$KEEPALIVE_WAS_ON" == 1 ]]; then ./ralphs/ralphctl.sh keepalive on >/dev/null 2>&1 && echo "$(ts) keepalive re-enabled"; fi; }
trap reenable EXIT

# --- 3. duel boundary
START_CID=$(cur_cid)
echo "$(ts) waiting for duel boundary (in_flight: ${START_CID:-none})"
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
echo "$(ts) boundary reached -> pause deadman, pm2 stop affine-validator"
mkdir -p "$(dirname "$PAUSE")" && touch "$PAUSE"
unpause() { rm -f "$PAUSE"; echo "$(ts) deadman pause removed"; }
trap 'unpause; reenable' EXIT
pm2 stop affine-validator >/dev/null
sleep 3

# --- 4. stale in_flight
cid=$(cur_cid)
if [[ -n "$cid" ]] && has_verdict "$cid"; then
  cp affine/state/state.json "$HERE/state.before_inflight_clear.$(date -u +%Y%m%dT%H%M%SZ).json"
  python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'
  echo "$(ts) cleared stale in_flight $cid (verdict in history)"
fi

# --- 5. contract flip: teacher / window / swarm / slicer / policies / band, then the wvk-25 scoring knobs
python ops/v19/teacher_swap_toml_edits.py --apply "$DIRECTIVE_DATE" --wvk-to "$WVK_TO"
# fork worker's rules script (box main b03fc00d/816af3d1): asserts the eight fixed knobs, flips
# miner_empty_rule / empty_gate_ratio / r_cap_teacher / seq_enabled, sets min_context_tokens 262144
# (idempotent), adds its history paragraph, mirrors the toml. Runs AFTER the wvk bump above.
if [[ -f ops/v20/wvk25_rules_toml_edits.py ]]; then
  python ops/v20/wvk25_rules_toml_edits.py --apply "$DIRECTIVE_DATE" && echo "$(ts) wvk-25 scoring knobs + context rule applied"
else
  echo "$(ts) ops/v20/wvk25_rules_toml_edits.py missing — scoring bundle NOT applied; abort"; exit 1
fi
grep -E '^(weight_version_key|repo|max_model_len|score_mode|max_thought_tokens|ref_max_tokens|miner_empty_rule|empty_gate_ratio|r_cap_teacher|seq_enabled|seq_look_every|seq_k|seq_consecutive|seq_shadow_full_first_n|min_context_tokens) ' affine/affine.toml

# --- 6. teacher swarm: the Qwen manager's targets -> 0 (its boxes drain), router must be GLM-only
pm2 restart affine-swarm-manager >/dev/null 2>&1 || true
for i in $(seq 1 60); do
  ROUTER_MODELS=$(curl -s -m 8 http://127.0.0.1:9100/v1/models | python3 -c 'import json,sys;print(",".join(sorted(set(m["id"] for m in json.load(sys.stdin).get("data",[])))))' || true)
  [[ "$ROUTER_MODELS" == "zai-org/GLM-5.3-Flash" ]] && break
  sleep 5
done
[[ "$ROUTER_MODELS" == "zai-org/GLM-5.3-Flash" ]] || { echo "$(ts) router still lists $ROUTER_MODELS; abort before the pod redeploy (toml flipped — revert with git checkout if you stop here)"; exit 1; }
echo "$(ts) router GLM-only"

# --- 7. llms.txt
[[ -f ops/v20/llms_wvk25_notice_edits.py ]] && python ops/v20/llms_wvk25_notice_edits.py --flip >/dev/null 2>&1 && echo "$(ts) llms 'Upcoming fork' -> 'Fork history: wvk 25'" || echo "$(ts) (llms flip: run ops/v20/llms_wvk25_notice_edits.py --flip by hand if the flag is absent)"
python affine/scripts/build_llms_txt.py >/dev/null && echo "$(ts) llms.txt rebuilt"
[[ -f ops/v20/banner_wvk25.py ]] && python ops/v20/banner_wvk25.py --remove >/dev/null 2>&1 && echo "$(ts) #fork-notice banner removed" || true
[[ -f ops/v20/verify_pod_max_model_len.sh ]] && bash ops/v20/verify_pod_max_model_len.sh && echo "$(ts) pod max_model_len verified" || true

# --- 8. eval pod
cd affine && python scripts/redeploy_pods.py --all && cd ..
echo "$(ts) pod redeployed; verifying"
POD_TOML=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["eval_machine"]["ssh"])')
echo "$(ts) (verify on pod: grep -E '^(weight_version_key|repo|max_model_len) ' /root/affine/affine.toml)"

# --- 9. validator + dash
pm2 start affine-validator >/dev/null
pm2 restart affine-dash >/dev/null 2>&1 || true
echo "$(ts) validator started"

# --- 10. corpus re-derive (tool turns re-baked under the GLM template) + datagen pods
python ops/corpus_build.py --rederive && echo "$(ts) fold --rederive done"
bash ops/king-datagen/deploy_pods.sh --restart --all && echo "$(ts) datagen pods restarted with the GLM teacher seat"
# --- 11. green watch: 6-hourly + the fork worker posts the live line after the first wvk-25 verdict
pm2 delete affine-green-watch >/dev/null 2>&1 || true
pm2 start --name affine-green-watch --cron-restart "0 */6 * * *" --no-autorestart -- .venv/bin/python ops/v19/green_watch.py --box >/dev/null && echo "$(ts) green watch scheduled (6-hourly); first read after the first wvk-$WVK_TO verdict: .venv/bin/python ops/v19/green_watch.py --box"
echo "$(ts) === done. Live line after the first wvk-$WVK_TO verdict; remove the #fork-notice banner and move llms.txt 'Upcoming fork' -> 'Fork history'."
