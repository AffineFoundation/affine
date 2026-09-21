#!/usr/bin/env bash
# wvk 19 -> 20 (thought_cap_ratio 1.25: teacher-relative miner thought cap) at
# the next duel boundary. Operator directive 2026-09-16 14:31 UTC.
# Modeled on ops/v13/deploy_wvk19.sh with the AGENTS.md pitfalls closed:
#   - env for redeploy_pods.py from the LIVE validator's /proc/<pm2 pid>/environ
#     BEFORE pm2 stop (never a pgrep -f match); ops/t0/validator_env.sh fallback
#   - keepalive ralph off for the window (it restarts a stopped validator)
#   - a stale in_flight is cleared only if its verdict is already in history.jsonl
#   - the deadman (systemd affine-deadman.timer) is paused with
#     ~/.affine/deadman.pause for the stop window and un-paused after
#   - toml flip happens while the validator is STOPPED
#   - pod toml + code verified (wvk 20, teacher_cap in dueling.py)
#   - affine-dash restarted so /api/v1/contract re-reads the toml
# Order inside the stop window: toml flip -> llms.txt build -> pod redeploy ->
# validator start. Discord is NOT posted here (after the first wvk-17 verdict).
# Run: bash ops/v15/deploy_wvk20.sh
set -euo pipefail
HERE=/home/const/subnet120/ops/v15
REPO=/home/const/subnet120
LOG=$HERE/deploy_wvk20.log
DIRECTIVE_DATE=2026-09-16
PAUSE=/home/const/.affine/deadman.pause
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_wvk20.sh start (thought_cap_ratio 1.25) (HEAD $(git rev-parse --short HEAD))"

# --- 0. preflight (nothing changes yet)
grep -q '^weight_version_key = 19$' affine/affine.toml || { echo "$(ts) toml is not at wvk 19; abort"; exit 1; }
grep -q '^max_thought_tokens = 2048$' affine/affine.toml || { echo "$(ts) max_thought_tokens is not 2048; abort"; exit 1; }
grep -q "def teacher_cap" affine/evalsrv/dueling.py || { echo "$(ts) dueling.py lacks teacher_cap (run ops/v15/wvk20_code_edits.py); abort"; exit 1; }
grep -q "from .chat import get_tokenizer" affine/evalsrv/dueling.py || { echo "$(ts) dueling.py lacks the get_tokenizer import; abort"; exit 1; }
grep -q "thought_cap_ratio" affine/affine/config.py || { echo "$(ts) config.py lacks the knob; abort"; exit 1; }
python -m py_compile affine/evalsrv/dueling.py affine/affine/config.py
PYTHONPATH=affine python -c "from evalsrv import dueling"
python ops/v15/wvk20_toml_edits.py --preview >/dev/null
python ops/v15/llms_wvk20_edits.py >/dev/null && echo "$(ts) llms builder patched (or already)"
python -c "import ast; ast.parse(open('affine/scripts/build_llms_txt.py').read())"
python ops/v10/replay_wvk16.py --last 30 > "$HERE/replay_before_flip.txt" && echo "$(ts) decision replay clean (ops/v15/replay_before_flip.txt)"
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
echo "$(ts) boundary reached (in_flight now: $(cur_cid), pod busy=$(busy_of "$(health)")) -> pause deadman, pm2 stop affine-validator"
mkdir -p "$(dirname "$PAUSE")" && touch "$PAUSE"
unpause() { rm -f "$PAUSE"; echo "$(ts) deadman pause removed"; }
trap 'unpause; reenable' EXIT
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
python ops/v15/wvk20_toml_edits.py --apply "$DIRECTIVE_DATE" --wvk-to 20
grep -E '^(weight_version_key|thought_cap_ratio|max_thought_tokens|max_action_tokens|ref_max_tokens|confirmation_required|crown_mode) ' affine/affine.toml
python -c 'from affine.config import load_config; c=load_config(); d=c.duel; print("config ok: wvk", c.weight_version_key, "thought_cap_ratio", d.thought_cap_ratio, "max_thought", d.max_thought_tokens, "ref", d.ref_max_tokens, "confirmation", d.confirmation_required)'
(cd affine && python scripts/build_llms_txt.py)
grep -c "Fork history: wvk 20" affine/website/llms.txt || { echo "$(ts) llms.txt lacks the wvk 20 section; abort before pod redeploy"; exit 1; }

# --- 7. redeploy the eval pod (code + toml)
echo "$(ts) redeploy eval pod"
(cd affine && python scripts/redeploy_pods.py)
echo "$(ts) verify pod tree"
"${POD_SSH[@]}" 'grep -E "^(weight_version_key|thought_cap_ratio|max_thought_tokens|ref_max_tokens|confirmation_required) " /root/affine/affine.toml; echo "teacher_cap in dueling.py: $(grep -c "def teacher_cap" /root/affine/evalsrv/dueling.py) (want 1)"'

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
print("contract: wvk", d["subnet"]["weight_version_key"], {k: du.get(k) for k in ("thought_cap_ratio","max_thought_tokens","max_action_tokens","ref_max_tokens","confirmation_required","crown_mode")})'
unpause
echo "$(ts) deploy done. Next: commit, verify the first wvk-20 verdict stamps duel_params.thought_cap_ratio = 1.25 / thought_cap_rule, per-side n_turns_cap_raised and forfeit rates, then Discord."
