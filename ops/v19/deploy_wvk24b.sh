#!/usr/bin/env bash
# wvk 23 -> 24 FLIP at the next duel boundary: [duel.sd_meter].forfeit_sd -12 -> -6.
# Explicit dated operator directive, Jacob Steeves 2026-09-23 20:17 UTC ("Lets do this").
# Same pitfall handling as deploy_wvk22.sh: env from /proc of the live
# validator (never pgrep), keepalive off for the window, deadman paused, stale
# in_flight cleared only with a verdict in history, eval pod ONLY redeployed,
# king pinned before/after. Order in the stop window: toml flip -> llms.txt ->
# pod redeploy -> validator start -> commit.
# Rollback rule: the teacher-vs-king control z flips SIGN vs the pre-fork
# verdicts on the first wvk-23 verdicts -> bash ops/v19/rollback_wvk24.sh.
# Run on the box:  bash ops/v19/deploy_wvk24b.sh   (FORCE_BOUNDARY=1 to skip the wait)
set -euo pipefail
HERE=/home/const/subnet120/ops/v19
REPO=/home/const/subnet120
LOG=$HERE/deploy_wvk24b.log
DIRECTIVE_DATE=2026-09-23
PAUSE=/home/const/.affine/deadman.pause
mkdir -p "$HERE"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_wvk24b.sh start (wvk 24 addendum: empty-ref rule + k-matched control) (HEAD $(git rev-parse --short HEAD))"

# --- 0. preflight (nothing changes yet)
grep -q '^weight_version_key = 24$' affine/affine.toml || { echo "$(ts) toml is not at wvk 24; abort"; exit 1; }
grep -q '^forfeit_sd = -6$' affine/affine.toml || { echo "$(ts) forfeit_sd is not -6; abort"; exit 1; }
grep -q 'ref_min_content' affine/evalsrv/sdmeter.py || { echo "$(ts) evalsrv code without ref_min_content; abort"; exit 1; }
grep -q '^score_mode = "sd_min_rga"$' affine/affine.toml || { echo "$(ts) score_mode is not sd_min_rga; abort"; exit 1; }
python ops/v19/wvk24b_toml_edits.py --preview >/dev/null
python -c "import ast; ast.parse(open('affine/scripts/build_llms_txt.py').read())"
cp affine/scripts/build_llms_txt.py /tmp/wvk24b_builder_check.py && python ops/v19/llms_wvk24b_edits.py --builder /tmp/wvk24b_builder_check.py >/dev/null && python -c "import ast; ast.parse(open('/tmp/wvk24b_builder_check.py').read())" && echo "$(ts) llms edit dry run ok"
python -c 'import sys; sys.path.insert(0,"affine"); from evalsrv import sdmeter, terms, dueling; print("evalsrv imports ok")'
QUEUE_N=$(python3 -c 'import json;s=json.load(open("affine/state/state.json"));print(len(s.get("queue",[])) + (1 if s.get("in_flight") else 0))')
echo "$(ts) challengers submitted under the old cap still to be judged (queue + in_flight): $QUEUE_N"
KING_HK_BEFORE=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["king"]["hotkey"])')
echo "$(ts) king before: $KING_HK_BEFORE"
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

# --- 3b. duel boundary (kept for the FORCE_BOUNDARY-less path)
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
echo "$(ts) boundary check (current in_flight: ${START_CID:-none})"
reached=0
if [[ "${FORCE_BOUNDARY:-0}" == 1 ]]; then reached=1; echo "$(ts) FORCE_BOUNDARY=1: pod busy=$(busy_of "$(health)"), in_flight=$(cur_cid) (post-cutoff; requeued on restart, judged under wvk 22)"; fi
for i in $(seq 1 4200); do   # up to 140 min: one full duel incl. load
  [[ "$reached" == 1 ]] && break
  cid=$(cur_cid); h=$(health); busy=$(busy_of "$h")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi
  if [[ -n "$cid" ]] && has_verdict "$cid"; then reached=1; break; fi
  (( i % 60 == 0 )) && echo "$(ts)   busy=$busy in_flight=${cid:-none}"
  sleep 2
done
[[ "$reached" == 1 ]] || { echo "$(ts) no boundary in 140 min; abort (nothing changed)"; exit 1; }
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
  if has_verdict "$cid"; then
    cp affine/state/state.json "$HERE/state.before_inflight_clear.$(date -u +%Y%m%dT%H%M%SZ).json"
    python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'
    echo "$(ts) cleared stale in_flight $cid (verdict already in history)"
  else
    echo "$(ts) in_flight $cid has NO verdict in history — leaving it; State.load requeues it"
  fi
fi

# --- 5. contract flip (toml + website mirror), llms.txt
python ops/v19/wvk24b_toml_edits.py --apply "$DIRECTIVE_DATE"
grep -E '^(weight_version_key|forfeit_sd|ref_min_content|typ_min_refs) ' affine/affine.toml
python -c "from affine.config import load_config; c=load_config(); d=c.duel; assert c.weight_version_key==24 and int(d.sd_meter['ref_min_content'])==10 and int(d.sd_meter['typ_min_refs'])==2; print('config ok: wvk', c.weight_version_key, d.sd_meter['ref_min_content'], d.sd_meter['typ_min_refs'])"
python ops/v19/llms_wvk24b_edits.py
python affine/scripts/build_llms_txt.py | tail -1
grep -q "k-matched" affine/website/llms.txt && echo "$(ts) llms.txt has the wvk 24 addendum"

# --- 5b. pods: code + flipped toml
# eval pod ONLY: the bench pod (reign 15's card pass) and the chat pod are not touched;
# the king seat (pm2 affine-king-datagen / kingctl) reads state.json and is not restarted.
cd affine && python scripts/redeploy_pods.py --role eval && cd "$REPO"
echo "$(ts) redeploy_pods done"
"${POD_SSH[@]}" 'grep -E "^(weight_version_key|forfeit_sd|ref_min_content|typ_min_refs) " /root/affine/affine.toml; grep -c "control_kmatched" /root/affine/evalsrv/sdmeter.py' || echo "$(ts) WARNING pod verify ssh failed"

# --- 6. validator back
pm2 start affine-validator >/dev/null
sleep 5
pm2 ls | grep affine-validator || true
for i in $(seq 1 60); do h=$(health); [[ -n "$h" ]] && { echo "$(ts) pod health: $h"; break; }; sleep 5; done
pm2 restart affine-dash >/dev/null 2>&1 && echo "$(ts) affine-dash restarted" || true
KING_HK_AFTER=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["king"]["hotkey"])')
[[ "$KING_HK_AFTER" == "$KING_HK_BEFORE" ]] && echo "$(ts) king unchanged: $KING_HK_AFTER" || echo "$(ts) WARNING king changed across the restart: $KING_HK_BEFORE -> $KING_HK_AFTER"
python ops/king-datagen/kingctl.py status 2>/dev/null | head -3 || true
git add affine/affine.toml affine/website/code/affine.toml affine/scripts/build_llms_txt.py affine/website/llms.txt && git commit -q -m "contract: wvk 24 addendum — empty-thought reference rule (ref_min_content 10, typ_min_refs 2) + k-matched floor-dropped teacher-vs-king control on every verdict (operator directive 2026-09-23 20:47 UTC)" && echo "$(ts) committed $(git rev-parse --short HEAD)"
echo "$(ts) deploy done. Next: first verdict must stamp sd_meter.ref_min_content 10 / typ_min_refs 2 and carry control_kmatched; then Discord live line + AGENTS.md."
echo DONE
