#!/usr/bin/env bash
# Incident reign-13 byte-copy (2026-09-13) — fix (a): weight-identity gate.
# Applies ops/incident-r13/weights_identity_gate.patch to ~/subnet120, then
# redeploys the eval pod at the next DUEL BOUNDARY (pod idle, no in_flight):
#   pm2 stop affine-validator -> scripts/redeploy_pods.py (eval) -> pm2 start.
# Modeled on /tmp/v9flip/deploy.sh with the AGENTS.md pitfalls closed:
#   - env for redeploy_pods.py is read from the LIVE validator's
#     /proc/<pm2 pid>/environ BEFORE pm2 stop (never from a pgrep -f match);
#     ops/t0/validator_env.sh (~/.affine-validator.env) is the fallback
#   - keepalive ralph is disabled for the window (it restarts a stopped validator)
#   - a stale in_flight is cleared only if its verdict is already in history.jsonl
#   - the pod tree is verified (gate present, knobs present, numpy importable)
#   - state.json is seeded with the reign-12/13 weight fingerprint while the
#     validator is stopped (belt and braces: the pod also compares against
#     the king snapshot directly)
# No weight_version_key change. Admission rule only. Run ONLY on Jacob's go:
#   bash ~/subnet120/ops/incident-r13/deploy_gate.sh
set -euo pipefail
HERE=/home/const/subnet120/ops/incident-r13
REPO=/home/const/subnet120
LOG=$HERE/deploy_gate.log
PATCH=$HERE/weights_identity_gate.patch
# Reign 12 == reign 13 weights (1,026 tensors, sha256 per tensor; see the
# incident report). Recorded under the crown that first carried them.
SEED_FP=8996cf7cca0d1c75e899501ccdb5352c957a3e603dd490111fcc4f223730cf29
SEED_CID=chal-00454
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_gate.sh start (HEAD $(git rev-parse --short HEAD))"

# --- 0. patch the box tree (idempotent: skip if the gate is already there)
if grep -q "def _weights_identity_gate" affine/evalsrv/server.py; then
  echo "$(ts) gate already in the tree; not re-applying the patch"
else
  [[ -f "$PATCH" ]] || { echo "$(ts) $PATCH missing; abort"; exit 1; }
  git apply --check "$PATCH" || { echo "$(ts) patch does not apply to $(git rev-parse --short HEAD); abort"; exit 1; }
  git apply "$PATCH"
  echo "$(ts) patch applied (working tree; commit it after the deploy)"
fi
for f in affine/evalsrv/r2store.py affine/evalsrv/server.py affine/evalsrv/engine.py \
         affine/affine/validator.py affine/affine/state.py affine/affine/eval_client.py affine/affine/config.py; do
  python -m py_compile "$f"
done
grep -q '^near_duplicate_min_changed_frac = ' affine/affine.toml || { echo "$(ts) toml lacks the [submission] knobs; abort"; exit 1; }
grep -q '^weight_version_key = 15$' affine/affine.toml || { echo "$(ts) weight_version_key is not 15 — unexpected; abort"; exit 1; }
python -c 'from affine.config import load_config; c=load_config(); s=c.submission; print("config ok:", s.near_duplicate_max_identical_bytes_frac, s.near_duplicate_min_changed_frac)'
echo "$(ts) tree ok"

POD_SSH_STR=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["eval_machine"]["ssh"])')
read -r POD_USERHOST _ POD_PORT <<<"$POD_SSH_STR"
POD_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$POD_PORT" "$POD_USERHOST")
echo "$(ts) eval pod: $POD_SSH_STR"

# --- 1. capture the live validator env for redeploy_pods.py (before any stop)
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
echo "$(ts) env ok: HF_TOKEN AFFINE_EVAL_TOKEN R2_ENDPOINT AFFINE_EVAL_R2_* present"

health() { curl -s -m 8 -H "X-Affine-Token: $AFFINE_EVAL_TOKEN" http://127.0.0.1:9000/health 2>/dev/null || true; }
busy_of() { python3 -c 'import json,sys
try: print(json.loads(sys.argv[1]).get("busy"))
except Exception: print("?")' "$1"; }
inflight() { python3 -c 'import json;print(json.load(open("affine/state/state.json")).get("in_flight") or "")'; }
# A window close runs its confirmation slice on the pod with in_flight EMPTY
# (busy=true covers it); the crown itself takes milliseconds after the
# verdict. Also refuse to stop inside the last 15 min before the window's
# last block: the close + confirmation must not straddle the restart.
window_blocks_left() { python3 - <<'PY'
import json
s = json.load(open("affine/state/state.json"))
cw = s.get("crown_window") or {}
try:
    import bittensor as bt
    b = bt.Subtensor("finney").block
    W = int(cw.get("window_blocks") or 3600)
    print((int(cw["window_id"]) + 1) * W - int(b))
except Exception:
    print(99999)
PY
}

# --- 2. keepalive ralph off for the window
KEEPALIVE_WAS_ON=0
if ./ralphs/ralphctl.sh keepalive status 2>/dev/null | head -1 | grep -q '^ON'; then KEEPALIVE_WAS_ON=1; fi
if [[ -f ralphs/keepalive/pid ]] && kill -0 "$(cat ralphs/keepalive/pid)" 2>/dev/null; then KEEPALIVE_WAS_ON=1; fi
./ralphs/ralphctl.sh keepalive off >/dev/null 2>&1 || true
echo "$(ts) keepalive ralph: was_on=$KEEPALIVE_WAS_ON, now off"
reenable() { if [[ "$KEEPALIVE_WAS_ON" == 1 ]]; then ./ralphs/ralphctl.sh keepalive on >/dev/null 2>&1 && echo "$(ts) keepalive ralph re-enabled"; fi; }
trap reenable EXIT

# --- 3. wait for a duel boundary: pod idle AND no in_flight AND not mid window-close
echo "$(ts) waiting for duel boundary (pod busy=false, in_flight empty, >75 blocks before the window close)"
busy=?; inf=x
for i in $(seq 1 240); do
  h=$(health); busy=$(busy_of "$h"); inf=$(inflight)
  if [[ "$busy" == "False" && -z "$inf" ]]; then
    left=$(window_blocks_left)
    if (( left > 75 )); then break; fi
    echo "$(ts)   boundary but only $left blocks before the window close; waiting for the close to run first"
  fi
  (( i % 6 == 0 )) && echo "$(ts)   busy=$busy in_flight=${inf:-none}"
  sleep 20
done
[[ "$busy" == "False" && -z "$inf" ]] || { echo "$(ts) no boundary in 80 min; abort (keepalive re-enabled by trap)"; exit 1; }
echo "$(ts) boundary reached -> pm2 stop affine-validator"
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

# --- 5. seed the fingerprint ledger (validator is stopped; State.load reads it)
cp affine/state/state.json "$HERE/state.before_seed.$(date -u +%Y%m%dT%H%M%SZ).json"
python3 - "$SEED_FP" "$SEED_CID" <<'PY'
import json, sys
fp, cid = sys.argv[1], sys.argv[2]
p = "affine/state/state.json"
s = json.load(open(p))
led = s.setdefault("weight_fingerprints", {})
led.setdefault(fp, cid)
json.dump(s, open(p, "w"), indent=1)
print("state.json weight_fingerprints:", {k[:16] + "…": v for k, v in led.items()})
PY

# --- 6. redeploy the eval pod (code + toml); env comes from this shell
echo "$(ts) redeploy eval pod (scripts/redeploy_pods.py, role eval)"
(cd affine && python scripts/redeploy_pods.py)
echo "$(ts) verify pod tree"
"${POD_SSH[@]}" 'set -e; cd /root/affine
echo "gate defs: $(grep -c "def _weights_identity_gate" evalsrv/server.py) (want 1)"
echo "reject_weight_fingerprints in DuelRequest: $(grep -c "reject_weight_fingerprints: list" evalsrv/server.py) (want 1)"
echo "near_duplicate_stats: $(grep -c "def near_duplicate_stats" evalsrv/r2store.py) (want 1)"
grep -E "^(weight_version_key|crown_mode|near_duplicate_max_identical_bytes_frac|near_duplicate_min_changed_frac) " affine.toml
/root/venv/bin/python -c "import numpy; print(\"pod numpy\", numpy.__version__)"
/root/venv/bin/python -m py_compile evalsrv/server.py evalsrv/r2store.py evalsrv/engine.py && echo "pod evalsrv compiles"'

# --- 7. validator + dash back up
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
echo "$(ts) deploy done. First duel after this pays one ~2 min tensor hash of the king snapshot."
echo "$(ts) Watch: pm2 logs affine-validator --lines 50 | grep -E 'weights_duplicate|fingerprint|model_copy'"
echo "$(ts) Remember: git add -A affine && git commit (patch is in the working tree)."
