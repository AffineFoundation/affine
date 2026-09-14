#!/usr/bin/env bash
# King payout window (72 h per crown) — validator-side deploy at a duel
# boundary. Operator directive (Jacob Steeves) 2026-09-14 11:09 UTC.
# No eval-pod code changes: the rule lives in the validator's weight sweep
# (affine/payout.py, chain.set_payout_weights). weight_version_key is NOT
# touched (payout rule, not a scoring-rule change).
#
# Modeled on ops/v11/deploy_wvk17.sh with the AGENTS.md pitfalls closed:
#   - waits for a duel boundary (pod idle + no in_flight, or the in_flight
#     duel's verdict already in history.jsonl); never stops mid-duel
#   - the deadman (systemd affine-deadman.timer) is paused with
#     ~/.affine/deadman.pause for the stop window and un-paused after
#   - keepalive ralph off for the window (it restarts a stopped validator)
#   - a stale in_flight is cleared only if its verdict is already in history
#   - the code lands while the validator is STOPPED: `git stash` the box's
#     uncommitted work, fast-forward main onto the payout branch, `stash pop`
#   - effective time = the instant `pm2 start` runs, stamped into
#     [subnet].king_payout_rule_effective_at (toml + llms.txt + api/contract)
#   - affine-dash restarted so /api/v1/contract + snapshot re-read the toml
#   - verifies the first payout sweep line in the validator log
# Preconditions checked: toml at wvk 17 (the wvk-17 flip has landed) and,
# unless REQUIRE_WVK17_VERDICT=0, at least one verdict stamped band_c = 4.0.
set -euo pipefail
HERE=/home/const/subnet120/ops/v12
REPO=/home/const/subnet120
LOG=$HERE/deploy_payout_window.log
BRANCH=${BRANCH:-staging/king-payout-window}
PAUSE=/home/const/.affine/deadman.pause
REQUIRE_WVK17_VERDICT=${REQUIRE_WVK17_VERDICT:-1}
BOUNDARY_WAIT_S=${BOUNDARY_WAIT_S:-4800}
mkdir -p "$HERE"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_payout_window.sh start (HEAD $(git rev-parse --short HEAD), branch $BRANCH $(git rev-parse --short "$BRANCH"))"

# --- 0. preflight (nothing changes yet)
grep -q '^weight_version_key = 17$' affine/affine.toml || { echo "$(ts) toml is not at wvk 17 — the wvk-17 flip has not landed; abort"; exit 1; }
git merge-base --is-ancestor HEAD "$BRANCH" || { echo "$(ts) $BRANCH does not contain HEAD (not a fast-forward); abort"; exit 1; }
git show "$BRANCH":affine/affine/payout.py >/dev/null || { echo "$(ts) branch lacks affine/payout.py; abort"; exit 1; }
git show "$BRANCH":affine/affine.toml | grep -q '^king_payout_window_hours = 72$' || { echo "$(ts) branch toml lacks king_payout_window_hours = 72; abort"; exit 1; }
git show "$BRANCH":affine/affine.toml | grep -q '^weight_version_key = 17$' || { echo "$(ts) branch toml is not at wvk 17; abort"; exit 1; }
has_wvk17_verdict() { python3 - <<'PY'
import json, sys
ok = False
for l in open("affine/state/history.jsonl"):
    if not l.strip():
        continue
    r = json.loads(l)
    if r.get("event") in ("verdict", "crowned") and float(((r.get("verdict") or {}).get("duel_params") or {}).get("band_c") or 0) >= 4.0:
        ok = True
sys.exit(0 if ok else 1)
PY
}
if has_wvk17_verdict; then
  echo "$(ts) a wvk-17 verdict (band_c 4.0) is in history"
elif [[ "$REQUIRE_WVK17_VERDICT" == 1 ]]; then
  echo "$(ts) no wvk-17 verdict in history yet; REQUIRE_WVK17_VERDICT=1 -> abort (re-run with REQUIRE_WVK17_VERDICT=0 to deploy at an idle boundary anyway)"; exit 1
else
  echo "$(ts) no wvk-17 verdict in history yet; REQUIRE_WVK17_VERDICT=0 -> proceeding (pod idle boundary)"
fi
echo "$(ts) preflight ok"

# --- 1. env for health checks (never a pgrep -f match)
VPID=$(pm2 pid affine-validator 2>/dev/null | tr -d '[:space:]' || true)
if [[ -n "$VPID" && "$VPID" != "0" && -r /proc/$VPID/environ ]]; then
  AFFINE_EVAL_TOKEN=$(tr '\0' '\n' < /proc/"$VPID"/environ | grep '^AFFINE_EVAL_TOKEN=' | cut -d= -f2- || true)
fi
[[ -n "${AFFINE_EVAL_TOKEN:-}" ]] || source ops/t0/validator_env.sh
[[ -n "${AFFINE_EVAL_TOKEN:-}" ]] || { echo "$(ts) AFFINE_EVAL_TOKEN missing; abort"; exit 1; }
export AFFINE_EVAL_TOKEN
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
if [[ -f ralphs/keepalive/pid ]] && kill -0 "$(cat ralphs/keepalive/pid)" 2>/dev/null; then KEEPALIVE_WAS_ON=1; fi
./ralphs/ralphctl.sh keepalive off >/dev/null 2>&1 || true
echo "$(ts) keepalive ralph: was_on=$KEEPALIVE_WAS_ON, now off"
reenable() { if [[ "$KEEPALIVE_WAS_ON" == 1 ]]; then ./ralphs/ralphctl.sh keepalive on >/dev/null 2>&1 && echo "$(ts) keepalive ralph re-enabled"; fi; }
trap reenable EXIT

# --- 3. wait for a duel boundary
START_CID=$(cur_cid)
echo "$(ts) waiting for duel boundary (current in_flight: ${START_CID:-none})"
reached=0
for i in $(seq 1 $((BOUNDARY_WAIT_S / 2))); do
  cid=$(cur_cid); h=$(health); busy=$(busy_of "$h")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi
  if [[ -n "$cid" ]] && has_verdict "$cid"; then reached=1; break; fi
  (( i % 60 == 0 )) && echo "$(ts)   busy=$busy in_flight=${cid:-none}"
  sleep 2
done
[[ "$reached" == 1 ]] || { echo "$(ts) no boundary in $BOUNDARY_WAIT_S s; abort (nothing changed)"; exit 1; }
echo "$(ts) boundary reached (in_flight now: $(cur_cid), pod busy=$(busy_of "$(health)")) -> pause deadman, pm2 stop affine-validator"
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
    echo "$(ts) in_flight $inf has NO verdict in history — leaving it; State.load requeues it"
  fi
fi

# --- 5. code lands: stash the box's uncommitted work, fast-forward, pop
STASHED=0
if ! git diff --quiet; then
  git stash push -q -m "payout-deploy $(ts)" && STASHED=1
  echo "$(ts) stashed the box's uncommitted tracked changes"
fi
git merge -q --ff-only "$BRANCH"
echo "$(ts) main fast-forwarded to $(git rev-parse --short HEAD)"
if [[ "$STASHED" == 1 ]]; then
  if git stash pop -q; then
    echo "$(ts) stash popped cleanly (uncommitted work restored)"
  else
    echo "$(ts) !!! stash pop CONFLICTED — the branch code is in place; resolve the stash by hand after the restart (git stash list)"
    git checkout -q -- . 2>/dev/null || true
  fi
fi

# --- 6. effective time stamp (toml) + llms.txt
EFFECTIVE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - "$EFFECTIVE" <<'PY'
import re, sys
eff = sys.argv[1]
p = "affine/affine.toml"
s = open(p).read()
s2, n = re.subn(r'^king_payout_rule_effective_at = "[^"]*"$',
                f'king_payout_rule_effective_at = "{eff}"', s, flags=re.M)
assert n == 1, "effective_at key not found"
open(p, "w").write(s2)
print("stamped", eff)
PY
python -m py_compile affine/affine/payout.py affine/affine/state.py affine/affine/chain.py affine/affine/validator.py affine/affine/dashboard.py affine/affine/dash/readers.py
python -c 'from affine.config import load_config; c=load_config(); print("config ok: wvk", c.weight_version_key, "window_h", c.king_payout_window_s/3600, "effective", c.king_payout_rule_effective_at)'
(cd affine && python scripts/build_llms_txt.py)
grep -c "^## Payout rule" affine/website/llms.txt >/dev/null || { echo "$(ts) llms.txt lacks the Payout rule section; abort before start"; exit 1; }
grep -q "effective $EFFECTIVE" affine/website/llms.txt && echo "$(ts) llms.txt carries the effective time"

# --- 7. validator + dash back up
echo "$(ts) pm2 start affine-validator"
pm2 start affine-validator --update-env >/dev/null
pm2 restart affine-dash >/dev/null && echo "$(ts) affine-dash restarted"
unpause
trap reenable EXIT

# --- 8. verify: first payout sweep line + set_weights + APIs
echo "$(ts) waiting for the first payout sweep in the validator log"
for i in $(seq 1 60); do
  if grep -q "payout sweep (window" affine/logs/validator.err.log && \
     [[ "$(grep "payout sweep (window" affine/logs/validator.err.log | tail -1 | cut -c1-19)" > "${EFFECTIVE:0:19}" ]]; then break; fi
  sleep 10
done
grep "payout set changed\|payout sweep (window" affine/logs/validator.err.log | tail -2
for i in $(seq 1 90); do
  line=$(grep -E "set_weights uids=|set_weights rate-limited|set_weights failed" affine/logs/validator.err.log | tail -1)
  if [[ "${line:0:19}" > "${EFFECTIVE:0:19}" ]]; then echo "$line"; [[ "$line" == *"uids="* ]] && break; fi
  sleep 20
done
sleep 3
curl -s -m 10 http://127.0.0.1:8787/api/v1/contract | python3 -c 'import json,sys
d=json.load(sys.stdin); print("contract: wvk", d["subnet"]["weight_version_key"], "window_h", d["subnet"].get("king_payout_window_hours"), "payout:", {k: d.get("payout",{}).get(k) for k in ("window_hours","effective_at")})'
curl -s -m 10 http://127.0.0.1:8787/api/v1/snapshot | python3 -c 'import json,sys
d=json.load(sys.stdin); p=d.get("payout") or {}
print("snapshot payout: burn", p.get("burn"), "n_paid", p.get("n_paid"), [(x.get("reign_number"), (x.get("hotkey") or "")[:8], x.get("uid"), x.get("share"), (x.get("paid_until") or "")[:16]) for x in p.get("paid", [])])'
echo "$(ts) deploy done. Next: commit the toml stamp on the box, push the branch, Discord notice (ops/v12/discord_payout_notice.py)."
