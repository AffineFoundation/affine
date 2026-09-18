#!/usr/bin/env bash
# wvk 21 -> 22 FLIP: score_mode min_rg -> sd_min_rga, n_turns 1300 -> 1000, δ /
# forfeit in sd units, at the next duel boundary. Explicit dated operator
# directive 2026-09-18 (Jacob 10:04 / 10:25 UTC); the GO to flip is relayed by
# the coordinator — do not run without it. Same pitfall handling as
# deploy_sd_shadow.sh (env from /proc, keepalive off, deadman paused, stale
# in_flight, validator stopped while the toml flips and the pods redeploy).
# Order inside the stop window: toml flip -> llms.txt (Upcoming -> Fork
# history) -> pod redeploy -> validator start. Discord "live" line is NOT
# posted here (after the first wvk-22 verdict stamps).
# Rollback rule (docs/wvk22-plan.md §3, Jacob 15:11 UTC): first 3 wvk-22 verdicts —
# SE > 2x shadow expectation, teacher-vs-king z <= -2, any leg binds > 80 %,
# or a leg dropped on > 5 % of valid turns -> bash ops/v17/rollback_wvk22.sh.
# Run on the box:
#   EXPECT_KING_HOTKEY=5FptWRWY4TM69QQTDdrMkc4L1ZdiFxm6krXELR14VybDnbxD DELTA_SD=0.07 FORFEIT_SD=-4.5 bash ops/v17/deploy_wvk22.sh
#   (FORCE_BOUNDARY=1 to skip the boundary wait; eval pod only — bench/chat pods untouched)
set -euo pipefail
HERE=/home/const/subnet120/ops/v17
REPO=/home/const/subnet120
LOG=$HERE/deploy_wvk22.log
DIRECTIVE_DATE=2026-09-18
DELTA_SD=${DELTA_SD:?set DELTA_SD (δ in sd units, e.g. 0.10)}
FORFEIT_SD=${FORFEIT_SD:?set FORFEIT_SD (forfeit floor in sd units, e.g. -2.4)}
PAUSE=/home/const/.affine/deadman.pause
mkdir -p "$HERE"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
echo "$(ts) === deploy_wvk22.sh start (wvk 21 -> 22, δ_sd=$DELTA_SD forfeit_sd=$FORFEIT_SD) (HEAD $(git rev-parse --short HEAD))"

# --- 0. preflight (nothing changes yet)
grep -q '^weight_version_key = 21$' affine/affine.toml || { echo "$(ts) toml is not at wvk 21; abort"; exit 1; }
grep -q '^score_mode = "min_rg"$' affine/affine.toml || { echo "$(ts) score_mode is not min_rg; abort"; exit 1; }
grep -q '^\[duel.sd_meter\]$' affine/affine.toml || { echo "$(ts) [duel.sd_meter] missing; abort"; exit 1; }
python ops/v17/wvk22_toml_edits.py --preview --delta-sd "$DELTA_SD" --forfeit-sd "$FORFEIT_SD" >/dev/null
python -c "import ast; ast.parse(open('affine/scripts/build_llms_txt.py').read())"
grep -q 'WVK22_NOTICE' affine/scripts/build_llms_txt.py || { echo "$(ts) llms notice section missing; abort"; exit 1; }
python -c 'import sys; sys.path.insert(0,"affine"); from evalsrv import sdmeter, dueling; print("evalsrv imports ok")'
# at least one shadow verdict must exist and be sane (gate of docs/wvk22-plan.md §2)
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("affine/state/history.jsonl") if l.strip()]
sh=[r for r in rows if r.get("event")=="verdict" and ((r.get("verdict") or {}).get("shadow") or {}).get("sd_meter")]
assert sh, "no verdict with shadow.sd_meter yet — gate not met"
for r in sh[-3:]:
    s=r["verdict"]["shadow"]["sd_meter"]; loo=s["by_anchor"].get("loo") or {}
    assert loo.get("available"), f"{r['challenge_id']}: loo anchor unavailable"
    print("shadow", r["challenge_id"], "margin", s.get("margin"), "se", s.get("se"), "z", s.get("z"),
          "t_vs_k z", (loo.get("teacher_vs_king") or {}).get("z"), "binds", loo["challenger"]["bind_frac"])
PY
# king pin: the flip must not disturb the standing king (reign 15, chal-00581, uid 222)
KING_HK_BEFORE=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["king"]["hotkey"])')
KING_REPO_BEFORE=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["king"]["repo"])')
echo "$(ts) king before: $KING_HK_BEFORE $KING_REPO_BEFORE"
[[ "$KING_HK_BEFORE" == "${EXPECT_KING_HOTKEY:-$KING_HK_BEFORE}" ]] || { echo "$(ts) king hotkey differs from EXPECT_KING_HOTKEY; abort"; exit 1; }
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

# --- 3a. Jacob's condition (go 2026-09-18 15:11 UTC: "We go on the new scoring
# but only when the current queued models have run"): every entry queued as of
# 15:11 UTC is judged under wvk 21. The flip boundary is the verdict of the
# LAST of them — not earlier, even if the queue empties faster; not later,
# even if post-cutoff entries were dispatched (the one in flight at the stop
# has no verdict, State.load requeues it, it is judged under wvk 22).
CUTOFF=(chal-00582 chal-00583 chal-00584 chal-00585 chal-00586 chal-00587 chal-00588)
cutoff_pending() { python3 - "${CUTOFF[@]}" <<'PY'
import json, sys
ids = set(sys.argv[1:])
s = json.load(open("affine/state/state.json"))
f = s.get("in_flight"); fid = (f or {}).get("challenge_id") if isinstance(f, dict) else f
q = [e.get("challenge_id") for e in s.get("queue", [])]
pending = [c for c in ids if c == fid or c in q]
hist = {json.loads(l).get("challenge_id") for l in open("affine/state/history.jsonl") if l.strip()}
missing = [c for c in ids if c not in pending and c not in hist]   # neither active nor decided (should not happen)
print(json.dumps({"pending": sorted(pending), "in_flight": fid, "queue": q, "missing": sorted(missing)}))
PY
}
echo "$(ts) waiting for the cutoff set to finish under wvk 21: ${CUTOFF[*]}"
flagged=0
for i in $(seq 1 4320); do   # up to 36 h at 30 s
  st=$(cutoff_pending)
  pend=$(python3 -c 'import json,sys;print(len(json.loads(sys.argv[1])["pending"]))' "$st")
  fid=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["in_flight"] or "")' "$st")
  if [[ "$pend" == 0 ]]; then echo "$(ts) cutoff set done: $st"; break; fi
  if [[ -n "$fid" && "$flagged" == 0 ]] && ! printf '%s\n' "${CUTOFF[@]}" | grep -qx "$fid"; then
    echo "$(ts) FLAG: non-cutoff entry $fid is in flight while cutoff entries are still pending ($st) — it runs under wvk 21 (deferred cutoff entry?); tell the coordinator"
    flagged=1
  fi
  (( i % 20 == 0 )) && echo "$(ts)   cutoff pending: $st"
  sleep 30
done
[[ "$pend" == 0 ]] || { echo "$(ts) cutoff set still pending after 36 h; abort (nothing changed)"; exit 1; }
# The last cutoff verdict IS the boundary: stop now. A post-cutoff duel that
# was dispatched in the meantime has no verdict and is requeued (wvk 22).
FORCE_BOUNDARY=1

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
for i in $(seq 1 2400); do
  [[ "$reached" == 1 ]] && break
  cid=$(cur_cid); h=$(health); busy=$(busy_of "$h")
  if [[ -z "$cid" && "$busy" == "False" ]]; then reached=1; break; fi
  if [[ -n "$START_CID" && "$cid" != "$START_CID" ]]; then reached=1; break; fi
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
  if has_verdict "$cid"; then
    cp affine/state/state.json "$HERE/state.before_inflight_clear.$(date -u +%Y%m%dT%H%M%SZ).json"
    python3 -c 'import json;p="affine/state/state.json";s=json.load(open(p));s["in_flight"]=None;json.dump(s,open(p,"w"),indent=1)'
    echo "$(ts) cleared stale in_flight $cid (verdict already in history)"
  else
    echo "$(ts) in_flight $cid has NO verdict in history — leaving it; State.load requeues it"
  fi
fi

# --- 5. contract flip (toml + website mirror), llms.txt
python ops/v17/wvk22_toml_edits.py --apply "$DIRECTIVE_DATE" --wvk-to 22 --delta-sd "$DELTA_SD" --forfeit-sd "$FORFEIT_SD"
grep -E '^(weight_version_key|score_mode|n_turns|min_margin_sd|forfeit_sd|k_sigma) ' affine/affine.toml
python -c "from affine.config import load_config; c=load_config(); d=c.duel; assert c.weight_version_key==22 and d.score_mode=='sd_min_rga' and d.n_turns==1000; print('config ok: wvk', c.weight_version_key, d.score_mode, 'n_turns', d.n_turns, 'delta_sd', d.sd_meter['min_margin_sd'], 'forfeit_sd', d.sd_meter['forfeit_sd'])"
python ops/v17/llms_wvk22_flip_edits.py --date "$DIRECTIVE_DATE" --delta-sd "$DELTA_SD" --forfeit-sd "$FORFEIT_SD"
python affine/scripts/build_llms_txt.py | tail -1
grep -q "^## Fork history: wvk 22" affine/website/llms.txt && echo "$(ts) llms.txt has Fork history: wvk 22"

# --- 5b. pods: code + flipped toml
# eval pod ONLY: the bench pod (reign 15's card pass) and the chat pod are not touched;
# the king seat (pm2 affine-king-datagen / kingctl) reads state.json and is not restarted.
cd affine && python scripts/redeploy_pods.py --role eval && cd "$REPO"
echo "$(ts) redeploy_pods done"
"${POD_SSH[@]}" 'grep -E "^(weight_version_key|score_mode|n_turns|min_margin_sd|forfeit_sd) " /root/affine/affine.toml' || echo "$(ts) WARNING pod verify ssh failed"

# --- 6. validator back
pm2 start affine-validator >/dev/null
sleep 5
pm2 ls | grep affine-validator || true
for i in $(seq 1 60); do h=$(health); [[ -n "$h" ]] && { echo "$(ts) pod health: $h"; break; }; sleep 5; done
pm2 restart affine-dash >/dev/null 2>&1 && echo "$(ts) affine-dash restarted" || true
KING_HK_AFTER=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["king"]["hotkey"])')
[[ "$KING_HK_AFTER" == "$KING_HK_BEFORE" ]] && echo "$(ts) king unchanged: $KING_HK_AFTER" || echo "$(ts) WARNING king changed across the restart: $KING_HK_BEFORE -> $KING_HK_AFTER"
python ops/king-datagen/kingctl.py status 2>/dev/null | head -3 || true
git add affine/affine.toml affine/website/code/affine.toml affine/scripts/build_llms_txt.py affine/website/llms.txt && git commit -q -m "contract: sd-meter min(z_R, typ_c, z_A) becomes the rule, n_turns 1300 -> 1000, wvk 21 -> 22 (operator directive 2026-09-18; δ_sd=$DELTA_SD forfeit_sd=$FORFEIT_SD)" && echo "$(ts) committed $(git rev-parse --short HEAD)"
echo "$(ts) deploy done. Next: first wvk-22 verdict must stamp duel_params.score_mode=sd_min_rga, n_turns 1000, shadow.sd_meter.role=rule; then Discord live line + AGENTS.md snapshot; rollback rule: ops/v17/rollback_wvk22.sh."
echo DONE
