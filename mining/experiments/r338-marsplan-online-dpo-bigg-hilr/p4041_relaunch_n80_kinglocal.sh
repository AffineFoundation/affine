#!/usr/bin/env bash
# p4041: R338 n80 died — vLLM king id is KING_LOCAL path, not hub repo string.
set -euo pipefail
LOG=/root/logs/p4041_r338_n80_relaunch.nohup
exec >>"$LOG" 2>&1
log(){ echo "[p4041-r338] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=/root/affine_data
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
KING_LOCAL=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
MERGED=/tmp/r338_merged
SIM_N80=/root/affine_data/r338_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r338_sim_progress_reign36_wvk7.json
SIM_DEC=/root/affine_data/r338_decision.json
# health
for p in 8000 8001 8002; do
  curl -sf -m 5 "http://127.0.0.1:$p/v1/models" >/dev/null || { log "FATAL :$p down"; exit 1; }
done
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
cid=$(curl -sf -m 3 http://127.0.0.1:8002/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
tid=$(curl -sf -m 3 http://127.0.0.1:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "ids teacher=$tid king=$kid chall=$cid"
# kill any stale sim
if [[ -f /root/logs/r338_sim_wvk7.pid ]]; then
  old=$(cat /root/logs/r338_sim_wvk7.pid)
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    log "stopping stale sim pid=$old"; kill "$old" || true; sleep 2
  fi
fi
BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r338-reign36-wvk7-p4041-{time.time()}".encode()).hexdigest())
PY
)
log "launch n80 king-repo=$kid chall=$cid block_hash=${BLOCK_HASH:0:16}…"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$tid" \
  --king-repo "$kid" \
  --king-rev "$KING_REV" \
  --chall-repo "$cid" \
  --chall-rev local \
  --chall-port 8002 \
  --n-turns 80 \
  --hotkey local-r338-reign36-wvk7-p4041 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
echo $! >/root/logs/r338_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r338_n80_launched.p4041
log "n80 pid=$(cat /root/logs/r338_sim_wvk7.pid) — waiting"
# brief smoke: must stay alive >20s and not 404 again
sleep 25
pid=$(cat /root/logs/r338_sim_wvk7.pid)
if ! kill -0 "$pid" 2>/dev/null; then
  log "FATAL sim died early"; tail -n 60 "$LOG"; exit 2
fi
if grep -q '404 Not Found' "$LOG"; then
  log "FATAL still 404"; tail -n 40 "$LOG"; exit 3
fi
log "SMOKE_OK sim alive pid=$pid"
# detach wait
nohup bash -c '
  while kill -0 '"$pid"' 2>/dev/null; do sleep 30; done
  echo "[p4041-r338] $(date -u +%Y-%m-%dT%H:%M:%SZ) SIM_EXIT"
  if [[ -f '"$SIM_N80"' ]]; then
    /root/venv/bin/python3 - <<PY
import json
from pathlib import Path
d=json.loads(Path("'"$SIM_N80"'").read_text())
v=d.get("verdict") if isinstance(d.get("verdict"), dict) else {}
dp=(v.get("duel_params") or {}) if isinstance(v, dict) else {}
margin = v.get("margin") if v else (d.get("margin") or d.get("mean_margin"))
se = v.get("se") if v else d.get("se")
print(json.dumps({"margin":margin,"se":se,"duel_params":dp}, indent=2))
PY
  else
    echo "[p4041-r338] SIM missing"
  fi
' >/root/logs/p4041_r338_n80_waiter.nohup 2>&1 &
echo $! >/root/logs/p4041_r338_n80_waiter.pid
log "waiter pid=$(cat /root/logs/p4041_r338_n80_waiter.pid) DONE_ARM"
