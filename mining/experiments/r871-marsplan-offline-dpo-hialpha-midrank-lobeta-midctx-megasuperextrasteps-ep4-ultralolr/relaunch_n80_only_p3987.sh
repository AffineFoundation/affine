#!/usr/bin/env bash
# p3987: R871 chall already LIVE on :8002; sync'd full affine_pkg; relaunch v4 n80 only.
# Never pkill -f. Do not touch teacher:8000, king:8001, GRPO on 2,3, or chall:8002.
set -euo pipefail

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi

export HF_HOME=${HF_HOME:-/root/hf}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r871_merged
CHALL_PORT=8002
LOG=/root/logs/p3987_r871_n80_wvk7.log
SIM_N80=/root/affine_data/r871_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r871_sim_progress_reign36_wvk7.json
SIM_DEC=/root/affine_data/r871_decision_reign36_wvk7.json
PIDF=/root/logs/r871_sim_wvk7.pid
mkdir -p /root/logs /root/affine_data

: >"$LOG"
log() { echo "[p3987-r871] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

log "START n80-only (affine_pkg sync + pyarrow; chall already :8002)"
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
cid=$(curl -sf -m 3 "http://127.0.0.1:${CHALL_PORT}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "king=$kid chall=$cid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { log "ERROR king not reign36"; exit 5; }

# stop prior dead/live sim by pidfile only
if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    log "stop prior sim pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
  rm -f "$PIDF"
fi

BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r871-reign36-wvk7-p3987-{time.time()}".encode()).hexdigest())
PY
)
log "launch n80 block_hash=${BLOCK_HASH:0:16}…"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC"
nohup env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$KING_REPO" \
  --king-rev "$KING_REV" \
  --chall-repo "$MERGE_DIR" \
  --chall-rev local \
  --chall-port "$CHALL_PORT" \
  --n-turns 80 \
  --hotkey local-r871-reign36-wvk7-p3987 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" >"$PIDF"
log "n80 pid=$SIM_PID"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r871_n80_launched.p3987

# brief smoke: process alive + progress file starts within ~3 min
for i in $(seq 1 36); do
  if [[ -f "$PROG" ]]; then
    log "PROGRESS_LIVE poll=$i $(head -c 200 "$PROG" | tr '\n' ' ')"
    exit 0
  fi
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    log "FATAL sim died early"
    tail -n 80 "$LOG"
    exit 1
  fi
  sleep 5
done
log "WARN no progress yet but sim alive pid=$SIM_PID — next pass poll"
exit 0
