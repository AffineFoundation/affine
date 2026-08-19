#!/usr/bin/env bash
# p4008: R338 MERGE backlog (R882/R883) idle — wait for reign36 vera HF DL, serve king GPUs 2,3,
# then chall+n80 R882 :8002 GPUs4,5 and R883 :8003 GPUs6,7. Never pkill -f.
# R884 stays parked until a GPU pair frees (see wait_r884_after_slot_p4008.sh).
set -euo pipefail
exec >/root/logs/p4008_r882_r883_wait_king_chall.nohup 2>&1
source /root/venv/bin/activate
[[ -f /root/mine.env ]] && set -a && source /root/mine.env && set +a
export HF_HOME=${HF_HOME:-/root/hf}
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_SNAP=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/$KING_REV
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
log(){ echo "[p4008-r882r883] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

hub_ok(){
  local path=$1
  local n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge 2 ]]
}

log "WAIT vera king hub_ok at $KING_SNAP"
for i in $(seq 1 720); do
  if hub_ok "$KING_SNAP"; then log "KING_HUB_OK poll=$i shards=$(ls "$KING_SNAP"/model-*-of-*.safetensors | wc -l)"; break; fi
  for d in /root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/*; do
    if hub_ok "$d"; then KING_SNAP=$d; KING_REV=$(basename "$d"); log "KING_HUB_OK alt=$d"; break 2; fi
  done
  if (( i % 12 == 0 )); then
    du -sh /root/hf/hub/models--vera6--affine-5g4yy75zuz-t6 2>/dev/null || true
    tail -n 3 /root/logs/p4008_king_dl.nohup 2>/dev/null || true
  fi
  sleep 10
done
hub_ok "$KING_SNAP" || { log "FATAL king download timeout"; exit 1; }

for i in $(seq 1 60); do
  curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 5
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { log FATAL teacher down; exit 1; }

need_king=1
if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
  if echo "$kid" | grep -qiE 'vera6|5g4yy75zuz'; then
    log "king already vera id=$kid"; need_king=0
  else
    log "stale king id=$kid — will replace"
    python3 - <<'PY'
import os,re,signal,subprocess,time
try:
  out=subprocess.check_output(["ss","-lptn","sport = :8001"],text=True,stderr=subprocess.DEVNULL)
except Exception:
  out=""
for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(2)
for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
PY
  fi
fi

if [[ "$need_king" -eq 1 ]]; then
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
    [[ "$used" -lt 4096 ]] && break
    sleep 2
  done
  export CUDA_VISIBLE_DEVICES=2,3
  export TRITON_CACHE_DIR=/root/.triton/cache/king_r338_vera
  mkdir -p "$TRITON_CACHE_DIR" /root/logs
  : >/root/logs/vllm_king_vera_p4008.log
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$KING_SNAP" \
    --port 8001 --tensor-parallel-size 2 --max-model-len 65536 \
    --gpu-memory-utilization 0.80 --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    --served-model-name "$KING_REPO" --revision "$KING_REV" \
    >/root/logs/vllm_king_vera_p4008.log 2>&1 &
  echo $! >/root/logs/vllm_king_vera_p4008.pid
  log "king serve pid=$(cat /root/logs/vllm_king_vera_p4008.pid)"
  for i in $(seq 1 240); do
    if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null; then log "KING_READY poll=$i"; break; fi
    kill -0 "$(cat /root/logs/vllm_king_vera_p4008.pid)" 2>/dev/null || { log FATAL king died; tail -80 /root/logs/vllm_king_vera_p4008.log; exit 1; }
    sleep 5
  done
fi
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz' || { log "FATAL king not vera ($kid)"; exit 5; }

launch_one(){
  local TAG=$1 GPUS=$2 PORT=$3 MERGE=$4
  local LOG=/root/logs/p4008_${TAG}_chall_n80_wvk7.log
  local CHALL_LOG=/root/logs/vllm_chall_${TAG}_p4008.log
  local PIDF=/root/logs/vllm_chall_${TAG}.pid
  local SIM=/root/affine_data/${TAG}_sim_result_reign36_wvk7.json
  local PROG=/root/affine_data/${TAG}_sim_progress_reign36_wvk7.json
  test -f "$MERGE/config.json" || { log "FATAL missing $MERGE"; return 1; }
  local n; n=$(ls "$MERGE"/model-*-of-*.safetensors 2>/dev/null | wc -l)
  [[ "$n" -ge 16 ]] || { log "FATAL $TAG merge shards=$n"; return 1; }
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPUS | awk '{s+=$1} END{print s+0}')
    [[ "$used" -lt 4096 ]] && break
    sleep 2
  done
  export CUDA_VISIBLE_DEVICES=$GPUS
  export TRITON_CACHE_DIR=/root/.triton/cache/chall_$TAG
  mkdir -p "$TRITON_CACHE_DIR" /root/logs /root/affine_data
  : >"$CHALL_LOG" : >"$LOG"
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
    --port "$PORT" --tensor-parallel-size 2 --max-model-len 65536 \
    --gpu-memory-utilization 0.72 --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' --enforce-eager \
    >"$CHALL_LOG" 2>&1 &
  echo $! >"$PIDF"
  log "$TAG chall pid=$(cat $PIDF) :$PORT"
  for i in $(seq 1 240); do
    if curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then log "$TAG CHALL_READY"; break; fi
    kill -0 "$(cat $PIDF)" 2>/dev/null || { log FATAL $TAG chall died; tail -60 "$CHALL_LOG"; return 1; }
    sleep 5
  done
  BLOCK_HASH=$(python3 -c "import hashlib,time; print(hashlib.sha256(f'${TAG}-reign36-wvk7-p4008-{time.time()}'.encode()).hexdigest())")
  rm -f "$SIM" "$PROG"
  nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    HF_HOME="$HF_HOME" PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" AFFINE_DATA_DIR=/root/affine_data \
    /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
    --teacher-repo "$TEACHER_REPO" --king-repo "$KING_REPO" --king-rev "$KING_REV" \
    --chall-repo "$MERGE" --chall-rev local --chall-port "$PORT" --n-turns 80 \
    --hotkey "local-${TAG}-reign36-wvk7" --block-hash "$BLOCK_HASH" \
    --out "$SIM" --progress-out "$PROG" --save-artifact >>"$LOG" 2>&1 &
  echo $! >/root/logs/${TAG}_sim_wvk7.pid
  log "$TAG n80 pid=$(cat /root/logs/${TAG}_sim_wvk7.pid)"
  date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/${TAG}_n80_launched.p4008
}

launch_one r882 4,5 8002 /tmp/r882_merged
launch_one r883 6,7 8003 /tmp/r883_merged
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4008_r882_r883_n80_launched
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4008_r882_r883_armed.done
log "ARMED R882+R883 n80 vs reign36 — waiting both"
wait || true
log "DONE waiters exited"
