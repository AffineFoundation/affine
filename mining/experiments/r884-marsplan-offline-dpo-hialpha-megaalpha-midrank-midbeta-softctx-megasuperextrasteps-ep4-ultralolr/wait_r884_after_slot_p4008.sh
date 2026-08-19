#!/usr/bin/env bash
# p4008: After R882 or R883 n80 finishes on R338, reclaim that GPU pair for R884 chall+n80.
# Never pkill -f. Do not touch teacher 0,1 or king 2,3.
set -euo pipefail
exec >/root/logs/p4008_r884_wait_slot.nohup 2>&1
source /root/venv/bin/activate
[[ -f /root/mine.env ]] && set -a && source /root/mine.env && set +a
export HF_HOME=${HF_HOME:-/root/hf}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE=/tmp/r884_merged
log(){ echo "[p4008-r884] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

hub_ok(){
  local path=$1 n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge 16 ]]
}
hub_ok "$MERGE" || { log "FATAL merge incomplete"; exit 1; }

log "WAIT for R882 or R883 n80 result so a GPU pair frees"
SLOT=""
for i in $(seq 1 720); do
  if [[ -f /root/affine_data/r882_sim_result_reign36_wvk7.json ]]; then SLOT=r882; GPUS=4,5; PORT=8002; break; fi
  if [[ -f /root/affine_data/r883_sim_result_reign36_wvk7.json ]]; then SLOT=r883; GPUS=6,7; PORT=8003; break; fi
  sleep 30
done
[[ -n "$SLOT" ]] || { log "FATAL timeout waiting sibling n80"; exit 1; }
log "slot free after $SLOT → R884 on GPUs=$GPUS :$PORT"
export PORT GPUS

# kill only the finished sibling's chall listener on that port (exact pid via ss)
python3 - <<'PY'
import os, re, signal, subprocess, time
port = os.environ["PORT"]
try:
  out = subprocess.check_output(["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
except Exception:
  out = ""
pids = set(int(x) for x in re.findall(r"pid=(\d+)", out))
for pid in sorted(pids):
  try:
    os.kill(pid, signal.SIGTERM)
  except ProcessLookupError:
    pass
time.sleep(3)
for pid in sorted(pids):
  try:
    os.kill(pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
print(f"reaped :{port} pids={sorted(pids)}")
PY

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPUS | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 4096 ]] && { log "GPUs $GPUS free used=$used"; break; }
  sleep 2
done

# ensure TK warm
for i in $(seq 1 60); do
  curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null \
    && curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null && break
  sleep 5
done
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz' || { log "FATAL king not vera ($kid)"; exit 5; }

export CUDA_VISIBLE_DEVICES=$GPUS
export TRITON_CACHE_DIR=/root/.triton/cache/chall_r884
mkdir -p "$TRITON_CACHE_DIR" /root/logs /root/affine_data
LOG=/root/logs/p4008_r884_chall_n80_wvk7.log
CHALL_LOG=/root/logs/vllm_chall_r884_p4008.log
PIDF=/root/logs/vllm_chall_r884.pid
SIM=/root/affine_data/r884_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r884_sim_progress_reign36_wvk7.json
: >"$CHALL_LOG" : >"$LOG"
nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
  --port "$PORT" --tensor-parallel-size 2 --max-model-len 65536 \
  --gpu-memory-utilization 0.72 --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' --enforce-eager \
  >"$CHALL_LOG" 2>&1 &
echo $! >"$PIDF"
log "r884 chall pid=$(cat $PIDF) :$PORT"
for i in $(seq 1 240); do
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then log "r884 CHALL_READY"; break; fi
  kill -0 "$(cat $PIDF)" 2>/dev/null || { log FATAL chall died; tail -60 "$CHALL_LOG"; exit 1; }
  sleep 5
done
BLOCK_HASH=$(python3 -c "import hashlib,time; print(hashlib.sha256(f'r884-reign36-wvk7-p4008-{time.time()}'.encode()).hexdigest())")
rm -f "$SIM" "$PROG"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="$HF_HOME" PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" AFFINE_DATA_DIR=/root/affine_data \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" --king-repo "$KING_REPO" --king-rev "$KING_REV" \
  --chall-repo "$MERGE" --chall-rev local --chall-port "$PORT" --n-turns 80 \
  --hotkey local-r884-reign36-wvk7 --block-hash "$BLOCK_HASH" \
  --out "$SIM" --progress-out "$PROG" --save-artifact >>"$LOG" 2>&1 &
echo $! >/root/logs/r884_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r884_n80_launched.p4008
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4008_r884_armed.done
log "ARMED R884 n80 pid=$(cat /root/logs/r884_sim_wvk7.pid) after $SLOT"
wait || true
log "DONE"
