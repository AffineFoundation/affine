#!/usr/bin/env bash
# p3924: R837 MERGE → chall :8002 + v4 n80 lunar GPUs 4,5 vs reign36 vera
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
GPUS=4,5; CHALL_PORT=8002; export CUDA_VISIBLE_DEVICES=$GPUS
export HF_HOME=${HF_HOME:-/root/hf} AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r837_merged
UTIL=${UTIL:-0.72}
LOG=/root/logs/p3924_r837_chall_n80_wvk7.log
CHALL_LOG=/root/logs/vllm_chall_r837_p3924.log
PIDF=/root/logs/vllm_chall_r837.pid
SIM_N80=/root/affine_data/r837_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r837_sim_progress_reign36_wvk7.json
mkdir -p /root/logs /root/affine_data
: >"$LOG"
log(){ echo "[p3924-r837] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
stop_pid(){ local pid=$1; [[ "$pid" =~ ^[0-9]+$ ]] || return 0; kill -0 "$pid" 2>/dev/null || return 0; kill "$pid" 2>/dev/null || true; for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -9 "$pid" 2>/dev/null || true; }
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]] || { log "FATAL merge shards=$n"; exit 2; }
# seed triton from king if needed
if [[ ! -d /root/.triton/cache/chall_r837 ]] || [[ $(find /root/.triton/cache/chall_r837 -name '__triton_launcher.so' 2>/dev/null | wc -l) -lt 1 ]]; then
  mkdir -p /root/.triton/cache
  if [[ -d /root/.triton/cache/king ]]; then cp -a /root/.triton/cache/king /root/.triton/cache/chall_r837; log "seeded triton from king"; fi
fi
export TRITON_CACHE_DIR=/root/.triton/cache/chall_r837
if [[ -f "$PIDF" ]]; then stop_pid "$(cat "$PIDF")" "old chall"; fi
# kill leftover Worker on 4,5 if any — by exact parent only; workers die with parent
nohup /root/venv/bin/vllm serve "$MERGE_DIR" --port "$CHALL_PORT" --tensor-parallel-size 2 \
  --max-model-len 65536 --gpu-memory-utilization "$UTIL" --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' --enforce-eager \
  >"$CHALL_LOG" 2>&1 &
echo $! >"$PIDF"
log "chall vllm pid=$(cat $PIDF)"
for i in $(seq 1 180); do
  if curl -sf -m 3 http://127.0.0.1:$CHALL_PORT/v1/models >/dev/null 2>&1; then log "chall READY iter=$i"; break; fi
  sleep 5
done
curl -sf -m 5 http://127.0.0.1:$CHALL_PORT/v1/models | tee -a "$LOG" >/dev/null
BH=1d2f2f1903eb67b6242c2a0075bd0e4965e8eebcea541a420fcd7aa2b45d2025
nohup /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" --king-repo "$KING_REPO" --king-rev "$KING_REV" \
  --chall-repo "$MERGE_DIR" --chall-rev local --chall-port "$CHALL_PORT" --n-turns 80 \
  --hotkey local-r837-reign36-wvk7 --block-hash "$BH" \
  --out "$SIM_N80" --progress-out "$PROG" --save-artifact \
  >>"$LOG" 2>&1 &
echo $! >/root/logs/r837_sim_wvk7.pid
log "n80 pid=$(cat /root/logs/r837_sim_wvk7.pid)"
wait "$(cat /root/logs/r837_sim_wvk7.pid)" || true
log "n80 exit"
