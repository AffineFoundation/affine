#!/usr/bin/env bash
# p4121: R973 p4119b refused on false key-layout sanity.
# HF keys model.language_model.* / model.visual.* / lm_head.* are CORRECT —
# vLLM Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper remaps them to
# language_model.model.* / visual.* / language_model.lm_head.*.
# Same layout as king + successful R944. Skip remap; serve existing merge + n80.
# Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4121_r926_r973_chall_n80.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4121-r973] $(date -u +%Y-%m-%dT%H:%M:%SZ) start — serve HF-layout merge (no remap)"

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}

MERGE=/tmp/r973_merged
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8

python3 - <<'PY'
import json
from pathlib import Path
out=Path("/tmp/r973_merged")
wm=json.load(open(out/"model.safetensors.index.json"))["weight_map"]
missing=[s for s in set(wm.values()) if not (out/s).exists()]
assert not missing, missing
print("MERGE_OK nkeys",len(wm),"shards",len(set(wm.values())), flush=True)
PY

for gi in 3 4; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gi" | awk '{print $1+0}')
  echo "[p4121-r973] gpu$gi used_mib=$used"
  [[ "$used" -lt 2048 ]] || { echo FATAL GPU$gi busy; exit 2; }
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { echo FATAL teacher; exit 2; }
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king; exit 2; }
kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo ERROR king not vera; exit 5; }

stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
[[ -f /root/logs/vllm_chall_r973.pid ]] && stop_pid "$(cat /root/logs/vllm_chall_r973.pid)"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
sleep 2

gpus=3,4; port=8002; tag=r973
chall_log=/root/logs/vllm_chall_${tag}_p4121.log
pidf=/root/logs/vllm_chall_${tag}.pid
tcache=/root/.triton/cache/chall_${tag}
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r944 /root/.triton/cache/chall_r926; do
  if [[ -d "$cand" ]] && find "$cand" -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
    rm -rf "$tcache"; cp -a "$cand" "$tcache"; break
  fi
done
mkdir -p "$tcache"
: >"$chall_log"
CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$tcache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
    --port "$port" --tensor-parallel-size 2 --max-model-len 65536 \
    --gpu-memory-utilization 0.72 --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton --additional-config '{"gdn_prefill_backend": "triton"}' \
    --enforce-eager >"$chall_log" 2>&1 &
echo $! >"$pidf"
chall_pid=$(cat "$pidf")
ready=0
for i in $(seq 1 360); do
  curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$chall_pid" 2>/dev/null || { echo FATAL chall died; tail -80 "$chall_log"; exit 1; }
  grep -q 'ValueError: There is no module' "$chall_log" 2>/dev/null && { echo FATAL layout; exit 1; }
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL not ready; exit 1; }

bh=$(python3 -c 'import hashlib,time;print(hashlib.sha256(f"r973-reign36-wvk7-p4121-{time.time()}".encode()).hexdigest())')
sim_out=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
prog=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
rm -f "$sim_out" "$prog"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 "$SIM" \
  --teacher-repo "$TEACHER_REPO" --king-repo "$kid" --king-rev "$KING_REV" \
  --chall-repo "$MERGE" --chall-rev local --chall-port "$port" --n-turns 80 \
  --hotkey "local-r973-reign36-wvk7-p4121" --block-hash "$bh" \
  --out "$sim_out" --progress-out "$prog" --save-artifact \
  >>/root/logs/p4121_${tag}_chall_n80_wvk7.log 2>&1 &
echo $! >/root/logs/${tag}_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/${tag}_n80_launched.p4121
echo "[p4121-r973] ARMED n80 pid=$(cat /root/logs/${tag}_sim_wvk7.pid)"
