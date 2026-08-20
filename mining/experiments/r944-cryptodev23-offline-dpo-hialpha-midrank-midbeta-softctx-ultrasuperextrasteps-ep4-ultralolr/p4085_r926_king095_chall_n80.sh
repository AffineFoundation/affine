#!/usr/bin/env bash
# p4085: R926 teacher READY; king OOM at gpu_mem=0.85 (model 65.5GiB → KV −2.7GiB).
# Relaunch king :8001 GPU2 @0.95; chall R944 :8002 GPUs3,4; v4 n80. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4085_r926_king095_chall_n80.log
mkdir -p /root/logs /root/affine_data /root/.triton/cache/king /root/.triton/cache/chall_r944
exec > >(tee -a "$LOG") 2>&1
echo "[p4085-r944] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}

_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
_CU13="${_SITE}/nvidia/cu13"
if [[ -x "${_CU13}/bin/nvcc" && -f "${_CU13}/include/cuda_fp16.h" ]]; then
  export CUDA_HOME=${CUDA_HOME:-$_CU13}
  export CUDA_PATH=$CUDA_HOME
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_LOCAL=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/${KING_REV}
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
KING_PIDF=/root/logs/vllm_king.pid
KING_LOG=/root/logs/vllm_king.log
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
MERGE=/tmp/r944_merged

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4085-r944] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1 why=${2:-}
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)" "$why pidf=$pidf"
  rm -f "$pidf"
}

hub_ok() {
  local path=$1 min=${2:-1} n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge "$min" ]]
}

wait_ready() {
  local port=$1 name=$2 pidf=$3 logf=$4 ready=0
  for i in $(seq 1 720); do
    if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1; echo "[p4085-r944] ${name}_READY poll=$i"; break
    fi
    if [[ -f "$pidf" ]]; then
      local pid; pid=$(cat "$pidf")
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[p4085-r944] ERROR $name died"; tail -80 "$logf"; exit 1
      fi
    fi
    if grep -qE 'OutOfMemoryError|CUDA out of memory|No available memory for the cache' "$logf" 2>/dev/null; then
      echo "[p4085-r944] ERROR $name mem fail"; tail -40 "$logf"; exit 1
    fi
    (( i % 12 == 0 )) && echo "[p4085-r944] wait $name :$port iter=$i last=$(tail -1 "$logf" 2>/dev/null | cut -c1-120)"
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || { echo "[p4085-r944] ERROR $name not ready"; tail -80 "$logf"; exit 1; }
}

# Teacher must already be up
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { echo FATAL teacher :8000 not READY; exit 2; }
echo "[p4085-r944] teacher READY kept"

hub_ok "$KING_LOCAL" 2 || { echo FATAL king incomplete; exit 3; }
hub_ok "$MERGE" 16 || { echo FATAL r944 merge incomplete; exit 3; }
[[ -f "$SIM" ]] || { echo FATAL missing $SIM; exit 3; }
echo "[p4085-r944] r944 shards=$(ls "$MERGE"/model-*-of-*.safetensors | wc -l)"

# Free GPU2 and :8001/:8002 only (exact PID)
stop_pidfile "$KING_PIDF" "stale king"
stop_pidfile /root/logs/vllm_chall_r944.pid "stale chall"
for port in 8001 8002; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "listener :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
done
sleep 2

used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | awk '{print $1+0}')
echo "[p4085-r944] gpu2 used_mib=$used2 (want <2048)"
[[ "$used2" -lt 2048 ]] || { echo FATAL GPU2 still busy; exit 2; }

VLLM_COMMON=(--max-model-len 65536 --max-num-batched-tokens 8192
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton
  --additional-config '{"gdn_prefill_backend": "triton"}'
  --enforce-eager)

# Ensure king triton cache has launcher
n_launch=$(find /root/.triton/cache/king -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
echo "[p4085-r944] king triton n_launcher=$n_launch"
if [[ "${n_launch:-0}" -lt 1 ]]; then
  if [[ -d /root/.triton/cache/teacher ]] && find /root/.triton/cache/teacher -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
    rm -rf /root/.triton/cache/king
    cp -a /root/.triton/cache/teacher /root/.triton/cache/king
    echo "[p4085-r944] seeded king Triton from teacher"
  fi
fi

echo "[p4085-r944] launch king :8001 GPU2 TP=1 gpu_mem=0.95 (was 0.85 OOM)"
: >"$KING_LOG"
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/root/.triton/cache/king \
  nohup /root/venv/bin/vllm serve "$KING_LOCAL" \
  --port 8001 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \
  "${VLLM_COMMON[@]}" --served-model-name "$KING_REPO" >>"$KING_LOG" 2>&1 &
echo $! >"$KING_PIDF"
echo "[p4085-r944] king pid=$(cat "$KING_PIDF")"
wait_ready 8001 king "$KING_PIDF" "$KING_LOG"

kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4085-r944] king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo ERROR king not vera; exit 5; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r944_tk_ready_p4085.done

tag=r944
merge=$MERGE
gpus=3,4
port=8002
chall_log=/root/logs/vllm_chall_${tag}_p4085.log
pidf=/root/logs/vllm_chall_${tag}.pid
tcache=/root/.triton/cache/chall_${tag}
sim_out=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
prog=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
sim_dec=/root/affine_data/${tag}_decision_reign36_wvk7.json
n80_log=/root/logs/p4085_${tag}_chall_n80_wvk7.log

if [[ -d /root/.triton/cache/king ]] && find /root/.triton/cache/king -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
  rm -rf "$tcache"
  cp -a /root/.triton/cache/king "$tcache"
  n_so=$(find "$tcache" -name '*.so' | wc -l)
  echo "[p4085-r944] $tag seeded triton from king n_so=$n_so"
else
  mkdir -p "$tcache"
fi

: >"$chall_log"
CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$tcache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$merge" \
    --port "$port" \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.72 \
    --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    --enforce-eager \
    >"$chall_log" 2>&1 &
echo $! >"$pidf"
chall_pid=$(cat "$pidf")
echo "[p4085-r944] $tag chall pid=$chall_pid"

ready=0
for i in $(seq 1 240); do
  if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    ready=1; echo "[p4085-r944] ${tag}_CHALL_READY poll=$i"; break
  fi
  if ! kill -0 "$chall_pid" 2>/dev/null; then
    echo "[p4085-r944] FATAL $tag chall died"; tail -100 "$chall_log"; exit 1
  fi
  if grep -q 'ImportError:.*__triton_launcher' "$chall_log" 2>/dev/null; then
    echo "[p4085-r944] FATAL $tag Triton ImportError"; tail -80 "$chall_log"; exit 1
  fi
  (( i % 12 == 0 )) && echo "[p4085-r944] wait chall :$port iter=$i"
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL $tag chall not ready; tail -80 "$chall_log"; exit 1; }

bh=$(python3 - <<PY
import hashlib, time
print(hashlib.sha256(f"r944-reign36-wvk7-p4085-{time.time()}".encode()).hexdigest())
PY
)
rm -f "$sim_out" "$prog" "$sim_dec"
: >"$n80_log"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 "$SIM" \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$kid" \
  --king-rev "$KING_REV" \
  --chall-repo "$merge" \
  --chall-rev local \
  --chall-port "$port" \
  --n-turns 80 \
  --hotkey "local-r944-reign36-wvk7-p4085" \
  --block-hash "$bh" \
  --out "$sim_out" \
  --progress-out "$prog" \
  --save-artifact \
  >>"$n80_log" 2>&1 &
echo $! >"/root/logs/${tag}_sim_wvk7.pid"
date -u +%Y-%m-%dT%H:%M:%SZ >"/root/logs/${tag}_n80_launched.p4085"
echo "[p4085-r944] $tag n80 LIVE pid=$(cat /root/logs/${tag}_sim_wvk7.pid) bh=${bh:0:16}…"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4085_r926_king095_n80_armed.done
echo "[p4085-r944] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
