#!/usr/bin/env bash
# p4119b: R973 after visual repair, chall died — keys were model.language_model.*
# / model.visual.* but vLLM Qwen3_5Moe wants language_model.model.* / visual.*.
# Cause: p4118 --no-save-original-format. Rematch MERGE on free GPUs 3,4 WITH
# default save_original_format + visual-only missing patch, then chall+n80.
# Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4119b_r926_r973_remerge_sof_n80.log
mkdir -p /root/logs /root/affine_data /root/.triton/cache
exec > >(tee -a "$LOG") 2>&1
echo "[p4119b-r973] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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

BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
ADAPTER=/root/r973/train/adapter
MERGE=/tmp/r973_merged
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
MERGE_PY=/root/mining_src/r973-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-ultrasuperextrasteps-ep4-hilr/merge_lora.py

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4119b-r973] kill pid=$pid ($why)"
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

[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no adapter; exit 3; }
[[ -f "$BASE/config.json" ]] || { echo FATAL no base; exit 3; }
[[ -f "$MERGE_PY" ]] || { echo FATAL no merge_py; exit 3; }
[[ -f "$SIM" ]] || { echo FATAL missing $SIM; exit 3; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { echo FATAL teacher :8000; exit 2; }
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king :8001; exit 2; }
echo "[p4119b-r973] TK READY kept"

for gi in 3 4; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gi" | awk '{print $1+0}')
  echo "[p4119b-r973] gpu$gi used_mib=$used"
  [[ "$used" -lt 2048 ]] || { echo FATAL GPU$gi busy; exit 2; }
done

stop_pidfile /root/logs/vllm_chall_r973.pid "stale r973 chall"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
sleep 2

rm -rf "$MERGE"
echo "[p4119b-r973] MERGE start CUDA=3,4 save_original_format=True (default) max_shard=4GB"
export CUDA_VISIBLE_DEVICES=3,4
python3 "$MERGE_PY" \
  --base "$BASE" \
  --adapter "$ADAPTER" \
  --out "$MERGE" \
  --device-map auto \
  --max-shard-size 4GB \
  --save-original-format
unset CUDA_VISIBLE_DEVICES
hub_ok "$MERGE" 16 || { echo FATAL merge incomplete shards=$(ls "$MERGE"/model-*-of-*.safetensors 2>/dev/null | wc -l); exit 4; }

# Sanity: keys must not be model.language_model / model.visual top-level for vLLM
python3 - <<'PY'
from pathlib import Path
from safetensors import safe_open
out=Path("/tmp/r973_merged")
shard=next(out.glob("model-*-of-*.safetensors"))
with safe_open(str(shard), framework="pt") as f:
    keys=list(f.keys())[:20]
print("[p4119b-r973] sample_keys", keys[:5], flush=True)
bad=sum(1 for k in keys if k.startswith("model.language_model") or k.startswith("model.visual"))
if bad:
    raise SystemExit(f"REFUSE still bad key layout bad_sample={bad}")
print("[p4119b-r973] key_layout_ok", flush=True)
PY

touch /root/logs/r973_merge.done
echo READY_FOR_N80 >/root/logs/r973_merge_ready
echo "[p4119b-r973] MERGE_DONE shards=$(ls "$MERGE"/model-*-of-*.safetensors | wc -l)"

kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4119b-r973] king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo ERROR king not vera; exit 5; }

tag=r973
gpus=3,4
port=8002
chall_log=/root/logs/vllm_chall_${tag}_p4119b.log
pidf=/root/logs/vllm_chall_${tag}.pid
tcache=/root/.triton/cache/chall_${tag}
sim_out=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
prog=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
sim_dec=/root/affine_data/${tag}_decision_reign36_wvk7.json
n80_log=/root/logs/p4119b_${tag}_chall_n80_wvk7.log

seeded=0
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r944 /root/.triton/cache/chall_r926 /root/.triton/cache/teacher; do
  if [[ -d "$cand" ]] && find "$cand" -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
    rm -rf "$tcache"
    cp -a "$cand" "$tcache"
    n_so=$(find "$tcache" -name '*.so' | wc -l)
    echo "[p4119b-r973] Triton seed from $cand n_so=$n_so"
    seeded=1
    break
  fi
done
[[ "$seeded" -eq 1 ]] || mkdir -p "$tcache"

: >"$chall_log"
CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$tcache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
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
echo "[p4119b-r973] chall pid=$chall_pid :$port GPUs=$gpus"

ready=0
for i in $(seq 1 360); do
  if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    ready=1; echo "[p4119b-r973] CHALL_READY poll=$i"; break
  fi
  if ! kill -0 "$chall_pid" 2>/dev/null; then
    echo "[p4119b-r973] FATAL chall died"; tail -100 "$chall_log"; exit 1
  fi
  if grep -q 'ImportError:.*__triton_launcher' "$chall_log" 2>/dev/null; then
    echo "[p4119b-r973] FATAL Triton ImportError"; tail -80 "$chall_log"; exit 1
  fi
  (( i % 12 == 0 )) && echo "[p4119b-r973] wait chall :$port iter=$i last=$(tail -1 "$chall_log" 2>/dev/null | cut -c1-120)"
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL chall not ready; tail -80 "$chall_log"; exit 1; }

bh=$(python3 - <<PY
import hashlib, time
print(hashlib.sha256(f"r973-reign36-wvk7-p4119b-{time.time()}".encode()).hexdigest())
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
  --chall-repo "$MERGE" \
  --chall-rev local \
  --chall-port "$port" \
  --n-turns 80 \
  --hotkey "local-r973-reign36-wvk7-p4119b" \
  --block-hash "$bh" \
  --out "$sim_out" \
  --progress-out "$prog" \
  --save-artifact \
  >>"$n80_log" 2>&1 &
echo $! >"/root/logs/${tag}_sim_wvk7.pid"
date -u +%Y-%m-%dT%H:%M:%SZ >"/root/logs/${tag}_n80_launched.p4119b"
echo "[p4119b-r973] n80 LIVE pid=$(cat /root/logs/${tag}_sim_wvk7.pid) bh=${bh:0:16}…"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4119b_r926_r973_remerge_n80_armed.done
echo "[p4119b-r973] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
