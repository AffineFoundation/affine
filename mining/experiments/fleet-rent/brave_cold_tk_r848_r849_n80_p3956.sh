#!/usr/bin/env bash
# p3956: idle brave-raven — cold TK (teacher+reign36 vera) then R848+R849 v4 n80.
# Merges already local: /tmp/r848_merged (4,5/:8002) Soft Mid Mid Soft MidRank Midβ SoftCtx UltraLoLR
#                      /tmp/r849_merged (6,7/:8003) Soft Mid Mid Soft HiRank Hiβ SoftCtx UltraLoLR
# Never pkill -f. Hard-pin GPUS/CHALL_PORT after mine.env in lean challs.
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
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
# p3960: offline — brave TP=2 teacher hung with CloudFront HF socket post-NCCL
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
_CU13="${_SITE}/nvidia/cu13"
if [[ -x "${_CU13}/bin/nvcc" && -f "${_CU13}/include/cuda_fp16.h" ]]; then
  export CUDA_HOME=${CUDA_HOME:-$_CU13}
  export CUDA_PATH=$CUDA_HOME
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
fi

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_LOCAL=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/${KING_REV}
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_848=/tmp/r848_merged
MERGE_849=/tmp/r849_merged

LOG=/root/logs/p3956_brave_cold_tk_r848_r849.log
TEACHER_LOG=/root/logs/vllm_teacher.log
KING_LOG=/root/logs/vllm_king.log
TEACHER_PIDF=/root/logs/vllm_teacher.pid
KING_PIDF=/root/logs/vllm_king.pid
mkdir -p /root/logs /root/affine_data /root/.triton/cache/teacher /root/.triton/cache/king
: >"$LOG"
log() { echo "[p3956-brave] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_pidfile() {
  local pidf=$1 why=${2:-}
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)" "$why pidf=$pidf"
  rm -f "$pidf"
}

# min shards: merges=16; vera king is 2-shard (hub_ok≥16 false-fails → HF re-prefetch stall).
hub_ok() {
  local path=$1 min=${2:-16} n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge "$min" ]]
}

wait_ready() {
  local port=$1 name=$2 pidf=$3 logf=$4 ready=0
  for i in $(seq 1 480); do
    if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1; log "${name}_READY poll=$i"; break
    fi
    if [[ -f "$pidf" ]]; then
      local pid; pid=$(cat "$pidf")
      if ! kill -0 "$pid" 2>/dev/null; then
        log "ERROR $name died"; tail -80 "$logf" | tee -a "$LOG"; exit 1
      fi
    fi
    (( i % 12 == 0 )) && log "wait $name :$port iter=$i last=$(tail -1 "$logf" 2>/dev/null | cut -c1-100)"
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || { log "ERROR $name not ready"; tail -80 "$logf" | tee -a "$LOG"; exit 1; }
}

# p3963: brave TP=2 king also NCCL-spins (same as teacher p3960) — king uses TP=1 on GPU1
# Challengers still TP=2 on 4,5 / 6,7 (COMMON unused for TK after p3963).
COMMON=(--tensor-parallel-size 2 --max-model-len 65536 --max-num-batched-tokens 8192
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton
  --additional-config '{"gdn_prefill_backend": "triton"}'
  --enforce-eager)
# p3960/p3963: brave TP≥2 NCCL-spins post-init; teacher+king use TP=1
TEACHER_COMMON=(--tensor-parallel-size 1 --max-model-len 65536 --max-num-batched-tokens 8192
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton
  --additional-config '{"gdn_prefill_backend": "triton"}'
  --enforce-eager)
KING_COMMON=("${TEACHER_COMMON[@]}")

log "START cold TK + R848/R849 n80 vs reign36 vera wvk7 (p3960/p3963 OFFLINE+eager+TP1 teacher+king)"
hub_ok "$MERGE_848" || { log "FATAL missing $MERGE_848"; exit 1; }
hub_ok "$MERGE_849" || { log "FATAL missing $MERGE_849"; exit 1; }

python3 - <<PY
from pathlib import Path
p = Path("/root/mine.env")
text = p.read_text() if p.exists() else ""
lines = text.splitlines()
want = {
    "KING_REPO": "$KING_REPO",
    "KING_REV": "$KING_REV",
    "KING_LOCAL": "$KING_LOCAL",
    "KING_SERVED_NAME": "$KING_REPO",
    "RESTART_KING": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SKIP_LOCAL_TKC": "0",
}
seen = set()
out = []
for line in lines:
    if line.startswith("export ") and "=" in line:
        key = line.split("=", 1)[0].removeprefix("export ").strip()
        if key in want:
            out.append(f"export {key}={want[key]}")
            seen.add(key)
            continue
    out.append(line)
for key, val in want.items():
    if key not in seen:
        out.append(f"export {key}={val}")
p.write_text("\n".join(out) + "\n")
print("[p3956] mine.env KING→vera reign36")
PY

# Prefer local lunar→brave SIZE_OK cache (p3958). Do NOT HF-prefetch if 2 shards present —
# marsplan/vera HF is gated/slow and idle-stalls GPUs (p3956/p3958).
if ! hub_ok "$KING_LOCAL" 2; then
  log "FATAL vera missing/incomplete at $KING_LOCAL (need ≥2 named shards; wait SIZE_OK)"
  exit 2
fi
log "vera shards=$(ls "$KING_LOCAL"/model-*-of-*.safetensors | wc -l) path=$KING_LOCAL"

if [[ ! -f "$KING_LOCAL/preprocessor_config.json" && -f "$KING_LOCAL/processor_config.json" ]]; then
  python3 - <<PY
import json
from pathlib import Path
p = Path("$KING_LOCAL")
proc = json.loads((p / "processor_config.json").read_text())
out = {
    "processor_class": proc.get("processor_class", "Qwen2VLProcessor"),
    "image_processor_type": "Qwen2VLImageProcessor",
    "auto_map": proc.get("auto_map", {}),
}
(p / "preprocessor_config.json").write_text(json.dumps(out, indent=2) + "\n")
print("[p3956] wrote preprocessor_config.json")
PY
fi

stop_pidfile "$TEACHER_PIDF" "stale teacher"
stop_pidfile "$KING_PIDF" "stale king"
for port in 8000 8001 8002 8003; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "listener :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
done
sleep 2

if [[ -z "$(ls -A /root/.triton/cache/teacher 2>/dev/null || true)" ]]; then
  for cand in /root/.triton/cache/chall /root/.triton/cache/king /root/.triton/cache/teacher_r477; do
    if [[ -d "$cand" && -n "$(ls -A "$cand" 2>/dev/null || true)" ]]; then
      cp -a "$cand"/. /root/.triton/cache/teacher/ || true
      log "seeded teacher Triton from $cand"
      break
    fi
  done
fi
if [[ -z "$(ls -A /root/.triton/cache/king 2>/dev/null || true)" ]]; then
  if [[ -n "$(ls -A /root/.triton/cache/teacher 2>/dev/null || true)" ]]; then
    cp -a /root/.triton/cache/teacher/. /root/.triton/cache/king/ || true
    log "seeded king Triton from teacher"
  fi
fi

log "launch teacher :8000 GPU 0 TP=1 (p3960 NCCL workaround)"
: >"$TEACHER_LOG"
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/root/.triton/cache/teacher \
  nohup /root/venv/bin/vllm serve "$TEACHER_REPO" \
  --port 8000 --gpu-memory-utilization 0.92 \
  "${TEACHER_COMMON[@]}" >>"$TEACHER_LOG" 2>&1 &
echo $! >"$TEACHER_PIDF"
log "teacher pid=$(cat "$TEACHER_PIDF")"
wait_ready 8000 teacher "$TEACHER_PIDF" "$TEACHER_LOG"

log "launch king :8001 GPU 1 TP=1 $KING_REPO@$KING_REV (p3963 NCCL workaround)"
: >"$KING_LOG"
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/root/.triton/cache/king \
  nohup /root/venv/bin/vllm serve "$KING_LOCAL" \
  --port 8001 --gpu-memory-utilization 0.80 \
  "${KING_COMMON[@]}" --served-model-name "$KING_REPO" >>"$KING_LOG" 2>&1 &
echo $! >"$KING_PIDF"
log "king pid=$(cat "$KING_PIDF")"
wait_ready 8001 king "$KING_PIDF" "$KING_LOG"

kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { log "ERROR king not vera"; exit 5; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/brave_tk_ready_p3956.done
log "TK READY — launch R848+R849 lean challs"

for s in /root/mining_src/fleet-rent/lean_chall_n80_brave_r848_gpus45_p3956.sh \
         /root/mining_src/fleet-rent/lean_chall_n80_brave_r849_gpus67_p3956.sh; do
  [[ -x "$s" ]] || chmod +x "$s"
  tag=$(basename "$s" .sh)
  nohup bash "$s" >/root/logs/${tag}.outer.log 2>&1 &
  echo $! >/root/logs/${tag}.outer.pid
  log "armed $s pid=$(cat /root/logs/${tag}.outer.pid)"
done

log "DONE arm — watch /root/affine_data/r848_sim_* /root/affine_data/r849_sim_*"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p3956_brave_cold_tk_armed.done
