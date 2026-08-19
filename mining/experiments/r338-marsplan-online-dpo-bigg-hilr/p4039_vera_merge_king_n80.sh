#!/usr/bin/env bash
# p4039: R338 TRAIN_DONE but post_train merge aborted — BASE still pointed at
# missing marsplan queen path. Re-merge LoRA onto the vera base that trained,
# serve reign36 king :8001, chall :8002, then v4 n80 (k=3 τ=0.03).
# Never pkill -f. Do not touch teacher GPUs 0,1.
set -euo pipefail
exec >>/root/logs/p4039_r338_vera_merge_n80.nohup 2>&1

source /root/venv/bin/activate
set -a; source /root/mine.env; set +a

export HF_HOME=${HF_HOME:-/root/hf}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

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
fi

BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_LOCAL=$BASE
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
ADAPTER=/root/r338/train/adapter
MERGED=/tmp/r338_merged
UTIL=${UTIL:-0.72}
LOG=/root/logs/p4039_r338_vera_merge_n80.nohup
SIM_N80=/root/affine_data/r338_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r338_sim_progress_reign36_wvk7.json
SIM_DEC=/root/affine_data/r338_decision_reign36_wvk7.json
mkdir -p /root/logs /root/affine_data /tmp

log() { echo "[p4039-r338] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
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

log "START R338 vera-base merge→king→chall→n80"
test -f "$ADAPTER/adapter_model.safetensors" || { log "FATAL missing adapter"; exit 1; }
test -f "$BASE/config.json" || { log "FATAL missing vera base"; exit 1; }
nbase=$(ls "$BASE"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${nbase:-0}" -ge 2 ]] || { log "FATAL vera shards=$nbase"; exit 1; }

# Patch mine.env so later pipes do not re-hit marsplan path
python3 - <<'PY'
from pathlib import Path
p = Path("/root/mine.env")
text = p.read_text() if p.exists() else ""
lines = []
for line in text.splitlines():
    if line.startswith("export HF_BASE_HUB=") or line.startswith("HF_BASE_HUB="):
        continue
    if line.startswith("export KING_REPO=") or line.startswith("KING_REPO="):
        continue
    if line.startswith("export KING_REV=") or line.startswith("KING_REV="):
        continue
    if line.startswith("export KING_LOCAL=") or line.startswith("KING_LOCAL="):
        continue
    if line.startswith("export BASE=") or (line.startswith("BASE=") and "export" not in line):
        continue
    lines.append(line)
block = """
# p4039: pin to vera reign36 (train base); marsplan queen gated
export HF_BASE_HUB=vera6/affine-5g4yy75zuz-t6
export KING_REPO=vera6/affine-5g4yy75zuz-t6
export KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
export KING_LOCAL=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
export BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
"""
p.write_text("\n".join(lines).rstrip() + "\n" + block)
print("mine.env pinned to vera")
PY

# --- MERGE (GPUs 4,5) ---
if [[ -f /root/logs/r338_merge.done ]] && [[ -f "$MERGED/config.json" ]]; then
  nm=$(ls "$MERGED"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ "${nm:-0}" -ge 2 ]]; then
    log "REUSE merge shards=$nm"
  else
    rm -f /root/logs/r338_merge.done
  fi
fi

if [[ ! -f /root/logs/r338_merge.done ]]; then
  log "MERGE start base=$BASE adapter=$ADAPTER → $MERGED"
  rm -rf "$MERGED"
  mkdir -p "$MERGED"
  export CUDA_VISIBLE_DEVICES=4,5
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  unset HF_TOKEN || true
  /root/venv/bin/python3 /root/mining_src/s4-h1-sft/merge_lora.py \
    --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" \
    >/root/logs/r338_merge.nohup 2>&1
  nm=$(ls "$MERGED"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$MERGED/config.json" && "${nm:-0}" -ge 2 ]] || {
    log "FATAL merge incomplete shards=$nm"; tail -n 80 /root/logs/r338_merge.nohup; exit 1
  }
  date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r338_merge.done
  rm -f /root/logs/r338_pipeline.aborted
  log "MERGE done shards=$nm"
fi

cat >"$MERGED/README.md" <<'CARD'
# R338 merged challenger (local)

- **Base:** `vera6/affine-5g4yy75zuz-t6` @ `8e3f1695…` (train base; p4027 pivot off gated marsplan queen)
- **Method:** Online DPO on teacher Reason rewards (group samples → preferred/rejected)
- **Axis:** HiLR BigG online-DPO (lr=2e-5, LoRA r=16/α32, β=0.1, G=8, temp=1.2, max_steps=300, max_len=6144)
- **Hardware:** `mine-r338` 8×B200; train GPUs 6,7; merge/chall 4,5; king 2,3
- **Train evidence:** 189 steps DONE @2026-08-19T20:29:21Z; adapter `/root/r338/train/adapter`
- **Experiment:** `mining/experiments/r338-marsplan-online-dpo-bigg-hilr/`
- **Note:** p4039 re-merged with vera BASE after marsplan path abort
CARD

# --- KING :8001 GPUs 2,3 ---
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { log "FATAL teacher :8000 down"; exit 1; }
kid_now=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
if echo "${kid_now:-}" | grep -qiE 'vera6|5g4yy75zuz|t6'; then
  log "KING already warm id=$kid_now"
else
  stop_pidfile /root/logs/vllm_king.pid "stale king"
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "stale :8001"
  done < <(ss -lptn 'sport = :8001' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
  for i in $(seq 1 60); do
    used=$(nvidia-smi -i 2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
    [[ "${used:-999999}" -lt 2000 ]] && break
    sleep 2
  done
  export CUDA_VISIBLE_DEVICES=2,3
  export TRITON_CACHE_DIR=/root/.triton/cache/king
  mkdir -p "$TRITON_CACHE_DIR"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  unset HF_TOKEN || true
  : >/root/logs/vllm_king.log
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$KING_LOCAL" \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization "$UTIL" \
    --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    >/root/logs/vllm_king.log 2>&1 &
  echo $! >/root/logs/vllm_king.pid
  log "king pid=$(cat /root/logs/vllm_king.pid)"
  for i in $(seq 1 240); do
    if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
      log "KING_READY poll=$i"; break
    fi
    if ! kill -0 "$(cat /root/logs/vllm_king.pid)" 2>/dev/null; then
      log "FATAL king died"; tail -n 80 /root/logs/vllm_king.log; exit 1
    fi
    sleep 5
  done
fi
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { log "FATAL king not vera ($kid)"; exit 5; }

# --- CHALL :8002 GPUs 4,5 ---
stop_pidfile /root/logs/vllm_chall_r338.pid "stale chall"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
for i in $(seq 1 60); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [[ "${used:-999999}" -lt 2000 ]] && break
  sleep 2
done
export CUDA_VISIBLE_DEVICES=4,5
export TRITON_CACHE_DIR=/root/.triton/cache/chall_r338
mkdir -p "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
: >/root/logs/vllm_chall_r338.log
nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGED" \
  --port 8002 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN \
  --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' \
  --enforce-eager \
  >/root/logs/vllm_chall_r338.log 2>&1 &
echo $! >/root/logs/vllm_chall_r338.pid
log "chall pid=$(cat /root/logs/vllm_chall_r338.pid) :8002"
for i in $(seq 1 240); do
  if curl -sf -m 3 http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
    log "CHALL_READY poll=$i"; break
  fi
  if ! kill -0 "$(cat /root/logs/vllm_chall_r338.pid)" 2>/dev/null; then
    log "FATAL chall died"; tail -n 100 /root/logs/vllm_chall_r338.log; exit 1
  fi
  sleep 5
done

BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r338-reign36-wvk7-p4039-{time.time()}".encode()).hexdigest())
PY
)
log "launch n80 vs $KING_REPO block_hash=${BLOCK_HASH:0:16}…"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$KING_REPO" \
  --king-rev "$KING_REV" \
  --chall-repo "$MERGED" \
  --chall-rev local \
  --chall-port 8002 \
  --n-turns 80 \
  --hotkey local-r338-reign36-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
echo $! >/root/logs/r338_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r338_n80_launched.p4039
log "n80 pid=$(cat /root/logs/r338_sim_wvk7.pid) — waiting"
while kill -0 "$(cat /root/logs/r338_sim_wvk7.pid)" 2>/dev/null; do sleep 30; done
wait "$(cat /root/logs/r338_sim_wvk7.pid)" || true
if [[ -f "$SIM_N80" ]]; then
  log "SIM_DONE $SIM_N80"
  /root/venv/bin/python3 - <<PY
import json
from pathlib import Path
d=json.loads(Path("$SIM_N80").read_text())
v=d.get("verdict") if isinstance(d.get("verdict"), dict) else {}
dp=(v.get("duel_params") or {}) if isinstance(v, dict) else {}
margin = v.get("margin") if v else (d.get("margin") or d.get("mean_margin"))
se = v.get("se") if v else d.get("se")
print(json.dumps({"margin":margin,"se":se,"duel_params":dp,"keys":list(d)[:20]}, indent=2))
PY
else
  log "SIM missing — check log"
fi
log "DONE"
