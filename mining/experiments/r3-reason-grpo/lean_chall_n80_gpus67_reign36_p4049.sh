#!/usr/bin/env bash
# p4049: R252/R3 GRPO MERGE hung chall (shm) on GPUs4,5 → exact-PID reap →
# chall :8002 on free GPUs 6,7 + v4 n80 vs reign36 vera (wvk=7).
# Also swap :8001 king justice→vera if needed. Never pkill -f.
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi

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
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
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
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r3_merged
CHALL_GPUS=6,7
CHALL_PORT=8002
KING_GPUS=2,3
UTIL=${UTIL:-0.72}
LOG=/root/logs/p4049_r252_chall_n80_wvk7.log
CHALL_LOG=/root/logs/vllm_chall_r252_p4049.log
PIDF=/root/logs/vllm_chall_r252_p4049.pid
KING_PIDF=/root/logs/vllm_king.pid
TCACHE=/root/.triton/cache/chall_r252_p4049
SIM_N80=/root/affine_data/r252_r3_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r252_r3_sim_progress_reign36_wvk7.json
SIM_DEC=/root/affine_data/r252_r3_decision_reign36_wvk7.json
mkdir -p /root/logs /root/affine_data "$TCACHE"

: >"$LOG"
log(){ echo "[p4049-r252] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
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
  local path=$1 n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge 16 ]]
}

log "START R3 GRPO chall+v4-n80 GPUs=$CHALL_GPUS merge=$MERGE_DIR vs $KING_REPO@$KING_REV"
hub_ok "$MERGE_DIR" || { log "FATAL merge incomplete"; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { log "FATAL teacher down"; exit 1; }

# --- Reap hung chall on 4,5 (exact PIDs only) ---
stop_pidfile /root/logs/vllm_chall.pid "stale chall pidf"
# APIServer + EngineCore + workers by known hung tree / port / GPU apps on 4,5
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
# argv match without pkill -f: scan /proc
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || echo '')
  if echo "$cmd" | grep -qE 'vllm serve .*/tmp/r3_merged|--port 8002'; then
    stop_pid "$pid" "argv r3_merged/8002"
  fi
done < <(ls /proc | grep -E '^[0-9]+$' || true)
# GPU 4,5 compute apps
python3 - <<'PY'
import os, signal, time, subprocess
want={4,5}
out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"], text=True)
idx_uuid={}
for line in out.strip().splitlines():
  a,b=[p.strip() for p in line.split(",")]
  idx_uuid[int(a)]=b
uuids={idx_uuid[i] for i in want if i in idx_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
pids=set()
for line in apps.strip().splitlines():
  if not line.strip(): continue
  u,p=[x.strip() for x in line.split(",")]
  if u in uuids:
    try: pids.add(int(p))
    except: pass
# climb one parent
extra=set()
for pid in list(pids):
  try:
    with open(f"/proc/{pid}/stat") as f: body=f.read()
    r=body.rfind(")"); fields=body[r+2:].split(); ppid=int(fields[1])
    if ppid>1: extra.add(ppid)
  except: pass
pids |= extra
for pid in sorted(pids):
  try:
    os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(pids):
  try:
    os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("reaped_gpu45", sorted(pids))
PY
sleep 5
log "after reap gpu45: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '$1==4||$1==5||$1==6||$1==7')"

# --- Ensure king is vera on :8001 ---
kid=""
if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
fi
if ! echo "${kid:-}" | grep -qiE 'vera6|5g4yy75zuz|t6'; then
  log "king swap needed (was=${kid:-down}) → $KING_REPO@$KING_REV"
  # stop current king exact-PID
  stop_pidfile "$KING_PIDF" "old king"
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "port8001"
  done < <(ss -lptn 'sport = :8001' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || echo '')
    if echo "$cmd" | grep -qE 'vllm serve .*--port 8001'; then
      stop_pid "$pid" "argv king8001"
    fi
  done < <(ls /proc | grep -E '^[0-9]+$' || true)
  # GPU 2,3 apps
  python3 - <<'PY'
import os, signal, time, subprocess
want={2,3}
out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"], text=True)
idx_uuid={}
for line in out.strip().splitlines():
  a,b=[p.strip() for p in line.split(",")]; idx_uuid[int(a)]=b
uuids={idx_uuid[i] for i in want if i in idx_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
pids=set()
for line in apps.strip().splitlines():
  if not line.strip(): continue
  u,p=[x.strip() for x in line.split(",")]
  if u in uuids:
    try: pids.add(int(p))
    except: pass
extra=set()
for pid in list(pids):
  try:
    with open(f"/proc/{pid}/stat") as f: body=f.read()
    r=body.rfind(")"); fields=body[r+2:].split(); ppid=int(fields[1])
    if ppid>1: extra.add(ppid)
  except: pass
pids |= extra
for pid in sorted(pids):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(pids):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("reaped_gpu23", sorted(pids))
PY
  sleep 5
  # pin mine.env
  if [[ -f /root/mine.env ]]; then
    sed -i 's|^export KING_REPO=.*|export KING_REPO=vera6/affine-5g4yy75zuz-t6|' /root/mine.env || true
    sed -i 's|^export KING_REV=.*|export KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e|' /root/mine.env || true
    grep -q '^export KING_REPO=' /root/mine.env || echo 'export KING_REPO=vera6/affine-5g4yy75zuz-t6' >>/root/mine.env
    grep -q '^export KING_REV=' /root/mine.env || echo 'export KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e' >>/root/mine.env
  fi
  KT=/root/.triton/cache/king
  mkdir -p "$KT"
  # seed triton from teacher if present
  if [[ -d /root/.triton/cache/teacher ]]; then
    rsync -a /root/.triton/cache/teacher/ "$KT/" 2>/dev/null || cp -a /root/.triton/cache/teacher/. "$KT/" 2>/dev/null || true
  fi
  export CUDA_VISIBLE_DEVICES=$KING_GPUS
  export TRITON_CACHE_DIR=$KT
  log "start vera king GPUs=$KING_GPUS"
  nohup vllm serve "$KING_REPO" \
    --revision "$KING_REV" \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.80 \
    --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend":"triton"}' \
    --enforce-eager \
    --served-model-name "$KING_REPO" \
    >/root/logs/vllm_king.log 2>&1 &
  echo $! >"$KING_PIDF"
  log "king pid=$(cat $KING_PIDF)"
  for i in $(seq 1 240); do
    if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
      kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
      log "king READY id=$kid poll=$i"
      break
    fi
    sleep 5
  done
  curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { log "FATAL king never ready"; exit 1; }
  echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { log "FATAL king id=$kid"; exit 5; }
else
  log "king already vera id=$kid"
fi

# --- FORCE seed chall triton from king cache ---
rm -rf "$TCACHE"
mkdir -p "$TCACHE"
for cand in /root/.triton/cache/king /root/.triton/cache/chall /root/.triton/cache/chall_r252; do
  if [[ -d "$cand" ]]; then
    n_so=$(find "$cand" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
    if [[ "${n_so:-0}" -ge 1 ]]; then
      log "FORCE seed triton from $cand n_so=$n_so"
      rsync -a "$cand/" "$TCACHE/" 2>/dev/null || cp -a "$cand"/. "$TCACHE"/ 2>/dev/null || true
      break
    fi
  fi
done

export CUDA_VISIBLE_DEVICES=$CHALL_GPUS
export TRITON_CACHE_DIR=$TCACHE
log "start chall GPUs=$CHALL_GPUS port=$CHALL_PORT util=$UTIL"
nohup vllm serve "$MERGE_DIR" \
  --port "$CHALL_PORT" \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN \
  --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --moe-backend triton \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --enforce-eager \
  >/root/logs/vllm_chall_r252_p4049.log 2>&1 &
echo $! >"$PIDF"
log "chall pid=$(cat $PIDF)"

for i in $(seq 1 360); do
  if curl -sf -m 3 http://127.0.0.1:$CHALL_PORT/v1/models >/dev/null 2>&1; then
    log "chall READY poll=$i"
    break
  fi
  if ! kill -0 "$(cat $PIDF)" 2>/dev/null; then
    log "FATAL chall died"; tail -80 "$CHALL_LOG" | tee -a "$LOG"; exit 1
  fi
  sleep 5
done
curl -sf -m 5 http://127.0.0.1:$CHALL_PORT/v1/models >/dev/null || { log "FATAL chall not ready"; tail -80 "$CHALL_LOG" | tee -a "$LOG"; exit 1; }

# served model id for --chall-repo / king-repo from /v1/models
cid=$(curl -sf -m 3 http://127.0.0.1:$CHALL_PORT/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "ids king=$kid chall=$cid"

BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r252-r3-reign36-wvk7-p4049-{time.time()}".encode()).hexdigest())
PY
)
log "launch v4 n80 vs king_id=$kid chall_id=$cid block_hash=${BLOCK_HASH:0:16}"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$kid" \
  --king-rev "$KING_REV" \
  --chall-repo "$MERGE_DIR" \
  --chall-rev local \
  --chall-port "$CHALL_PORT" \
  --n-turns 80 \
  --hotkey local-r252-r3-reign36-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >/root/logs/r252_r3_sim_n80_wvk7.nohup 2>&1 &
echo $! >/root/logs/r252_r3_sim_n80_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r252_r3_n80_launched.p4049
log "n80 pid=$(cat /root/logs/r252_r3_sim_n80_wvk7.pid) bh=${BLOCK_HASH:0:16}"
log "DONE armed"
