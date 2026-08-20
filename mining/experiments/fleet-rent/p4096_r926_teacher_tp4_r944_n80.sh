#!/usr/bin/env bash
# p4096: R944 n80 DEAD again — teacher EngineDead mid-duel (CUDA OOM @ TP2/0.88
# under concurrent k=3 load; K:8001 + chall:8002@0.65 still up; GPUs 0,1,5,6,7 free).
# Relaunch teacher TP=4 on GPUs 0,1,5,6 @ gpu_mem=0.85 (spread weights; leave
# activation headroom), FORCE seed Triton, probe, re-arm v4 n80. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4096_r926_teacher_tp4_r944_n80.log
mkdir -p /root/logs /root/affine_data /root/r944
exec > >(tee -a "$LOG") 2>&1
echo "[p4096-r944] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

source /root/venv/bin/activate
[[ -f /root/mine.env ]] && set -a && source /root/mine.env && set +a
export HF_HOME=${HF_HOME:-/root/hf}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}

KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
TEACHER_PIDF=/root/logs/vllm_teacher.pid
TEACHER_LOG=/root/logs/vllm_teacher.log
TCACHE=/root/.triton/cache/teacher
MERGE=/tmp/r944_merged
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
PORT_CHALL=8002
GPU_MEM=0.85
TEACHER_GPUS=0,1,5,6
TEACHER_TP=4

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4096-r944] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
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

curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king down; exit 2; }
curl -sf -m 5 http://127.0.0.1:${PORT_CHALL}/v1/models >/dev/null || { echo FATAL chall down; exit 2; }
kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4096-r944] king+chall ok king_id=$kid"

n_merge=$(ls "$MERGE"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f "$MERGE/config.json" && "${n_merge:-0}" -ge 16 ]] || { echo FATAL merge incomplete n=$n_merge; exit 4; }

stop_pidfile "$TEACHER_PIDF" "dead teacher"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8000"
done < <(ss -lptn "sport = :8000" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
# Reap orphans only on teacher GPU set (never touch king GPU2 / chall 3,4).
for pid in $(pgrep -f 'VLLM::Worker|VLLM::EngineCore' || true); do
  envf="/proc/$pid/environ"
  [[ -r "$envf" ]] || continue
  if tr '\0' '\n' <"$envf" | grep -qE '^CUDA_VISIBLE_DEVICES=0,1$|^CUDA_VISIBLE_DEVICES=0,1,5,6$'; then
    stop_pid "$pid" "orphan teacher Worker/Engine"
  fi
done
sleep 2
IFS=',' read -r -a TGPU_ARR <<<"$TEACHER_GPUS"
for idx in "${TGPU_ARR[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | awk '{print $1+0}')
  echo "[p4096-r944] gpu$idx used_mib=$used"
  if [[ "$used" -ge 8192 ]]; then
    echo FATAL GPU $idx still busy; exit 3
  fi
done

SEED_SRC=""
if [[ -d /tmp/r926_triton_seed/teacher ]] && find /tmp/r926_triton_seed/teacher -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
  SEED_SRC=/tmp/r926_triton_seed/teacher
elif [[ -f /tmp/teacher_triton_crown_p4081.tgz ]]; then
  rm -rf /tmp/r926_triton_seed/teacher
  mkdir -p /tmp/r926_triton_seed
  tar xzf /tmp/teacher_triton_crown_p4081.tgz -C /tmp/r926_triton_seed
  SEED_SRC=/tmp/r926_triton_seed/teacher
elif [[ -d "$TCACHE" ]] && find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
  SEED_SRC=""
  echo "[p4096-r944] REUSE existing teacher cache"
fi
if [[ -n "$SEED_SRC" ]]; then
  rm -rf "$TCACHE"
  mkdir -p "$(dirname "$TCACHE")"
  cp -a "$SEED_SRC" "$TCACHE"
  echo "[p4096-r944] FORCE seed teacher from $SEED_SRC n_so=$(find "$TCACHE" -name '__triton_launcher*.so' | wc -l)"
else
  mkdir -p "$TCACHE"
  echo "[p4096-r944] REUSE teacher cache n_so=$(find "$TCACHE" -name '__triton_launcher*.so' | wc -l)"
fi

VLLM_COMMON=(--max-model-len 65536 --max-num-batched-tokens 4096
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton
  --additional-config '{"gdn_prefill_backend": "triton"}'
  --enforce-eager)

: >"$TEACHER_LOG"
CUDA_VISIBLE_DEVICES=$TEACHER_GPUS TRITON_CACHE_DIR=$TCACHE \
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$TEACHER_REPO" \
  --port 8000 --tensor-parallel-size "$TEACHER_TP" --gpu-memory-utilization "$GPU_MEM" \
  "${VLLM_COMMON[@]}" --served-model-name "$TEACHER_REPO" >>"$TEACHER_LOG" 2>&1 &
echo $! >"$TEACHER_PIDF"
tp=$(cat "$TEACHER_PIDF")
echo "[p4096-r944] teacher pid=$tp gpu_mem=$GPU_MEM TP=$TEACHER_TP GPUs=$TEACHER_GPUS"

ready=0
for i in $(seq 1 360); do
  if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    ready=1; echo "[p4096-r944] TEACHER_READY poll=$i"; break
  fi
  if ! kill -0 "$tp" 2>/dev/null; then
    echo FATAL teacher died; tail -80 "$TEACHER_LOG"; exit 6
  fi
  if grep -qE 'OutOfMemoryError|CUDA out of memory|KV cache is needed' "$TEACHER_LOG" 2>/dev/null; then
    echo FATAL teacher OOM/KV; tail -40 "$TEACHER_LOG"; exit 6
  fi
  (( i % 12 == 0 )) && echo "[p4096-r944] wait teacher iter=$i last=$(tail -1 "$TEACHER_LOG" 2>/dev/null | cut -c1-120)"
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL teacher not ready; tail -80 "$TEACHER_LOG"; exit 6; }

python3 - <<'PY'
import json, urllib.request
req=urllib.request.Request(
  "http://127.0.0.1:8000/v1/completions",
  data=json.dumps({
    "model":"zai-org/GLM-4.5-Air-FP8",
    "prompt":"ping",
    "max_tokens":4,
    "temperature":0.0,
  }).encode(),
  headers={"Content-Type":"application/json"},
)
try:
  with urllib.request.urlopen(req, timeout=300) as r:
    body=r.read()[:120]
  print("[p4096-r944] teacher_probe_ok", body[:80])
except Exception as e:
  print("[p4096-r944] FATAL teacher probe failed", e)
  raise SystemExit(7)
PY

curl -sf -m 5 http://127.0.0.1:${PORT_CHALL}/v1/models >/dev/null || { echo FATAL chall died during teacher relaunch; exit 8; }
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king died during teacher relaunch; exit 8; }

stop_pidfile /root/logs/r944_sim_wvk7.pid "stale sim"
stop_pidfile /root/r944/r944_sim_wvk7.pid "stale sim r944/"
# exact-PID: any leftover run_sim for r944 (never pkill -f)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  case "$cmd" in
    *run_sim_duel.py*r944*|*run_sim_duel.py*/tmp/r944_merged*) stop_pid "$pid" "stale r944 sim cmdline" ;;
  esac
done < <(pgrep -f 'run_sim_duel.py' || true)

bh=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r944-reign36-wvk7-p4096-{time.time()}".encode()).hexdigest())
PY
)
sim_out=/root/affine_data/r944_sim_reign36_wvk7.json
prog=/root/affine_data/r944_sim_progress_reign36_wvk7.json
n80_log=/root/logs/p4096_r944_chall_n80_wvk7.log
rm -f "$sim_out" "$prog"
: >"$n80_log"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 "$SIM" \
  --toml /root/mining_src/affine_pkg/affine.toml \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$kid" \
  --king-rev "$KING_REV" \
  --chall-repo "$MERGE" \
  --chall-rev local \
  --chall-port "$PORT_CHALL" \
  --n-turns 80 \
  --hotkey "local-r944-reign36-wvk7-p4096" \
  --block-hash "$bh" \
  --out "$sim_out" \
  --progress-out "$prog" \
  --save-artifact \
  >>"$n80_log" 2>&1 &
echo $! >/root/logs/r944_sim_wvk7.pid
echo $! >/root/r944/r944_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r944_n80_launched.p4096
echo "[p4096-r944] n80 pid=$(cat /root/logs/r944_sim_wvk7.pid) bh=${bh:0:16}…"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4096_r944_teacher_n80_armed.done
echo "[p4096-r944] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
