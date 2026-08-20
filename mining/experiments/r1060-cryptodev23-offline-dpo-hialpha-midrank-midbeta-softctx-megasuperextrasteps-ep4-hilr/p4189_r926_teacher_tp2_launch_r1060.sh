#!/usr/bin/env bash
# p4189: mine-r926 — shrink teacher TP4→TP2 (GPUs0,1), keep king GPU2 + R1051 GPUs5,6,
# free GPUs3,4 for R1060 SoftCtx MidRank Midβ Mega HiLR. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4189_r926_teacher_tp2_launch_r1060.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4189] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4189] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}

TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
TEACHER_PIDF=/root/logs/vllm_teacher.pid
TEACHER_LOG=/root/logs/vllm_teacher.log

curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king :8001 down; exit 1; }
kill -0 "$(cat /root/logs/r1051_train.pid 2>/dev/null)" 2>/dev/null && echo "[p4189] R1051 train alive" || echo "[p4189] WARN R1051 train not alive"

# Exact-PID stop teacher on GPUs 0,1,3,4 only — protect king(2) and R1051(5,6)
python3 - <<'PY'
import os, signal, subprocess, time
want={0,1,3,4}
uu={}
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True).splitlines():
  idx,u=line.split(","); uu[u.strip()]=int(idx.strip())
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader"], text=True)
kill=set(); keep=set()
for line in apps.splitlines():
  parts=[p.strip() for p in line.split(",")]
  if len(parts)<2: continue
  pid=int(parts[0]); gi=uu.get(parts[1])
  if gi in (2,5,6,7):
    keep.add(pid)
  elif gi in want:
    kill.add(pid)
print(f"[p4189] reap gpus={sorted(want)} kill={sorted(kill-keep)} keep={sorted(keep)}", flush=True)
for pid in sorted(kill-keep):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(4)
for pid in sorted(kill-keep):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
PY

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8000"
done < <(ss -lptn 'sport = :8000' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
rm -f "$TEACHER_PIDF"

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,3,4 | awk '{s+=$1} END{print s+0}')
  echo "[p4189] wait free 0,1,3,4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL teacher GPUs still busy; nvidia-smi; exit 1; }

curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king died during teacher reap; exit 1; }
kill -0 "$(cat /root/logs/r1051_train.pid 2>/dev/null)" 2>/dev/null && echo "[p4189] R1051 still alive after reap" || echo "[p4189] WARN R1051 gone"

VLLM_COMMON=(--max-model-len 65536 --max-num-batched-tokens 8192
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton
  --additional-config '{"gdn_prefill_backend": "triton"}'
  --enforce-eager)

echo "[p4189] launch teacher :8000 GPUs0,1 TP=2"
: >"$TEACHER_LOG"
CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/root/.triton/cache/teacher \
  nohup /root/venv/bin/vllm serve "$TEACHER_REPO" \
  --port 8000 --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
  "${VLLM_COMMON[@]}" --served-model-name "$TEACHER_REPO" >>"$TEACHER_LOG" 2>&1 &
echo $! >"$TEACHER_PIDF"
echo "[p4189] teacher pid=$(cat "$TEACHER_PIDF")"

ready=0
for i in $(seq 1 180); do
  if curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "[p4189] teacher ready iter=$i"; ready=1; break
  fi
  if ! kill -0 "$(cat "$TEACHER_PIDF")" 2>/dev/null; then
    echo FATAL teacher died; tail -80 "$TEACHER_LOG"; exit 1
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL teacher timeout; tail -80 "$TEACHER_LOG"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4189] TK warm (TP2 teacher + king)"

EXP=r1060-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-hilr
test -f /root/mining_src/$EXP/lean_train_h100_gpus34_p4189.sh
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_h100_gpus34_p4189.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4189_r1060_lean_outer.pid
echo "[p4189] R1060 lean launched outer_pid=$(cat /root/logs/p4189_r1060_lean_outer.pid)"
sleep 30
tail -80 /root/logs/r1060_lean_warm.log || true
ps -p "$(cat /root/logs/r1060_train.pid 2>/dev/null)" -o pid,etime,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4189_r926_teacher_tp2_r1060_armed.done
echo "[p4189] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
