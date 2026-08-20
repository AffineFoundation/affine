#!/usr/bin/env bash
# p4066: R930 chall died mid-probe on Triton missing __triton_launcher*.so
# (king-seeded cache had a stale hash dir). Teacher+king stay up.
# FORCE wipe+reseed chall_r930 from king, relaunch :8002 + v4 n80.
# Never pkill -f. Exact-PID only.
set -euo pipefail
LOG=/root/logs/p4066_r924_repair_r930.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4066-r930] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

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
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE=/tmp/r930_merged
PORT=8002
GPUS=4,5
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
CHALL_LOG=/root/logs/vllm_chall_r930_p4066.log
PIDF=/root/logs/vllm_chall_r930.pid
TCACHE=/root/.triton/cache/chall_r930
KING_CACHE=/root/.triton/cache/king

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4066-r930] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
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

# Confirm TK still healthy
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { echo FATAL teacher down; exit 2; }
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king down; exit 2; }
kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4066-r930] TK ok king_id=$kid"

# GPUs 4,5 must be free (R930 slot)
for idx in 4 5; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | awk '{print $1+0}')
  echo "[p4066-r930] gpu$idx used_mib=$used"
  if [[ "$used" -ge 8192 ]]; then
    echo "[p4066-r930] FATAL GPU $idx busy used_mib=$used"; exit 3
  fi
done

n_merge=$(ls "$MERGE"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f "$MERGE/config.json" && "${n_merge:-0}" -ge 16 ]] || { echo FATAL merge incomplete n=$n_merge; exit 4; }

# Exact-PID reap stale r930 chall/sim listeners on :8002
stop_pidfile "$PIDF" "stale r930 chall"
stop_pidfile /root/logs/r930_sim_wvk7.pid "stale r930 sim"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
sleep 2

# FORCE wipe + seed from king (lesson p4052 / p4046)
if [[ ! -d "$KING_CACHE" ]]; then
  echo FATAL missing king triton cache; exit 5
fi
n_so_k=$(find "$KING_CACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
n_cubin_k=$(find "$KING_CACHE" -name '*.cubin' 2>/dev/null | wc -l || true)
echo "[p4066-r930] king cache n_so=$n_so_k n_cubin=$n_cubin_k"
[[ "${n_so_k:-0}" -ge 1 ]] || { echo FATAL king cache has no launcher .so; exit 5; }

rm -rf "$TCACHE"
cp -a "$KING_CACHE" "$TCACHE"
n_so=$(find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
n_cubin=$(find "$TCACHE" -name '*.cubin' 2>/dev/null | wc -l || true)
echo "[p4066-r930] FORCE seed chall_r930 from king n_so=$n_so n_cubin=$n_cubin"
[[ "$n_so" -eq "$n_so_k" ]] || { echo FATAL seed so mismatch; exit 5; }

: >"$CHALL_LOG"
CUDA_VISIBLE_DEVICES=$GPUS TRITON_CACHE_DIR=$TCACHE \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
    --port "$PORT" \
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
    >"$CHALL_LOG" 2>&1 &
echo $! >"$PIDF"
chall_pid=$(cat "$PIDF")
echo "[p4066-r930] chall pid=$chall_pid"

ready=0
for i in $(seq 1 240); do
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1; echo "[p4066-r930] CHALL_READY poll=$i"; break
  fi
  if ! kill -0 "$chall_pid" 2>/dev/null; then
    echo FATAL chall died; tail -100 "$CHALL_LOG"; exit 6
  fi
  if grep -q 'ImportError:.*__triton_launcher' "$CHALL_LOG" 2>/dev/null; then
    echo FATAL Triton ImportError during load; tail -80 "$CHALL_LOG"; exit 6
  fi
  (( i % 12 == 0 )) && echo "[p4066-r930] wait chall :$PORT iter=$i last=$(tail -1 "$CHALL_LOG" 2>/dev/null | cut -c1-120)"
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL chall not ready; tail -80 "$CHALL_LOG"; exit 6; }

# Probe one completion to flush Triton compile before n80 (fail-closed)
python3 - <<'PY'
import json, urllib.request
req=urllib.request.Request(
  "http://127.0.0.1:8002/v1/completions",
  data=json.dumps({
    "model":"/tmp/r930_merged",
    "prompt":"ping",
    "max_tokens":4,
    "temperature":0.0,
  }).encode(),
  headers={"Content-Type":"application/json"},
)
try:
  with urllib.request.urlopen(req, timeout=180) as r:
    body=r.read()[:200]
  print("[p4066-r930] probe_ok", body[:80])
except Exception as e:
  print("[p4066-r930] FATAL probe failed", e)
  raise SystemExit(7)
PY

bh=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r930-reign36-wvk7-p4066-{time.time()}".encode()).hexdigest())
PY
)
sim_out=/root/affine_data/r930_sim_result_reign36_wvk7.json
prog=/root/affine_data/r930_sim_progress_reign36_wvk7.json
sim_dec=/root/affine_data/r930_decision_reign36_wvk7.json
n80_log=/root/logs/p4066_r930_chall_n80_wvk7.log
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
  --chall-port "$PORT" \
  --n-turns 80 \
  --hotkey "local-r930-reign36-wvk7-p4066" \
  --block-hash "$bh" \
  --out "$sim_out" \
  --progress-out "$prog" \
  --save-artifact \
  >>"$n80_log" 2>&1 &
echo $! >/root/logs/r930_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r930_n80_launched.p4066
echo "[p4066-r930] n80 pid=$(cat /root/logs/r930_sim_wvk7.pid) bh=${bh:0:16}…"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4066_r930_repair_armed.done
echo "[p4066-r930] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
