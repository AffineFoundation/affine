#!/usr/bin/env bash
# p4347: revive r924 teacher :8000 GPU0 at util0.85 (prior OOM@0.90), then arm R1222/23/24 chall+v4 n80.
# Leave king:8001 GPU2. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4347_r924_revive_arm.log
mkdir -p /root/logs /root/affine_data
: >"$LOG"
log(){ echo "[p4347-r924] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_TOKEN || true

TEACHER_SNAP=/root/hf/hub/models--zai-org--GLM-4.5-Air-FP8/snapshots/f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc
[[ -f "$TEACHER_SNAP/config.json" ]] || TEACHER_SNAP=zai-org/GLM-4.5-Air-FP8

stop_pid() {
  local pid=$1; [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill teacher-ish pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# Reap only stale :8000 listeners (exact PID) — never pkill -f
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -qE 'vllm serve.*(GLM-4.5-Air|/GLM-4\.5-Air-FP8)'; then
    stop_pid "$pid"
  fi
done < <(ss -lptn 'sport = :8000' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
# also clear stale pidfile if process gone
if [[ -f /root/logs/vllm_teacher.pid ]]; then
  opid=$(cat /root/logs/vllm_teacher.pid 2>/dev/null || true)
  if [[ -n "${opid:-}" && "$opid" =~ ^[0-9]+$ ]]; then
    if ! kill -0 "$opid" 2>/dev/null; then rm -f /root/logs/vllm_teacher.pid
    else
      # if still alive but not answering, kill exact
      if ! curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then stop_pid "$opid"; rm -f /root/logs/vllm_teacher.pid; fi
    fi
  fi
fi

# Wait GPU0 free
for i in $(seq 1 60); do
  used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1+0}')
  log "gpu0 used_mib=$used poll=$i"
  [[ "$used" -lt 2000 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1+0}')
[[ "$used" -lt 2000 ]] || { log "FATAL gpu0 busy used=$used"; exit 1; }

# King must stay up
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { log "FATAL king :8001 down"; exit 1; }
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz' || { log "FATAL wrong king"; exit 5; }

# Merge hubs present
for m in /tmp/r1222_merged /tmp/r1223_merged /tmp/r1224_merged; do
  n=$(ls "$m"/model-*-of-*.safetensors 2>/dev/null | wc -l)
  [[ -f "$m/config.json" && "$n" -ge 16 ]] || { log "FATAL $m incomplete n=$n"; exit 1; }
  log "merge ok $m shards=$n"
done

# Launch teacher TP1 util0.85 enforce-eager (lesson: 0.90 OOM)
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: >/root/logs/vllm_teacher_p4347.log
nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$TEACHER_SNAP" \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN \
  --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' \
  --enforce-eager \
  --served-model-name zai-org/GLM-4.5-Air-FP8 \
  >/root/logs/vllm_teacher_p4347.log 2>&1 &
echo $! >/root/logs/vllm_teacher.pid
TPID=$(cat /root/logs/vllm_teacher.pid)
log "teacher pid=$TPID util=0.85 enforce-eager"

for i in $(seq 1 240); do
  if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    log "TEACHER_READY poll=$i"
    break
  fi
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "FATAL teacher died"; tail -n 80 /root/logs/vllm_teacher_p4347.log | tee -a "$LOG"; exit 1
  fi
  sleep 5
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
log "TK warm"

# Sync scripts into mining_src and launch three chall+n80 (background, non-blocking outer)
for rid in 1222 1223 1224; do
  src=$(ls -d /root/mining_src/r${rid}-vera-* 2>/dev/null | head -1 || true)
  [[ -n "$src" ]] || { log "FATAL missing mining_src r$rid"; exit 1; }
  scr="$src/lean_chall_n80_r924_r${rid}_p4347.sh"
  [[ -x "$scr" ]] || chmod +x "$scr"
  nohup bash "$scr" >/root/logs/p4347_r${rid}_outer.nohup 2>&1 &
  echo $! >/root/logs/p4347_r${rid}_outer.pid
  log "armed R$rid outer pid=$(cat /root/logs/p4347_r${rid}_outer.pid) scr=$scr"
  sleep 2
done

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4347_r924_r1222_23_24_armed.done
log "ARMED teacher+R1222/23/24 chall+n80"
