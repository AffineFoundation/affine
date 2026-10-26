#!/usr/bin/env bash
# Track T bring-up (idempotent). 8xB300 Blackwell -> Docker vLLM :latest
# (pip vLLM cannot JIT for sm_103a).
#   GPU 0,1  27B TP2 :8000   teacher (base) + judge (d_v* runtime LoRA)
#   GPU 2    student :8001   raw Albedo-genesis 35B-A3B (+ m_r* LoRA)
#   GPU 3,4  g_sft per round (judge pretrain borrows them once, DDPx3)
#   GPU 5    trackT_d_train online updates
#   GPU 6,7  student TP2 :8006  bench server (panels; loop unpaused)
set -uo pipefail
cd /root/work
W=/dshare/trackT
mkdir -p "$W"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
STUDENT="dendriteholdings/albedo-qwen3.6-35b-king-genesis"

log() { echo "[phaseT $(date -u +%T)] $*"; }
note() { echo "$(date -u +%FT%TZ) [infra] $*" >> "$W/status.log"; }
healthy() { curl -sf -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
wait_health() {
  local p="$1" t="${2:-1800}" i=0
  while ! healthy "$p"; do
    sleep 15; i=$((i+15)); [ "$i" -ge "$t" ] && return 1
  done
  log "port $p healthy after ${i}s"
}

# wait for model downloads + data copy
i=0
while ! grep -q SETUP_DONE /root/work/setup.log 2>/dev/null; do
  sleep 20; i=$((i+20))
  [ "$i" -ge 5400 ] && { log "setup not done after 90min; proceeding"; break; }
done
i=0
while [ ! -s /root/work/turns.jsonl.gz ] || [ ! -s /root/work/teacher_train.jsonl ]; do
  sleep 20; i=$((i+20))
  [ "$i" -ge 3600 ] && { note "FATAL data files missing"; exit 1; }
done

note "phaseT: bringing up servers (vllm :latest docker)"
if ! healthy 8000; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/trackT bash /root/work/pod_serve_docker.sh \
    Qwen/Qwen3.8-27B 8000 0,1 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4 --max-logprobs 25
fi
if ! healthy 8001; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/trackT bash /root/work/pod_serve_docker.sh \
    "$STUDENT" 8001 2 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4
fi
if ! healthy 8006; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/trackT bash /root/work/pod_serve_docker.sh \
    "$STUDENT" 8006 6,7 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4
fi
wait_health 8000 1800 || { note "FATAL teacher/judge :8000 failed"; exit 1; }
wait_health 8001 1800 || { note "FATAL student :8001 failed"; exit 1; }
wait_health 8006 1800 || { note "FATAL bench server :8006 failed"; exit 1; }
note "servers up: 0,1=teacher+judge :8000 | 2=student :8001 | 3,4=g_sft | 5=d_train | 6,7=bench :8006"

# RAW BASELINE (the control for the whole experiment): full-150 panel on the
# untouched student, on the bench server, before/alongside judge pretrain.
if [ ! -s "$W/bench_full_baseline.json" ] && [ ! -f "$W/bench.lock" ]; then
  echo full_baseline > "$W/bench.lock"
  nohup /root/benchenv/bin/python /root/work/trackT_bench.py \
    --tag full_baseline --model "$STUDENT" --adapter "" --work "$W" \
    > "$W/bench_full_baseline.driver.log" 2>&1 < /dev/null &
  disown
  note "RAW BASELINE full-150 launched on :8006 (background)"
fi

if ! pgrep -f 'trackT_driver[.]py' >/dev/null; then
  nohup $PY /root/work/trackT_driver.py > "$W/driver.log" 2>&1 < /dev/null &
  disown
  note "trackT driver launched (pid $!)"
fi
if ! pgrep -f 'work/watchdog[.]sh' >/dev/null; then
  nohup bash /root/work/watchdog.sh > /root/work/watchdog_nohup.log 2>&1 < /dev/null &
  disown
  note "watchdog launched"
fi
log "PHASET_COMPLETE"
