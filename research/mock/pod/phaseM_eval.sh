#!/usr/bin/env bash
# Track M EVAL BOX bring-up (idempotent). H200 box -> image :latest with
# --enforce-eager --no-async-scheduling (v0.10.1.1/v0.11.0 predate the
# qwen3_5/qwen3_6 archs; :latest needs these flags to be stable on H200).
#   GPU 0,1  27B TP2 :8000 (teacher + judge, runtime LoRA)
#   GPU 2    35B king :8001    GPU 3  35B challenger :8003
#   GPU 4,5  d_train | GPU 6,7 bench
set -uo pipefail
cd /root/work
W=/dshare/koth
mkdir -p "$W"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf

log() { echo "[phaseM $(date -u +%T)] $*"; }
note() { echo "$(date -u +%FT%TZ) [eval] $*" >> "$W/status.log"; }
healthy() { curl -sf -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
wait_health() {
  local p="$1" t="${2:-1200}" i=0
  while ! healthy "$p"; do
    docker ps --format '{{.Names}}' | grep -q "^vllm_$p\$" || return 1
    sleep 15; i=$((i+15)); [ "$i" -ge "$t" ] && return 1
  done
  log "port $p healthy after ${i}s"
}
serve() { # repo port gpus extra...
  local repo="$1" port="$2" gpus="$3"; shift 3
  healthy "$port" && { log "port $port already healthy"; return 0; }
  # CUDA graphs on (audit: eager cost throughput); async scheduling still off
  # (H200+:latest mitigation); prefix caching on (same turn prefixes recur,
  # ~2-3x on judge scoring; vLLM salts cache blocks by LoRA id).
  VLLM_IMAGE=vllm/vllm-openai:latest bash /root/work/serve_trackP.sh \
    "$repo" "$port" "$gpus" --no-async-scheduling --enable-prefix-caching "$@"
  wait_health "$port" 1200
}

note "phaseM_eval: bringing up judge/king/challenger servers"
serve Qwen/Qwen3.8-27B 8000 0,1 \
  --max-model-len 32768 --enable-lora --max-lora-rank 16 --max-loras 4 \
  --max-logprobs 25 &
P1=$!
serve Qwen/Qwen3.6-35B-A3B 8001 2 \
  --max-model-len 32768 --enable-lora --max-lora-rank 16 --max-loras 4 &
P2=$!
serve Qwen/Qwen3.6-35B-A3B 8003 3 \
  --max-model-len 32768 --enable-lora --max-lora-rank 16 --max-loras 4 &
P3=$!
wait $P1 || { note "FATAL judge server :8000 failed"; exit 1; }
wait $P2 || { note "FATAL king server :8001 failed"; exit 1; }
wait $P3 || { note "FATAL challenger server :8003 failed"; exit 1; }
note "servers up: 0,1=judge/teacher :8000 | 2=king :8001 | 3=chal :8003"

# ---- seed archive + judge v0 (skipped once published) ----------------------
if [ ! -s "$W/VERSION" ]; then
  if [ ! -s "$W/seed_pairs.jsonl" ]; then
    note "seeding archive from cached king0 rollouts"
    $PY /root/work/build_seed.py 2>&1 | tail -3
    note "seed done: $(wc -l < "$W/seed_pairs.jsonl") pairs, $(wc -l < "$W/held_pairs.jsonl") held"
  fi
  if [ ! -s "$W/d_versions/v0/adapter_model.safetensors" ]; then
    note "training judge v0 from scratch (GPUs 4,5; from-scratch LoRA on 27B)"
    T0=$(date +%s)
    CUDA_VISIBLE_DEVICES=4,5 $PY /root/work/d_train.py \
      --base Qwen/Qwen3.8-27B --pairs "$W/seed_pairs.jsonl" \
      --lora-out "$W/d_versions/v0" --max-steps 150 --batch 4 --accum 2 \
      --lr 1e-5 --max-len 3584 --seed 0 > "$W/d_train_v0.log" 2>&1
    RC=$?
    note "judge v0 train exit=$RC wall=$(( $(date +%s) - T0 ))s"
    [ $RC -ne 0 ] && { note "FATAL judge v0 training failed"; exit 1; }
  fi
  $PY /root/work/publish_v0.py 2>&1 | tail -5
fi

# ---- eval driver ------------------------------------------------------------
if ! pgrep -f 'koth_eval[.]py' >/dev/null; then
  nohup $PY /root/work/koth_eval.py > "$W/eval_driver.log" 2>&1 < /dev/null &
  disown
  note "eval driver launched (pid $!)"
fi
log "PHASEM_EVAL_COMPLETE"
