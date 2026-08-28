#!/usr/bin/env bash
# Track B phase 0 + loop launch. Run under nohup; idempotent-ish (guards on
# output files). Steps:
#  1. wait for setup_pod.sh to finish
#  2. serve Qwen3-32B teacher on GPUs 0-3 (4 x TP1 replicas)
#  3. generate teacher rollouts: k=3 x 3990 train turns, k=2 x 300 held-out
#  4. retire GPUs 0-2 -> G server (GPU0) + driver w/ D (GPU1) + SFT (GPU2)
#  5. teacher SWE bench on the surviving replica (GPU3), then G0 SWE (GPU3)
#  6. hand GPU3 to the checkpoint SWE watcher
set -uo pipefail
cd /root/work
W=/dshare/gad
mkdir -p "$W"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
export HF_TOKEN="$(cat /root/.hf_token)"

log() { echo "[phase0 $(date -u +%T)] $*"; }
note() { echo "$(date -u +%FT%TZ) $*" >> "$W/status.log"; }

wait_health() { # port timeout_s
  local p="$1" t="${2:-900}" i=0
  while ! curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1; do
    sleep 10; i=$((i+10))
    if [ "$i" -ge "$t" ]; then log "TIMEOUT waiting port $p"; return 1; fi
  done
  log "port $p healthy after ${i}s"
}

# 1. setup barrier ----------------------------------------------------------
i=0
while ! grep -q SETUP_DONE /root/work/setup.log 2>/dev/null; do
  sleep 20; i=$((i+20))
  if [ "$i" -ge 5400 ]; then
    log "setup still not done after 90min; proceeding anyway"; break
  fi
done
log "setup barrier passed"
note "[trackB] setup done; starting phase 0 (teacher rollouts)"

# 2. teacher replicas -------------------------------------------------------
mkdir -p "$W/swe"
ensure_teachers() {
  # 6 x TP1 replicas on GPUs 0-5; GPUs 6-7 stay free for the loop smoke test
  # --enforce-eager: vLLM 0.27.1's inductor-compiled kernels hit a device-side
  # "index out of bounds < 40960" assert on Qwen3-32B under load, killing the
  # engine minutes into generation. Eager mode is ~30% slower but stable.
  # v0.11.0 image: 0.27.1 (:latest) kept dying (async-sched assert, inductor
  # index assert, silent engine deaths) on the 32B under load. Eager mode for
  # belt and suspenders; teacher gen is a one-time cost.
  if ! curl -sf http://127.0.0.1:8010/v1/models >/dev/null 2>&1; then
    for g in 0 1 2 3 4 5; do
      VLLM_IMAGE=vllm/vllm-openai:v0.11.0 \
        bash /root/work/serve_h200.sh Qwen/Qwen3-32B "801$g" "$g" --enforce-eager
    done
    for g in 0 1 2 3 4 5; do wait_health "801$g" 900 || true; done
  fi
}
TURLS=http://127.0.0.1:8010,http://127.0.0.1:8011,http://127.0.0.1:8012,http://127.0.0.1:8013,http://127.0.0.1:8014,http://127.0.0.1:8015

# docker image pre-pull in the background while GPUs generate
(/root/benchenv/bin/python - <<'PY'
import json
pin = json.load(open('/root/work/evalsrv/data/swe_rebench_lite_ids.json'))
for iid in pin['instance_ids']:
    print('swerebench/sweb.eval.x86_64.' + iid.replace('__','_1776_').lower() + ':latest')
PY
) | xargs -P 4 -n 1 docker pull > "$W/prepull.log" 2>&1 &

# 3. teacher rollouts (retry loop: sampler is resumable, servers can die) ---
attempt=0
while [ "$(wc -l < "$W/teacher_train.jsonl" 2>/dev/null || echo 0)" -lt 3000 ] && [ "$attempt" -lt 5 ]; do
  attempt=$((attempt+1))
  ensure_teachers
  log "teacher rollouts: train split (attempt $attempt)"
  $PY /root/work/sample_rollouts.py \
    --urls "$TURLS" \
    --model Qwen/Qwen3-32B --tokenizer Qwen/Qwen3-32B \
    --split train --k 3 --workers 144 --rescue \
    --out "$W/teacher_train.jsonl" 2>&1 | tee -a "$W/teacher_gen_train.log"
  note "[trackB] teacher train rollouts attempt $attempt: $(wc -l < "$W/teacher_train.jsonl" 2>/dev/null || echo 0) turns"
done
if [ "$(wc -l < "$W/teacher_train.jsonl" 2>/dev/null || echo 0)" -lt 3000 ]; then
  note "[trackB] FATAL: teacher train rollouts failed after 5 attempts; stopping phase0"
  exit 1
fi

attempt=0
while [ "$(wc -l < "$W/teacher_heldout.jsonl" 2>/dev/null || echo 0)" -lt 250 ] && [ "$attempt" -lt 5 ]; do
  attempt=$((attempt+1))
  ensure_teachers
  log "teacher rollouts: held-out split (attempt $attempt)"
  $PY /root/work/sample_rollouts.py \
    --urls "$TURLS" \
    --model Qwen/Qwen3-32B --tokenizer Qwen/Qwen3-32B \
    --split test --k 2 --limit 320 --seed 1 --workers 96 --rescue \
    --out "$W/teacher_heldout.jsonl" 2>&1 | tee -a "$W/teacher_gen_heldout.log"
  note "[trackB] teacher held-out rollouts attempt $attempt: $(wc -l < "$W/teacher_heldout.jsonl" 2>/dev/null || echo 0) turns"
done

# 4. retire replicas except the last (kept on GPU5 for the teacher SWE run) --
for p in 8010 8011 8012 8013 8014; do
  docker rm -f "vllm_$p" >/dev/null 2>&1 || true
done
sleep 10

log "starting G server (GPU0) with runtime-LoRA"
bash /root/work/serve_h200.sh Qwen/Qwen3-4B 8001 0 \
  --enforce-eager --enable-lora --max-lora-rank 16 --max-loras 4
wait_health 8001 900

log "starting driver (GPU1: discriminator)"
CUDA_VISIBLE_DEVICES=1 nohup $PY /root/work/gad_driver.py \
  --sft-gpu 2 > "$W/driver.log" 2>&1 &
note "[trackB] loop driver launched"
nohup bash /root/work/babysit_g.sh > "$W/babysit_g.log" 2>&1 &

# 5. SWE benches: teacher (GPU3 replica), then G0 ----------------------------
export PATH=/root/benchenv/bin:$PATH
if [ ! -s "$W/swe/swe_teacher.json" ]; then
  wait_health 8015 300 && \
  /root/benchenv/bin/python /root/work/run_swe.py \
    --model Qwen/Qwen3-32B --port 8015 --workers 24 --tag teacher \
    > "$W/swe_teacher_run.log" 2>&1
  note "[trackB] teacher SWE done: $(cat "$W/swe/swe_teacher.json" 2>/dev/null | head -c 300)"
fi
docker rm -f vllm_8015 >/dev/null 2>&1 || true
sleep 10

if [ ! -s "$W/swe/swe_G0.json" ]; then
  bash /root/work/serve_h200.sh Qwen/Qwen3-4B 8030 3 --enforce-eager
  wait_health 8030 900 && \
  /root/benchenv/bin/python /root/work/run_swe.py \
    --model Qwen/Qwen3-4B --port 8030 --workers 24 --tag G0 \
    > "$W/swe_G0_run.log" 2>&1
  note "[trackB] G0 SWE done: $(cat "$W/swe/swe_G0.json" 2>/dev/null | head -c 300)"
  docker rm -f vllm_8030 >/dev/null 2>&1 || true
  sleep 10
fi

# 6. checkpoint SWE watcher --------------------------------------------------
nohup bash /root/work/swe_watcher.sh > "$W/swe_watcher.log" 2>&1 &
note "[trackB] swe watcher launched; phase0 orchestration complete"
log "PHASE0_COMPLETE"
