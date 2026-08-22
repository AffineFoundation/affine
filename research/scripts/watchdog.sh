#!/usr/bin/env bash
# Keep the adversarial loop and its sampler alive overnight.
#
# The loop resumes from its newest checkpoint, so a restart costs one round
# rather than the night's training. The sampler is restarted first when it is
# missing, because the loop cannot make progress without it.
set -u

WORK=/root/work
# Arm B: discriminator initialised from the pre-trained judge (held-out
# AUC 0.985). Arm A used a randomly initialised judge that stayed at chance,
# and its checkpoints are kept as the no-signal control.
CKPT=/opt/ckpt/gadB
DISC_INIT=/opt/ckpt/disc_pre
LOG=$WORK/watchdog.log
SAMPLER_PORT=8004
SAMPLER_GPU=3
GEN=Qwen/Qwen3.6-35B-A3B

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) | $*" >> "$LOG"; }

start_loop() {
  cd "$WORK" || return
  nohup python3 gad_loop.py \
    --turns data/disc_pairs/turns.jsonl.gz \
    --teacher-rollouts data/teacher_rollouts.jsonl \
    --gen-model "$GEN" --gen-url "http://127.0.0.1:$SAMPLER_PORT" \
    --disc-model Qwen/Qwen3-14B \
    --gen-device cuda:4 --disc-device cuda:6 \
    --rounds 1000 --turns-per-round 16 --k 6 \
    --g-steps-cap 8 --d-samples 24 \
    --gen-lr 1e-5 --disc-lr 5e-5 \
    --ckpt-dir "$CKPT" --status-log "$WORK/loop_status.log" --resume \
    --disc-init "$DISC_INIT" --reward-floor 0.0 \
    >> "$WORK/loop.log" 2>&1 &
  say "loop restarted pid=$!"
}

say "watchdog up"
while true; do
  sleep 180

  if ! curl -s -m 8 "http://127.0.0.1:$SAMPLER_PORT/v1/models" | grep -q Qwen; then
    say "sampler down, restarting"
    MAX_MODEL_LEN=32768 "$WORK/pod_serve_docker.sh" "$GEN" "$SAMPLER_PORT" \
      "$SAMPLER_GPU" --enable-lora --max-lora-rank 16 --max-loras 2 \
      >> "$LOG" 2>&1
    for _ in $(seq 1 40); do
      sleep 15
      curl -s -m 5 "http://127.0.0.1:$SAMPLER_PORT/v1/models" | grep -q Qwen && break
    done
    say "sampler back up"
  fi

  if ! pgrep -f "gad_loo[p].py" > /dev/null; then
    say "loop not running, resuming"
    start_loop
  fi
done
