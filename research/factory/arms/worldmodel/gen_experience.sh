#!/usr/bin/env bash
# Factory experience generation (runs on the grounded pod, 8xB300).
# One corpus for the whole fleet: teacher rollouts (train + heldout) from
# Qwen3.6-35B-A3B with PLAIN prompts (no <think> injection), plus witness
# observation data, fanned out to every other pod over the pod network.
set -uo pipefail
cd /root/work
D=$(cat /root/.dshare_path 2>/dev/null || echo /dshare)
W="$D/factory"; mkdir -p "$W"
PY=/root/trainenv/bin/python
export HF_HOME="$D/hf"
export HF_TOKEN="$(cat /root/.hf_token)"
TEACHER=Qwen/Qwen3.6-35B-A3B

log() { echo "[genexp $(date -u +%T)] $*"; }
note() { echo "$(date -u +%FT%TZ) $*" >> "$W/status.log"; }

wait_health() { local p="$1" t="${2:-1800}" i=0
  while ! curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1; do
    sleep 10; i=$((i+10)); [ "$i" -ge "$t" ] && { log "TIMEOUT port $p"; return 1; }
  done; log "port $p healthy after ${i}s"; }

# barrier: setup done
i=0
while ! grep -q SETUP_DONE /root/work/setup.log 2>/dev/null; do
  sleep 20; i=$((i+20)); [ "$i" -ge 7200 ] && { log "setup slow; proceeding"; break; }
done
note "[factory] gen_experience start"

# 1. teacher replicas: GPUs 0-5, TP1, ports 8010-8015
for g in 0 1 2 3 4 5; do
  curl -sf "http://127.0.0.1:801$g/v1/models" >/dev/null 2>&1 && continue
  MAX_MODEL_LEN=32768 HF_DIR="$D/hf" CKPT_DIR="$D/ckpt" \
    bash /root/work/pod_serve_docker.sh "$TEACHER" "801$g" "$g"
done
for g in 0 1 2 3 4 5; do wait_health "801$g" 2400 || note "[factory] WARN replica 801$g down"; done

TURLS=http://127.0.0.1:8010,http://127.0.0.1:8011,http://127.0.0.1:8012,http://127.0.0.1:8013,http://127.0.0.1:8014,http://127.0.0.1:8015

# 2. rollouts: train split
attempt=0
while [ "$(wc -l < "$W/teacher_train.jsonl" 2>/dev/null || echo 0)" -lt 3000 ] && [ "$attempt" -lt 4 ]; do
  attempt=$((attempt+1))
  log "train rollouts attempt $attempt"
  $PY /root/work/sample_rollouts.py --urls "$TURLS" \
    --model "$TEACHER" --tokenizer "$TEACHER" --plain \
    --split train --k 2 --workers 192 \
    --out "$W/teacher_train.jsonl" 2>&1 | tee -a "$W/gen_train.log"
  note "[factory] train rollouts attempt $attempt: $(wc -l < "$W/teacher_train.jsonl" 2>/dev/null || echo 0) turns"
done

# 3. rollouts: held-out split
attempt=0
while [ "$(wc -l < "$W/teacher_heldout.jsonl" 2>/dev/null || echo 0)" -lt 150 ] && [ "$attempt" -lt 4 ]; do
  attempt=$((attempt+1))
  log "held-out rollouts attempt $attempt"
  $PY /root/work/sample_rollouts.py --urls "$TURLS" \
    --model "$TEACHER" --tokenizer "$TEACHER" --plain \
    --split test --k 2 --limit 220 --seed 1 --workers 96 \
    --out "$W/teacher_heldout.jsonl" 2>&1 | tee -a "$W/gen_heldout.log"
  note "[factory] held-out rollouts attempt $attempt: $(wc -l < "$W/teacher_heldout.jsonl" 2>/dev/null || echo 0) turns"
done

# 4. style report: does this teacher think?
$PY - <<'EOF' >> "$W/status.log"
import json, datetime
D = open("/root/.dshare_path").read().strip()
think = tot = 0
for line in open(f"{D}/factory/teacher_train.jsonl"):
    r = json.loads(line)
    for roll in r.get("rollouts", []):
        tot += 1
        if roll["raw"].lstrip().startswith("<think>"):
            think += 1
ts = datetime.datetime.utcnow().strftime("%FT%TZ")
print(f"{ts} [factory] STYLE: {think}/{tot} rollouts start with <think>")
EOF

# 5. witness observation data (train + held-out splits)
$PY /root/work/build_witness_data.py \
  --turns /root/work/turns.jsonl.gz --meta /root/work/turn_meta.jsonl.gz \
  --out-turns "$W/witness_turns.jsonl.gz" --out-rolls "$W/witness_obs.jsonl" \
  2>&1 | tee -a "$W/gen_witness.log"
$PY /root/work/build_witness_data.py --split test --limit 500 \
  --turns /root/work/turns.jsonl.gz --meta /root/work/turn_meta.jsonl.gz \
  --out-turns "$W/witness_turns_test.jsonl.gz" \
  --out-rolls "$W/witness_obs_test.jsonl" \
  2>&1 | tee -a "$W/gen_witness.log"
note "[factory] witness data: $(wc -l < "$W/witness_obs.jsonl" 2>/dev/null || echo 0) train + $(wc -l < "$W/witness_obs_test.jsonl" 2>/dev/null || echo 0) test obs examples"

# 6. fan out to the fleet
SSHK="-i /root/.ssh/factory_key -o StrictHostKeyChecking=no"
while read -r host port; do
  (
    scp $SSHK -q -P "$port" /root/work/turns.jsonl.gz "$host:/root/work/" &&
    scp $SSHK -q -P "$port" "$W/teacher_train.jsonl" "$W/teacher_heldout.jsonl" \
        "$W/witness_turns.jsonl.gz" "$W/witness_obs.jsonl" \
        "$W/witness_turns_test.jsonl.gz" "$W/witness_obs_test.jsonl" \
        "$host:/root/work/" &&
    ssh $SSHK -p "$port" "$host" "touch /root/work/EXPERIENCE_READY" &&
    echo "fanout ok: $host" || echo "fanout FAIL: $host"
  ) &
done <<'PODS'
root@REDACTED_IP 40050
root@REDACTED_IP 40050
root@REDACTED_IP 40051
root@REDACTED_IP 40299
root@REDACTED_IP 40300
root@REDACTED_IP 40301
root@REDACTED_IP 36784
PODS
wait
cp "$W/teacher_train.jsonl" "$W/teacher_heldout.jsonl" /root/work/ 2>/dev/null
cp "$W/witness_turns.jsonl.gz" "$W/witness_obs.jsonl" /root/work/ 2>/dev/null
touch /root/work/EXPERIENCE_READY
note "[factory] EXPERIENCE_READY fanned out"
echo "GEN_EXPERIENCE_DONE"
