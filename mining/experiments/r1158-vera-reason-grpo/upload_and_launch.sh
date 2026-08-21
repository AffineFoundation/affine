#!/usr/bin/env bash
# Host → mine-r1158-vera-reason-grpo-1: upload stack + data, start bootstrap under nohup.
# After BOOT_HF_DONE: serve_teacher_tp1 + start_r1158 (next pass if GPUs still warming).
set -euo pipefail

ROOT=/home/const/subnet120
POD_NAME=${POD_NAME:-mine-r1158-vera-reason-grpo-1}
DST_HOST=${DST_HOST:-192.9.163.79}
DST_PORT=${DST_PORT:-20532}
KNOWN=${KNOWN:-/tmp/${POD_NAME}.known_hosts}
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes
     -p "$DST_PORT" "root@$DST_HOST")
SCP=(scp -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes
     -P "$DST_PORT")

STAGE=$(mktemp -d /tmp/mine-r1158-stack.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT

EXP=r3-reason-grpo
R1158=r1158-vera-reason-grpo
mkdir -p "$STAGE/affine_pkg/affine" "$STAGE/affine_pkg/evalsrv" \
         "$STAGE/s3-duel-sim" "$STAGE/s4-h2-merge" "$STAGE/s4-h1-sft" \
         "$STAGE/r1-reason-distill" "$STAGE/$EXP" "$STAGE/$R1158"

cp -a "$ROOT/affine/affine.toml" "$STAGE/affine_pkg/"
cp -a "$ROOT/affine/affine/." "$STAGE/affine_pkg/affine/"
cp -a "$ROOT/affine/evalsrv/." "$STAGE/affine_pkg/evalsrv/"
cp -a "$ROOT/mining/experiments/s3-duel-sim/"*.sh "$STAGE/s3-duel-sim/" 2>/dev/null || true
cp -a "$ROOT/mining/experiments/s3-duel-sim/"*.py "$STAGE/s3-duel-sim/" 2>/dev/null || true
cp -a "$ROOT/mining/experiments/s4-h2-merge/run_sim_duel.py" "$STAGE/s4-h2-merge/"
cp -a "$ROOT/mining/experiments/s4-h2-merge/"*.sh "$STAGE/s4-h2-merge/" 2>/dev/null || true
cp -a "$ROOT/mining/experiments/s4-h1-sft/merge_lora.py" "$STAGE/s4-h1-sft/"
cp -a "$ROOT/mining/experiments/s4-h1-sft/salvage_adapter.py" "$STAGE/s4-h1-sft/" 2>/dev/null || true
cp -a "$ROOT/mining/experiments/s4-h1-sft/push_merged.py" "$STAGE/s4-h1-sft/" 2>/dev/null || true
cp -a "$ROOT/mining/experiments/r1-reason-distill/write_reason_decision.py" \
      "$STAGE/r1-reason-distill/" 2>/dev/null || true
cp -a "$ROOT/mining/experiments/$EXP/"*.sh "$STAGE/$EXP/"
cp -a "$ROOT/mining/experiments/$EXP/"*.py "$STAGE/$EXP/"
cp -a "$ROOT/mining/experiments/$R1158/"*.sh "$STAGE/$R1158/"
test -f "$STAGE/$EXP/train_reason_grpo.py"
test -f "$STAGE/$R1158/bootstrap_r1158.sh"
test -f "$STAGE/$R1158/start_r1158.sh"
test -f "$STAGE/$R1158/serve_teacher_tp1_r1158.sh"

TAR=/tmp/mine-r1158-stack.tar.gz
tar -C "$STAGE" -czf "$TAR" .
ls -lh "$TAR"

ENV_TMP=$(mktemp /tmp/mine-r1158.env.XXXXXX)
# shellcheck disable=SC1091
set -a
source "$ROOT/mining/.env"
set +a
umask 077
{
  echo "export HF_TOKEN=${HF_TOKEN}"
  echo "export HF_HOME=/root/hf"
  echo "export HF_XET_HIGH_PERFORMANCE=1"
  echo "export AFFINE_DATA_DIR=/root/affine_data"
  echo "export PASS=${PASS:-4281}"
} >"$ENV_TMP"
chmod 600 "$ENV_TMP"

DATA="$ROOT/mining/experiments/s4-h27-clip-l1-shape/results/winner_za_high_l1.jsonl"
test -s "$DATA"
test "$(wc -l <"$DATA")" -ge 300

"${SSH[@]}" 'mkdir -p /root/mining_src /root/affine_data /root/logs /root/r1158 /root/hf/hub'
"${SCP[@]}" "$TAR" "root@${DST_HOST}:/tmp/mine-r1158-stack.tar.gz"
"${SCP[@]}" "$ENV_TMP" "root@${DST_HOST}:/root/mine.env"
"${SCP[@]}" "$DATA" "root@${DST_HOST}:/root/r1158/winner_za_high_l1.jsonl"
rm -f "$ENV_TMP"

"${SSH[@]}" 'set -e
  tar -C /root/mining_src -xzf /tmp/mine-r1158-stack.tar.gz
  chmod 600 /root/mine.env
  chmod +x /root/mining_src/r1158-vera-reason-grpo/*.sh \
           /root/mining_src/r3-reason-grpo/*.sh \
           /root/mining_src/s4-h2-merge/*.sh 2>/dev/null || true
  test -f /root/mining_src/affine_pkg/affine/score.py
  test -f /root/mining_src/r3-reason-grpo/train_reason_grpo.py
  test -s /root/r1158/winner_za_high_l1.jsonl
  test -x /root/mining_src/r1158-vera-reason-grpo/bootstrap_r1158.sh
  echo STACK_UPLOAD_OK
  nohup env PASS=4280 bash /root/mining_src/r1158-vera-reason-grpo/bootstrap_r1158.sh \
    >/root/logs/r1158_pipeline.nohup 2>&1 &
  echo $! > /root/logs/r1158_pipeline.pid
  echo PIPELINE_PID=$(cat /root/logs/r1158_pipeline.pid)
  sleep 8
  head -n 60 /root/logs/bootstrap_r1158.log 2>/dev/null || head -n 60 /root/logs/r1158_pipeline.nohup || true
  ps -p "$(cat /root/logs/r1158_pipeline.pid)" -o pid,etime,cmd || true
  nvidia-smi -L || true
'

echo "UPLOAD_AND_LAUNCH_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
