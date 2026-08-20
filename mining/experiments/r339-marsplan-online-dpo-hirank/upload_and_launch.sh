#!/usr/bin/env bash
# Host → mine-r339-marsplan-online-dpo-hirank-1: H138 online-DPO stack + R339 Marsplan overlays, bootstrap.
# Axis R339: marsplan-init online DPO (R13 method; ≠ R204–R227 / ≠ R13 Tok / ≠ R11).
set -euo pipefail

ROOT=/home/const/subnet120
POD_NAME=${POD_NAME:-mine-r339-marsplan-online-dpo-hirank-1}
DST_HOST=${DST_HOST:?set DST_HOST}
DST_PORT=${DST_PORT:?set DST_PORT}
KNOWN=${KNOWN:-/tmp/${POD_NAME}.known_hosts}
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -p "$DST_PORT" "root@$DST_HOST")
SCP=(scp -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -P "$DST_PORT")

SOFT=$(date -u -d '+23 hours' +%Y-%m-%dT%H:%M:%SZ)
DEAD=$(date -u -d '+23 hours 30 minutes' +%Y-%m-%dT%H:%M:%SZ)
EXP=s4-h139-f44-tok-online-dpo-l2
PREBUILT=${R339_PREBUILT_TAR:-$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/artifacts/mine-r339-stack_latest.tar.gz}
TAR=/tmp/mine-r339-stack.tar.gz
STAGE=""
if [[ -s "$PREBUILT" ]] \
  && tar -tzf "$PREBUILT" | grep -q 's4-h139-f44-tok-online-dpo-l2/train_online_dpo.py' \
  && tar -xOf "$PREBUILT" ./s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh 2>/dev/null | grep -qE "DOWNLOAD (marsplan-init|vera-reign36-init)" \
  && tar -xOf "$PREBUILT" ./s4-h139-f44-tok-online-dpo-l2/start_h139.sh 2>/dev/null | grep -qE "R339: Marsplan-init online DPO HiRank" \
  && tar -xOf "$PREBUILT" ./s4-h139-f44-tok-online-dpo-l2/post_train_pipeline.sh 2>/dev/null | grep -q "/root/r339/train"; then
  cp -f "$PREBUILT" "$TAR"
  echo "R339_PREBUILT_USED $PREBUILT -> $TAR ($(du -h "$TAR" | awk '{print $1}')) soft=$SOFT"
else
  STAGE=$(mktemp -d /tmp/mine-r339-stack.XXXXXX)
  trap 'rm -rf "$STAGE"' EXIT
  mkdir -p "$STAGE/affine_pkg/affine" "$STAGE/affine_pkg/evalsrv" \
           "$STAGE/s3-duel-sim" "$STAGE/s4-h2-merge" "$STAGE/s4-h1-sft" \
           "$STAGE/s4-h1v2-sft" "$STAGE/$EXP" "$STAGE/r339-marsplan-online-dpo-hirank"
  cp -a "$ROOT/affine/affine.toml" "$STAGE/affine_pkg/"
  cp -a "$ROOT/affine/affine/." "$STAGE/affine_pkg/affine/"
  cp -a "$ROOT/affine/evalsrv/." "$STAGE/affine_pkg/evalsrv/"
  cp -a "$ROOT/mining/experiments/s3-duel-sim/"*.sh "$STAGE/s3-duel-sim/"
  cp -a "$ROOT/mining/experiments/s3-duel-sim/"*.py "$STAGE/s3-duel-sim/" 2>/dev/null || true
  cp -a "$ROOT/mining/experiments/s4-h2-merge/restart_for_h2.sh" "$STAGE/s4-h2-merge/"
  cp -a "$ROOT/mining/experiments/s4-h2-merge/run_sim_duel.py" "$STAGE/s4-h2-merge/"
  cp -a "$ROOT/mining/experiments/s4-h2-merge/write_merge_decision.py" "$STAGE/s4-h2-merge/"
  cp -a "$ROOT/mining/experiments/s4-h2-merge/watch_form_decision.sh" "$STAGE/s4-h2-merge/"
  cp -a "$ROOT/mining/experiments/s4-h2-merge/watch_n80_retry.sh" "$STAGE/s4-h2-merge/"
  cp -a "$ROOT/mining/experiments/s4-h1-sft/merge_lora.py" "$STAGE/s4-h1-sft/"
  cp -a "$ROOT/mining/experiments/s4-h1-sft/salvage_adapter.py" "$STAGE/s4-h1-sft/"
  cp -a "$ROOT/mining/experiments/s4-h1-sft/push_merged.py" "$STAGE/s4-h1-sft/"
  cp -a "$ROOT/mining/experiments/s4-h1v2-sft/thought_mask.py" "$STAGE/s4-h1v2-sft/" 2>/dev/null || true
  cp -a "$ROOT/mining/experiments/s4-h1v2-sft/verify_thought_mask.py" "$STAGE/s4-h1v2-sft/" 2>/dev/null || true
  cp -a "$ROOT/mining/experiments/$EXP/"*.sh "$STAGE/$EXP/"
  cp -a "$ROOT/mining/experiments/$EXP/"*.py "$STAGE/$EXP/"
  cp -a "$ROOT/mining/experiments/$EXP/plan.md" "$STAGE/$EXP/"
  cp -a "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/plan.md" "$STAGE/r339-marsplan-online-dpo-hirank/"
  cp -a "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/start_r339.sh" "$STAGE/r339-marsplan-online-dpo-hirank/"
  cp -a "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/bootstrap_r339.sh" "$STAGE/r339-marsplan-online-dpo-hirank/"
  cp -a "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/post_train_pipeline_r339.sh" "$STAGE/r339-marsplan-online-dpo-hirank/"
  # Overlay H138 entrypoints → R339 Marsplan Online-DPO.
  cp -a "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/start_r339.sh" "$STAGE/$EXP/start_h139.sh"
  cp -a "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/bootstrap_r339.sh" "$STAGE/$EXP/bootstrap_h139.sh"
  cp -a "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/post_train_pipeline_r339.sh" "$STAGE/$EXP/post_train_pipeline.sh"
  export STAGE_EXP_POST="$STAGE/$EXP/post_train_pipeline.sh"
  export SOFT
  python3 - <<'PY'
from pathlib import Path
import re, os
soft = os.environ["SOFT"]
p = Path(os.environ["STAGE_EXP_POST"])
t = p.read_text()
t2, n = re.subn(
    r"SOFT_DEADLINE_UTC=\$\{SOFT_DEADLINE_UTC:-[^}]+\}",
    f"SOFT_DEADLINE_UTC=${{SOFT_DEADLINE_UTC:-{soft}}}",
    t,
    count=1,
)
if n != 1:
    print("SOFT_DEADLINE_PATTERN_MISS n=", n)
else:
    p.write_text(t2)
    print("SOFT_DEADLINE_SET", soft)
PY
  tar -C "$STAGE" -czf "$TAR" .
  echo "R339_REBUILT_TAR $TAR"
fi
ls -lh "$TAR"
test -n "$(tar -tzf "$TAR" | grep 's4-h139-f44-tok-online-dpo-l2/train_online_dpo.py' || true)"
if [[ -n "$STAGE" ]]; then
  grep -qE "DOWNLOAD (marsplan-init|vera-reign36-init)" "$STAGE/$EXP/bootstrap_h139.sh"
  grep -qE "R339: Marsplan-init online DPO HiRank" "$STAGE/$EXP/start_h139.sh"
  grep -q "/root/r339/train" "$STAGE/$EXP/post_train_pipeline.sh"
else
  tar -xOf "$TAR" ./s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh | grep -qE "DOWNLOAD (marsplan-init|vera-reign36-init)"
  tar -xOf "$TAR" ./s4-h139-f44-tok-online-dpo-l2/start_h139.sh | grep -qE "R339: Marsplan-init online DPO HiRank"
  tar -xOf "$TAR" ./s4-h139-f44-tok-online-dpo-l2/post_train_pipeline.sh | grep -q "/root/r339/train"
fi

ENV_TMP=$(mktemp /tmp/mine-r339.env.XXXXXX)
# shellcheck disable=SC1091
set -a
source "$ROOT/mining/.env"
set +a
umask 077
# p4171: pin live reign36 vera (marsplan@556d02a2 RevisionNotFound; queen gated)
KING_REPO_DEFAULT=vera6/affine-5g4yy75zuz-t6
KING_REV_DEFAULT=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_LOCAL_DEFAULT=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
{
  echo "export HF_TOKEN=${HF_TOKEN}"
  echo "export HF_HOME=/root/hf"
  echo "export HF_HUB_ENABLE_HF_TRANSFER=1"
  echo "export HF_XET_HIGH_PERFORMANCE=1"
  echo "export AFFINE_DATA_DIR=/root/affine_data"
  echo "export SOFT_DEADLINE_UTC=${SOFT}"
  echo "export DEADMAN_UTC=${DEAD}"
  echo "export HF_MERGED_REPO=unconst/Affine-5czsc2fc98-r339-online-dpo-merged"
  echo "export HF_LORA_REPO=unconst/Affine-5czsc2fc98-r339-online-dpo-lora"
  echo "export HF_BASE_HUB=${KING_REPO_DEFAULT}"
  echo "export KING_SERVED_NAME=${KING_REPO_DEFAULT}"
  echo "export R339_AXIS=vera_online_dpo_hirank"
  echo "export R339_GROUP=4"
  echo "export R339_MAX_STEPS=300"
  echo "export R339_TEMP=1.2"
  echo "export R339_MIN_GAP=0.0"
  echo "export R339_LR=5e-6"
  echo "export R339_LORA_R=64"
  echo "export R339_LORA_ALPHA=32"
  echo "export R339_BETA=0.1"
  echo "export KING_REPO=${KING_REPO_DEFAULT}"
  echo "export KING_REV=${KING_REV_DEFAULT}"
  echo "export KING_LOCAL=${KING_LOCAL_DEFAULT}"
  echo "export BASE=${KING_LOCAL_DEFAULT}"
} >"$ENV_TMP"
chmod 600 "$ENV_TMP"

DATA="$ROOT/mining/experiments/s4-h27-clip-l1-shape/results/winner_za_high_l1.jsonl"
test -s "$DATA"
test "$(wc -l <"$DATA")" -ge 300

"${SSH[@]}" 'mkdir -p /root/mining_src /root/affine_data /root/logs /root/h139 /root/r339 /root/r13 /root/hf'
"${SCP[@]}" "$TAR" "root@${DST_HOST}:/tmp/mine-r339-stack.tar.gz"
"${SCP[@]}" "$ENV_TMP" "root@${DST_HOST}:/root/mine.env"
"${SCP[@]}" "$DATA" "root@${DST_HOST}:/root/h139/winner_za_high_l1.jsonl"
"${SCP[@]}" "$DATA" "root@${DST_HOST}:/root/h139/dpo_duel_l2.jsonl"
"${SCP[@]}" "$DATA" "root@${DST_HOST}:/root/r339/winner_za_high_l1.jsonl"
rm -f "$ENV_TMP"

python3 - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for repo in (
    "unconst/Affine-5czsc2fc98-r339-online-dpo-merged",
    "unconst/Affine-5czsc2fc98-r339-online-dpo-lora",
):
    try:
        api.create_repo(repo, private=False, exist_ok=True, repo_type="model")
        print("HF_OK", repo)
    except Exception as e:
        print("HF_ERR", repo, type(e).__name__, e)
PY

"${SSH[@]}" 'set -e
  tar -C /root/mining_src -xzf /tmp/mine-r339-stack.tar.gz
  chmod 600 /root/mine.env
  chmod +x /root/mining_src/s3-duel-sim/*.sh \
           /root/mining_src/s4-h2-merge/*.sh \
           /root/mining_src/s4-h139-f44-tok-online-dpo-l2/*.sh \
           /root/mining_src/r339-marsplan-online-dpo-hirank/*.sh
  test -f /root/mining_src/affine_pkg/affine/score.py
  test -s /root/h139/winner_za_high_l1.jsonl
  test -x /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh
  grep -qE "R339: Marsplan-init online DPO HiRank" /root/mining_src/s4-h139-f44-tok-online-dpo-l2/start_h139.sh
  grep -qE "DOWNLOAD (marsplan-init|vera-reign36-init)" /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh
  grep -q "/root/r339/train" /root/mining_src/s4-h139-f44-tok-online-dpo-l2/post_train_pipeline.sh
  set -a; source /root/mine.env; set +a
  echo "R339_DEADLINES soft=$SOFT_DEADLINE_UTC dead=$DEADMAN_UTC axis=$R339_AXIS"
  echo STACK_UPLOAD_OK
  nohup bash /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh \
    >/root/logs/r339_pipeline.nohup 2>&1 &
  echo $! > /root/logs/r339_pipeline.pid
  cp -f /root/logs/r339_pipeline.pid /root/logs/h139_pipeline.pid
  nohup bash /root/mining_src/s4-h2-merge/watch_form_decision.sh r339 \
    /root/affine_data/r339_sim_result.json /root/affine_data/r339_decision.json \
    /root/logs/r339_form_decision.nohup \
    >/root/logs/r339_form_decision.launch.out 2>&1 &
  echo $! > /root/logs/r339_form_decision.pid
  echo PIPELINE_PID=$(cat /root/logs/r339_pipeline.pid)
  sleep 5
  head -n 40 /root/logs/bootstrap_h139.log 2>/dev/null || head -n 40 /root/logs/r339_pipeline.nohup || true
  ps -p "$(cat /root/logs/r339_pipeline.pid)" -o pid,etime,cmd || true
  nvidia-smi -L | wc -l
'

echo "UPLOAD_AND_LAUNCH_OK pod=$POD_NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)"
