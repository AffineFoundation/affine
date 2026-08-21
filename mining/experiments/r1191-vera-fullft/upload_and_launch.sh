#!/usr/bin/env bash
# Host → mine-r1191-vera-fullft-1: H121 FullFT stack + R1191 Genesis overlays, bootstrap.
# Axis R1191: vera-init×FullFT thought-only lr=1e-6 @8192 (R4/H121 method; ≠ LoRA axes / ≠ R4 Tok).
set -euo pipefail

ROOT=/home/const/subnet120
POD_NAME=${POD_NAME:-mine-r1191-vera-fullft-1}
DST_HOST=${DST_HOST:?set DST_HOST}
DST_PORT=${DST_PORT:?set DST_PORT}
KNOWN=${KNOWN:-/tmp/${POD_NAME}.known_hosts}
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -p "$DST_PORT" "root@$DST_HOST")
SCP=(scp -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -P "$DST_PORT")

SOFT=$(date -u -d '+23 hours' +%Y-%m-%dT%H:%M:%SZ)
DEAD=$(date -u -d '+23 hours 30 minutes' +%Y-%m-%dT%H:%M:%SZ)
EXP=s4-h121-f26-full-ft
PREBUILT=${R1191_PREBUILT_TAR:-$ROOT/mining/experiments/r1191-vera-fullft/artifacts/mine-r1191-stack_latest.tar.gz}
TAR=/tmp/mine-r1191-stack.tar.gz
STAGE=""
if [[ -s "$PREBUILT" ]] \
  && tar -tzf "$PREBUILT" | grep -q 's4-h121-f26-full-ft/train_full.py' \
  && tar -xOf "$PREBUILT" ./s4-h121-f26-full-ft/bootstrap_h121.sh 2>/dev/null | grep -q "DOWNLOAD vera-init" \
  && tar -xOf "$PREBUILT" ./s4-h121-f26-full-ft/start_h121.sh 2>/dev/null | grep -q "R1191: Vera-FullFT" \
  && tar -xOf "$PREBUILT" ./s4-h121-f26-full-ft/post_train_pipeline.sh 2>/dev/null | grep -q "/root/r1191/train"; then
  cp -f "$PREBUILT" "$TAR"
  echo "R1191_PREBUILT_USED $PREBUILT -> $TAR ($(du -h "$TAR" | awk '{print $1}')) soft=$SOFT"
else
  STAGE=$(mktemp -d /tmp/mine-r1191-stack.XXXXXX)
  trap 'rm -rf "$STAGE"' EXIT
  mkdir -p "$STAGE/affine_pkg/affine" "$STAGE/affine_pkg/evalsrv" \
           "$STAGE/s3-duel-sim" "$STAGE/s4-h2-merge" "$STAGE/s4-h1-sft" \
           "$STAGE/s4-h1v2-sft" "$STAGE/r1-reason-distill" \
           "$STAGE/$EXP" "$STAGE/r1191-vera-fullft"
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
  cp -a "$ROOT/mining/experiments/r1-reason-distill/write_reason_decision.py" \
        "$STAGE/r1-reason-distill/"
  cp -a "$ROOT/mining/experiments/s4-h1-sft/push_merged.py" "$STAGE/s4-h1-sft/"
  cp -a "$ROOT/mining/experiments/s4-h1v2-sft/thought_mask.py" "$STAGE/s4-h1v2-sft/"
  cp -a "$ROOT/mining/experiments/s4-h1v2-sft/verify_thought_mask.py" "$STAGE/s4-h1v2-sft/"
  cp -a "$ROOT/mining/experiments/$EXP/"*.sh "$STAGE/$EXP/"
  cp -a "$ROOT/mining/experiments/$EXP/"*.py "$STAGE/$EXP/"
  cp -a "$ROOT/mining/experiments/$EXP/plan.md" "$STAGE/$EXP/"
  cp -a "$ROOT/mining/experiments/r1191-vera-fullft/plan.md" "$STAGE/r1191-vera-fullft/"
  cp -a "$ROOT/mining/experiments/r1191-vera-fullft/start_r1191.sh" "$STAGE/r1191-vera-fullft/"
  cp -a "$ROOT/mining/experiments/r1191-vera-fullft/bootstrap_r1191.sh" "$STAGE/r1191-vera-fullft/"
  cp -a "$ROOT/mining/experiments/r1191-vera-fullft/post_train_pipeline_r1191.sh" "$STAGE/r1191-vera-fullft/"
  # Overlay H121 entrypoints → R1191 Vera FullFT.
  cp -a "$ROOT/mining/experiments/r1191-vera-fullft/start_r1191.sh" "$STAGE/$EXP/start_h121.sh"
  cp -a "$ROOT/mining/experiments/r1191-vera-fullft/bootstrap_r1191.sh" "$STAGE/$EXP/bootstrap_h121.sh"
  cp -a "$ROOT/mining/experiments/r1191-vera-fullft/post_train_pipeline_r1191.sh" "$STAGE/$EXP/post_train_pipeline.sh"
  export STAGE_EXP_POST="$STAGE/$EXP/post_train_pipeline.sh"
  export SOFT
  python3 - <<'PY'
from pathlib import Path
import re, os
soft = os.environ["SOFT"]
p = Path(os.environ["STAGE_EXP_POST"])
t = p.read_text()
pat = r"SOFT_DEADLINE_UTC=\$\{SOFT_DEADLINE_UTC:-[^}]+\}"
t2, n = re.subn(pat, f"SOFT_DEADLINE_UTC=${{SOFT_DEADLINE_UTC:-{soft}}}", t, count=1)
if n != 1:
    t2, n = re.subn(
        r"SOFT_DEADLINE_UTC=\$\{SOFT_DEADLINE_UTC:-[^}]+\}",
        f"SOFT_DEADLINE_UTC=${{SOFT_DEADLINE_UTC:-{soft}}}",
        t,
        count=1,
    )
# also try raw bash form without double-escape confusion
if n != 1:
    import re as _re
    t2, n = _re.subn(
        r"SOFT_DEADLINE_UTC=\$\{SOFT_DEADLINE_UTC:-[^}]+\}",
        "SOFT_DEADLINE_UTC=${SOFT_DEADLINE_UTC:-%s}" % soft,
        t,
        count=1,
    )
if n != 1:
    # direct string replace of known default
    needle = "SOFT_DEADLINE_UTC=${SOFT_DEADLINE_UTC:-2026-08-16T08:00:00Z}"
    if needle in t:
        t2 = t.replace(needle, f"SOFT_DEADLINE_UTC=${{SOFT_DEADLINE_UTC:-{soft}}}", 1)
        n = 1
if n != 1:
    print("SOFT_DEADLINE_PATTERN_MISS n=", n)
else:
    p.write_text(t2)
    print("SOFT_DEADLINE_SET", soft)
PY
  tar -C "$STAGE" -czf "$TAR" .
  echo "R1191_REBUILT_TAR $TAR"
fi
ls -lh "$TAR"
test -n "$(tar -tzf "$TAR" | grep 's4-h121-f26-full-ft/train_full.py' || true)"
if [[ -n "$STAGE" ]]; then
  grep -q "DOWNLOAD vera-init" "$STAGE/$EXP/bootstrap_h121.sh"
  grep -q "R1191: Vera-FullFT" "$STAGE/$EXP/start_h121.sh"
  grep -q "/root/r1191/train" "$STAGE/$EXP/post_train_pipeline.sh"
else
  tar -xOf "$TAR" ./s4-h121-f26-full-ft/bootstrap_h121.sh | grep -q "DOWNLOAD vera-init"
  tar -xOf "$TAR" ./s4-h121-f26-full-ft/start_h121.sh | grep -q "R1191: Vera-FullFT"
  tar -xOf "$TAR" ./s4-h121-f26-full-ft/post_train_pipeline.sh | grep -q "/root/r1191/train"
fi

ENV_TMP=$(mktemp /tmp/mine-r1191.env.XXXXXX)
# shellcheck disable=SC1091
set -a
source "$ROOT/mining/.env"
set +a
umask 077
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
  echo "export HF_MERGED_REPO=unconst/Affine-5czsc2fc98-r1191-fullft"
  echo "export R1191_AXIS=vera_fullft"
  echo "export R1191_LR=1e-6"
  echo "export R1191_MAX_LEN=8192"
  echo "export KING_REPO=${KING_REPO_DEFAULT}"
  echo "export KING_REV=${KING_REV_DEFAULT}"
  echo "export KING_LOCAL=${KING_LOCAL_DEFAULT}"
  echo "export BASE=${KING_LOCAL_DEFAULT}"
  echo "export HF_BASE_HUB=${KING_REPO_DEFAULT}"
  echo "export RESTART_KING=1"
  echo "export HYP=R1191"
  echo "export AXIS_HYP=R1191"
} >"$ENV_TMP"
chmod 600 "$ENV_TMP"

DATA="$ROOT/mining/experiments/r1191-vera-fullft/results/winner_za_high_l2.jsonl"
test -s "$DATA"
test "$(wc -l <"$DATA")" -ge 200

"${SSH[@]}" 'mkdir -p /root/mining_src /root/affine_data /root/logs /root/h121 /root/r1191 /root/hf'
"${SCP[@]}" "$TAR" "root@${DST_HOST}:/tmp/mine-r1191-stack.tar.gz"
"${SCP[@]}" "$ENV_TMP" "root@${DST_HOST}:/root/mine.env"
"${SCP[@]}" "$DATA" "root@${DST_HOST}:/root/h121/winner_za_high_l2.jsonl"
"${SCP[@]}" "$DATA" "root@${DST_HOST}:/root/r1191/winner_za_high_l2.jsonl"
"${SCP[@]}" "$DATA" "root@${DST_HOST}:/root/r1191/winner_za_high_l1.jsonl"
rm -f "$ENV_TMP"

python3 - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "unconst/Affine-5czsc2fc98-r1191-fullft"
try:
    api.create_repo(repo, private=False, exist_ok=True, repo_type="model")
    print("HF_OK", repo)
except Exception as e:
    print("HF_ERR", repo, type(e).__name__, e)
PY

"${SSH[@]}" 'set -e
  tar -C /root/mining_src -xzf /tmp/mine-r1191-stack.tar.gz
  chmod 600 /root/mine.env
  chmod +x /root/mining_src/s3-duel-sim/*.sh \
           /root/mining_src/s4-h2-merge/*.sh \
           /root/mining_src/s4-h121-f26-full-ft/*.sh \
           /root/mining_src/r1191-vera-fullft/*.sh
  test -f /root/mining_src/affine_pkg/affine/score.py
  test -s /root/r1191/winner_za_high_l2.jsonl
  test -x /root/mining_src/s4-h121-f26-full-ft/bootstrap_h121.sh
  test -x /root/mining_src/s4-h121-f26-full-ft/start_h121.sh
  grep -q "R1191: Vera-FullFT" /root/mining_src/s4-h121-f26-full-ft/start_h121.sh
  grep -q "DOWNLOAD vera-init" /root/mining_src/s4-h121-f26-full-ft/bootstrap_h121.sh
  grep -q "/root/r1191/train" /root/mining_src/s4-h121-f26-full-ft/post_train_pipeline.sh
  test -f /root/mining_src/r1-reason-distill/write_reason_decision.py
  set -a; source /root/mine.env; set +a
  test "$KING_REPO" = "vera6/affine-5g4yy75zuz-t6"
  echo "R1191_DEADLINES soft=$SOFT_DEADLINE_UTC dead=$DEADMAN_UTC axis=$R1191_AXIS king=$KING_REPO@$KING_REV"
  echo "R1191_KNOBS lr=${R1191_LR} max_len=${R1191_MAX_LEN}"
  echo STACK_UPLOAD_OK
  nohup bash /root/mining_src/s4-h121-f26-full-ft/bootstrap_h121.sh \
    >/root/logs/r1191_pipeline.nohup 2>&1 &
  echo $! > /root/logs/r1191_pipeline.pid
  cp -f /root/logs/r1191_pipeline.pid /root/logs/h121_pipeline.pid
  if [[ -s /root/affine_data/r1191_sim_result.json || -s /root/affine_data/r1191_decision.json || -s /root/affine_data/h121_sim_result.json ]]; then
    _stale=/root/affine_data/stale_pre_form_$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p "$_stale"
    for _f in r1191_sim_result.json r1191_sim_progress.json r1191_decision.json h121_sim_result.json h121_decision.json; do
      [[ -e /root/affine_data/$_f ]] && mv -f /root/affine_data/$_f "$_stale/" || true
    done
  fi
  : >/root/logs/r1191_form_decision.nohup
  nohup bash /root/mining_src/s4-h2-merge/watch_form_decision.sh r1191 \
    /root/affine_data/r1191_sim_result.json /root/affine_data/r1191_decision.json \
    /root/logs/r1191_form_decision.nohup \
    >/root/logs/r1191_form_decision.launch.out 2>&1 &
  echo $! > /root/logs/r1191_form_decision.pid
  echo PIPELINE_PID=$(cat /root/logs/r1191_pipeline.pid)
  echo FORM_PID=$(cat /root/logs/r1191_form_decision.pid)
  sleep 5
  head -n 40 /root/logs/bootstrap_h121.log 2>/dev/null || head -n 40 /root/logs/r1191_pipeline.nohup || true
  ps -p "$(cat /root/logs/r1191_pipeline.pid)" -o pid,etime,cmd || true
  nvidia-smi -L | wc -l
'

echo "UPLOAD_AND_LAUNCH_OK pod=$POD_NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)"
