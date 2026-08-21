#!/usr/bin/env bash
# R1209: After Vera FullFT train.done: finalize visual → chall:8002 → n80 vs live genesis.
# Overlay: upload_and_launch copies this to s4-h121-f26-full-ft/post_train_pipeline.sh.
set -euo pipefail

# shellcheck disable=SC1091
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=${HF_HOME:-/root/hf}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}

# p3007: hard-pin R1209 HF+BASE+KING after mine.env
export HF_MERGED_REPO=${HF_MERGED_REPO:-unconst/Affine-5czsc2fc98-r1209-fullft-ultralolr}
export HF_BASE_HUB=${HF_BASE_HUB:-vera6/affine-5g4yy75zuz-t6}
export KING_REPO=${KING_REPO:-vera6/affine-5g4yy75zuz-t6}
export KING_REV=${KING_REV:-8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
export KING_LOCAL=${KING_LOCAL:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
export BASE=${BASE:-$KING_LOCAL}
export AXIS_HYP=R1209
export HYP=R1209
export RESTART_KING=${RESTART_KING:-1}
export SKIP_LOCAL_TKC=${SKIP_LOCAL_TKC:-0}
export SKIP_HF_PUSH=${SKIP_HF_PUSH:-1}

BASE=${BASE:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
KING_REPO=${KING_REPO:-vera6/affine-5g4yy75zuz-t6}
KING_REV=${KING_REV:-8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
KING_LOCAL=${KING_LOCAL:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
TRAIN_DIR=${TRAIN_DIR:-/root/r1209/train}
FULL_FT=${FULL_FT:-/tmp/r1209_full_ft_save}
if [[ ! -d "$FULL_FT" ]]; then
  FULL_FT=$TRAIN_DIR/full_ft
fi
MERGED=${MERGED:-/tmp/r1209_merged}
MERGED_LINK=${MERGED_LINK:-/root/r1209/merged}
SIM_N80=/root/affine_data/r1209_sim_result.json
PROG=/root/affine_data/r1209_sim_progress.json
SIM_DEC=${SIM_DEC:-/root/affine_data/r1209_decision.json}
LOG=/root/logs/r1209_pipeline.nohup
SOFT_DEADLINE_UTC=${SOFT_DEADLINE_UTC:-2026-08-22T11:00:00Z}
DEADMAN_UTC=${DEADMAN_UTC:-2026-08-22T11:30:00Z}

log() { echo "[r1209-pipe] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

_train_alive() {
  if [[ -f /root/logs/r1209_train.pid ]]; then
    local tpid
    tpid=$(cat /root/logs/r1209_train.pid 2>/dev/null || true)
    if [[ -n "${tpid:-}" ]] && kill -0 "$tpid" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -f "python3 /root/mining_src/s4-h121-f26-full-ft/train_full.py --base" >/dev/null 2>&1
}

_abort_on_exit() {
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    return 0
  fi
  if [[ -f /root/logs/r1209_pipeline.done ]]; then
    return 0
  fi
  if [[ ! -f /root/logs/r1209_pipeline.aborted ]]; then
    echo "aborted_err_rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >/root/logs/r1209_pipeline.aborted
    echo "[r1209-pipe] $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT trap wrote aborted_err_rc=${rc}" \
      | tee -a "$LOG" >/dev/null 2>&1 || true
  fi
}
trap _abort_on_exit EXIT

mkdir -p /root/logs /root/affine_data /root/r1209 /root/h121
rm -f /root/logs/r1209_pipeline.aborted /root/logs/r1209_pipeline.done \
  /root/logs/r1209_merge.done /root/logs/r1209_chall_serve.done \
  /root/logs/r1209_sim_n80.done

log "waiting for $TRAIN_DIR/train.done (or full_ft + no train proc)"
_wait_i=0
while true; do
  if [[ -f "$TRAIN_DIR/train.done" ]]; then
    log "train.done present"
    break
  fi
  if [[ -f "$FULL_FT/config.json" ]] && ! _train_alive; then
    log "full_ft present and train proc gone - proceed"
    break
  fi
  now=$(date -u +%s)
  soft=$(date -u -d "$SOFT_DEADLINE_UTC" +%s)
  if (( now > soft - 3600 )); then
    log "WARN: <60m to soft and train not done; abort"
    echo "aborted_no_train $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /root/logs/r1209_pipeline.aborted
    exit 1
  fi
  _wait_i=$((_wait_i + 1))
  if (( _wait_i % 10 == 0 )); then
    log "still waiting for train.done (poll #$_wait_i)"
  fi
  sleep 30
done

log "waiting for train pid to exit and release GPUs"
for _ in $(seq 1 180); do
  if ! _train_alive; then
    log "train proc gone"
    break
  fi
  sleep 5
done
if _train_alive; then
  log "ERROR: train still alive >15m after train.done; abort"
  echo "aborted_train_stuck $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /root/logs/r1209_pipeline.aborted
  exit 1
fi
sleep 15
log "GPU settle done; finalize full-FT → $MERGED"

if [[ ! -d "$FULL_FT" ]]; then
  log "ERROR: no full_ft dir"
  echo "aborted_no_full_ft $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/root/logs/r1209_pipeline.aborted
  exit 1
fi

if [[ "${SKIP_MERGE:-0}" == "1" && -f "$MERGED/config.json" ]] \
  && ls "$MERGED"/model-*-of-*.safetensors >/dev/null 2>&1; then
  log "SKIP_MERGE=1 — reuse $MERGED"
else
  rm -rf "$MERGED"
  python3 /root/mining_src/s4-h121-f26-full-ft/finalize_full_ft.py \
    --base "$BASE" \
    --full-ft "$FULL_FT" \
    --out "$MERGED" \
    --king "$KING_LOCAL" \
    | tee -a "$LOG"
fi
mkdir -p "$(dirname "$MERGED_LINK")"
ln -sfn "$MERGED" "$MERGED_LINK"
log "merged link $MERGED_LINK → $MERGED"
cp -f "$MERGED/finalize_meta.json" /root/affine_data/r1209_finalize_meta.json 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1209_merge.done

HF_MERGED_REPO=${HF_MERGED_REPO:-unconst/Affine-5czsc2fc98-r1209-fullft-ultralolr}
# p4317: this H200 is known-good; HF public storage full → skip push, local TKC.
SKIP_LOCAL_TKC=${SKIP_LOCAL_TKC:-0}
SKIP_HF_PUSH=${SKIP_HF_PUSH:-1}
if [[ "${SKIP_HF_PUSH}" == "1" ]]; then
  log "SKIP_HF_PUSH=1 — skip push_merged; local TKC only"
  date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1209_hf_skipped.done
elif [[ -n "${HF_TOKEN:-}" ]]; then
  if [[ "$SKIP_LOCAL_TKC" == "1" ]]; then
    log "foreground HF push full-FT → $HF_MERGED_REPO (SKIP_LOCAL_TKC=1 bad-host salvage)"
    python3 /root/mining_src/s4-h1-sft/push_merged.py \
      --merged "$MERGED" \
      --repo "$HF_MERGED_REPO" \
      --public \
      --commit-message "R1209 genesis FullFT bad-host salvage p3191 (not a submission)" \
      --out-meta /root/affine_data/r1209_merged_salvage.json \
      | tee -a /root/logs/r1209_push_merged.nohup "$LOG"
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1209_hf_pushed.done
  else
    log "background HF push full-FT → $HF_MERGED_REPO"
    nohup python3 /root/mining_src/s4-h1-sft/push_merged.py \
      --merged "$MERGED" \
      --repo "$HF_MERGED_REPO" \
      --public \
      --commit-message "R1209 genesis FullFT salvage (TTL insurance; not a submission)" \
      --out-meta /root/affine_data/r1209_merged_salvage.json \
      >>/root/logs/r1209_push_merged.nohup 2>&1 &
    echo $! >/root/logs/r1209_push_merged.pid
  fi
fi

if [[ "$SKIP_LOCAL_TKC" == "1" ]]; then
  log "SKIP_LOCAL_TKC=1 — no serve_three/n80 on blacklisted host; remote n80 next"
  python3 - <<'PY' | tee -a "$LOG"
import json
from datetime import datetime, timezone
meta = {
    "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hyp": "R1209",
    "reason": "executor_blacklist_8f34559f_moe_hang",
    "hf_repo": "unconst/Affine-5czsc2fc98-r1209-fullft-ultralolr",
    "king_repo": "vera6/affine-5g4yy75zuz-t6",
    "king_rev": "8e3f1695e058837ed80fec3238ff439fdc2d0f0e",
    "action": "pull_HF_on_warm_mine_serve_chall_n80_vs_genesis",
    "pass": 3191,
}
open("/root/affine_data/r1209_need_remote_n80.json", "w").write(json.dumps(meta, indent=2) + "\n")
open("/root/logs/r1209_need_remote_n80.stamp", "w").write(meta["utc"] + "\n")
print(json.dumps(meta))
PY
  date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1209_pipeline.done
  log "PIPELINE_DONE (HF salvage; remote n80 pending)"
  exit 0
fi

log "wait teacher.done"
for _ in $(seq 1 240); do
  [[ -f /root/logs/teacher.done ]] && break
  sleep 30
done
test -f /root/logs/teacher.done
test -f /root/logs/tok331102.done

unset CUDA_VISIBLE_DEVICES
log "serve teacher + genesis king + full-FT chall"
TEACHER_REPO=${TEACHER_REPO:-zai-org/GLM-4.5-Air-FP8} \
  TEACHER_REV=${TEACHER_REV:-} \
  KING_REPO="$KING_REPO" \
  KING_REV="$KING_REV" \
  CHALL_REPO="$MERGED" \
  CHALL_REV=local \
  bash /root/mining_src/s3-duel-sim/serve_three.sh | tee -a "$LOG"

date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1209_chall_serve.done
log "CHALL_SERVE_DONE — Reason n80 vs genesis"

N80_MAX_ATTEMPTS=${N80_MAX_ATTEMPTS:-3}
n80_ok=0
for attempt in $(seq 1 "$N80_MAX_ATTEMPTS"); do
  now=$(date -u +%s)
  dead=$(date -u -d "$DEADMAN_UTC" +%s)
  if (( now > dead - 2400 )); then
    log "ABORT: <40m to deadman before n80 attempt $attempt; stop"
    echo "aborted_no_n80_budget attempt=$attempt $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >/root/logs/r1209_pipeline.aborted
    exit 1
  fi
  if [[ -f "$SIM_N80" ]]; then
    log "sim already present — skip relaunch"
    n80_ok=1
    break
  fi
  bh=$(python3 -c 'import hashlib,time; print(hashlib.sha256(f"r1209-{time.time()}".encode()).hexdigest())')
  log "launch n80 sim attempt $attempt/$N80_MAX_ATTEMPTS block_hash=${bh:0:16}… → $SIM_N80"
  set +e
  python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
    --n-turns 80 \
    --block-hash "$bh" \
    --out "$SIM_N80" \
    --progress-out "$PROG" \
    --king-repo "$KING_REPO" \
    --king-rev "$KING_REV" \
    --chall-repo "$MERGED" \
    --teacher-repo "${TEACHER_REPO:-zai-org/GLM-4.5-Air-FP8}" \
    --hotkey "local-r1209-sim" \
    >>"$LOG" 2>&1
  sim_rc=$?
  set -e
  if [[ "$sim_rc" -eq 0 && -f "$SIM_N80" ]]; then
    n80_ok=1
    log "n80 attempt $attempt OK"
    break
  fi
  log "WARN: n80 attempt $attempt failed rc=$sim_rc"
  rm -f "$SIM_N80" "$PROG"
  sleep 30
done
if [[ "$n80_ok" -ne 1 ]]; then
  log "ERROR: n80 failed after $N80_MAX_ATTEMPTS attempts"
  echo "aborted_n80_failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/root/logs/r1209_pipeline.aborted
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1209_sim_n80.done

if [[ -f /root/mining_src/r1-reason-distill/write_reason_decision.py ]]; then
  python3 /root/mining_src/r1-reason-distill/write_reason_decision.py \
    --sim-result "$SIM_N80" --out "$SIM_DEC" --hyp "$HYP" --k-sigma 2.0 \
    || true
fi
log "SIM_DONE dec=$(python3 -c "import json;print(json.load(open('$SIM_DEC')).get('decision'))" 2>/dev/null || echo '?')"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1209_pipeline.done
log "PIPELINE_DONE"
