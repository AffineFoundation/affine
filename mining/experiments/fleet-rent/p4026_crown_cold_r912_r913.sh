#!/usr/bin/env bash
# p4026: cold-bootstrap empty mine-crown-1 (8×B300) and re-arm stranded
# R912 (GPUs 6,7 Midβ MidCtx) + R913 (GPUs 4,5 MidLoβ ShortCtx) Offline-DPO.
# Soft Mid Mid Soft data from local r886. Never pkill -f. Leave GPUs 0–3 free for TK later.
set -euo pipefail

ROOT=/home/const/subnet120
MINING=$ROOT/mining
EXP=$MINING/experiments/fleet-rent
LOGDIR=$EXP/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/p4026_crown_cold_r912_r913.log
: >"$LOG"
log() { echo "[p4026-crown] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

DST_HOST=${DST_HOST:-95.133.252.28}
DST_PORT=${DST_PORT:-40298}
KNOWN=${KNOWN:-/tmp/mine-crown-1.p4026.known_hosts}
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes
     -p "$DST_PORT" "root@$DST_HOST")
SCP=(scp -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes
     -P "$DST_PORT")

R912=r912-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr
R913=r913-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr
DATA_SRC=$MINING/experiments/r886-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl
test -s "$DATA_SRC"
test "$(wc -l <"$DATA_SRC")" -ge 200

log "START crown cold → R912+R913 host=$DST_HOST:$DST_PORT"

STAGE=$(mktemp -d /tmp/mine-crown-p4026.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/affine_pkg/affine" "$STAGE/affine_pkg/evalsrv" \
         "$STAGE/s3-duel-sim" "$STAGE/s4-h1-sft" \
         "$STAGE/s4-h138-f43-tok-dpo-l2" \
         "$STAGE/$R912" "$STAGE/$R913"

cp -a "$ROOT/affine/affine.toml" "$STAGE/affine_pkg/"
cp -a "$ROOT/affine/affine/." "$STAGE/affine_pkg/affine/"
cp -a "$ROOT/affine/evalsrv/." "$STAGE/affine_pkg/evalsrv/"
cp -a "$MINING/experiments/s3-duel-sim/"*.sh "$STAGE/s3-duel-sim/"
cp -a "$MINING/experiments/s3-duel-sim/"*.py "$STAGE/s3-duel-sim/" 2>/dev/null || true
cp -a "$MINING/experiments/s4-h1-sft/merge_lora.py" "$STAGE/s4-h1-sft/"
cp -a "$MINING/experiments/s4-h1-sft/salvage_adapter.py" "$STAGE/s4-h1-sft/" 2>/dev/null || true
cp -a "$MINING/experiments/s4-h138-f43-tok-dpo-l2/train_dpo.py" "$STAGE/s4-h138-f43-tok-dpo-l2/"
for d in "$R912" "$R913"; do
  cp -a "$MINING/experiments/$d/"*.sh "$STAGE/$d/"
  cp -a "$MINING/experiments/$d/train_dpo.py" "$STAGE/$d/"
done
cp -a "$DATA_SRC" "$STAGE/dpo_duel_reason.jsonl"

# Pod-side cold bootstrap: pip → DL vera king → launch R912+R913 lean trains + waiters.
cat >"$STAGE/bootstrap_crown_r912_r913_p4026.sh" <<'POD'
#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/bootstrap_crown_r912_r913_p4026.log
mkdir -p /root/logs /root/hf /root/affine_data /root/mining_src /root/r912 /root/r913 /root/r886
exec > >(tee -a "$LOG") 2>&1
echo "[bootstrap-crown-p4026] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_TOKEN

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo FATAL missing HF_TOKEN; exit 1
fi

nvidia-smi -L || true
NGPU=$(nvidia-smi -L | wc -l)
echo "[bootstrap-crown-p4026] GPU_COUNT=$NGPU"
test "$NGPU" -ge 8

test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
test -f /root/mining_src/s4-h1-sft/merge_lora.py
test -s /root/r886/dpo_duel_reason.jsonl

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -d /root/venv ]]; then
  uv venv /root/venv --python 3.12
fi
# shellcheck disable=SC1091
source /root/venv/bin/activate

uv pip install \
  "torch==2.11.0" \
  "transformers==5.14.1" \
  "vllm==0.22.1" \
  "peft" \
  "accelerate" \
  "httpx" \
  "huggingface_hub[hf_transfer]" \
  "hf_transfer" \
  "safetensors" \
  "numpy" \
  "scipy" \
  2>&1 | tee /root/logs/pip_crown_p4026.log | tail -40

python - <<'PY'
import torch, transformers, vllm, peft
print("[bootstrap-crown-p4026] VERSIONS",
      "torch", torch.__version__,
      "transformers", transformers.__version__,
      "vllm", vllm.__version__,
      "peft", peft.__version__)
assert vllm.__version__.startswith("0.22.1"), vllm.__version__
assert transformers.__version__.startswith("5.14"), transformers.__version__
PY

# Live king parent (reign36 vera).
python - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
repo = "vera6/affine-5g4yy75zuz-t6"
rev = "8e3f1695e058837ed80fec3238ff439fdc2d0f0e"
print("[bootstrap-crown-p4026] DOWNLOAD king start", repo, rev, flush=True)
path = snapshot_download(repo, revision=rev, token=token)
print("[bootstrap-crown-p4026] DOWNLOAD king done", path, flush=True)
open("/root/logs/vera_king_dl.done", "w").write(path + "\n")
PY

BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
test -e "$BASE/config.json"

# Soft Mid Mid Soft preference rows for both axes (lean_train also accepts r901).
mkdir -p /root/r901 /root/r912 /root/r913
cp -f /root/r886/dpo_duel_reason.jsonl /root/r912/dpo_duel_reason.jsonl
cp -f /root/r886/dpo_duel_reason.jsonl /root/r913/dpo_duel_reason.jsonl
cp -f /root/r886/dpo_duel_reason.jsonl /root/r901/dpo_duel_reason.jsonl

chmod +x /root/mining_src/r912-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/*.sh
chmod +x /root/mining_src/r913-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr/*.sh

# R913 GPUs 4,5 then R912 GPUs 6,7 (lean trains set SKIP_LOCAL_TKC).
nohup bash /root/mining_src/r913-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus45_p4017.sh \
  >/root/logs/p4026_r913_lean_train.outer.nohup 2>&1 &
echo $! >/root/logs/p4026_r913_lean_train.outer.pid
sleep 2
nohup bash /root/mining_src/r912-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus67_p4017.sh \
  >/root/logs/p4026_r912_lean_train.outer.nohup 2>&1 &
echo $! >/root/logs/p4026_r912_lean_train.outer.pid

# Wait→merge (n80 waiters need TK later — leave 0–3 free).
nohup bash /root/mining_src/r913-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr/wait_r913_train_then_merge_p4017.sh \
  >/root/logs/p4026_r913_wait_merge.nohup 2>&1 &
echo $! >/root/logs/p4026_r913_wait_merge.pid
nohup bash /root/mining_src/r912-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/wait_r912_train_then_merge_p4017.sh \
  >/root/logs/p4026_r912_wait_merge.nohup 2>&1 &
echo $! >/root/logs/p4026_r912_wait_merge.pid

sleep 5
echo "R913_TRAIN_PID=$(cat /root/logs/r913_train.pid 2>/dev/null || echo missing)"
echo "R912_TRAIN_PID=$(cat /root/logs/r912_train.pid 2>/dev/null || echo missing)"
echo "R913_WAIT=$(cat /root/logs/p4026_r913_wait_merge.pid)"
echo "R912_WAIT=$(cat /root/logs/p4026_r912_wait_merge.pid)"
tail -20 /root/logs/r913_lean_warm.log 2>/dev/null || true
tail -20 /root/logs/r912_lean_warm.log 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4026_crown_r912_r913_armed.done
echo "[bootstrap-crown-p4026] ARMED R912+R913 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
POD
chmod +x "$STAGE/bootstrap_crown_r912_r913_p4026.sh"

TAR=/tmp/mine-crown-p4026-stack.tar.gz
tar -C "$STAGE" -czf "$TAR" .
ls -lh "$TAR" | tee -a "$LOG"

# shellcheck disable=SC1091
set -a
source "$MINING/.env"
set +a
ENV_TMP=$(mktemp /tmp/mine-crown.env.XXXXXX)
umask 077
{
  echo "HF_TOKEN=${HF_TOKEN}"
  echo "HF_HOME=/root/hf"
  echo "HF_HUB_ENABLE_HF_TRANSFER=1"
  echo "HF_XET_HIGH_PERFORMANCE=1"
  echo "AFFINE_DATA_DIR=/root/affine_data"
} >"$ENV_TMP"
chmod 600 "$ENV_TMP"

"${SSH[@]}" 'mkdir -p /root/mining_src /root/affine_data /root/logs /root/hf /root/r886 /root/r912 /root/r913'
"${SCP[@]}" "$TAR" "root@${DST_HOST}:/tmp/mine-crown-p4026-stack.tar.gz"
"${SCP[@]}" "$ENV_TMP" "root@${DST_HOST}:/root/mine.env"
rm -f "$ENV_TMP"

"${SSH[@]}" 'set -e
  tar -C /root/mining_src -xzf /tmp/mine-crown-p4026-stack.tar.gz
  # data landed inside mining_src; move Soft Mid Mid Soft rows to /root/r886
  if [[ -f /root/mining_src/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/dpo_duel_reason.jsonl /root/r886/dpo_duel_reason.jsonl
  fi
  chmod 600 /root/mine.env
  chmod +x /root/mining_src/bootstrap_crown_r912_r913_p4026.sh \
           /root/mining_src/s3-duel-sim/*.sh \
           /root/mining_src/r912-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/*.sh \
           /root/mining_src/r913-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr/*.sh
  test -f /root/mining_src/affine_pkg/affine/score.py
  test -s /root/r886/dpo_duel_reason.jsonl
  echo STACK_UPLOAD_OK
  nohup bash /root/mining_src/bootstrap_crown_r912_r913_p4026.sh \
    >/root/logs/bootstrap_crown_r912_r913_p4026.outer.nohup 2>&1 &
  echo $! >/root/logs/bootstrap_crown_r912_r913_p4026.outer.pid
  echo BOOTSTRAP_PID=$(cat /root/logs/bootstrap_crown_r912_r913_p4026.outer.pid)
  sleep 3
  head -n 30 /root/logs/bootstrap_crown_r912_r913_p4026.log 2>/dev/null || \
    head -n 30 /root/logs/bootstrap_crown_r912_r913_p4026.outer.nohup || true
'

rm -f "$TAR"
log "UPLOAD_AND_LAUNCH_OK"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p4026_crown_cold_r912_r913.done"

# Replace needs_axis_uploader stamp with real bootstrap note.
python3 - <<PY
import json, time
from pathlib import Path
p = Path("$EXP/artifacts/bootstrapped/rented_mine-crown-1.json.bootstrapped")
p.write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "pass": 4026,
  "name": "mine-crown-1",
  "axis": "r912+r913-offline-dpo",
  "host": "$DST_HOST",
  "port": $DST_PORT,
  "note": "cold bootstrap LIVE: pip→vera DL→R913@4,5 + R912@6,7 Soft Mid Mid Soft; GPUs 0-3 free for TK",
}, indent=2) + "\n")
print("BOOTSTRAP_STAMP", p)
PY

log "DONE stamp updated; poll /root/logs/bootstrap_crown_r912_r913_p4026.log on crown"
