#!/usr/bin/env bash
# R1191: vera-init → FullFT; n80 king = live vera reign36 (p3007).
# Overlay: upload_and_launch copies this to s4-h121-f26-full-ft/bootstrap_h121.sh.
# Order: pip → DL marsplan (train init + n80 king) → launch full-FT → bg teacher → post_train.
set -euo pipefail

LOG=/root/logs/bootstrap_h121.log
mkdir -p /root/logs /root/hf /root/affine_data /root/mining_src /root/h121 /root/r1191
exec > >(tee -a "$LOG") 2>&1

echo "[bootstrap-r1191] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

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
export HF_TOKEN
export PYTHONPATH=/root/mining_src/affine_pkg:/root/mining_src/r1191-vera-fullft:${PYTHONPATH:-}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[bootstrap-r1191] FATAL: HF_TOKEN missing in /root/mine.env"
  exit 1
fi

nvidia-smi -L || true
NGPU=$(nvidia-smi -L | wc -l)
echo "[bootstrap-r1191] GPU_COUNT=$NGPU"
test "$NGPU" -ge 8
test -s /root/r1191/winner_za_high_l2.jsonl
ln -sfn /root/r1191/winner_za_high_l2.jsonl /root/h121/winner_za_high_l2.jsonl
test -f /root/mining_src/s4-h121-f26-full-ft/train_full.py
test -f /root/mining_src/s4-h121-f26-full-ft/finalize_full_ft.py
test -x /root/mining_src/s4-h121-f26-full-ft/start_h121.sh
test -x /root/mining_src/s4-h121-f26-full-ft/post_train_pipeline.sh
# Prove overlay is R1191 Vera FullFT, not stock H121 Tok.
grep -q "DOWNLOAD vera-init" /root/mining_src/s4-h121-f26-full-ft/bootstrap_h121.sh
grep -q "R1191: Vera-FullFT" /root/mining_src/s4-h121-f26-full-ft/start_h121.sh
grep -q "/root/r1191/train" /root/mining_src/s4-h121-f26-full-ft/post_train_pipeline.sh

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
  "accelerate" \
  "huggingface_hub[hf_transfer]" \
  "hf_transfer" \
  "safetensors" \
  "numpy" \
  "scipy" \
  2>&1 | tee /root/logs/pip_r1191.log | tail -40

python - <<'PY'
import torch, transformers, vllm, accelerate
print("[bootstrap-r1191] VERSIONS",
      "torch", torch.__version__,
      "transformers", transformers.__version__,
      "vllm", vllm.__version__,
      "accelerate", accelerate.__version__)
assert vllm.__version__.startswith("0.22.1"), vllm.__version__
assert transformers.__version__.startswith("5.14"), transformers.__version__
PY

if [[ -x /root/mining_src/s3-duel-sim/patch_b300_sm103_flash_attn.sh ]]; then
  bash /root/mining_src/s3-duel-sim/patch_b300_sm103_flash_attn.sh || true
  date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/b300_flash_patch.done
fi

python3 - <<'PY'
from pathlib import Path
p = Path("/root/mining_src/affine_pkg/evalsrv/vllm_client.py")
if p.is_file():
    txt = p.read_text()
    orig = txt
    txt = txt.replace("httpx.Timeout(180.0, connect=10.0)", "httpx.Timeout(600.0, connect=10.0)")
    txt = txt.replace("httpx.Timeout(360.0, connect=10.0)", "httpx.Timeout(600.0, connect=10.0)")
    txt = txt.replace("httpx.Timeout(480.0, connect=10.0)", "httpx.Timeout(600.0, connect=10.0)")
    txt = txt.replace("for attempt in range(3):", "for attempt in range(5):")
    txt = txt.replace("if attempt == 2:", "if attempt == 4:")
    if txt != orig:
        p.write_text(txt)
        print("[bootstrap-r1191] patched vllm_client timeout=600 retries=5", flush=True)
    else:
        print("[bootstrap-r1191] vllm_client already patched or pattern miss", flush=True)
PY

# Blocking DL: marsplan once (train init + live n80 king).
python - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
repo = "vera6/affine-5g4yy75zuz-t6"
rev = "8e3f1695e058837ed80fec3238ff439fdc2d0f0e"
print("[bootstrap-r1191] DOWNLOAD vera-init start", repo, rev, flush=True)
path = snapshot_download(repo, revision=rev, token=token)
print(f"[bootstrap-r1191] DOWNLOAD vera-init done -> {path}", flush=True)
open("/root/logs/genesis.done", "w").write(path + "\n")
open("/root/r1191/r1191_base.path", "w").write(path + "\n")
# Compat stamps for H121-era gates that look for tok*.done.
open("/root/logs/tok_init.done", "w").write(path + "\n")
open("/root/logs/tok331102.done", "w").write(path + "\n")
print("[bootstrap-r1191] DOWNLOAD vera-king done (same path; n80 king=train init)", flush=True)
assert rev in path, path
PY

export BASE=$(cat /root/r1191/r1191_base.path)
if [[ -f /root/mine.env ]]; then
  grep -q '^export BASE=' /root/mine.env \
    && sed -i "s|^export BASE=.*|export BASE=${BASE}|" /root/mine.env \
    || echo "export BASE=${BASE}" >>/root/mine.env
fi

echo "[bootstrap-r1191] $(date -u +%Y-%m-%dT%H:%M:%SZ) launching R1191 Vera FullFT train BASE=$BASE"
BASE="$BASE" bash /root/mining_src/s4-h121-f26-full-ft/start_h121.sh
touch /root/logs/h121_train_launched.stamp
touch /root/logs/r1191_train_launched.stamp

# Teacher + corpus in background (disk only; GPUs busy with full-FT).
nohup bash -lc '
  set -euo pipefail
  set -a; source /root/mine.env; set +a
  source /root/venv/bin/activate
  export HF_HOME=/root/hf HF_TOKEN
  python - <<PY
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
print("[bootstrap-r1191] DOWNLOAD teacher start", flush=True)
path = snapshot_download("zai-org/GLM-4.5-Air-FP8", token=token)
print(f"[bootstrap-r1191] DOWNLOAD teacher done -> {path}", flush=True)
open("/root/logs/teacher.done", "w").write(path + "\n")
PY
  bash /root/mining_src/s3-duel-sim/sync_corpus.sh || true
' >/root/logs/r1191_extra_dl.nohup 2>&1 &
echo $! >/root/logs/r1191_extra_dl.pid
cp -f /root/logs/r1191_extra_dl.pid /root/logs/h121_extra_dl.pid

nohup bash /root/mining_src/s4-h121-f26-full-ft/post_train_pipeline.sh \
  >/root/logs/r1191_post_train.nohup 2>&1 &
echo $! >/root/logs/r1191_post_train.pid
cp -f /root/logs/r1191_post_train.pid /root/logs/h121_post_train.pid

echo "[bootstrap-r1191] $(date -u +%Y-%m-%dT%H:%M:%SZ) BOOTSTRAP_DONE train=$(cat /root/logs/r1191_train.pid) post=$(cat /root/logs/r1191_post_train.pid)"
