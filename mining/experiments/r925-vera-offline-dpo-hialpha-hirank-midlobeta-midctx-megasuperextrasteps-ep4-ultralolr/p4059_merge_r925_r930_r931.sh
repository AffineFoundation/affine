#!/usr/bin/env bash
# p4059: remerge R925/R930/R931 after wait FATAL (looked for flat adapter_model;
# peft wrote …/train/adapter/). Sequential merges — one 35B base load at a time.
set -euo pipefail
exec >>/root/logs/p4059_merge_r925_r930_r931.nohup 2>&1
echo "[p4059-merge] $(date -u +%Y-%m-%dT%H:%M:%SZ) START"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/merge_lora.py

merge_one() {
  local id="$1"
  local adapter="/root/${id}/train/adapter"
  local out="/tmp/${id}_merged"
  if [[ -f "${out}/model.safetensors.index.json" ]]; then
    echo "[p4059-merge] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${id} already merged → skip"
    date -u +%Y-%m-%dT%H:%M:%SZ >"/root/logs/${id}_merge.done"
    return 0
  fi
  if [[ ! -f "${adapter}/adapter_config.json" || ! -f "${adapter}/adapter_model.safetensors" ]]; then
    echo "[p4059-merge] FATAL ${id} missing adapter under ${adapter}"
    exit 1
  fi
  # stamp train.done if missing (wait FATAL'd before marking)
  if [[ ! -f /root/logs/${id}_train.done ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ >"/root/logs/${id}_train.done"
  fi
  echo "[p4059-merge] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE ${id} adapter=${adapter}"
  rm -rf "$out"
  mkdir -p "$out"
  python3 "$MERGE_PY" --base "$BASE" --adapter "$adapter" --out "$out" \
    >"/root/logs/${id}_merge.nohup" 2>&1
  local n
  n=$(ls "${out}"/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  echo "[p4059-merge] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${id} shards=${n}"
  [[ "$n" -ge 8 ]] || { echo "[p4059-merge] FATAL ${id} too few shards"; exit 1; }
  date -u +%Y-%m-%dT%H:%M:%SZ >"/root/logs/${id}_merge.done"
  echo "[p4059-merge] MERGE_DONE ${id}"
}

merge_one r925
merge_one r930
merge_one r931
echo "[p4059-merge] $(date -u +%Y-%m-%dT%H:%M:%SZ) ALL_DONE"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4059_merge_r925_r930_r931.done
