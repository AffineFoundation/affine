#!/usr/bin/env bash
# p4045: R927 TRAIN_DONE but wait FATAL — peft wrote adapter under
# /root/r927/train/adapter/ while wait checked flat …/train/adapter_model.safetensors
# and merge used --adapter …/train. Fix path → merge on free GPUs 2,3.
# Never pkill -f. Do not touch R926/R933/R934 trains on other GPUs.
set -euo pipefail
exec >>/root/logs/p4045_r927_merge.nohup 2>&1

source /root/venv/bin/activate
set -a; source /root/mine.env; set +a
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_TOKEN || true

BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
ADAPTER=/root/r927/train/adapter
OUT=/tmp/r927_merged
MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
[[ -f /root/mining_src/r927-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/merge_lora.py ]] \
  && MERGE_PY=/root/mining_src/r927-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/merge_lora.py

log() { echo "[p4045-r927] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "START adapter-path fix merge"
test -f "$ADAPTER/adapter_model.safetensors" || { log "FATAL missing $ADAPTER/adapter_model.safetensors"; exit 1; }
test -f "$BASE/config.json" || { log "FATAL missing base"; exit 1; }
test -f "$MERGE_PY" || { log "FATAL missing merge_lora.py"; exit 1; }

# Mark train done if waiter never did (peft subdir)
if [[ ! -f /root/logs/r927_train.done ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r927_train.done
  log "stamped r927_train.done (adapter under train/adapter/)"
fi

if [[ -f /root/logs/r927_merge.done ]] && [[ $(ls "$OUT"/model-*-of-*.safetensors 2>/dev/null | wc -l) -ge 16 ]]; then
  log "MERGE already done shards=$(ls "$OUT"/model-*-of-*.safetensors | wc -l)"
  exit 0
fi

# Free VRAM on 2,3 for merge (train slot)
export CUDA_VISIBLE_DEVICES=2,3
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  log "wait VRAM2+3 used_mib=$used iter=$i"
  [[ "$used" -lt 4096 ]] && break
  sleep 2
done

rm -rf "$OUT"
mkdir -p "$OUT"
log "MERGE start base=$BASE adapter=$ADAPTER → $OUT"
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out "$OUT"
n=$(ls "$OUT"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { log "FATAL shards=$n"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r927_merge.done
log "MERGE done shards=$n"
ls -lah "$OUT" | head -30
