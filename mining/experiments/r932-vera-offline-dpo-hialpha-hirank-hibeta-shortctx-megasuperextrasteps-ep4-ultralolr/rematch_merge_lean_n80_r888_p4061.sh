#!/usr/bin/env bash
# p4061: R932 TRAIN_DONE sat idle (wait FATAL flat adapter) → merge adapter/ → lean chall+n80 GPUs5,6 :8002
# Never pkill -f. Do not touch teacher:8000 GPU0 or king:8001 GPU4.
set -euo pipefail
exec >>/root/logs/r932_rematch_p4061.nohup 2>&1
echo "[p4061-r932] $(date -u +%Y-%m-%dT%H:%M:%SZ) START rematch merge→lean"

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=/root/hf
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
ADAPTER=/root/r932/train/adapter
MERGE=/tmp/r932_merged
LEAN=/root/mining_src/r932-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r888_gpus56_p4061.sh

test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
test -x "$LEAN" || test -f "$LEAN"
test -f /root/mining_src/s4-h1-sft/merge_lora.py || test -f /root/mining_src/r932-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr/merge_lora.py

MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/r932-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr/merge_lora.py

echo "[p4061-r932] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE start adapter=$ADAPTER"
rm -rf "$MERGE"
mkdir -p "$MERGE"
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out "$MERGE" \
  >/root/logs/r932_merge_p4061.nohup 2>&1
n=$(ls "$MERGE"/model-*-of-*.safetensors | wc -l)
test "$n" -ge 16
test -f "$MERGE/config.json"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r932_merge.done
echo "[p4061-r932] MERGE_DONE shards=$n"

# warm TK check
for i in $(seq 1 30); do
  if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null \
    && curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null; then
    echo "[p4061-r932] TK warm poll=$i"; break
  fi
  sleep 2
done

chmod +x "$LEAN"
nohup bash "$LEAN" >/root/logs/r932_lean_n80_outer_p4061.nohup 2>&1 &
echo $! >/root/logs/r932_lean_n80_outer_p4061.pid
echo "[p4061-r932] LEAN_LAUNCHED pid=$(cat /root/logs/r932_lean_n80_outer_p4061.pid)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r932_rematch_p4061.armed
