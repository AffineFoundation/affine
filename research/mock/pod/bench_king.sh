#!/usr/bin/env bash
# Bench a crowned king on GPUs 6,7: proxy panel (16) then full panel (150).
# usage: bench_king.sh <king_name> <adapter_dir> <reign>
set -uo pipefail
KING="${1:?king name}"; ADAPTER="${2:?adapter dir}"; REIGN="${3:?reign}"
W=/dshare/koth
PYB=/root/benchenv/bin/python
note() { echo "$(date -u +%FT%TZ) [eval] $*" >> "$W/status.log"; }

if [ ! -x "$PYB" ]; then
  note "BENCH SKIPPED reign=$REIGN: benchenv not ready"
  exit 0
fi

# serve the king (base + adapter) on GPU 6
VLLM_IMAGE=vllm/vllm-openai:latest bash /root/work/serve_trackP.sh \
  Qwen/Qwen3.6-35B-A3B 8006 6 --enforce-eager --no-async-scheduling \
  --max-model-len 65536 --enable-lora --max-lora-rank 16 \
  --lora-modules "$KING=$ADAPTER"
for i in $(seq 1 60); do
  curl -sf -m 5 http://127.0.0.1:8006/v1/models >/dev/null 2>&1 && break
  sleep 15
done
if ! curl -sf -m 5 http://127.0.0.1:8006/v1/models >/dev/null 2>&1; then
  note "BENCH reign=$REIGN: server :8006 failed to come up"
  docker rm -f vllm_8006 >/dev/null 2>&1
  exit 1
fi

export PATH=/root/benchenv/bin:$PATH
$PYB /root/work/run_panel.py --pin /root/work/panels/panel_proxy.json \
  --model "$KING" --port 8006 --tag "proxy_r$REIGN" --workers 8 \
  --timeout 7200
$PYB /root/work/run_panel.py --pin /root/work/panels/panel_full.json \
  --model "$KING" --port 8006 --tag "full_r$REIGN" --workers 12 \
  --timeout 21600

docker rm -f vllm_8006 >/dev/null 2>&1
note "BENCH reign=$REIGN complete; :8006 released"
