#!/usr/bin/env bash
# p4049 sidecar: fire v4 n80 once R252 chall:8002 + vera king:8001 are READY.
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf AFFINE_DATA_DIR=/root/affine_data
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r3_merged
CHALL_PORT=8002
SIM_N80=/root/affine_data/r252_r3_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r252_r3_sim_progress_reign36_wvk7.json
kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo "FATAL king=$kid"; exit 5; }
curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null
BLOCK_HASH=$(python3 -c 'import hashlib,time; print(hashlib.sha256(f"r252-r3-reign36-wvk7-p4049-{time.time()}".encode()).hexdigest())')
rm -f "$SIM_N80" "$PROG"
nohup env -u HF_TOKEN PYTHONPATH="/root/mining_src/affine_pkg" AFFINE_DATA_DIR=/root/affine_data HF_HOME=/root/hf \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" --king-repo "$kid" --king-rev "$KING_REV" \
  --chall-repo "$MERGE_DIR" --chall-rev local --chall-port "$CHALL_PORT" \
  --n-turns 80 --hotkey local-r252-r3-reign36-wvk7 --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" --progress-out "$PROG" --save-artifact \
  >/root/logs/r252_r3_sim_n80_wvk7.nohup 2>&1 &
echo $! >/root/logs/r252_r3_sim_n80_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r252_r3_n80_launched.p4049
echo "launched:$(cat /root/logs/r252_r3_sim_n80_wvk7.pid) king=$kid bh=${BLOCK_HASH:0:16}"
