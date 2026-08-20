#!/usr/bin/env bash
# p4177: Stage-5 HF push for R1032 CROWN_OK (merge stays on disk; chall vllm optional).
set -euo pipefail
LOG=/root/logs/p4177_r1032_hf_push.nohup
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4177-r1032-push] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
test -n "${HF_TOKEN:-}" || { echo FATAL missing HF_TOKEN; exit 1; }
test -f /tmp/r1032_merged/config.json || { echo FATAL missing /tmp/r1032_merged; exit 1; }
EXP=r1032-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/push_r1032_hf.py
python3 /root/mining_src/$EXP/push_r1032_hf.py
echo "[p4177-r1032-push] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
