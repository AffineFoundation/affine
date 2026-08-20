#!/usr/bin/env bash
# p4208: Stage-5 HF push for R1064 CROWN_OK (merge stays on disk; chall vllm optional).
set -euo pipefail
LOG=/root/logs/p4208_r1064_hf_push.nohup
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4208-r1064-push] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
test -n "${HF_TOKEN:-}" || { echo FATAL missing HF_TOKEN; exit 1; }
test -f /tmp/r1064_merged/config.json || { echo FATAL missing /tmp/r1064_merged; exit 1; }
EXP=r1064-vera-offline-dpo-hialpha-midrank-midbeta-midctx-ultrasuperextrasteps-ep4-hilr
test -f /root/mining_src/$EXP/push_r1064_hf.py
python3 /root/mining_src/$EXP/push_r1064_hf.py
echo "[p4208-r1064-push] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
