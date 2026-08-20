#!/usr/bin/env bash
# p4150: Stage-5 HF push for R1008 CROWN_OK (keep chall:8003 until push starts).
set -euo pipefail
LOG=/root/logs/p4150_r1008_hf_push.nohup
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4150-r1008-push] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
test -n "${HF_TOKEN:-}" || { echo FATAL missing HF_TOKEN; exit 1; }
test -f /tmp/r1008_merged/config.json || { echo FATAL missing /tmp/r1008_merged; exit 1; }
EXP=r1008-vera-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/push_r1008_hf.py
python3 /root/mining_src/$EXP/push_r1008_hf.py
echo "[p4150-r1008-push] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
