#!/usr/bin/env bash
# p4101: Stage-5 HF push for R959 CROWN_OK (keep chall:8002 until push starts; do not touch :8003).
set -euo pipefail
LOG=/root/logs/p4101_r959_hf_push.nohup
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4101-r959-push] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
test -n "${HF_TOKEN:-}" || { echo FATAL missing HF_TOKEN; exit 1; }
test -f /tmp/r959_merged/config.json || { echo FATAL missing /tmp/r959_merged; exit 1; }
EXP=r959-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
test -f /root/mining_src/$EXP/push_r959_hf.py
python3 /root/mining_src/$EXP/push_r959_hf.py
echo "[p4101-r959-push] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
