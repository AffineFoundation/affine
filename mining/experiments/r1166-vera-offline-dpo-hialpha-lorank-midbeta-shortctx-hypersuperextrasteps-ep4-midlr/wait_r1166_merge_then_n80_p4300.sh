#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4300_r1166_merge_then_n80.nohup
mkdir -p /root/logs; exec > >(tee -a "$LOG") 2>&1
echo "[p4300-r1166-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START (merge already done; TP1 relaunch)"
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
LEAN=/root/mining_src/r1166-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-midlr/lean_chall_n80_r339_gpu6_tp1_p4300.sh
chmod +x "$LEAN"; bash "$LEAN"
