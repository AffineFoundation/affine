#!/usr/bin/env bash
# p4031: wait for R926 bootstrap (venv + cryptoDev DL done), then TRAIN R927 on GPUs 2,3
set -euo pipefail
LOG=/root/logs/arm_r927_p4031.log
mkdir -p /root/logs /root/r927
exec > >(tee -a "$LOG") 2>&1
echo "[arm-r927-p4031] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"
EXP=r927-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
# wait up to ~3h for pip+full DL marker
for i in $(seq 1 360); do
  venv_ok=0; dl_ok=0; pkg_ok=0; base_ok=0
  [[ -x /root/venv/bin/python3 ]] && venv_ok=1
  [[ -f /root/logs/cryptodev_dl.done ]] && dl_ok=1
  [[ -f /root/mining_src/affine_pkg/evalsrv/chat.py ]] && pkg_ok=1
  # require at least one large weight shard present
  if [[ -e "$BASE/config.json" ]] && ls "$BASE"/model*.safetensors >/dev/null 2>&1; then base_ok=1; fi
  echo "[arm-r927-p4031] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait venv=$venv_ok dl=$dl_ok pkg=$pkg_ok base=$base_ok iter=$i"
  if [[ "$venv_ok" -eq 1 && "$dl_ok" -eq 1 && "$pkg_ok" -eq 1 && "$base_ok" -eq 1 ]]; then break; fi
  sleep 30
done
[[ -x /root/venv/bin/python3 ]] || { echo FATAL no venv; exit 1; }
[[ -f /root/logs/cryptodev_dl.done ]] || { echo FATAL no cryptodev_dl.done; exit 1; }
[[ -f /root/mining_src/affine_pkg/evalsrv/chat.py ]] || { echo FATAL no affine_pkg evalsrv; exit 1; }
[[ -e "$BASE/config.json" ]] || { echo FATAL no cryptoDev base; exit 1; }
ls "$BASE"/model*.safetensors >/dev/null || { echo FATAL no weight shards; exit 1; }
test -s /root/mining_src/$EXP/dpo_duel_reason.jsonl
test -f /root/mining_src/$EXP/train_dpo.py
test -f /root/mining_src/$EXP/merge_lora.py
chmod +x /root/mining_src/$EXP/*.sh
if [[ -f /root/logs/r927_train.pid ]]; then
  old=$(cat /root/logs/r927_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "[arm-r927-p4031] already training pid=$old"; exit 0
  fi
fi
# clear stale failure markers from premature launch
rm -f /root/logs/r927_train.done /root/logs/r927_merge.done
nohup bash /root/mining_src/$EXP/lean_train_h100_gpus23_p4031.sh \
  >/root/logs/p4031_r927_lean_train.outer.nohup 2>&1 &
echo $! >/root/logs/p4031_r927_lean_train.outer.pid
nohup bash /root/mining_src/$EXP/wait_r927_train_then_merge_p4031.sh \
  >/root/logs/p4031_r927_wait_merge.outer.nohup 2>&1 &
echo $! >/root/logs/p4031_r927_wait_merge.outer.pid
sleep 8
echo "R927_TRAIN_PID=$(cat /root/logs/r927_train.pid 2>/dev/null || echo pending)"
tail -40 /root/logs/r927_lean_warm.log 2>/dev/null || true
tail -20 /root/logs/r927_train.nohup 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4031_r927_armed.done
echo "[arm-r927-p4031] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
