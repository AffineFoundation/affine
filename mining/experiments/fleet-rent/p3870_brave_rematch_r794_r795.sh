#!/usr/bin/env bash
# p3870: R794/R795 merge ENOSPC → rematch after disk free (keep r783/r784).
# Sequential rematch on idle GPUs 0,1 then 2,3. Never pkill -f.
set -euo pipefail
log() { echo "[p3870-rematch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "START rematch R794 then R795"
df -h / | tail -1
test -f /root/r794/train/adapter/adapter_model.safetensors
test -f /root/r795/train/adapter/adapter_model.safetensors
rm -f /root/logs/r794_merge.done /root/logs/r795_merge.done
rm -rf /tmp/r794_merged /tmp/r795_merged
# R794 on 0,1
nohup bash /root/mining_src/r794-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_merge_brave_gpus01_p3850.sh \
  >/root/logs/p3870_r794_rematch.outer.nohup 2>&1 &
echo $! >/root/logs/p3870_r794_rematch.outer.pid
log "R794 rematch pid=$(cat /root/logs/p3870_r794_rematch.outer.pid)"
# Wait for R794 MERGE_DONE before R795 (disk headroom for 66G write)
for i in $(seq 1 720); do
  if [[ -f /root/logs/r794_merge.done ]] && [[ -f /tmp/r794_merged/config.json ]]; then
    n=$(ls /tmp/r794_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "R794 MERGE_OK shards=$n poll=$i"
      break
    fi
  fi
  (( i % 20 == 0 )) && log "wait R794 poll=$i df=$(df -h / | awk 'NR==2{print $4}')"
  sleep 15
done
[[ -f /root/logs/r794_merge.done ]] || { log "FATAL R794 rematch timeout"; exit 1; }
n=$(ls /tmp/r794_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { log "FATAL R794 incomplete n=$n"; exit 2; }
# R795 on 2,3
nohup bash /root/mining_src/r795-r252-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_merge_brave_gpus23_p3850.sh \
  >/root/logs/p3870_r795_rematch.outer.nohup 2>&1 &
echo $! >/root/logs/p3870_r795_rematch.outer.pid
log "R795 rematch pid=$(cat /root/logs/p3870_r795_rematch.outer.pid)"
for i in $(seq 1 720); do
  if [[ -f /root/logs/r795_merge.done ]] && [[ -f /tmp/r795_merged/config.json ]]; then
    n=$(ls /tmp/r795_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "R795 MERGE_OK shards=$n poll=$i"
      break
    fi
  fi
  (( i % 20 == 0 )) && log "wait R795 poll=$i"
  sleep 15
done
[[ -f /root/logs/r795_merge.done ]] || { log "FATAL R795 rematch timeout"; exit 3; }
n=$(ls /tmp/r795_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { log "FATAL R795 incomplete n=$n"; exit 4; }
df -h / | tail -1
du -sh /tmp/r794_merged /tmp/r795_merged /tmp/r783_merged /tmp/r784_merged
log "DONE rematch R794+R795 — ready host-relay n80"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p3870_r794_r795_rematch.done
