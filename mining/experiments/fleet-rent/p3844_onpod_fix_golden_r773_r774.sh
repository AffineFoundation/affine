#!/usr/bin/env bash
# Runs ON golden-comet-78 (uploaded by host p3844 fix). Never pkill -f.
set -euo pipefail
log(){ echo "[p3844-onpod] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

stop_pid() {
  local pid=$1 why=${2:-}
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  log "stop pid=$pid ($why)"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}

log "START kill clash + relaunch R774:8002 + R773:8003"

for pf in /root/logs/vllm_chall_r773.pid /root/logs/vllm_chall_r774.pid \
          /root/logs/r773_sim_wvk7.pid /root/logs/r774_sim_wvk7.pid \
          /root/logs/p3822_r773_lean.outer.pid /root/logs/p3822_r774_lean.outer.pid; do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "pidf=$pf"
  rm -f "$pf"
done
for pid in 597674 598725; do
  stop_pid "$pid" "orphan chall"
done

for i in $(seq 1 90); do
  used=$(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 4000 ]]; then
    log "GPUs 4-7 free used_mib=$used"
    break
  fi
  sleep 2
done

rm -f /root/affine_data/r773_sim_result_reign35_wvk7.json \
      /root/affine_data/r773_sim_result_reign35_wvk7_artifact.json \
      /root/affine_data/r773_sim_progress_reign35_wvk7.json \
      /root/affine_data/r773_decision_reign35_wvk7.json \
      /root/affine_data/r774_sim_result_reign35_wvk7.json \
      /root/affine_data/r774_sim_result_reign35_wvk7_artifact.json \
      /root/affine_data/r774_sim_progress_reign35_wvk7.json \
      /root/affine_data/r774_decision_reign35_wvk7.json \
      /root/logs/r773_n80_launched.p3822 \
      /root/logs/r774_n80_launched.p3822

# Ensure R774 lean uses :8002 even if scp lagged
sed -i 's/^CHALL_PORT=8003$/CHALL_PORT=8002/' \
  /root/mining_src/r774-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus45_p3822.sh || true
grep -n '^CHALL_PORT=' \
  /root/mining_src/r774-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus45_p3822.sh \
  /root/mining_src/r773-r252-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus67_p3822.sh

chmod +x /root/mining_src/r774-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus45_p3822.sh
chmod +x /root/mining_src/r773-r252-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus67_p3822.sh

nohup bash /root/mining_src/r774-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus45_p3822.sh \
  >/root/logs/p3844_r774_lean.outer.log 2>&1 &
echo $! >/root/logs/p3844_r774_lean.outer.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r774_n80_launched.p3822
log "armed R774 lean pid=$(cat /root/logs/p3844_r774_lean.outer.pid) :8002"

nohup bash /root/mining_src/r773-r252-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus67_p3822.sh \
  >/root/logs/p3844_r773_lean.outer.log 2>&1 &
echo $! >/root/logs/p3844_r773_lean.outer.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r773_n80_launched.p3822
log "armed R773 lean pid=$(cat /root/logs/p3844_r773_lean.outer.pid) :8003"

for i in $(seq 1 90); do
  h2=$(curl -sf -m 2 http://127.0.0.1:8002/v1/models 2>/dev/null | head -c 80 || true)
  h3=$(curl -sf -m 2 http://127.0.0.1:8003/v1/models 2>/dev/null | head -c 80 || true)
  if [[ -n "$h2" && -n "$h3" ]]; then
    log "BOTH_READY :8002 and :8003 poll=$i"
    echo "8002:$h2"
    echo "8003:$h3"
    exit 0
  fi
  (( i % 15 == 0 )) && log "warming poll=$i h2=${#h2} h3=${#h3}"
  sleep 10
done
log "WARN timeout waiting both ports"
ss -tlnp | grep -E ':800[23]' || true
tail -30 /root/logs/p3844_r774_lean.outer.log || true
tail -30 /root/logs/p3844_r773_lean.outer.log || true
exit 0
