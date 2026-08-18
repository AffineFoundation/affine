#!/usr/bin/env bash
# p3836: R767 MERGE_DONE idle on brave → wait R762 SCP_READY+host-done, then solo-relay → R252.
# R252 :40299 flaky — lium exec + SSH retry. Never pkill -f. Leave R782 6,7 alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r767-r252-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r767_resume_after_r762_p3836.log
: >"$LOG"
log() { echo "[p3836-r767] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299
R762_LOG=$LOGDIR/host_relay_r762_resume_after_r761_p3835.log
HUID=gentle-wolf-8c

r252_probe() {
  local out=""
  out=$(timeout 55 lium exec "$HUID" 'if [[ -f /root/logs/r762_scp_ready.done ]] && [[ -f /tmp/r762_merged/config.json ]]; then
    n=$(ls /tmp/r762_merged/model-*-of-*.safetensors 2>/dev/null | wc -l); echo "ok:$n"
  else
    n=$(ls /tmp/r762_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || echo 0); echo "wait:$n"
  fi' 2>/dev/null | grep -E '^(ok|wait):' | tail -1 || true)
  if [[ -n "$out" ]]; then echo "$out"; return 0; fi
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'if [[ -f /root/logs/r762_scp_ready.done ]] && [[ -f /tmp/r762_merged/config.json ]]; then
       n=$(ls /tmp/r762_merged/model-*-of-*.safetensors 2>/dev/null | wc -l); echo "ok:$n"
     else
       n=$(ls /tmp/r762_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || echo 0); echo "wait:$n"
     fi' 2>/dev/null || true)
  if [[ -n "$out" ]]; then echo "$out"; return 0; fi
  echo "sshfail"
}

r762_host_done() {
  if grep -qE 'DONE relay|tar pipe rc=0' "$R762_LOG" 2>/dev/null; then return 0; fi
  if grep -qE 'FATAL tar pipe' "$R762_LOG" 2>/dev/null; then return 2; fi
  if grep -q 'begin tar pipe' "$R762_LOG" 2>/dev/null; then
    if ! pgrep -f 'host_relay_r762_resume_after_r761_p3835' >/dev/null 2>&1 \
       && ! pgrep -f 'tar cf - r762_merged' >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

log "armed: wait R762 SCP_READY (lium/SSH + host-log) then solo-relay R767 (timeout 8h)"
ready=""
for i in $(seq 1 1920); do
  hd=0
  set +e; r762_host_done; hd_rc=$?; set -e
  if [[ "$hd_rc" -eq 2 ]]; then log "FATAL R762 host relay failed"; exit 1
  elif [[ "$hd_rc" -eq 0 ]]; then hd=1; fi
  ready=$(r252_probe)
  if [[ "$ready" == ok:* ]]; then
    shards=${ready#ok:}
    if [[ "${shards:-0}" -ge 16 ]]; then
      log "R762 SCP_READY shards=$shards — begin R767 relay (poll=$i host_done=$hd)"
      break
    fi
  fi
  if [[ "$hd" -eq 1 && "$ready" == sshfail ]]; then
    [[ $((i % 4)) -eq 0 ]] && log "poll i=$i host_DONE but r252=$ready (retry)"
  else
    [[ $((i % 6)) -eq 0 ]] && log "poll i=$i r762=$ready host_done=$hd"
  fi
  if [[ "$i" -eq 1920 ]]; then log "TIMEOUT waiting R762 SCP_READY last=$ready"; exit 1; fi
  sleep 15
done

for i in $(seq 1 180); do
  if ! pgrep -f 'host_relay_r762_resume_after_r761_p3835' >/dev/null 2>&1 \
     && ! pgrep -f 'tar cf - r762_merged' >/dev/null 2>&1; then
    log "R762 host tar/pipe gone — uplink free"
    break
  fi
  [[ $((i % 6)) -eq 0 ]] && log "wait R762 host pipe exit poll=$i"
  sleep 10
done

n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r767_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r767 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

stage_ok=0
for attempt in $(seq 1 30); do
  if ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
      "mkdir -p /root/mining_src/r767-chall /root/logs /root/affine_data" \
    && scp "${SSH_OPTS[@]}" -P "$R252_PORT" \
      "$EXP/lean_chall_n80_r252_gpus45_p3836.sh" \
      "$EXP/wait_r767_after_r762_then_n80_p3836.sh" \
      "root@$R252_HOST:/root/mining_src/r767-chall/" \
    && ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
chmod +x /root/mining_src/r767-chall/*.sh
bash -n /root/mining_src/r767-chall/lean_chall_n80_r252_gpus45_p3836.sh
bash -n /root/mining_src/r767-chall/wait_r767_after_r762_then_n80_p3836.sh
echo SYNTAX_OK
rm -rf /tmp/r767_merged
rm -f /root/logs/r767_scp_ready.done /root/logs/r767_chall_n80_launched.p3836
mkdir -p /tmp/r767_merged
df -h / | tail -1
nohup bash /root/mining_src/r767-chall/wait_r767_after_r762_then_n80_p3836.sh \
  >/root/logs/p3836_r767_wait.nohup 2>&1 &
echo $! >/root/logs/p3836_r767_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3836_r767_wait.pid)"
REMOTE
  then stage_ok=1; log "stage scripts ok attempt=$attempt"; break
  fi
  log "stage scripts fail attempt=$attempt — sleep 20"; sleep 20
done
[[ "$stage_ok" -eq 1 ]] || { log "FATAL could not stage R767 scripts on R252"; exit 3; }

log "begin tar pipe brave→R252 ~66G R767 (SSH retry)"
rc=1
for attempt in $(seq 1 12); do
  set +e
  ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r767_merged" \
    | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
      "cd /tmp && tar xf - && n=\$(ls /tmp/r767_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r767_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r767_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r767_merged"
  rc=$?
  set -e
  log "tar pipe attempt=$attempt rc=$rc"
  [[ "$rc" -eq 0 ]] && break
  log "tar pipe fail — sleep 30 then retry"; sleep 30
done
[[ "$rc" -eq 0 ]] || { log "FATAL tar pipe failed after retries"; exit 4; }
log "DONE relay — waiter queues R767 n80 after R762 decision"
