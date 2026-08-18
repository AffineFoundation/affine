#!/usr/bin/env bash
# p3872: free lunar R790/R789 idle REFUTE challs; parallel×2 size-checked
# host-relay R794+R795 MERGE_DONE brave→lunar; arm wait→n80. Leave R784×4.
# Never pkill -f. Never touch teacher/king.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r794_r795_brave_to_lunar_p3872.log
: >"$LOG"
log() { echo "[p3872-relay] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
LUNAR_HOST=150.136.46.118
LUNAR_PORT=20299
NPARA=2

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

log "START free lunar R790/R789 challs + relay R794 then R795"

# Free idle REFUTE challs by exact PID (never pkill -f)
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
log(){ echo "[p3872-lunar] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill $pid"; kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
# Exact chall PIDs from p3872 inspect + any wait/sim for r789/r790
for pid in 764445 766959; do stop_pid "$pid"; done
while read -r pid cmd; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "$cmd" | grep -qE 'r78[90]|wait_r78[90]|vllm_chall_r78[90]|r78[90]_sim' || continue
  echo "$cmd" | grep -qE 'GLM-4.5|tammy2|Affine-5hmwh' && continue
  stop_pid "$pid"
done < <(ps -eo pid=,args=)
rm -f /root/logs/r794_scp_ready.done /root/logs/r795_scp_ready.done
rm -rf /tmp/r794_merged /tmp/r795_merged
mkdir -p /tmp/r794_merged /tmp/r795_merged /root/logs /root/mining_src/r794-chall /root/mining_src/r795-chall
# Drop old REFUTE merges to reclaim space (optional; keep if disk tight later)
# leave r789/r790 merges for now — 4.8T free
df -h / | tail -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8
REMOTE

# Deploy lean+wait scripts
scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" \
  "$ROOT/experiments/r794-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_lunar_gpus45_p3872.sh" \
  "$ROOT/experiments/r794-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/wait_r794_scp_then_chall_p3872.sh" \
  "root@$LUNAR_HOST:/root/mining_src/r794-chall/"
scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" \
  "$ROOT/experiments/r795-r252-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_lunar_gpus67_p3872.sh" \
  "$ROOT/experiments/r795-r252-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/wait_r795_scp_then_chall_p3872.sh" \
  "root@$LUNAR_HOST:/root/mining_src/r795-chall/"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
  'chmod +x /root/mining_src/r794-chall/*.sh /root/mining_src/r795-chall/*.sh
   nohup bash /root/mining_src/r794-chall/wait_r794_scp_then_chall_p3872.sh >/root/logs/wait_r794_scp_then_chall_p3872.outer.nohup 2>&1 &
   echo $! >/root/logs/wait_r794_scp_then_chall_p3872.outer.pid
   nohup bash /root/mining_src/r795-chall/wait_r795_scp_then_chall_p3872.sh >/root/logs/wait_r795_scp_then_chall_p3872.outer.nohup 2>&1 &
   echo $! >/root/logs/wait_r795_scp_then_chall_p3872.outer.pid
   echo waiters=$(cat /root/logs/wait_r794_scp_then_chall_p3872.outer.pid) $(cat /root/logs/wait_r795_scp_then_chall_p3872.outer.pid)'

xfer_one() {
  local hypo=$1 f=$2 want=$3
  local dest="/tmp/${hypo}_merged"
  local attempt rc got
  for attempt in 1 2 3 4 5; do
    log "PIPE $hypo $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/${hypo}_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
        "mkdir -p $dest && cat > $dest/$f.tmp && mv -f $dest/$f.tmp $dest/$f"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "PIPE fail $hypo $f rc=$rc"
      sleep $((attempt * 3))
      continue
    fi
    got=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
      "stat -c%s $dest/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
    got=${got//[^0-9]/}; got=${got:-0}
    if [[ "$got" == "$want" ]]; then
      log "PIPE ok $hypo $f ($got)"
      return 0
    fi
    log "PIPE size mismatch $hypo $f got=$got want=$want"
    sleep $((attempt * 3))
  done
  log "FATAL $hypo $f"
  return 1
}

relay_hypo() {
  local hypo=$1
  log "=== relay $hypo parallel×$NPARA ==="
  mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "cd /tmp/${hypo}_merged && for f in model-*-of-*.safetensors model-visual-restored.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json README.md; do
       [[ -f \$f ]] && printf '%s %s\n' \"\$f\" \"\$(stat -c%s \"\$f\")\"
     done")
  log "$hypo src_files=${#SRC_LINES[@]}"
  [[ "${#SRC_LINES[@]}" -ge 17 ]] || { log "FATAL $hypo source incomplete count=${#SRC_LINES[@]}"; return 1; }

  need_list=()
  for line in "${SRC_LINES[@]}"; do
    f=${line%% *}; want=${line##* }
    have=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
      "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
    have=${have//[^0-9]/}; have=${have:-0}
    if [[ "$have" == "$want" ]]; then
      log "KEEP $hypo $f ($have)"
      continue
    fi
    if ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
        "test -f /tmp/${hypo}_merged/$f.tmp && echo yes || echo no" 2>/dev/null | grep -q yes; then
      log "BUSY skip $hypo $f"
      continue
    fi
    log "NEED $hypo $f (have=$have want=$want)"
    need_list+=("$f:$want")
  done
  log "$hypo need_count=${#need_list[@]}"

  fail=0; active=0; pids=()
  for item in "${need_list[@]}"; do
    f=${item%%:*}; want=${item##*:}
    while [[ "$active" -ge "$NPARA" ]]; do
      if ! wait -n; then fail=1; fi
      active=0; live=()
      for p in "${pids[@]:-}"; do
        if kill -0 "$p" 2>/dev/null; then live+=("$p"); active=$((active + 1)); fi
      done
      pids=("${live[@]:-}")
    done
    xfer_one "$hypo" "$f" "$want" &
    pids+=("$!"); active=$((active + 1))
  done
  for p in "${pids[@]:-}"; do
    if ! wait "$p"; then fail=1; fi
  done

  local manif=/tmp/p3872_${hypo}_sizes.txt
  printf '%s\n' "${SRC_LINES[@]}" >"$manif"
  scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" "$manif" "root@$LUNAR_HOST:/tmp/p3872_${hypo}_sizes.txt"
  stamp_out=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "hypo=$hypo bash -s" <<'REMOTE'
set -euo pipefail
ok=1
while read -r f want; do
  [[ -z "${f:-}" ]] && continue
  got=$(stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0)
  if [[ "$got" != "$want" ]]; then
    echo "bad:$f:got=$got:want=$want"
    ok=0
  fi
done < /tmp/p3872_${hypo}_sizes.txt
n=$(ls /tmp/${hypo}_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
vis=0; [[ -f /tmp/${hypo}_merged/model-visual-restored.safetensors ]] && vis=1
cfg=0; [[ -f /tmp/${hypo}_merged/config.json ]] && cfg=1
if [[ "$ok" -eq 1 && "$n" -ge 16 && "$vis" -eq 1 && "$cfg" -eq 1 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/${hypo}_scp_ready.done
  # hypo is r794 / r795 — stamp path uses that name
  echo "ok:$n:vis=$vis:$(du -sh /tmp/${hypo}_merged | awk '{print $1}')"
else
  echo "partial:$n:vis=$vis:cfg=$cfg:ok=$ok"
fi
REMOTE
)
  log "stamp_check $hypo $stamp_out"
  [[ "$stamp_out" == ok:* ]] || { log "FATAL $hypo verify fail=$fail stamp=$stamp_out"; return 1; }
  log "DONE $hypo SCP_READY"
  return 0
}

relay_hypo r794
relay_hypo r795

log "DONE dual parallel brave→lunar — waiters should arm lean challs"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r794_r795_brave_to_lunar_p3872.done"
exit 0
