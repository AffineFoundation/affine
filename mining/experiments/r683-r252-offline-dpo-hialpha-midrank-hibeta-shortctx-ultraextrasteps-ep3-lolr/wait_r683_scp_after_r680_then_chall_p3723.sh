#!/usr/bin/env bash
# p3723: wait R683 SCP_READY + R680 n80 finished (4,5 free) + reign34 king → lean chall.
# Never pkill -f. Do not launch while R680 owns :8003.
set -euo pipefail

log() { echo "[p3723-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a /root/logs/p3723_r683_wait.log; }

SCP_DONE=/root/logs/r683_scp_ready.done
KING_DONE=/root/logs/swap_king_reign34_golden_p3719.done
LAUNCHED=/root/logs/r683_chall_n80_launched.p3723
LEAN=/root/mining_src/r683-chall/lean_chall_n80_golden_gpus45_p3723.sh
R680_DONE=/root/logs/r680_reign34_wvk7_pipeline.done
R680_DEC=/root/affine_data/r680_decision_reign34_wvk7.json
mkdir -p /root/logs
: >>/root/logs/p3723_r683_wait.log

r680_clear() {
  # R680 finished (decision/pipeline) OR no live R680 chall on :8003
  [[ -f "$R680_DONE" || -f "$R680_DEC" ]] && return 0
  if ss -lptn 'sport = :8003' 2>/dev/null | grep -q r680_merged; then
    return 1
  fi
  if [[ -f /root/logs/vllm_chall_r680.pid ]]; then
    local p; p=$(cat /root/logs/vllm_chall_r680.pid 2>/dev/null || true)
    if [[ "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; then
      return 1
    fi
  fi
  # if :8003 has any listener that is not yet R683, treat as busy unless empty
  if ss -lptn 'sport = :8003' 2>/dev/null | grep -q LISTEN; then
    if ss -lptn 'sport = :8003' 2>/dev/null | grep -q r683_merged; then
      return 0
    fi
    return 1
  fi
  return 0
}

log "armed: wait R683 SCP + R680 clear + reign34 → chall 4,5/:8003 (timeout 10h)"
for i in $(seq 1 3600); do
  if [[ -f "$LAUNCHED" ]]; then
    log "already launched — exit"
    exit 0
  fi

  if [[ ! -f "$KING_DONE" ]]; then
    code=$(curl -s -o /tmp/king_models_p3723.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
    id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3723.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
    if echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
      date -u +%Y-%m-%dT%H:%M:%SZ >"$KING_DONE"
      log "king already reign34 id=$id — mark KING_DONE"
    fi
  fi

  r680_ok=0
  if r680_clear; then r680_ok=1; fi

  if [[ -f "$SCP_DONE" ]] && [[ -f "$KING_DONE" ]] && [[ "$r680_ok" -eq 1 ]]; then
    if [[ -f /tmp/r683_merged/config.json ]] \
      && [[ $(ls /tmp/r683_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) -ge 16 ]]; then
      sz15=$(stat -c%s /tmp/r683_merged/model-00015-of-00016.safetensors 2>/dev/null || echo 0)
      if [[ "$sz15" -ne 4988775104 ]]; then
        [[ $((i % 6)) -eq 0 ]] && log "SCP marker but shard15 trunc have=$sz15 (i=$i)"
      else
        code=$(curl -s -o /tmp/king_models_p3723b.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
        id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3723b.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
        if [[ "$code" = "200" ]] && echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
          date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
          log "GO king=$id — launch lean chall+v4-n80 (R680 clear)"
          nohup bash "$LEAN" >/root/logs/p3723_r683_lean_outer.nohup 2>&1 &
          echo $! >/root/logs/p3723_r683_lean_outer.pid
          log "lean outer pid=$(cat /root/logs/p3723_r683_lean_outer.pid)"
          exit 0
        else
          log "markers ok but king not ready code=$code id=$id (i=$i)"
        fi
      fi
    else
      log "SCP done marker but merge incomplete (i=$i)"
    fi
  else
    s=missing; k=missing; r=busy
    [[ -f "$SCP_DONE" ]] && s=ok
    [[ -f "$KING_DONE" ]] && k=ok
    [[ "$r680_ok" -eq 1 ]] && r=clear
    n=$(ls /tmp/r683_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ $((i % 6)) -eq 0 ]] && log "poll i=$i scp=$s king=$k r680=$r shards=${n:-0}"
  fi
  sleep 10
done
log "TIMEOUT waiting scp+r680+king"
exit 1
