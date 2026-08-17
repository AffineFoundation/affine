#!/usr/bin/env bash
# p3714: wait R679 SCP_READY + R681 n80 finished (4,5 free) + reign34 king → lean chall.
# Never pkill -f. Do not launch while R681 owns :8003.
set -euo pipefail

log() { echo "[p3714-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a /root/logs/p3714_r679_wait.log; }

SCP_DONE=/root/logs/r679_scp_ready.done
KING_DONE=/root/logs/swap_king_reign34_golden_p3711.done
LAUNCHED=/root/logs/r679_chall_n80_launched.p3714
LEAN=/root/mining_src/r679-chall/lean_chall_n80_golden_gpus45_p3714.sh
R681_DONE=/root/logs/r681_reign34_wvk7_pipeline.done
R681_DEC=/root/affine_data/r681_decision_reign34_wvk7.json
mkdir -p /root/logs
: >>/root/logs/p3714_r679_wait.log

r681_clear() {
  # R681 finished (decision/pipeline) OR no live R681 chall on :8003
  [[ -f "$R681_DONE" || -f "$R681_DEC" ]] && return 0
  if ss -lptn 'sport = :8003' 2>/dev/null | grep -q r681_merged; then
    return 1
  fi
  # if :8003 empty and no r681 vllm pidfile live, treat as clear
  if [[ -f /root/logs/vllm_chall_r681.pid ]]; then
    local p; p=$(cat /root/logs/vllm_chall_r681.pid 2>/dev/null || true)
    if [[ "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; then
      return 1
    fi
  fi
  return 0
}

log "armed: wait R679 SCP + R681 clear + reign34 → chall 4,5/:8003 (timeout 8h)"
for i in $(seq 1 2880); do
  if [[ -f "$LAUNCHED" ]]; then
    log "already launched — exit"
    exit 0
  fi

  if [[ ! -f "$KING_DONE" ]]; then
    code=$(curl -s -o /tmp/king_models_p3714.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
    id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3714.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
    if echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
      date -u +%Y-%m-%dT%H:%M:%SZ >"$KING_DONE"
      log "king already reign34 id=$id — mark KING_DONE"
    fi
  fi

  r681_ok=0
  if r681_clear; then r681_ok=1; fi

  if [[ -f "$SCP_DONE" ]] && [[ -f "$KING_DONE" ]] && [[ "$r681_ok" -eq 1 ]]; then
    if [[ -f /tmp/r679_merged/config.json ]] \
      && [[ $(ls /tmp/r679_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) -ge 16 ]]; then
      # exact size gate for shard15
      sz15=$(stat -c%s /tmp/r679_merged/model-00015-of-00016.safetensors 2>/dev/null || echo 0)
      if [[ "$sz15" -ne 4988775104 ]]; then
        [[ $((i % 6)) -eq 0 ]] && log "SCP marker but shard15 trunc have=$sz15 (i=$i)"
      else
        code=$(curl -s -o /tmp/king_models_p3714b.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
        id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3714b.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
        if [[ "$code" = "200" ]] && echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
          date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
          log "GO king=$id — launch lean chall+v4-n80 (R681 clear)"
          nohup bash "$LEAN" >/root/logs/p3714_r679_lean_outer.nohup 2>&1 &
          echo $! >/root/logs/p3714_r679_lean_outer.pid
          log "lean outer pid=$(cat /root/logs/p3714_r679_lean_outer.pid)"
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
    [[ "$r681_ok" -eq 1 ]] && r=clear
    n=$(ls /tmp/r679_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ $((i % 6)) -eq 0 ]] && log "poll i=$i scp=$s king=$k r681=$r shards=${n:-0}"
  fi
  sleep 10
done
log "TIMEOUT waiting scp+r681+king"
exit 1
