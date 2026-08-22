#!/usr/bin/env bash
# Keep the G sampling replicas alive all night. If a replica is unhealthy for
# 3 consecutive checks: recreate its container and re-load the current G
# adapter from the driver's state.json so rounds keep working.
#
# Health = a real 1-token completion. /v1/models is NOT enough: when the vLLM
# engine core dies the API frontend keeps answering /v1/models while every
# completion 500s, which hid a dead engine for 40+ rounds.
set -uo pipefail
W=/dshare/gad
PORTS=(8001 8002 8003 8004 8005)
GPUS=(0 4 5 6 7)

healthy() { # port
  curl -sf -m 30 "http://127.0.0.1:$1/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-4B","prompt":"hi","max_tokens":1}' \
    >/dev/null 2>&1
}

reload_adapter() { # port
  python3 - "$1" <<'PY' >> "$W/status.log" 2>&1 || true
import json, sys, urllib.request
port = sys.argv[1]
st = json.load(open("/dshare/gad/state.json"))
name, path = st.get("g_serve", ""), st.get("g_lora", "")
if name and path and not name.startswith("Qwen"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/load_lora_adapter",
        data=json.dumps({"lora_name": name, "lora_path": path}).encode(),
        headers={"Content-Type": "application/json"})
    print(f"[babysit_g] {port} reloaded adapter:", name,
          urllib.request.urlopen(req, timeout=120).status)
PY
}

declare -A DOWN
while true; do
  sleep 60
  for i in "${!PORTS[@]}"; do
    p="${PORTS[$i]}"
    if healthy "$p"; then
      DOWN[$p]=0
      continue
    fi
    DOWN[$p]=$(( ${DOWN[$p]:-0} + 1 ))
    [ "${DOWN[$p]}" -lt 3 ] && continue
    echo "$(date -u +%FT%TZ) [babysit_g] $p down 3x; restarting" >> "$W/status.log"
    bash /root/work/serve_h200.sh Qwen/Qwen3-4B "$p" "${GPUS[$i]}" \
      --enforce-eager --enable-lora --max-lora-rank 16 --max-loras 4
    for j in $(seq 1 60); do
      sleep 10
      healthy "$p" && break
    done
    reload_adapter "$p"
    DOWN[$p]=0
  done
done
