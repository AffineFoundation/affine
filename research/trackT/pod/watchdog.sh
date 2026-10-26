#!/usr/bin/env bash
# trackT watchdog.sh: restart dead vLLM containers (docker start keeps the
# original run flags) and relaunch a dead driver. Adapter re-registration is
# the driver's job (its stall branch reloads m_rN / d_vN after restarts).
W=/dshare/trackT
export HF_HOME=/dshare/hf
note() { echo "$(date -u +%FT%TZ) [infra] $*" >> "$W/status.log"; }
declare -A NAMES=(
  [8000]="vllm_Qwen_Qwen3_8-27B_8000"
  [8001]="vllm_dendriteholdings_albedo-qwen3_6-35b-king-genesis_8001"
  [8006]="vllm_dendriteholdings_albedo-qwen3_6-35b-king-genesis_8006"
)
while true; do
  for port in 8000 8001 8006; do
    name="${NAMES[$port]}"
    if ! curl -sf -m 5 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)"
      if [ "$state" != "running" ]; then
        note "watchdog: $name is $state -- docker start"
        docker start "$name" >/dev/null 2>&1
      fi
    fi
  done
  if ! pgrep -f "trackT_driver[.]py" >/dev/null 2>&1; then
    note "watchdog: driver dead -- relaunching"
    nohup /root/trainenv/bin/python /root/work/trackT_driver.py >> "$W/driver.log" 2>&1 < /dev/null &
  fi
  # stale bench lock (crashed bench process): clear after 4h
  if [ -f "$W/bench.lock" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$W/bench.lock") ))
    if [ "$age" -gt 14400 ] && ! pgrep -f "trackT_bench[.]py" >/dev/null 2>&1; then
      note "watchdog: clearing stale bench.lock (age ${age}s, no bench proc)"
      rm -f "$W/bench.lock"
    fi
  fi
  sleep 60
done
