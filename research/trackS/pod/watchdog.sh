#!/usr/bin/env bash
# Track S watchdog: restart a dead vLLM container (docker start keeps the
# original run flags) and note it in the status log. Adapter reloads are the
# driver's job (its stall branch re-registers g_r0 / g_rN / d_vN).
W=/dshare/trackS
export HF_HOME=/dshare/hf
note() { echo "$(date -u +%FT%TZ) [infra] $*" >> "$W/status.log"; }
while true; do
  for port in 8000 8001; do
    name="vllm_Qwen_Qwen3_8-27B_${port}"
    if ! curl -sf -m 5 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)"
      if [ "$state" != "running" ]; then
        note "watchdog: $name is $state -- docker start"
        docker start "$name" >/dev/null 2>&1
      fi
    fi
  done
  if ! pgrep -f "trackS_driver[.]py" >/dev/null 2>&1; then
    note "watchdog: driver dead -- relaunching"
    nohup /root/trainenv/bin/python /root/work/trackS_driver.py >> "$W/driver.log" 2>&1 < /dev/null &
  fi
  sleep 60
done
