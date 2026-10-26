#!/bin/bash
# Poll the reason-arm pod: server health + EXPERIENCE_READY marker.
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p 40050 root@REDACTED_IP"
while true; do
  OUT=$($SSH 'T=$(curl -sf -m 5 http://127.0.0.1:8010/v1/models >/dev/null && echo OK || echo DOWN); M=$(curl -sf -m 5 http://127.0.0.1:8021/v1/models >/dev/null && echo OK || echo DOWN); D=$(test -e /root/work/EXPERIENCE_READY && echo YES || echo NO); echo "teacher=$T miner=$M data=$D"' 2>&1)
  echo "$(date -u +%H:%M:%S) $OUT"
  if echo "$OUT" | grep -q "data=YES"; then
    echo "DATA_READY"
    exit 0
  fi
  sleep 300
done
