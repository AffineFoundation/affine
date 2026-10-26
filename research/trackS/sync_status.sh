#!/usr/bin/env bash
# Sync Track S status from the pod into the local coordinator-readable log.
# Same mechanism as Track M (scp + header concat).
POD="-P 40299 root@REDACTED_IP"
LOGS=/home/const/subnet120/research/logs
while true; do
  scp -q -o StrictHostKeyChecking=no $POD:/dshare/trackS/status.log \
    /tmp/ts_status.log 2>/dev/null
  { cat "$LOGS/trackS_header.log" 2>/dev/null;
    sort -s -k1,1 /tmp/ts_status.log 2>/dev/null; } \
    > "$LOGS/trackS_status.log" 2>/dev/null
  sleep 120
done
