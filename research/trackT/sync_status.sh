#!/usr/bin/env bash
# Sync Track T status from the pod into the local coordinator-readable log.
# Same mechanism as Tracks M/S (scp + header concat).
POD="-P 40300 root@REDACTED_IP"
LOGS=/home/const/subnet120/research/logs
while true; do
  scp -q -o StrictHostKeyChecking=no $POD:/dshare/trackT/status.log \
    /tmp/tt_status.log 2>/dev/null
  { cat "$LOGS/trackT_header.log" 2>/dev/null;
    sort -s -k1,1 /tmp/tt_status.log 2>/dev/null; } \
    > "$LOGS/trackT_status.log" 2>/dev/null
  sleep 120
done
