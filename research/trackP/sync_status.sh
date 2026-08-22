#!/usr/bin/env bash
# Sync remote Track P status log to the local coordinator-readable file.
# Same mechanism as Track B: scp the pod's status.log, prepend the header.
while true; do
  scp -q -P 40301 root@86.38.182.105:/dshare/gad/status.log /tmp/tp_remote_status.log 2>/dev/null
  cat /home/const/subnet120/research/logs/trackP_header.log /tmp/tp_remote_status.log \
    > /home/const/subnet120/research/logs/trackP_status.log 2>/dev/null
  sleep 120
done
