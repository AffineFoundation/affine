#!/usr/bin/env bash
# Sync remote Track B status log to the local coordinator-readable file.
while true; do
  scp -q -P 19050 root@216.48.189.107:/dshare/gad/status.log /tmp/tb_remote_status.log 2>/dev/null
  cat /home/const/subnet120/research/logs/trackB_header.log /tmp/tb_remote_status.log \
    > /home/const/subnet120/research/logs/trackB_status.log 2>/dev/null
  sleep 120
done
