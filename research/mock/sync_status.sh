#!/usr/bin/env bash
# Sync Track M status from BOTH boxes (eval + miner) into the local
# coordinator-readable log, merged by timestamp. Same mechanism as the
# other tracks (scp + header concat).
EVAL="-P 40301 root@86.38.182.105"
MINER="-P 40301 root@204.9.206.214"
LOGS=/home/const/subnet120/research/logs
while true; do
  scp -q $EVAL:/dshare/koth/status.log /tmp/tm_eval_status.log 2>/dev/null
  scp -q $MINER:/dshare/koth/status.log /tmp/tm_miner_status.log 2>/dev/null
  { cat "$LOGS/trackM_header.log";
    sort -m /tmp/tm_eval_status.log /tmp/tm_miner_status.log 2>/dev/null \
      | sort -s -k1,1; } > "$LOGS/trackM_status.log" 2>/dev/null
  sleep 120
done
