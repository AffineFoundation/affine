#!/bin/bash
# Watch loop_reason.log for round summaries; print new ROUND/kept/CKPT lines.
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p 40050 root@REDACTED_IP"
SEEN=0
while true; do
  OUT=$($SSH 'grep -nE "ROUND|kept=|CKPT_SAVED|WARN|Error|Traceback|DONE" /root/work/loop_reason.log' 2>/dev/null)
  N=$(echo "$OUT" | grep -c . )
  if [ "$N" -gt "$SEEN" ]; then
    echo "$OUT" | tail -n $((N - SEEN))
    SEEN=$N
  fi
  ALIVE=$($SSH 'pgrep -f "reason_family_loo[p]" | head -1' 2>/dev/null)
  if [ -z "$ALIVE" ]; then
    echo "LOOP_EXITED"
    $SSH 'tail -n 30 /root/work/loop_reason.log'
    exit 1
  fi
  echo "$OUT" | grep -q "REASON_LOOP_DONE" && exit 0
  sleep 180
done
