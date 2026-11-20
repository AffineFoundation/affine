#!/bin/bash
# Stop the model server by its exact PID (process-group leader started by serve.sh; its DP engine children exit with it).
W=/data/bench/serve; P=$(cat $W/serve.pid 2>/dev/null) || exit 0
[ -n "$P" ] && kill -0 $P 2>/dev/null || { rm -f $W/serve.pid; exit 0; }
kill -TERM -- -$P 2>/dev/null || kill $P
for i in $(seq 60); do kill -0 $P 2>/dev/null || break; sleep 1; done
kill -0 $P 2>/dev/null && kill -KILL -- -$P 2>/dev/null
rm -f $W/serve.pid $W/serving_at; echo stopped $P
