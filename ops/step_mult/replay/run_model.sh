#!/bin/bash
# RA/20261102-step-mult pod side: serve one model dir on GPU 0 (perf run_king.sh flags), run perf_eval.py on the replay task subset,
# stop the server (exact PIDs). Stall guard: if no new graded task for 600 s, stop and keep the partial grades.
# usage: run_model.sh CKPT_DIR OUT_DIR SEED K_HOLDOUT
set -u
. /data/perf/env.sh
CK=$1; W=$2; SEED=$3; KH=$4; mkdir -p $W; cd $W
[ -f $CK/.verified ] || { echo "no verified ckpt $CK"; exit 3; }
[ -f $W/result.json ] && { echo "already done"; exit 0; }
CUDA_VISIBLE_DEVICES=0 nohup nice -n 5 /data/venv/bin/vllm serve "$CK" --host 127.0.0.1 --port 8170 --served-model-name perf \
  --dtype bfloat16 --max-model-len 12288 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 512 > $W/serve.log 2>&1 &
SP=$!; echo $SP > $W/serve.pid
stop() { kill $SP 2>/dev/null; [ -n "${EP:-}" ] && kill $EP 2>/dev/null; sleep 10; kill -9 $SP 2>/dev/null; [ -n "${EP:-}" ] && kill -9 $EP 2>/dev/null; true; }
trap stop EXIT
for i in $(seq 360); do curl -sf -m 3 http://127.0.0.1:8170/v1/models >/dev/null && break; kill -0 $SP 2>/dev/null || { echo "server died"; tail -5 $W/serve.log; exit 4; }; sleep 5; done
GW=$(( $(nproc) - 4 ))
cat > $W/plan.json <<J
{"king":"$(basename $CK)","ckpt_dir":"$CK","urls":["http://127.0.0.1:8170"],"tasks_file":"/data/smult/tasks.sub.json",
 "k":{"full":0,"holdout":$KH},"seed":"$SEED","concurrency":1100,"grade_workers":$GW}
J
nice -n 5 /data/venv/bin/python /data/perf/perf_eval.py $W/plan.json $W/result.json > $W/eval.log 2>&1 &
EP=$!; echo $EP > $W/eval.pid
last=0; lastt=$(date +%s)
while kill -0 $EP 2>/dev/null; do
  sleep 20; n=$(wc -l < $W/result.json.grades.jsonl 2>/dev/null || echo 0)
  if [ "$n" != "$last" ]; then last=$n; lastt=$(date +%s); fi
  if [ $(( $(date +%s) - lastt )) -gt 600 ]; then echo "STALL at $n tasks; stopping" >> $W/eval.log; echo stall > $W/stalled; kill $EP 2>/dev/null; sleep 5; kill -9 $EP 2>/dev/null; break; fi
done
wait $EP 2>/dev/null; echo done
