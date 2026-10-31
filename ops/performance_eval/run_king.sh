#!/bin/bash
# Pod side: serve one king on both GPUs (probe serve.sh flags), run perf_eval.py, stop the servers (exact PIDs). Plain disk only.
# usage: run_king.sh KING SEED K_FULL K_HOLDOUT
set -u
. /data/perf/env.sh
K=$1; SEED=$2; KF=$3; KH=$4; W=/data/perf/runs/$K; CK=/data/perf/ckpt/$K
mkdir -p $W; cd $W
[ -f $CK/.verified ] || { echo "no verified ckpt $K"; exit 3; }
echo $$ > $W/run.pid
PIDS=()
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g nohup nice -n 5 /data/venv/bin/vllm serve "$CK" --host 127.0.0.1 --port 810$g --served-model-name perf \
    --dtype bfloat16 --max-model-len 12288 --gpu-memory-utilization 0.90 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 512 > $W/serve$g.log 2>&1 &
  PIDS+=($!)
done
echo "${PIDS[@]}" > $W/serve.pids
stop() { for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; sleep 10; for p in "${PIDS[@]}"; do kill -9 $p 2>/dev/null; done; }
trap stop EXIT
for i in $(seq 360); do
  ok=0; for g in 0 1; do curl -sf -m 3 http://127.0.0.1:810$g/v1/models >/dev/null && ok=$((ok+1)); done
  [ $ok = 2 ] && break
  for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null || { echo "server died"; tail -5 $W/serve*.log; exit 4; }; done
  sleep 5
done
date +%s > $W/serving_at
cat > $W/plan.json <<J
{"king":"$K","ckpt_dir":"$CK","urls":["http://127.0.0.1:8100","http://127.0.0.1:8101"],"tasks_file":"/data/perf/tasks.private.json",
 "k":{"full":$KF,"holdout":$KH},"seed":"$SEED","concurrency":1100,"grade_workers":48}
J
nice -n 5 /data/venv/bin/python /data/perf/perf_eval.py $W/plan.json $W/result.json > $W/eval.log 2>&1 &
EP=$!; echo $EP > $W/eval.pid
trap 'kill $EP 2>/dev/null; stop' TERM
wait $EP; rc=$?; echo $rc > $W/eval.rc; exit $rc
