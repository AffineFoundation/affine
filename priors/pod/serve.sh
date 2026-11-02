#!/bin/bash
# Prior-table pod: serve checkpoint CKPT on every GPU, one vLLM replica per GPU (ports 8100+gpu), loopback only.
# Same vLLM flags as the live probe pod's serve.sh (bf16, max-model-len 12288, prefix caching, max-num-seqs 512, gmu 0.90, no mm).
# usage: serve.sh CKPT_ID [GMU]      stop: serve.sh --stop      (kills only the exact PIDs recorded in /opt/prior/serve/*.pid)
set -u
D=/opt/prior; S=$D/serve; mkdir -p $S
stop() {
  for f in $S/r*.pid; do [ -f "$f" ] || continue; p=$(cat "$f"); kill "$p" 2>/dev/null; done
  for _ in $(seq 60); do alive=0; for f in $S/r*.pid; do [ -f "$f" ] && kill -0 "$(cat $f)" 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 2; done
  for f in $S/r*.pid; do [ -f "$f" ] || continue; p=$(cat "$f"); kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; rm -f "$f"; done
  rm -f $S/SERVING.json
}
if [ "${1:-}" = "--stop" ]; then stop; echo stopped; exit 0; fi
CK=$1; GMU=${2:-0.90}; M=$D/ckpt/$CK
# optional overrides (e.g. MAX_MODEL_LEN=16384 when the gate's max output moves 8192 -> 12288): /opt/prior/serve/serve.env
[ -f $S/serve.env ] && . $S/serve.env
MML=${MAX_MODEL_LEN:-12288}
[ -f $M/.verified ] || { echo "ckpt $CK not verified"; exit 3; }
stop
NG=$(nvidia-smi -L | wc -l)
export VLLM_LOGGING_LEVEL=WARNING HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TMPDIR=$D/tmp
for g in $(seq 0 $((NG-1))); do
  C=$D/vcache/r$g; mkdir -p $C/vllm $C/triton $C/inductor
  if [ $g -gt 0 ] && [ ! -e $C/.seeded ] && [ -d $D/vcache/r0/vllm ]; then cp -al $D/vcache/r0/. $C/ 2>/dev/null; touch $C/.seeded; fi
  CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor \
    setsid nohup $D/venv/bin/vllm serve "$M" --host 127.0.0.1 --port $((8100+g)) --served-model-name prior \
      --dtype bfloat16 --max-model-len $MML --gpu-memory-utilization "$GMU" --enable-prefix-caching \
      --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs ${MAX_SEQS:-512} > $D/logs/serve-r$g.log 2>&1 < /dev/null &
  echo $! > $S/r$g.pid
  # replica 0 first (compile caches), then the rest in parallel
  if [ $g -eq 0 ]; then for _ in $(seq 240); do curl -sf -m 3 http://127.0.0.1:8100/v1/models >/dev/null && break; kill -0 $(cat $S/r0.pid) 2>/dev/null || { echo r0 died; tail -30 $D/logs/serve-r0.log; exit 4; }; sleep 5; done; fi
done
for g in $(seq 0 $((NG-1))); do
  for _ in $(seq 240); do curl -sf -m 3 http://127.0.0.1:$((8100+g))/v1/models >/dev/null && break; sleep 5; done
done
up=0; for g in $(seq 0 $((NG-1))); do curl -sf -m 3 http://127.0.0.1:$((8100+g))/v1/models >/dev/null && up=$((up+1)); done
python3 -c "import json,time;json.dump(dict(ckpt='$CK',replicas=$up,gpus=$NG,gmu=float('$GMU'),max_model_len=$MML,at=time.time(),ports=[8100+i for i in range($NG)]),open('$S/SERVING.json','w'))"
echo "serving $CK replicas=$up/$NG"
