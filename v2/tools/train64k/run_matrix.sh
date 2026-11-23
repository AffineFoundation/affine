#!/bin/bash
# phase 1: memory at 32k/48k/64k on the long row pair; worst-case miner settings; short-batch equivalence.
cd /root/t64
PY=venv/bin/python
run() { # gpu tag args...
  local g=$1 t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=/root/t64/triton/gpu$g PYTHONUNBUFFERED=1 $PY bench.py --tag $t "$@" > out/$t.log 2>&1
}
C32="--out 32768 --ctx 36864"; C48="--out 49152 --ctx 53248"; C64="--out 65536 --ctx 69632"
ALL="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,out_proj"
( run 0 long32 --data long.json $C32; run 0 b128_32 --data batch128.json $C32 ) &
( run 1 long48 --data long.json $C48; run 1 b128_64 --data batch128.json $C64 ) &
( run 2 long64 --data long.json $C64 ) &
( run 3 long64_worst --data long.json $C64 --rank 64 --alpha 128 --targets $ALL ) &
( run 4 long32_worst --data long.json $C32 --rank 64 --alpha 128 --targets $ALL ) &
( run 5 short32 --data short.json $C32; run 5 short32b --data short.json $C32 ) &
( run 6 short64 --data short.json $C64 ) &
( run 7 long64_r2 --data long.json $C64 ) &
wait
echo MATRIX_DONE
