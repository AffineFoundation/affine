#!/bin/bash
# real-file subset proof: the 36 groups (108 rollouts) of the e526 GREEN file with prompt+output <= 3500 tokens (single B200 memory)
set -x
P=/data/smult/prove; cd $P; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=3eceb8902d80a82762a5503742915a4f99509e878eb78ddc591b2fccd98aa117; HK=5EgeoEQjdBkWRVei2ib8HRrmnJaawv9nEFUhLdSJ8wLQmWPb
run() { /data/tvenv/bin/python $2/miner/kit/train_step.py --head $P/e526-m0 --ref $P/ref --batch $3 --hotkey $HK --out $P/out-$1 \
     --master /data/smult/fp32/$H --report $P/rep-$1.json --token-budget 4096 > log-$1.txt 2>&1 && touch out-$1/.done; echo "RUN $1 rc=$?"; }
run J  $P/kit-j $P/b526s.json
run L0 $P/kit-l $P/b526s.json
run LD $P/kit-l $P/b526s-v2-alldefaults.json
run LK $P/kit-l $P/b526s-v2-knob.json
/data/tvenv/bin/python $P/compare.py J L0 LD LK > compare.json; tail -40 compare.json
echo PROVE-DONE >> $P/prove.log
