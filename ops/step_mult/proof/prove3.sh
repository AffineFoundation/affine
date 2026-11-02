#!/bin/bash
# defaults proof from the CURRENT head 771eff20 (E567 GREEN): same real-file subset b526s (36 groups / 108 rollouts), kit v833j vs v833l (no knobs) vs v833l (all knobs explicit default)
set -x
P=/data/smult/prove; cd $P; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=771eff20a1d6a1b46d04c86e2d35aca165678b5c01b203a84a6b122e4af18111; HK=5EgeoEQjdBkWRVei2ib8HRrmnJaawv9nEFUhLdSJ8wLQmWPb
HD=$P/h771
while [ $(ls /data/smult/fp32/$H/*.ok 2>/dev/null | wc -l) -lt 9 ] || [ $(ls /data/smult/bf16/$H/model*.ok 2>/dev/null | wc -l) -lt 11 ]; do sleep 20; done
if [ ! -d $HD ]; then mkdir -p $HD.partial && for f in $P/e526-m0/*; do case $(basename $f) in model*.safetensors|model.safetensors.index.json) ;; *) cp $f $HD.partial/ ;; esac; done
  cp /data/smult/bf16/$H/model*.safetensors /data/smult/bf16/$H/model.safetensors.index.json $HD.partial/ && mv $HD.partial $HD; fi
ls $HD
while [ ! -f $P/GO ]; do sleep 10; done   # set by switcher.sh once the replay driver is stopped between evals
run() { /data/tvenv/bin/python $2/miner/kit/train_step.py --head $HD --ref $P/ref --batch $3 --hotkey $HK --out $P/out3-$1 \
     --master /data/smult/fp32/$H --report $P/rep3-$1.json --token-budget 4096 > log3-$1.txt 2>&1 && touch out3-$1/.done; echo "RUN $1 rc=$?"; }
run J  $P/kit-j $P/b526s.json
run L0 $P/kit-l $P/b526s.json
run LD $P/kit-l $P/b526s-v2-alldefaults.json
mkdir -p c3 && for n in J L0 LD; do ln -sfn $P/out3-$n $P/out-c3$n; cp rep3-$n.json rep-c3$n.json; done
/data/tvenv/bin/python $P/compare.py c3J c3L0 c3LD > compare3.json; tail -5 compare3.json
echo PROVE-DONE >> $P/prove.log
