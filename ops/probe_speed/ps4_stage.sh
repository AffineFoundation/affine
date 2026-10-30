#!/bin/bash
# probe-speed ps4 stage (probe pod): byte-identical overlay copy of the vLLM venv site-packages, low priority. No live change.
set -euo pipefail
SP=/root/affine-env-probe/venv/lib/python3.12/site-packages; NEW=/opt/affine-vllm-venv/site-packages
L=/opt/affine-vllm-venv/stage.log; mkdir -p /opt/affine-vllm-venv
echo "start $(date -u +%T)" > $L
t0=$(date +%s)
[ -L $SP ] && { echo "already a symlink" >> $L; exit 1; }
rm -rf $NEW.tmp; ionice -c3 nice -n 19 cp -a $SP $NEW.tmp
echo "copied $(( $(date +%s)-t0 ))s $(du -sh $NEW.tmp | cut -f1)" >> $L
ionice -c3 nice -n 19 diff -rq --no-dereference $SP $NEW.tmp >> $L 2>&1 && echo "DIFF-OK $(( $(date +%s)-t0 ))s" >> $L || { echo "DIFF-FAIL" >> $L; exit 1; }
mv -T $NEW.tmp $NEW; echo "STAGED $(date -u +%T)" >> $L
