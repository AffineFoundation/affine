#!/bin/sh
# revert the after-GREEN record publisher (20261031T190221Z): restore k1_transparency.py, remove the green.json/tasks.json files, restart the sidecar
cp -p /home/const/subnet120-rewrite/state/root-audits/20261031-green-publish/backup/20261031T190221Z/k1_transparency.py /home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/code/k1_transparency.py
rm -f /home/const/subnet120/affine/website/k1/turns/*/green.json /home/const/subnet120/affine/website/k1/turns/*/tasks.json
rm -f /home/const/subnet120/affine/website/k1/turns/index.json   # rebuilt without the green links on the next tick
systemctl --user restart affine-k1-transparency.service
cp -p /home/const/subnet120-rewrite/state/root-audits/20261031-green-publish/backup/llms.txt.pre-20261031T190424Z /home/const/subnet120/affine/website/llms.txt   # llms.txt before the green-publish section
