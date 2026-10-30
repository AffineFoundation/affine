#!/bin/bash
# probe-speed ps4 (Jacob t840u): vLLM venv site-packages off gocryptfs -> byte-identical overlay copy (/opt/affine-vllm-venv/site-packages)
# via a symlink at the same path. Running vLLM procs keep their mapped files; new serves import from the overlay. Same bytes, same paths
# -> sampling/fingerprint unchanged. Installs only after e$WAIT_EPOCH DECISION with no run on the pod. Revert: MODE=revert bash activate_ps4.sh
set -euo pipefail; umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed; GV=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate
PP(){ ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o UserKnownHostsFile=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d -p 40198 root@38.88.142.89 "$@"; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
prog(){ echo "- $(TZ=America/Costa_Rica date "+%H:%M CR"): activate_ps4 ${MODE:-switch}: $*" >> $D/PROGRESS.md; }
SP=/root/affine-env-probe/venv/lib/python3.12/site-packages; NEW=/opt/affine-vllm-venv/site-packages
if [[ ${MODE:-switch} == revert ]]; then
  PP "set -e; [ -L $SP ] && [ -d $SP.gocryptfs-orig ] && mv $SP $SP.link-off && mv $SP.gocryptfs-orig $SP && rm $SP.link-off; ls -ld $SP"
  prog "REVERTED (vLLM venv site-packages back on gocryptfs)"; exit 0; fi
E=${WAIT_EPOCH:?}; echo "$(cr) armed, waiting e$E DECISION"
while [[ ! -f $GV/state/epochs/$E/DECISION.json ]]; do sleep 1; done
sleep 5; echo "$(cr) e$E DECISION seen"
PP "grep -q STAGED /opt/affine-vllm-venv/stage.log" || { prog "copy not staged - NOT installed"; exit 1; }
[[ ${FORCE:-0} == 1 ]] || PP "python3 /root/affine-env-probe/podctl.py status" | python3 -c 'import json,sys;d=json.load(sys.stdin);r=[k for k,v in d["runs"].items() if v.get("running")];sys.exit(1 if r else 0)' || { prog "pod run active after e$E DECISION - NOT installed"; exit 1; }
PP "set -e; [ -d $SP ] && [ ! -L $SP ]; ln -sfn $NEW $SP.ps4link; mv $SP $SP.gocryptfs-orig && mv -T $SP.ps4link $SP; ls -ld $SP; /root/affine-env-probe/venv/bin/python -c 'import time;t=time.time();import vllm,torch;print(vllm.__version__,torch.__version__,vllm.__file__,\"import_s\",round(time.time()-t,1))'" | tee -a $D/data/ps4.txt \
  || { MODE=revert bash $D/activate_ps4.sh; prog "smoke import failed - reverted"; exit 1; }
prog "LIVE after e$E DECISION: vLLM venv site-packages -> overlay copy ($SP -> $NEW). Effect shows at the next cold serve. Revert: MODE=revert bash $D/activate_ps4.sh"
