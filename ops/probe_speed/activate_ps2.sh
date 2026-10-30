#!/bin/bash
# probe-speed step 2 (Jacob t836u/t838u): native MATH grader runtime off gocryptfs. The uv grader env's site-packages (51 MB, pure .py,
# byte-identical copy verified by diff -r and by RUNTIME_LOCK) is served from the overlay root (/opt/affine-grader-uv) via a symlink at
# the SAME path, so the gate probe fingerprint, interpreter, script and argv are unchanged. Install only right after DECISION of e$WAIT_EPOCH.
# Revert: MODE=revert bash activate_ps2.sh
set -euo pipefail; umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed; GV=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate
PP(){ ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o UserKnownHostsFile=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d -p 40198 root@38.88.142.89 "$@"; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
prog(){ echo "- $(TZ=America/Costa_Rica date "+%H:%M CR"): activate_ps2 ${MODE:-switch}: $*" >> $D/PROGRESS.md; }
N=97322a8c223cfacd3691a794fa0c412cc1c6d012ab316bad91ae56ff84e7d34e-ac6818d574e35a95
SP=/root/.cache/uv/environments-v2/$N/lib/python3.12/site-packages; NEWSP=/opt/affine-grader-uv/environments-v2/$N/lib/python3.12/site-packages
if [[ ${MODE:-switch} == revert ]]; then
  PP "set -e; [ -L $SP ] && [ -d $SP.gocryptfs-orig ] && mv $SP $SP.link-off && mv $SP.gocryptfs-orig $SP && rm $SP.link-off; ls -ld $SP"
  prog "REVERTED (site-packages back on gocryptfs)"; exit 0; fi
[[ ${SKIP_SYNTH:-0} == 1 ]] || grep -q "mismatches 0" $D/data/identity-test-1.txt || { echo "identity test not passed"; exit 1; }
E=${WAIT_EPOCH:?}; echo "$(cr) armed, waiting e$E DECISION"
while [[ ! -f $GV/state/epochs/$E/DECISION.json ]]; do sleep 0.5; done
echo "$(cr) e$E DECISION seen"
PP "set -e; [ -d $SP ] && [ ! -L $SP ]; diff -rq $SP $NEWSP; ln -sfn $NEWSP $SP.ps2link; mv $SP $SP.gocryptfs-orig && mv -T $SP.ps2link $SP; ls -ld $SP"
prog "LIVE after e$E DECISION: grader site-packages -> overlay copy (symlink $SP -> $NEWSP). Revert: MODE=revert bash $D/activate_ps2.sh"
echo "$(cr) done"
