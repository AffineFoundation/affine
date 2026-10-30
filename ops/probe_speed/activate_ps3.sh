#!/bin/bash
# probe-speed step 3 (Jacob t840u): probe_worker fbb15543 + v2e setup-once (GRADE-SETUP-ONCE.json on) + live-math budget skip (pw_ps3 77081239).
# Lands only at a round boundary (round already closed -> next round baselines under it). Revert: MODE=revert bash activate_ps3.sh
# (restores fbb15543, switch off, closes the open round so the next one re-baselines on the old worker).
set -euo pipefail; umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed
PP(){ ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o UserKnownHostsFile=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d -p 40198 root@38.88.142.89 "$@"; }
prog(){ echo "- $(TZ=America/Costa_Rica date "+%H:%M CR"): activate_ps3 ${MODE:-switch}: $*" >> $D/PROGRESS.md; }
A=/root/affine-env-probe
if [[ ${MODE:-switch} == revert ]]; then
  PP "set -e; cp -p $A/probe_worker.py.bak-pre-ps3 $A/probe_worker.py; echo '{\"on\": false}' > $A/GRADE-SETUP-ONCE.json; sha256sum $A/probe_worker.py | cut -c1-16"
  r=$(bash /home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed/close_round.sh "probe-speed ps3 revert (probe_worker back to fbb15543) -> re-baseline" 0 2>&1 || true)
  prog "REVERTED probe_worker -> fbb15543, setup-once off; round: $r"; exit 0; fi
PP "set -e; sha256sum $A/probe_worker.py | cut -c1-16 | grep -q fbb155439672beeb; cp -p $A/probe_worker.py $A/probe_worker.py.bak-pre-ps3; [ -f $A/GRADE-SETUP-ONCE.json ] && cp -p $A/GRADE-SETUP-ONCE.json $A/GRADE-SETUP-ONCE.json.bak-pre-ps3 || true"
PP "cat > $A/probe_worker.py.ps3new" < $D/stage/pw_ps3.py
PP "set -e; cd $A; sha256sum probe_worker.py.ps3new | cut -c1-16 | grep -q 770812395998dd40; mv probe_worker.py.ps3new probe_worker.py; echo '{\"on\": true}' > GRADE-SETUP-ONCE.json; sha256sum probe_worker.py GRADE-SETUP-ONCE.json | cut -c1-16"
prog "LIVE: probe_worker fbb15543 -> 77081239 (setup-once + budget skip), GRADE-SETUP-ONCE on; lands in the fresh round opened after the 12:41 close. Revert: MODE=revert bash $D/activate_ps3.sh"
