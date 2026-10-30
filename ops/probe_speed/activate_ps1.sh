#!/bin/bash
# probe-speed step 1 (Jacob t835u). After gate DECISION of e$WAIT_EPOCH, in the same idle gap as the fp32v verifier round reset:
#  (a) probe pod podctl.py 156ac8fa -> 511142f1 (8 vLLM replicas start in parallel instead of replica 0 first)
#  (b) probe pod PREP-CACHE.json on (v2d prompt-id cache; ids identical by construction, verified 10-29)
#  (c) gate EPOCH-TIME.json delta_timeout 900 -> 240 (slow trainer-side delta make -> R2 full fetch fallback)
# Revert: MODE=revert bash activate_ps1.sh   (restores podctl backup, PREP-CACHE off, delta_timeout 900)
set -euo pipefail; umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed; GV=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate
PODD=/root/affine-env-probe; PY=/home/const/subnet120-rewrite/.venv/bin/python
PP(){ ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o UserKnownHostsFile=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d -p 40198 root@38.88.142.89 "$@"; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
prog(){ echo "- $(TZ=America/Costa_Rica date "+%H:%M CR"): activate_ps1 ${MODE:-switch}: $*" >> $D/PROGRESS.md; }
NEW=511142f1d0ebe8c5; OLD=156ac8fa8389cc3f
set_dt(){ cp -p $GV/state/EPOCH-TIME.json $D/backups/EPOCH-TIME.json.pre-ps1-$(date -u +%Y%m%dT%H%M%SZ)
  $PY - "$1" "$GV/state/EPOCH-TIME.json" <<'PYX'
import json,os,sys,time
p=sys.argv[2]; d=json.load(open(p)); d["delta_timeout"]=float(sys.argv[1]); d["ps1_delta_timeout_at"]=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()); d["ps1_by"]="20261030-probe-speed (Jacob t835u)"
t=p+".tmp"; json.dump(d,open(t,"w"),indent=1); os.chmod(t,0o644); os.replace(t,p); print("delta_timeout",d["delta_timeout"])
PYX
}
PC_ON='{"on": true, "by": "20261030-probe-speed ps1 (Jacob t835u)"}'
PC_OFF='{"on": false, "by": "20261030-probe-speed ps1 revert"}'
if [[ ${MODE:-switch} == revert ]]; then
  PP "cd $PODD && b=\$(ls -t podctl.py.bak-pre-ps1-* | head -1) && cp -p \$b .podctl.new && mv .podctl.new podctl.py && printf '%s' '$PC_OFF' > .pc.new && mv .pc.new PREP-CACHE.json && sha256sum podctl.py | cut -c1-16"
  set_dt 900; prog "REVERTED (podctl backup, PREP-CACHE off, delta_timeout 900)"; exit 0; fi
E=${WAIT_EPOCH:?}
echo "$(cr) armed, waiting e$E DECISION"
while [[ ! -f $GV/state/epochs/$E/DECISION.json ]]; do sleep 0.3; done
echo "$(cr) e$E DECISION seen"
PP "cd $PODD && test \"\$(sha256sum < podctl.py | cut -c1-16)\" = $OLD && ts=\$(date -u +%Y%m%dT%H%M%SZ) && cp -p podctl.py podctl.py.bak-pre-ps1-\$ts && cp -p PREP-CACHE.json PREP-CACHE.json.bak-pre-ps1-\$ts && cat > .podctl.new && test \"\$(sha256sum < .podctl.new | cut -c1-16)\" = $NEW && chmod --reference=podctl.py .podctl.new && mv .podctl.new podctl.py && printf '%s' '$PC_ON' > .pc.new && mv .pc.new PREP-CACHE.json && sha256sum podctl.py | cut -c1-16 && cat PREP-CACHE.json" < $D/stage/podctl.py
set_dt 240
prog "LIVE after e$E DECISION: podctl $NEW (parallel replica start), PREP-CACHE on, delta_timeout 240. Revert: MODE=revert bash $D/activate_ps1.sh"
echo "$(cr) done"
