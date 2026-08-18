#!/usr/bin/env bash
# p3810: wait for zesty→R337 marsplan cache, patch bootstrap, relaunch.
set -euo pipefail
ROOT=/home/const/subnet120/mining
KNOWN=/tmp/r337_p3810.known_hosts
GOLD_HOST=192.9.163.79
GOLD_PORT=20296
OLD=556d02a2adfa9bd42a02de3c766f98be7e44ca46
SSH_G=(ssh -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
       -o StrictHostKeyChecking=accept-new -p "$GOLD_PORT" "root@$GOLD_HOST")
SCP=(scp -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$KNOWN"
     -o StrictHostKeyChecking=accept-new -P "$GOLD_PORT")
log() { echo "[p3810-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "armed wait for local marsplan cache on golden-lion-72"
while true; do
  n=$("${SSH_G[@]}" "ls /root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/$OLD/model-*-of-*.safetensors 2>/dev/null | wc -l" || echo 0)
  if "${SSH_G[@]}" "test -f /root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/$OLD/config.json" \
    && [[ "${n:-0}" -ge 16 ]]; then
    log "cache ready shards=$n"
    break
  fi
  du=$("${SSH_G[@]}" "du -sh /root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen 2>/dev/null | awk '{print \$1}'" || echo '?')
  log "waiting shards=${n:-0} du=$du"
  sleep 30
done

"${SCP[@]}" "$ROOT/experiments/r337-marsplan-online-dpo-hilr/p3810_patch_bootstrap.py" \
  "root@$GOLD_HOST:/tmp/p3810_patch_bootstrap.py"
"${SSH_G[@]}" "python3 /tmp/p3810_patch_bootstrap.py"

"${SSH_G[@]}" 'bash -lc "
grep -E \"BASE=|KING_REPO=|KING_REV=\" /root/mine.env
rm -f /root/logs/r337_pipeline.p3810c.pid
nohup bash /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh \
  >/root/logs/r337_pipeline.p3810c.nohup 2>&1 &
echo \$! >/root/logs/r337_pipeline.p3810c.pid
echo RELAUNCH_PID=\$(cat /root/logs/r337_pipeline.p3810c.pid)
sleep 5
tail -30 /root/logs/r337_pipeline.p3810c.nohup
"'
log "bootstrap relaunched"
