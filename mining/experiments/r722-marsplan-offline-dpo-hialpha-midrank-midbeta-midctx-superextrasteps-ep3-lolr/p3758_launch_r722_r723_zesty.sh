#!/usr/bin/env bash
set -euo pipefail
source /home/const/subnet120/.venv/bin/activate
ROOT=/home/const/subnet120/mining/experiments
E722=r722-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-superextrasteps-ep3-lolr
E723=r723-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-superextrasteps-ep3-lolr
log(){ echo "[p3758-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "pack"
tar czf /tmp/r722_exp_p3758.tar.gz -C "$ROOT" "$E722"
tar czf /tmp/r723_exp_p3758.tar.gz -C "$ROOT" "$E723"
ls -lh /tmp/r722_exp_p3758.tar.gz /tmp/r723_exp_p3758.tar.gz
log "scp tarballs + pod script"
lium scp /tmp/r722_exp_p3758.tar.gz zesty-comet-da:/tmp/r722_exp_p3758.tar.gz
lium scp /tmp/r723_exp_p3758.tar.gz zesty-comet-da:/tmp/r723_exp_p3758.tar.gz
lium scp "$ROOT/$E722/p3758_pod_reap_launch.sh" zesty-comet-da:/tmp/p3758_pod_reap_launch.sh
log "exec pod"
lium exec zesty-comet-da 'bash /tmp/p3758_pod_reap_launch.sh'
log "host DONE"
