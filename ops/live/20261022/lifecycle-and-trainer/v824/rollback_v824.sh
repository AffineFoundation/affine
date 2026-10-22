#!/usr/bin/env bash
# Roll back v8.2.4.
#   rollback_v824.sh --trainer     owned worker back to v822s (drain-restart; never mid-job unless --force)
#   rollback_v824.sh --lifecycle   lifecycle + observer back to v8.2.3 drop-ins. Requires CURRENT max_tasks 128 (if an epoch
#                                  is planned at 256 use: activate_v824.sh --lifecycle v8.2.4-t128 and roll back after
#                                  that epoch finalizes; the t128 layer re-stamps 128 at the boundary).
set -euo pipefail
D=/home/const/subnet120-rewrite/state/root-audits/20261017-v824; U=/home/const/.config/systemd/user
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; PY=/home/const/subnet120-rewrite/.venv/bin/python
POD_SSH=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o UserKnownHostsFile=/home/const/.lium/known_hosts/0a7aba3d-9443-445b-9601-c78ac0d14b1c -p 40301 root@89.169.120.92)
POD=/workspace/affine-mimo-trainer-prod
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $D/ROLLBACK.log; }
cur(){ "$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['$1'])"; }
case "${1:-}" in
--trainer)
  EI=$(cur epoch_index)
  if [[ -e "$STATE/epochs/$EI/TRAINING-JOB.json" && ! -e "$STATE/epochs/$EI/TRAINING-RESULT.json" && "${2:-}" != --force ]]; then log "REFUSED: e$EI job in flight (use --force only if the v824 worker is broken)"; exit 1; fi
  NEW_SHA=$("$PY" -c "import json;print(json.load(open('$D/activation-v824/PREPARE-RESULT.json'))['owned_worker_v824s_sha256'])")
  rm -f "$U/affine-mimo-trainer-supervised.service.d/zzzz~~~~~v824s-owned-worker.conf"; systemctl --user daemon-reload
  "${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v824s.json --sha256 $NEW_SHA --stop-owned-after-drain" || true
  for i in $(seq 1 120); do
    "${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v824s.json --sha256 $NEW_SHA --status" | grep -q '"alive": false' && break; sleep 5
  done
  systemctl --user restart affine-mimo-trainer-supervised.service
  log "trainer rolled back to owned-worker-v822s (drain-restart)";;
--lifecycle)
  [[ "$(cur max_tasks)" == 128 ]] || { log "REFUSED: CURRENT max_tasks=$(cur max_tasks); switch to v8.2.4-t128 first and roll back at the next boundary"; exit 1; }
  rm -f "$U/affine-mimo-lifecycle.service.d/zzzz~~~~~~~~~v824-release.conf" "$U/affine-mining-discovery.service.d/zzzz~~~~mimo-observer-v824.conf"
  systemctl --user daemon-reload && systemctl --user restart affine-mimo-lifecycle.service && sleep 20 && systemctl --user restart affine-mining-discovery.service
  log "lifecycle + observer rolled back to v8.2.3 drop-ins";;
*) echo "usage: rollback_v824.sh --trainer [--force] | --lifecycle"; exit 2;;
esac
