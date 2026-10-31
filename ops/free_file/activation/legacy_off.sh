#!/usr/bin/env bash
# SN120 legacy-off (RA/20261030-free-file): after the old-format cutoff, stop the old-format machinery by its existing
# no-restart switches. Staged 2026-10-30 ~19:50 CR, NOT applied. Changes only two flag files (backed up first):
#   staged/FF.json                  legacy_capture=false  -> ff_lifecycle._capture returns {} (no old commitments collected)
#   staged/EPOCH-TIME-LIFECYCLE.json seal_pregrade=false, seal_prefetch=false, capture_put_fanout=false, dual_extended=false
#                                   (v829n pregrade/regrade, v829k seal prefetch, v829n capture fan-out, v829u dual-accept)
# MODE=check  : prints whether the preconditions hold (no change)
# MODE=apply  : preconditions -> backup -> set flags -> log (PROGRESS.md)
# MODE=revert : restore both files from the newest backup (no restart needed: every switch is read on each use)
# Preconditions for apply: now >= cutoff + 30 min; >= 2 epochs FINALIZED after the cutoff, none RECOVERY-REQUIRED;
# no route=legacy turn after the cutoff; lifecycle active; ff-watchdog active.
set -euo pipefail
MODE=${MODE:?MODE=check|apply|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261030-free-file; AV=$D/activation
S=$RA/20261023-toploc-credit-train-gate/staged; FF=$S/FF.json; LO=$S/EPOCH-TIME-LIFECYCLE.json
E=/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs; PY=/home/const/subnet120-rewrite/.venv/bin/python
CUTOFF=1793437200
cr(){ TZ=America/Costa_Rica date "+%H:%M CR"; }
prog(){ echo "- $(cr): legacy_off $MODE: $*" >> $D/PROGRESS.md; echo "$*"; }
check(){ $PY - "$E" "$CUTOFF" <<'PY'
import json, os, sys, time, subprocess
E, cut = sys.argv[1], int(sys.argv[2]); bad = []
if time.time() < cut + 1800: bad.append('less than 30 min after the cutoff')
fin = []
for n in sorted((int(x) for x in os.listdir(E) if x.isdigit()), reverse=True)[:12]:
    p = os.path.join(E, str(n))
    if os.path.exists(os.path.join(p, 'RECOVERY-REQUIRED.json')): bad.append('e%d RECOVERY-REQUIRED' % n)
    f = os.path.join(p, 'FINALIZED.json')
    if os.path.exists(f) and os.path.getmtime(f) > cut: fin.append(n)
    s = os.path.join(p, 'FF-SELECT.json')
    if os.path.exists(s) and os.path.getmtime(s) > cut and json.load(open(s)).get('route') == 'legacy': bad.append('e%d legacy turn after cutoff' % n)
if len(fin) < 2: bad.append('only %d epochs finalized after the cutoff' % len(fin))
for u in ('affine-mimo-lifecycle', 'ff-watchdog'):
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', u + '.service']).returncode: bad.append(u + ' not active')
print('OK' if not bad else 'NOT READY: ' + '; '.join(bad))
sys.exit(1 if bad else 0)
PY
}
jset(){ $PY - "$@" <<'PY'
import json, os, sys
p = sys.argv[1]; d = json.load(open(p))
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1); d[k] = json.loads(v)
t = p + '.tmp'; open(t, 'w').write(json.dumps(d, indent=1) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
PY
}
case $MODE in
check) check ;;
apply)
  check || { prog "not applied (preconditions)"; exit 1; }
  B=$AV/legacy-off-backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; cp -a $FF $LO $B/; echo $B > $AV/LEGACY-OFF-LAST-BACKUP
  jset $FF legacy_capture=false "legacy_off_at_cr=\"$(cr)\""
  jset $LO seal_pregrade=false seal_prefetch=false capture_put_fanout=false dual_extended=false "legacy_off_at_cr=\"$(cr)\""
  prog "applied (backup $B). Watch the next 2 epochs: each must seal, select, finalize in < 30 min; else MODE=revert" ;;
revert)
  B=$(cat $AV/LEGACY-OFF-LAST-BACKUP); cp -a $B/FF.json $FF; cp -a $B/EPOCH-TIME-LIFECYCLE.json $LO
  prog "reverted from $B" ;;
esac
