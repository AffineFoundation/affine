#!/usr/bin/env bash
# Public text install (RA/20261031-legacy-removal): mining.json no longer publishes old_format_cutoff / old_format_notice,
# FF-PUBLIC.json texts drop the old-format/old-client wording, llms.txt drops the cutoff section (anchored in-place edits, llms_patch.py). No lifecycle change:
# only affine-mining-discovery (the read-only mining.json publisher) is restarted.
# MODE=apply : backup -> copy staged files -> restart affine-mining-discovery -> verify mining.json (fresh, batch_file, no old_format_cutoff)
#              -> on failure revert automatically
# MODE=revert: restore the three files from PUBLIC-LAST-BACKUP and restart affine-mining-discovery
set -uo pipefail
MODE=${MODE:?MODE=apply|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261031-legacy-removal; S=$D/stage
PUB=$RA/20261023-toploc-credit-train-gate/observer-v829/publish_mimo_public_discovery.py
FFP=$RA/20261030-free-file/release/FF-PUBLIC.json; LL=/home/const/subnet120/affine/website/llms.txt
PY=/home/const/subnet120-rewrite/.venv/bin/python
cr(){ TZ=America/Costa_Rica date "+%H:%M CR"; }; log(){ echo "- $(cr): public $MODE: $*" >> $D/PROGRESS.md; echo "$*"; }
restart(){ systemctl --user restart affine-mining-discovery.service; sleep 20; systemctl --user is-active --quiet affine-mining-discovery.service; }
verify(){ $PY - <<'PY'
import json, time, sys
m = json.load(open('/home/const/subnet120/affine/website/mining.json'))
bad = []
if time.time() - float(m.get('status_updated_at', 0)) > 120: bad.append('stale')
if 'batch_file' not in m: bad.append('no batch_file')
if m.get('status') == 'discovery_binding_error': bad.append('discovery_binding_error')
if 'old_format_cutoff' in m or 'old_format_notice' in m: bad.append('old_format keys still present')
print('OK' if not bad else 'BAD: ' + ', '.join(bad), m.get('status'), m.get('protocol')); sys.exit(1 if bad else 0)
PY
}
revert(){ B=$(cat $D/PUBLIC-LAST-BACKUP); cp -a $B/publish_mimo_public_discovery.py $PUB; cp -a $B/FF-PUBLIC.json $FFP
  # llms.txt is also patched by other jobs: restore it only if nobody changed it after this install
  if [ -f $B/llms.after.sha256 ] && sha256sum -c --quiet $B/llms.after.sha256 2>/dev/null; then cp -a $B/llms.txt $LL; else echo "llms.txt changed since install: not restored (backup $B/llms.txt)"; fi
  restart; log "reverted from $B (discovery active: $(systemctl --user is-active affine-mining-discovery.service))"; }
case $MODE in
apply)
  B=$D/public-backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; cp -a $PUB $FFP $LL $B/; echo $B > $D/PUBLIC-LAST-BACKUP
  install -m 644 $S/publish_mimo_public_discovery.py $PUB; install -m 664 $S/FF-PUBLIC.json $FFP; $PY $D/llms_patch.py $LL && sha256sum $LL > $B/llms.after.sha256
  if restart && sleep 15 && verify; then log "applied (backup $B); mining.json verified"; else log "verify FAILED -> reverting"; revert; exit 1; fi ;;
revert) revert ;;
esac
