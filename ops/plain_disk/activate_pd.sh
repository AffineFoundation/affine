#!/bin/bash
# plain-disk switch (host side). usage: [MODE=install|revert|verify|status|stage] bash activate_pd.sh <probe|eval|trainer> <item>
# one-line revert of an item: MODE=revert bash activate_pd.sh <pod> <item>
set -uo pipefail
RA=/home/const/subnet120-rewrite/state/root-audits/20261030-plain-disk; POD=${1:?pod}; IT=${2:?item}; MODE=${MODE:-install}
source $RA/items.conf; eval "LIST=\${PATHS_${POD}_${IT}:?unknown item}"
prog(){ echo "- $(TZ=America/Costa_Rica date "+%H:%M:%S CR"): activate_pd $POD $IT MODE=$MODE: $*" >> $RA/PROGRESS.md; echo "$*"; }
$RA/podin.sh $POD "mkdir -p /opt/plain && cat > /opt/plain/pd_pod.sh && chmod +x /opt/plain/pd_pod.sh" < $RA/pd_pod.sh
$RA/podin.sh $POD "cat > /opt/plain/pdman.py" < $RA/pdman.py
rc=0
for P in $LIST; do
  N=/opt/plain/$POD${P#/root}
  OUT=$($RA/pod.sh $POD "MODE=$MODE MINFREE_G=${MINFREE_G:-20} /opt/plain/pd_pod.sh $P $N" 2>&1 | tail -5)
  prog "$P: $(echo "$OUT" | tr '\n' ' ' | cut -c1-600)"
  if [[ $MODE == install ]] && echo "$OUT" | grep -q '^sha256 ' && ! echo "$OUT" | grep -q '"bad": 0,'; then
    prog "$P sha256 MISMATCH -> auto-revert"; MODE=revert $RA/pod.sh $POD "MODE=revert /opt/plain/pd_pod.sh $P $N"; rc=5; fi
done
exit $rc
