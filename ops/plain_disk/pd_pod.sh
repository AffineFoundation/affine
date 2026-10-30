#!/bin/bash
# pod side of the plain-disk switch. usage: MODE=ensure|stage|install|verify|revert|status pd_pod.sh <path-on-gocryptfs> <plain-disk-target>
# ensure (default for NEW pod setup): P missing -> create N on plain disk + symlink P -> N; P present on gocryptfs -> install; already plain -> no-op.
# install: final rsync -aH --delete, mv P P.gocryptfs-orig, ln -s N P, sha256 compare (files present in both; pruned ones reported).
set -uo pipefail; P=$1; N=$2; MODE=${MODE:-install}; MINFREE_G=${MINFREE_G:-20}
need(){ local s; s=$(du -sx --apparent-size -B1G "$P" 2>/dev/null | cut -f1); local a; a=$(df --output=avail -B1G "$(dirname "$N")" | tail -1)
  [ $((a - ${s:-0})) -ge $MINFREE_G ] || { echo "REFUSE: disk: need ${s}G, avail ${a}G, keep >= ${MINFREE_G}G"; exit 6; }; }
gate_busy(){ pgrep -f 'gate_(delta|fetch)\.py' >/dev/null || { [ -e /root/affine-env-probe/fetch.pid ] && kill -0 "$(cat /root/affine-env-probe/fetch.pid)" 2>/dev/null; }; }
if [ "$MODE" = ensure ]; then
  if [ -L "$P" ] || { [ -e "$P" ] && [ "$(findmnt -n -o FSTYPE -T "$P")" != fuse.gocryptfs ]; }; then echo "plain already: $P"; exit 0; fi
  if [ ! -e "$P" ]; then mkdir -p "$N" "$(dirname "$P")" && ln -s "$N" "$P" && echo "created on plain disk: $P -> $N"; exit 0; fi
  MODE=install
fi
case $MODE in
 stage) [ -L "$P" ] && { echo "already live -> $(readlink "$P")"; exit 0; }; mkdir -p "$(dirname "$N")"; need
   nice -n 10 ionice -c3 rsync -aH "$P"/ "$N"/ && echo "staged $(du -sh "$N" | cut -f1)";;
 install) [ -L "$P" ] && { echo "already live -> $(readlink "$P")"; exit 0; }
   case $P in */ckpt) gate_busy && { echo "REFUSE: gate delta/fetch running"; exit 3; };; esac
   mkdir -p "$(dirname "$N")"; [ -d "$N" ] || need
   nice -n 10 rsync -aH --delete "$P"/ "$N"/ || { echo rsync-fail; exit 4; }
   mv "$P" "$P.gocryptfs-orig" && ln -s "$N" "$P" && echo "LIVE -> $N"
   nice -n 10 rsync -aH --ignore-existing "$P.gocryptfs-orig"/ "$N"/   # backfill files written between final sync and mv
   echo "sha256 $(python3 /opt/plain/pdman.py "$P.gocryptfs-orig" "$N")";;
 verify) echo "sha256 $(python3 /opt/plain/pdman.py "$P.gocryptfs-orig" "$N")";;
 revert) [ -L "$P" ] && [ -d "$P.gocryptfs-orig" ] || { echo "not switched"; exit 0; }
   rsync -aH "$N"/ "$P.gocryptfs-orig"/; rm "$P" && mv "$P.gocryptfs-orig" "$P" && echo "REVERTED (plain copy kept at $N)";;
 status) ls -ld "$P" "$P.gocryptfs-orig" 2>&1;;
esac
