#!/bin/bash
# usage (on the pod): bash delete_orig.sh <path>.gocryptfs-orig ...   (host: podin.sh <pod> "bash -s <paths>" < delete_orig.sh)
# Only deletes when <path> is already a symlink to /opt/... and no process has a cwd/fd/mapping in the orig; append the rows to
# state/root-audits/20261030-disk-cleanup/DELETED.tsv. After this the item's MODE=revert no longer works.
# delete *.gocryptfs-orig copies whose live path is a symlink to plain disk and nothing holds them. TSV rows on stdout.
set -u; H=$(hostname)
held(){ for p in /proc/[0-9]*; do for f in $p/cwd $p/exe $p/root $p/fd/*; do l=$(readlink $f 2>/dev/null) || continue; case $l in "$1"|"$1"/*) echo ${p#/proc/}; return;; esac; done; grep -q " $1/" $p/maps 2>/dev/null && { echo ${p#/proc/}; return; }; done; }
df -B1 --output=used,avail /root | tail -1 | sed 's/^/DF-BEFORE /' >&2
for o in "$@"; do
  case $o in /root/*.gocryptfs-orig) ;; *) echo "skip-bad-name $o" >&2; continue;; esac
  live=${o%.gocryptfs-orig}
  [ -L "$live" ] && case $(readlink "$live") in /opt/*) ;; *) false;; esac || { printf '%s\t%s\t%s\t\t\tSKIPPED-NOT-SWITCHED\n' "$(date -u +%FT%TZ)" "$H" "$o"; continue; }
  hp=$(held "$o"); [ -n "$hp" ] && { printf '%s\t%s\t%s\t\t\tSKIPPED-HELD-pid%s\n' "$(date -u +%FT%TZ)" "$H" "$o" "$hp"; continue; }
  b=$(du -sb "$o" | cut -f1); m=$(stat -c %y "$o" | cut -c1-19)
  nice -n 19 ionice -c3 rm -rf -- "$o" && st=DELETED || st=RM-FAILED
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$H" "$o" "$b" "$m" "$st"
done
df -B1 --output=used,avail /root | tail -1 | sed 's/^/DF-AFTER /' >&2
