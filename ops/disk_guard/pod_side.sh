#!/usr/bin/env bash
# disk_guard pod side. Env: ROLE (probe|trainer|eval), PROTECT (space-separated 64-hex ids), APPLY (0|1),
# MIN_AGE_MIN (default 180), LOG_CAP_MB (default 1024). Prints TSV: action path bytes mtime status ; then FREE lines.
# Never touches anything a process has open/mapped/cwd/argv-referenced. Never kills processes.
set -u
APPLY=${APPLY:-0}; MIN_AGE_MIN=${MIN_AGE_MIN:-180}; LOG_CAP_MB=${LOG_CAP_MB:-1024}
N="nice -n 19 ionice -c3"
INUSE=$(for q in /proc/[0-9]*; do ls -l $q/fd 2>/dev/null | awk '{print $NF}'; awk '{print $6}' $q/maps 2>/dev/null; readlink $q/cwd 2>/dev/null; tr '\0' '\n' < $q/cmdline 2>/dev/null; done | grep -oE '[0-9a-f]{64}' | sort -u)
prot() { local h=$1; for k in $PROTECT $INUSE; do [ "$k" = "$h" ] && return 0; done; return 1; }
old() { [ -n "$(find "$1" -maxdepth 0 -mmin +$MIN_AGE_MIN 2>/dev/null)" ]; }
act() { # act <path>
  local p=$1 sz mt; sz=$(du -sxb "$p" 2>/dev/null | cut -f1); mt=$(stat -c %y "$p" | cut -c1-19)
  if [ "$APPLY" = 1 ]; then $N rm -rf --one-file-system -- "$p" && st=DELETED || st=RM-FAIL; else st=WOULD-DELETE; fi
  printf 'prune\t%s\t%s\t%s\t%s\n' "$p" "$sz" "$mt" "$st"; }
prune_hash_dir() { # prune_hash_dir <dir> : entries named <64hex>[.suffix]
  local d=$1; [ -d "$d" ] || return 0
  for e in "$d"/*; do b=$(basename "$e"); h=${b:0:64}
    [[ $h =~ ^[0-9a-f]{64}$ ]] || continue; prot "$h" && continue; old "$e" || continue
    [ -d "$e" ] || [[ $b == *.partial ]] || continue   # only weight dirs and stale partial downloads (descriptors/logs are tiny)
    act "$e"; done; }
case "$ROLE" in
  probe) prune_hash_dir /root/affine-env-probe/ckpt/; prune_hash_dir /root/affine-env-probe/ckpt.gocryptfs-orig ;;
  trainer) prune_hash_dir /workspace/affine-mimo-trainer-prod/private/promotion-gate/keep ;;
  eval) : ;;
esac
# log size caps (copytruncate: keep the last 64 MB in <log>.1, truncate in place; safe for O_APPEND writers)
for g in ${LOG_GLOBS:-}; do for f in $g; do [ -f "$f" ] || continue
  s=$(stat -c %s "$f"); [ "$s" -gt $((LOG_CAP_MB*1048576)) ] || continue
  if [ "$APPLY" = 1 ]; then tail -c 67108864 "$f" > "$f.1" && : > "$f" && st=ROTATED || st=ROT-FAIL; else st=WOULD-ROTATE; fi
  printf 'rotate\t%s\t%s\t%s\t%s\n' "$f" "$s" "$(stat -c %y "$f" | cut -c1-19)" "$st"; done; done
for m in / /workspace /root; do mountpoint -q $m || [ $m = / ] || continue; df -B1 --output=target,size,avail $m | tail -1 | awk '{printf "free\t%s\t%s\t%s\n",$1,$2,$3}'; done | sort -u
