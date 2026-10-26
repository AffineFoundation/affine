#!/usr/bin/env bash
# /root is gocryptfs FUSE on these pods: docker bind mounts out of it fail.
# Relocate everything container-visible to /dshare on the overlay root.
set -u
mkdir -p /dshare
# don't move the hf cache while setup is still downloading into it
i=0
while ! grep -q SETUP_DONE /root/work/setup.log 2>/dev/null; do
  sleep 30; i=$((i+30))
  [ "$i" -ge 7200 ] && { echo "setup still running after 2h; proceeding"; break; }
done
if [ -d /root/dshare ]; then
  echo "moving /root/dshare -> /dshare ($(du -sh /root/dshare 2>/dev/null | cut -f1))"
  ( cd /root/dshare && for d in *; do
      [ -e "$d" ] || continue
      mv "$d" /dshare/ 2>/dev/null || { cp -a "$d" /dshare/ && rm -rf "${d:?}"; }
    done )
  rmdir /root/dshare 2>/dev/null || true
fi
echo /dshare > /root/.dshare_path
mkdir -p /dshare/hf /dshare/ckpt /dshare/factory /dshare/ship
du -sh /dshare/hf 2>/dev/null
echo "FIX_DSHARE_DONE"
