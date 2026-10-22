#!/usr/bin/env bash
# Read-only pod disk/GPU sampler for the v8.2.10 verification (every 60 s, <= 7 h). One persistent SSH master.
D=/home/const/subnet120-rewrite/state/root-audits/20261022-async-optimizer-save
S=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20
   -o UserKnownHostsFile=/home/const/.lium/known_hosts/9486fae6-c6d2-44ef-933e-af7b8b8e6711
   -o ControlMaster=auto -o ControlPath=/tmp/v8210mon.sock -o ControlPersist=8h -p 20301 root@150.136.210.177)
for i in $(seq 1 420); do
  out=$(timeout 40 "${S[@]}" 'echo "$(date -u +%FT%TZ) psi_full_avg10=$(awk "/full/{print \$2}" /proc/pressure/io | cut -d= -f2) $(awk "\$3==\"loop0\"||\$3==\"dm-0\"||\$3==\"vda\"{printf \"%s_wsect=%s %s_rsect=%s \", \$3, \$10, \$3, \$6}" /proc/diskstats) dirtyKB=$(awk "/^Dirty:/{print \$2}" /proc/meminfo) wbKB=$(awk "/^Writeback:/{print \$2}" /proc/meminfo) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d " ")"' 2>&1) || out="$(date -u +%FT%TZ) sample-failed: ${out:0:120}"
  echo "$out" >> "$D/diskmon.log"
  sleep 60
done
