#!/usr/bin/env bash
# p4195: R1050 merge DONE but chall :8002 died — Triton partial cache reuse
# (missing __triton_launcher.so under chall_r1050). Force wipe+seed from a
# known-good cache on mine-r338, then relaunch lean chall+v4 n80 on GPUs 6,7.
# Never pkill -f. Do not touch teacher:8000 or king:8001 or R1061 on 4,5.
set -euo pipefail

POD=${POD:-calm-fox-6a}
EXP_DIR=/home/const/subnet120/mining/experiments/r1050-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-hilr
LEAN_LOCAL=$EXP_DIR/lean_chall_n80_r338_gpus67_p4180.sh
LEAN_REMOTE=/root/mining_src/r1050-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-hilr/lean_chall_n80_r338_gpus67_p4180.sh

# Patch seed candidate list on the local lean script (also sync to pod).
python3 - <<'PY'
from pathlib import Path
p = Path("/home/const/subnet120/mining/experiments/r1050-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-hilr/lean_chall_n80_r338_gpus67_p4180.sh")
t = p.read_text()
old = "for cand in /root/.triton/cache/chall_r969 /root/.triton/cache/chall_r953 /root/.triton/cache/chall_r952 /root/.triton/cache/king /root/.triton/cache/chall; do"
new = "for cand in /root/.triton/cache/chall_r978 /root/.triton/cache/chall_r964 /root/.triton/cache/chall_r959 /root/.triton/cache/chall_r955 /root/.triton/cache/chall_r948 /root/.triton/cache/chall_r338 /root/.triton/cache/king /root/.triton/cache/chall; do"
if old in t:
    t = t.replace(old, new)
    # Force wipe even if preseed looks fat — p4195: REUSE hid a missing .so
    t = t.replace(
        'if [[ "${_pre_n:-0}" -ge 1 && "${_pre_sz:-0}" -ge 50 ]]; then\n  log "REUSE preseed $TCACHE n_so=$_pre_n size_mb=$_pre_sz — skip wipe"\nelse',
        'if [[ "${FORCE_TRITON_RESEED:-0}" != "1" && "${_pre_n:-0}" -ge 1 && "${_pre_sz:-0}" -ge 50 ]]; then\n  log "REUSE preseed $TCACHE n_so=$_pre_n size_mb=$_pre_sz — skip wipe"\nelse',
    )
    # bump block-hash tag so slice differs from failed attempt
    t = t.replace("r1050-reign36-wvk7-p4180-", "r1050-reign36-wvk7-p4195-")
    t = t.replace('p4180 chall+v4-n80', 'p4195 chall+v4-n80')
    t = t.replace("r1050_n80_launched.p4180", "r1050_n80_launched.p4195")
    p.write_text(t)
    print("patched", p)
else:
    print("seed list already patched or missing; continue")
PY

# Sync patched lean + this launcher to pod
lium exec "$POD" "bash -lc 'mkdir -p $(dirname $LEAN_REMOTE)'" >/dev/null
# Use base64 to avoid quoting issues
b64=$(base64 -w0 "$LEAN_LOCAL")
lium exec "$POD" "bash -lc 'echo $b64 | base64 -d > $LEAN_REMOTE && chmod +x $LEAN_REMOTE'" >/dev/null

lium exec "$POD" "bash -lc '
set -euo pipefail
date -u +%Y-%m-%dT%H:%M:%SZ
# Confirm merge + free GPUs + TK warm
test -f /tmp/r1050_merged/config.json
n=\$(ls /tmp/r1050_merged/model-*-of-*.safetensors | wc -l)
echo merge_shards=\$n
curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null
used=\$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk \"{s+=\\\$1} END{print s+0}\")
echo gpus67_used_mib=\$used
test \"\${used:-999}\" -lt 2000

# Exact-PID reap any leftover chall on :8002 / r1050 (never pkill -f)
for pidf in /root/logs/vllm_chall_r1050.pid /root/logs/r1050_sim_wvk7.pid; do
  if [[ -f \$pidf ]]; then
    pid=\$(cat \$pidf 2>/dev/null || true)
    if [[ \"\$pid\" =~ ^[0-9]+\$ ]] && kill -0 \"\$pid\" 2>/dev/null; then
      echo kill_leftover \$pidf=\$pid
      kill \"\$pid\" 2>/dev/null || true
      sleep 2
      kill -9 \"\$pid\" 2>/dev/null || true
    fi
    rm -f \$pidf
  fi
done
# Also reap any GPU6/7 compute apps that are r1050_merged only
python3 - <<\"PY\"
import os, signal, subprocess, time
want={6,7}
out=subprocess.check_output([\"nvidia-smi\",\"--query-compute-apps=pid,gpu_uuid,used_memory\",\"--format=csv,noheader\"], text=True)
# map uuid->index
uu={}
for line in subprocess.check_output([\"nvidia-smi\",\"--query-gpu=index,uuid\",\"--format=csv,noheader\"], text=True).splitlines():
    idx,u=line.split(\",\")
    uu[u.strip()]=int(idx.strip())
kill=set()
for line in out.splitlines():
    parts=[p.strip() for p in line.split(\",\")]
    if len(parts)<2: continue
    pid=int(parts[0]); u=parts[1]
    gi=uu.get(u)
    if gi not in want: continue
    try:
        cmd=open(f\"/proc/{pid}/cmdline\",\"rb\").read().decode(\"utf-8\",\"replace\")
    except Exception:
        continue
    if \"r1050_merged\" in cmd or \":8002\" in cmd or \"chall_r1050\" in cmd:
        kill.add(pid)
print(\"reap\", sorted(kill))
for pid in kill:
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in kill:
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
PY
sleep 2
used=\$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk \"{s+=\\\$1} END{print s+0}\")
echo gpus67_after_reap=\$used

# Force wipe+seed Triton from best known-good cache
SEED=\"\"
for cand in chall_r978 chall_r964 chall_r959 chall_r955 chall_r948 chall_r338 king; do
  d=/root/.triton/cache/\$cand
  n=\$(find \"\$d\" -name \"__triton_launcher*.so\" 2>/dev/null | wc -l || true)
  if [[ \"\${n:-0}\" -ge 20 ]]; then SEED=\$d; break; fi
done
echo SEED=\$SEED
test -n \"\$SEED\"
rm -rf /root/.triton/cache/chall_r1050
cp -a \"\$SEED\" /root/.triton/cache/chall_r1050
n_so=\$(find /root/.triton/cache/chall_r1050 -name \"__triton_launcher*.so\" | wc -l)
sz=\$(du -sm /root/.triton/cache/chall_r1050 | awk \"{print \\\$1}\")
echo seeded_n_so=\$n_so size_mb=\$sz
test \"\$n_so\" -ge 20

# Relaunch lean with FORCE_TRITON_RESEED so it will not trust a stale fat tree
export FORCE_TRITON_RESEED=1
nohup bash $LEAN_REMOTE > /root/logs/p4195_r1050_lean_n80.nohup 2>&1 &
echo \$! > /root/logs/p4195_r1050_lean_n80.pid
echo LAUNCHED pid=\$(cat /root/logs/p4195_r1050_lean_n80.pid)
sleep 3
head -20 /root/logs/p4195_r1050_lean_n80.nohup || true
tail -5 /root/logs/p4180_r1050_chall_n80_wvk7.log 2>/dev/null || true
'" 2>&1 | sed '/^Executing on /d'
