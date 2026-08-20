#!/usr/bin/env bash
# p4200: R1052 merge DONE but chall :8003 hung — Triton partial cache
# (missing __triton_launcher.so under chall_r1052; Worker ImportError + shm hang).
# Exact-PID reap GPUs6,7 only; FORCE wipe+seed from known-good cache; relaunch
# lean chall+v4 n80. Never pkill -f. Do not touch teacher:8000 king:8001 or R1070 on 4,5.
set -euo pipefail

POD=${POD:-noble-raven-a7}
EXP_DIR=/home/const/subnet120/mining/experiments/r1052-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr
LEAN_LOCAL=$EXP_DIR/lean_chall_n80_r339_gpus67_p4182.sh
LEAN_REMOTE=/root/mining_src/r1052-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr/lean_chall_n80_r339_gpus67_p4182.sh

# Patch lean: FORCE_TRITON_RESEED gate + broader seed list + fresh slice tag
python3 - <<'PY'
from pathlib import Path
p = Path("/home/const/subnet120/mining/experiments/r1052-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr/lean_chall_n80_r339_gpus67_p4182.sh")
t = p.read_text()
old = "for cand in /root/.triton/cache/chall_r969 /root/.triton/cache/chall_r953 /root/.triton/cache/chall_r952 /root/.triton/cache/king /root/.triton/cache/chall; do"
new = "for cand in /root/.triton/cache/chall_r339 /root/.triton/cache/chall_r1053 /root/.triton/cache/chall_r1032 /root/.triton/cache/chall_r978 /root/.triton/cache/king /root/.triton/cache/chall; do"
if old in t:
    t = t.replace(old, new)
    print("patched seed list")
else:
    print("seed list already patched or missing; continue")
if 'FORCE_TRITON_RESEED' not in t:
    t = t.replace(
        'if [[ "${_pre_n:-0}" -ge 1 && "${_pre_sz:-0}" -ge 50 ]]; then\n  log "REUSE preseed $TCACHE n_so=$_pre_n size_mb=$_pre_sz — skip wipe"\nelse',
        'if [[ "${FORCE_TRITON_RESEED:-0}" != "1" && "${_pre_n:-0}" -ge 1 && "${_pre_sz:-0}" -ge 50 ]]; then\n  log "REUSE preseed $TCACHE n_so=$_pre_n size_mb=$_pre_sz — skip wipe"\nelse',
    )
    print("patched FORCE_TRITON_RESEED gate")
# bump block-hash tag so slice differs from hung attempt
t = t.replace("r1052-reign36-wvk7-p4182-", "r1052-reign36-wvk7-p4200-")
t = t.replace("p4182 chall+v4-n80", "p4200 chall+v4-n80")
t = t.replace("r1052_n80_launched.p4182", "r1052_n80_launched.p4200")
p.write_text(t)
print("wrote", p)
PY

# Sync patched lean to pod
source /home/const/subnet120/.venv/bin/activate
lium exec "$POD" "bash -lc 'mkdir -p $(dirname $LEAN_REMOTE)'" >/dev/null
b64=$(base64 -w0 "$LEAN_LOCAL")
lium exec "$POD" "bash -lc 'echo $b64 | base64 -d > $LEAN_REMOTE && chmod +x $LEAN_REMOTE'" >/dev/null

lium exec "$POD" "bash -lc '
set -euo pipefail
date -u +%Y-%m-%dT%H:%M:%SZ
test -f /tmp/r1052_merged/config.json || test -f /tmp/r1052_merged/model.safetensors.index.json
n=\$(ls /tmp/r1052_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
echo merge_shards=\$n
curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null
echo TK_warm_ok

# Exact-PID reap hung chall/lean/sim on :8003 / r1052 (never pkill -f)
for pidf in /root/logs/vllm_chall_r1052.pid /root/logs/r1052_sim_wvk7.pid /root/logs/p4182_r1052_lean.pid /root/logs/p4200_r1052_lean_n80.pid; do
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
# Also reap lean_chall parent + any GPU6/7 apps that are r1052_merged / :8003 only
python3 - <<\"PY\"
import os, signal, subprocess, time
want={6,7}
out=subprocess.check_output([\"nvidia-smi\",\"--query-compute-apps=pid,gpu_uuid,used_memory\",\"--format=csv,noheader\"], text=True)
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
    if \"r1052_merged\" in cmd or \":8003\" in cmd or \"chall_r1052\" in cmd or \"lean_chall_n80_r339_gpus67\" in cmd:
        kill.add(pid)
# also kill known lean bash parents by cmdline scan (not GPU-bound)
for p in os.listdir(\"/proc\"):
    if not p.isdigit(): continue
    try:
        cmd=open(f\"/proc/{p}/cmdline\",\"rb\").read().decode(\"utf-8\",\"replace\")
    except Exception:
        continue
    if \"lean_chall_n80_r339_gpus67_p4182\" in cmd or \"wait_r1052_merge_then_n80_p4182\" in cmd:
        # keep wait_merge only if train waiters for OTHER jobs — wait_r1052 is done; kill lean only
        if \"lean_chall_n80_r339_gpus67_p4182\" in cmd:
            kill.add(int(p))
print(\"reap\", sorted(kill))
for pid in kill:
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in kill:
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
PY
sleep 3
used=\$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk \"{s+=\\\$1} END{print s+0}\")
echo gpus67_after_reap=\$used
# wait for free
for i in \$(seq 1 30); do
  used=\$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk \"{s+=\\\$1} END{print s+0}\")
  if [[ \"\${used:-999}\" -lt 2000 ]]; then echo gpus67_free=\$used; break; fi
  sleep 2
done
used=\$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk \"{s+=\\\$1} END{print s+0}\")
test \"\${used:-999}\" -lt 2000

# Force wipe+seed Triton from best known-good cache (n_so>=20)
SEED=\"\"
for cand in chall_r339 chall_r1053 chall_r1032 chall_r978 king chall; do
  d=/root/.triton/cache/\$cand
  n=\$(find \"\$d\" -name \"__triton_launcher*.so\" 2>/dev/null | wc -l || true)
  echo cand=\$cand n_so=\$n
  if [[ \"\${n:-0}\" -ge 20 ]]; then SEED=\$d; break; fi
done
echo SEED=\$SEED
test -n \"\$SEED\"
rm -rf /root/.triton/cache/chall_r1052
cp -a \"\$SEED\" /root/.triton/cache/chall_r1052
n_so=\$(find /root/.triton/cache/chall_r1052 -name \"__triton_launcher*.so\" | wc -l)
sz=\$(du -sm /root/.triton/cache/chall_r1052 | awk \"{print \\\$1}\")
echo seeded_n_so=\$n_so size_mb=\$sz
test \"\$n_so\" -ge 20

# Clear stale launch marker so lean will run n80 again
rm -f /root/logs/r1052_n80_launched.p4182 /root/logs/r1052_n80_launched.p4200

export FORCE_TRITON_RESEED=1
nohup bash $LEAN_REMOTE > /root/logs/p4200_r1052_lean_n80.nohup 2>&1 &
echo \$! > /root/logs/p4200_r1052_lean_n80.pid
echo LAUNCHED pid=\$(cat /root/logs/p4200_r1052_lean_n80.pid)
sleep 5
head -30 /root/logs/p4200_r1052_lean_n80.nohup || true
'" 2>&1 | sed '/^Executing on /d'
