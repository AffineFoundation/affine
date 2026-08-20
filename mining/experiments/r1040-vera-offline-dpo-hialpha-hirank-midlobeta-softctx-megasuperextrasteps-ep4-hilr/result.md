
chmod +x "$EXP_DIR"/*.sh
cd /home/const/subnet120/mining/experiments
EXP=r1058-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-megasuperextrasteps-ep4-hilr
tar czf /tmp/${EXP}.tgz $EXP
scp -o StrictHostKeyChecking=accept-new -P 40299 /tmp/${EXP}.tgz root@38.127.229.127:/tmp/${EXP}.tgz

ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p 40299 root@38.127.229.127 "bash -s" <<EOF
set -euo pipefail
EXP=$EXP
cd /root/mining_src && tar xzf /tmp/\${EXP}.tgz
chmod +x /root/mining_src/\$EXP/*.sh
mkdir -p /root/r1058
cp -f /root/mining_src/\$EXP/dpo_duel_reason.jsonl /root/r1058/
# exact-PID reap R1040 on GPUs 6,7 / :8003 — do NOT touch R1055 on 4,5
python3 - <<'PY'
import os,signal,time,subprocess
want={6,7}
kill_set=set()
for pf in ["/root/logs/vllm_chall_r1040.pid"]:
  if os.path.exists(pf):
    try: kill_set.add(int(open(pf).read().strip()))
    except Exception: pass
out=subprocess.check_output("ss -lntp 2>/dev/null | grep ':8003' || true", shell=True, text=True)
for tok in out.replace(',',' ').split():
  if tok.startswith('pid='): kill_set.add(int(tok.split('=')[1].split(')')[0]))
idx={}
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True).splitlines():
  i,u=[x.strip() for x in line.split(",")]; idx[u]=int(i)
for line in subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"], text=True).splitlines():
  if not line.strip(): continue
  u,pid=[x.strip() for x in line.split(",")]
  if idx.get(u) in want: kill_set.add(int(pid))
# also kill sim duel if still on these gpus - only if in kill_set from nvidia
print(f"reap gpu={sorted(want)} kill={sorted(kill_set)}", flush=True)
for pid in sorted(kill_set):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
for _ in range(40):
  if not any(os.path.exists(f"/proc/{p}") for p in kill_set): break
  time.sleep(1)
for pid in sorted(kill_set):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
time.sleep(2)
print(subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7", shell=True, text=True).strip())
# confirm R1055 still alive
print("r1055", open("/root/logs/r1055_train.pid").read().strip(), "alive", os.path.exists("/proc/"+open("/root/logs/r1055_train.pid").read().strip()))
PY
for i in \$(seq 1 60); do
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=\$1} END{print s+0}')
  echo wait used=\$used; [[ "\$used" -lt 8192 ]] && break; sleep 2
done
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=\$1} END{print s+0}')
[[ "\$used" -lt 8192 ]] || exit 1
nohup bash /root/mining_src/\$EXP/lean_train_r252_gpus67_p4186.sh >/root/logs/p4186_r1058_lean_outer.nohup 2>&1 &
echo \$! >/root/logs/p4186_r1058_lean_outer.pid
sleep 8
echo TRAIN=\$(cat /root/logs/r1058_train.pid)
tail -12 /root/logs/r1058_lean_warm.log
ps -p \$(cat /root/logs/r1058_train.pid) -o pid,cmd | tail -1
ps -p \$(cat /root/logs/r1055_train.pid) -o pid,cmd | tail -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
