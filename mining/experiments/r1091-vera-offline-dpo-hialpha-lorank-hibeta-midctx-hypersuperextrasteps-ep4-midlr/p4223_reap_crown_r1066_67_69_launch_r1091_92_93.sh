#!/usr/bin/env bash
# p4223: R1066/R1067/R1069 REFUTE idle challs on crown → reap :8004/:8002/:8003
# → R1091/R1092/R1093 Hyper MidLR TRAIN on GPUs 1,3 / 6,7 / 4,5.
# Never pkill -f. Leave teacher:8000 GPU0 and king:8001 GPU2 alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
source /home/const/subnet120/.venv/bin/activate
cd "$ROOT"

SSH_HOST=95.133.252.28
SSH_PORT=40298
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$ROOT/experiments/fleet-rent/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"

TGZ=/tmp/r1091_92_93_p4223.tgz
tar -C "$ROOT/experiments" -czf "$TGZ" \
  r1091-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-midlr \
  r1092-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr \
  r1093-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-midlr

scp $SSH_OPTS -P "$SSH_PORT" "$TGZ" "root@${SSH_HOST}:/tmp/r1091_92_93_p4223.tgz"

ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
mkdir -p /root/mining_src /root/logs /root/affine_data /root/r1091 /root/r1092 /root/r1093
cd /tmp
tar -xzf /tmp/r1091_92_93_p4223.tgz
for exp in \
  r1091-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-midlr \
  r1092-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr \
  r1093-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-midlr
do
  rm -rf "/root/mining_src/$exp"
  mv "/tmp/$exp" "/root/mining_src/$exp"
  chmod +x /root/mining_src/$exp/*.sh
done

python3 - <<'PY'
import os, signal, subprocess, time, re

def stop(pid, why):
    if not pid:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    print(f"stop {pid} ({why})", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except ProcessLookupError:
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

# pidfiles for idle REFUTE challs + stale sim waiters
for pf in [
    "/root/logs/vllm_chall_r1066.pid",
    "/root/logs/vllm_chall_r1067.pid",
    "/root/logs/vllm_chall_r1069.pid",
    "/root/logs/r1066_sim_wvk7.pid",
    "/root/logs/r1067_sim_wvk7.pid",
    "/root/logs/r1069_sim_wvk7.pid",
    "/root/logs/r1066_merge_then_n80.pid",
    "/root/logs/r1067_merge_then_n80.pid",
    "/root/logs/r1069_merge_then_n80.pid",
    "/root/logs/r1066_wait_merge.pid",
    "/root/logs/r1067_wait_merge.pid",
    "/root/logs/r1069_wait_merge.pid",
]:
    if os.path.exists(pf):
        try:
            stop(int(open(pf).read().strip()), pf)
        except Exception as e:
            print("pidf", pf, e)
        try:
            os.remove(pf)
        except Exception:
            pass

# ports 8002/8003/8004 only (never 8000/8001)
for port in (8002, 8003, 8004):
    try:
        out = subprocess.check_output(
            ["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        out = ""
    for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
        stop(pid, f":{port}")

ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
for line in ps.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if len(parts) < 2:
        continue
    pid = int(parts[0])
    cmd = parts[1]
    # never touch teacher/king/train
    if "train_dpo" in cmd or "train_online" in cmd:
        continue
    if "GLM-4.5-Air" in cmd and "8000" in cmd:
        continue
    if ":8001" in cmd or "/8001" in cmd:
        continue
    if any(
        tok in cmd
        for tok in [
            "r1066_merged",
            "r1067_merged",
            "r1069_merged",
            "vllm_chall_r1066",
            "vllm_chall_r1067",
            "vllm_chall_r1069",
            "local-r1066",
            "local-r1067",
            "local-r1069",
            "r1066_sim",
            "r1067_sim",
            "r1069_sim",
            "wait_r1066",
            "wait_r1067",
            "wait_r1069",
            "lean_chall_n80_crown_r1066",
            "lean_chall_n80_crown_r1067",
            "lean_chall_n80_crown_r1069",
        ]
    ):
        if any(x in cmd for x in ("r1091", "r1092", "r1093")):
            continue
        stop(pid, "argv")

# free GPUs 1,3,4,5,6,7 of leftover compute apps (not 0/2 teacher/king)
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
)
idx_to_uuid = {}
for line in out.strip().splitlines():
    a, b = [p.strip() for p in line.split(",")]
    idx_to_uuid[int(a)] = b
want = {idx_to_uuid[i] for i in (1, 3, 4, 5, 6, 7) if i in idx_to_uuid}
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    text=True,
)
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or parts[0] not in want:
        continue
    pid = int(parts[1])
    # skip if it is teacher/king (should be on 0/2 only)
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "ignore")
    except Exception:
        cmd = ""
    if "8000" in cmd or "8001" in cmd or "GLM-4.5-Air" in cmd:
        continue
    if "vera6" in cmd and "8001" in cmd:
        continue
    stop(pid, "gpu-app")

# wait VRAM free on train GPUs
for _ in range(60):
    used = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            "1,3,4,5,6,7",
        ],
        text=True,
    )
    total = sum(int(x.strip()) for x in used.strip().splitlines() if x.strip())
    print(f"VRAM1+3+4+5+6+7 used_mib={total}", flush=True)
    if total < 48000:
        break
    time.sleep(2)
else:
    raise SystemExit("FATAL VRAM still busy after reap")

print("REAP_OK", flush=True)
PY

# seed data from parent dirs if present
for pair in 1091:1066 1092:1067 1093:1069; do
  rid=${pair%%:*}; parent=${pair##*:}
  mkdir -p /root/r$rid
  if [[ -s /root/r$parent/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r$parent/dpo_duel_reason.jsonl /root/r$rid/dpo_duel_reason.jsonl
  fi
done

# launch three axes
bash /root/mining_src/r1091-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-midlr/lean_train_crown_gpus13_p4223.sh
bash /root/mining_src/r1092-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr/lean_train_crown_gpus67_p4223.sh
bash /root/mining_src/r1093-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-midlr/lean_train_crown_gpus45_p4223.sh

echo '===LAUNCHED==='
for rid in 1091 1092 1093; do
  echo "r$rid train.pid=$(cat /root/logs/r${rid}_train.pid 2>/dev/null) wait=$(cat /root/logs/r${rid}_wait_merge.pid 2>/dev/null) n80=$(cat /root/logs/r${rid}_merge_then_n80.pid 2>/dev/null)"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ss -lptn '( sport = :8000 or sport = :8001 or sport = :8002 or sport = :8003 or sport = :8004 )' 2>/dev/null | head -10
REMOTE
