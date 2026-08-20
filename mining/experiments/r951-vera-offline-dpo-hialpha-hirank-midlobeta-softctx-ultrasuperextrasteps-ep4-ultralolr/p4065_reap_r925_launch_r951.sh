#!/usr/bin/env bash
# p4065: exact-PID reap R925 chall :8002 GPUs6,7 after REFUTE → launch R951 TRAIN. Never pkill -f. Leave TK+R942 alone.
set -euo pipefail
exec >/root/logs/p4065_reap_r925_launch_r951.nohup 2>&1
echo "[p4065-r925] $(date -u +%Y-%m-%dT%H:%M:%SZ) START"
stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4065-r925] kill $pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
# known chall parents / workers from p4065 poll
for pid in 65373 65866 65867 66268 66269; do stop_pid "$pid"; done
[[ -f /root/logs/vllm_chall_r925.pid ]] && stop_pid "$(cat /root/logs/vllm_chall_r925.pid)"
while read -r pid; do stop_pid "$pid"; done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)
python3 - <<'PY'
import os, signal, subprocess, time
want = {6, 7}
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
idx = {}
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        idx[int(parts[0])] = parts[1]
uuids = {idx[i] for i in want if i in idx}
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    text=True,
)
pids = []
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or parts[0] not in uuids:
        continue
    try:
        pid = int(parts[1])
    except ValueError:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
    except Exception:
        cmd = ""
    if any(t in cmd for t in ["train_dpo", "train_online", "train_full", "train_reason"]):
        print("SKIP train", pid)
        continue
    if ("GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd) and "r925_merged" not in cmd and ":8002" not in cmd:
        print("SKIP TK", pid)
        continue
    print("kill gpu app", pid, cmd[:120])
    pids.append(pid)
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
print("reap python done")
PY
for i in $(seq 1 40); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  echo "[p4065-r925] vram67=$used iter=$i"
  [[ "$used" -lt 2000 ]] && break
  sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
if curl -sf -m 2 http://127.0.0.1:8002/v1/models >/dev/null; then echo STILL_UP; else echo PORT8002_DOWN; fi
curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null && echo T_OK || echo T_BAD
curl -sf -m 2 http://127.0.0.1:8001/v1/models >/dev/null && echo K_OK || echo K_BAD
# write decision stamp
python3 - <<'PY'
import json
from pathlib import Path
v=json.load(open("/root/affine_data/r925_sim_result_reign36_wvk7.json"))["verdict"]
c=v["challenger"]; se=float(v["se"]); m=float(v["margin"]); bar=max(2*se,0.002)
dec={
  "utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
  "hypo":"R925","contract":"wvk7","king":"reign36",
  "margin":m,"se":se,"z":v["z"],"n":v["n_paired_turns"],"bar":bar,"ratio":m/bar,
  "thought_median":c["median_len_z"],"b_pass":c["b_gate_pass_rate"],
  "n_teacher_samples":v["duel_params"]["n_teacher_samples"],"tau":v["duel_params"]["tau"],
  "wins":v["challenger_wins"],
  "note":"p4065 R925 REFUTE ~-0.001× → exact-PID reap :8002 → R951 SoftCtx HiRank MidLoβ UltraExtra",
}
Path("/root/affine_data/r925_decision_reign36_wvk7.json").write_text(json.dumps(dec,indent=2)+"\n")
print(json.dumps(dec,indent=2))
PY
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4065_reap_r925.done
echo "[p4065-r925] launch R951"
chmod +x /root/mining_src/r951-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-ultrasuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r951-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_r252_gpus67_p4065.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4065_r951_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r951-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-ultrasuperextrasteps-ep4-ultralolr/wait_r951_train_then_merge_p4065.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4065_r951_wait_outer.pid
echo "[p4065-r925] DONE lean=$(cat /root/logs/p4065_r951_lean_outer.pid) wait=$(cat /root/logs/p4065_r951_wait_outer.pid)"
