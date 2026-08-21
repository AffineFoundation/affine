#!/usr/bin/env bash
# p4295: mine-r340 R1142/R1144 NCCL-stall orphans + R1156 TP1 util0.90 OOM.
# Kill by PID (never pkill -f). Patch lean → TP1 util≤0.85 + FORCE Triton wipe
# + /v1/completions probe (p4285 pattern). Relaunch all three n80s.
set -euo pipefail
ROOT=/home/const/subnet120/mining
KH="$ROOT/.ralph/known_hosts"
KEY=/home/const/.ssh/id_ed25519
SSH_BASE=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)
SCP_BASE=(scp -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)

EXP1142=r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr
EXP1144=r1144-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr
EXP1156=r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr

HOST=18.118.83.97
PORT=40127

echo "[p4295] $(date -u +%Y-%m-%dT%H:%M:%SZ) patch lean scripts on host"

python3 - <<'PY'
from pathlib import Path

def patch(path: Path, rid: str, gpus: str, port: str, merge: str, util: str = "0.85", tp: str = "1"):
    t = path.read_text()
    # TP1 util floor (p4268 NCCL / p4269 OOM)
    t = t.replace("GPUS=1,2\n", f"GPUS={gpus}\n", 1) if "GPUS=1,2" in t else t
    t = t.replace("GPUS=6,7\n", f"GPUS={gpus}\n", 1) if "GPUS=6,7" in t else t
    if rid == "r1156":
        t = t.replace("GPUS=3\n", f"GPUS={gpus}\n", 1)
        t = t.replace("UTIL=${UTIL:-0.90}", f"UTIL=${{UTIL:-{util}}}", 1)
    else:
        if f"UTIL=${{UTIL:-0.72}}" in t:
            t = t.replace("UTIL=${UTIL:-0.72}", f"UTIL=${{UTIL:-{util}}}", 1)
        elif "UTIL=${UTIL:-0.72}" in t:
            t = t.replace("UTIL=${UTIL:-0.72}", f"UTIL=${{UTIL:-{util}}}", 1)
    t = t.replace("--tensor-parallel-size 2 \\", f"--tensor-parallel-size {tp} \\", 1)
    t = t.replace("--tensor-parallel-size 1 \\", f"--tensor-parallel-size {tp} \\", 1)

    # Fix broken awk (slash inside regex)
    old_awk = f"done < <(ps -eo pid=,args= | awk '/vllm serve .*/tmp/{rid}_merged/ && !/awk/ {{print $1}}')"
    new_awk = (
        f"done < <(ps -eo pid=,args= | awk 'index($0,\"/tmp/{rid}_merged\") "
        f"&& /vllm serve/ && !/awk/ {{print $1}}')"
    )
    if old_awk in t:
        t = t.replace(old_awk, new_awk, 1)
    else:
        # already fixed or variant
        pass

    # FORCE wipe+seed — never reuse half-written chall cache
    import re
    seed_re = re.compile(
        r'_seed_src=""\nfor cand in .*?\nlog "skip triton purge; keep seeded tree intact"',
        re.S,
    )
    new_seed = f'''_seed_src=""
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r969 /root/.triton/cache/chall_r954 /root/.triton/cache/chall_r949 /root/.triton/cache/chall_r941 /root/.triton/cache/chall_r340 /root/.triton/cache/chall; do
  if [[ -d "$cand" ]]; then
    _n=$(find "$cand" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
    if [[ "${{_n:-0}}" -ge 1 ]]; then
      _seed_src=$cand
      break
    fi
  fi
done
rm -rf "$TCACHE"
mkdir -p "$(dirname "$TCACHE")"
if [[ -n "$_seed_src" ]]; then
  log "p4295 FORCE wipe+seed $TCACHE from $_seed_src"
  cp -a "$_seed_src" "$TCACHE"
else
  log "WARN no triton seed; empty $TCACHE"
  mkdir -p "$TCACHE"
fi
chmod -R a+rX "$TCACHE" 2>/dev/null || true
_post_n=$(find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
log "triton seed n_so=$_post_n"'''
    t2, n = seed_re.subn(new_seed, t, count=1)
    if n != 1:
        raise SystemExit(f"{rid}: seed block replace failed n={n}")
    t = t2

    # Probe before n80 if missing
    needle = f'curl -sf -m 5 "http://127.0.0.1:${{CHALL_PORT}}/v1/models" >/dev/null\n\nBLOCK_HASH=$(python3 - <<\'PY\''
    if "probe sample before n80" not in t and needle in t:
        probe = f'''curl -sf -m 5 "http://127.0.0.1:${{CHALL_PORT}}/v1/models" >/dev/null

# p4295: probe one completion before n80
log "probe sample before n80"
if ! CHALL_PORT="$CHALL_PORT" python3 - <<'PY'
import json, os, urllib.request
port = os.environ["CHALL_PORT"]
req = urllib.request.Request(
    f"http://127.0.0.1:{{port}}/v1/completions",
    data=json.dumps({{
        "model": "{merge}",
        "prompt": "Next command:\\n",
        "max_tokens": 8,
        "temperature": 0.0,
    }}).encode(),
    headers={{"Content-Type": "application/json"}},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as r:
    body = json.loads(r.read().decode())
assert body.get("choices"), body
print("PROBE_OK", (body["choices"][0].get("text") or "")[:80])
PY
then
  log "FATAL probe sample failed — see $CHALL_LOG"
  tail -n 80 "$CHALL_LOG" | tee -a "$LOG" || true
  exit 1
fi
log "PROBE_OK — launch n80"

BLOCK_HASH=$(python3 - <<'PY\''''
        t = t.replace(needle, probe, 1)

    t = t.replace(f"[p4267-{rid}]", f"[p4295-{rid}]")
    t = t.replace(f"[p4270-{rid}]", f"[p4295-{rid}]")
    t = t.replace(f"[p4278-{rid}]", f"[p4295-{rid}]")
    path.write_text(t)
    print("patched", path.name, "gpus", gpus, "tp", tp, "util", util)

root = Path("/home/const/subnet120/mining/experiments")
patch(root / "r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr" / "lean_chall_n80_r340_gpus12_p4267.sh",
      "r1142", "1", "8002", "/tmp/r1142_merged")
patch(root / "r1144-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr" / "lean_chall_n80_r340_gpus67_p4270.sh",
      "r1144", "6", "8004", "/tmp/r1144_merged")
patch(root / "r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr" / "lean_chall_n80_r340_gpus34_p4278.sh",
      "r1156", "3", "8003", "/tmp/r1156_merged")
PY

echo "[p4295] $(date -u +%Y-%m-%dT%H:%M:%SZ) sync+kill+relaunch on r340"

"${SCP_BASE[@]}" -P "$PORT" \
  "$ROOT/experiments/$EXP1142/lean_chall_n80_r340_gpus12_p4267.sh" \
  "root@${HOST}:/root/mining_src/$EXP1142/"
"${SCP_BASE[@]}" -P "$PORT" \
  "$ROOT/experiments/$EXP1144/lean_chall_n80_r340_gpus67_p4270.sh" \
  "root@${HOST}:/root/mining_src/$EXP1144/"
"${SCP_BASE[@]}" -P "$PORT" \
  "$ROOT/experiments/$EXP1156/lean_chall_n80_r340_gpus34_p4278.sh" \
  "root@${HOST}:/root/mining_src/$EXP1156/"

"${SSH_BASE[@]}" -p "$PORT" root@"$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
# verify merges
for r in r1142 r1144 r1156; do
  [[ -f /tmp/${r}_merged/config.json ]] || { echo FATAL no $r merge; exit 1; }
  n=$(ls /tmp/${r}_merged/model-*-of-*.safetensors | wc -l)
  echo "shards $r=$n"
  [[ "$n" -ge 16 ]] || exit 1
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null

# Kill orphaned R1142 / R1144 vLLM trees by PID (never pkill -f)
kill_tree() {
  local root=$1
  [[ -n "$root" && "$root" =~ ^[0-9]+$ ]] || return 0
  local kids
  kids=$(ps -eo pid=,ppid= | awk -v p="$root" '$2==p {print $1}')
  for k in $kids; do kill_tree "$k"; done
  if kill -0 "$root" 2>/dev/null; then
    echo "kill $root"
    kill "$root" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$root" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$root" 2>/dev/null || true
  fi
}
for pidf in /root/logs/vllm_chall_r1142.pid /root/logs/vllm_chall_r1144.pid /root/logs/vllm_chall_r1156.pid; do
  if [[ -f "$pidf" ]]; then
    kill_tree "$(cat "$pidf" 2>/dev/null || true)"
    rm -f "$pidf"
  fi
done
# known orphans from p4295 diagnose
for pid in 73638 73974 74256 74257 76735 77060 77350 77351; do
  kill_tree "$pid"
done
# also any leftover workers holding GPUs 1,2,3,6
sleep 3
for i in 1 2 3 6; do
  used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  echo "GPU$i used=$used"
  # reap leftover compute apps on these GPUs (not teacher/king)
done
# reap by uuid for GPUs 1,2,3,6 if still >2GiB
python3 - <<'PY'
import os, signal, subprocess, time
want = {1, 2, 3, 6}
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
idx_to_uuid = {}
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        idx_to_uuid[int(parts[0])] = parts[1]
uuids = {idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
    text=True,
)
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or parts[0] not in uuids:
        continue
    pid = int(parts[1])
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
    except Exception:
        cmd = ""
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1142_merged" not in cmd and "r1144_merged" not in cmd and "r1156_merged" not in cmd:
            print(f"SKIP TK pid={pid}", flush=True)
            continue
    if "train_dpo" in cmd:
        print(f"SKIP train pid={pid}", flush=True)
        continue
    print(f"reap leftover pid={pid} cmd={cmd[:80]}", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or parts[0] not in uuids:
        continue
    pid = int(parts[1])
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
print("reap done", flush=True)
PY

for i in 1 2 3 6; do
  used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  echo "post-reap GPU$i used=$used"
  [[ "$used" -lt 4096 ]] || { echo FATAL GPU$i still busy; exit 1; }
done

chmod +x /root/mining_src/r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r340_gpus12_p4267.sh
chmod +x /root/mining_src/r1144-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r340_gpus67_p4270.sh
chmod +x /root/mining_src/r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r340_gpus34_p4278.sh

nohup bash /root/mining_src/r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r340_gpus12_p4267.sh \
  >/root/logs/p4295_r1142_chall_n80_relaunch.nohup 2>&1 &
echo $! | tee /root/logs/p4295_r1142_chall_n80_relaunch.pid
nohup bash /root/mining_src/r1144-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r340_gpus67_p4270.sh \
  >/root/logs/p4295_r1144_chall_n80_relaunch.nohup 2>&1 &
echo $! | tee /root/logs/p4295_r1144_chall_n80_relaunch.pid
nohup bash /root/mining_src/r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r340_gpus34_p4278.sh \
  >/root/logs/p4295_r1156_chall_n80_relaunch.nohup 2>&1 &
echo $! | tee /root/logs/p4295_r1156_chall_n80_relaunch.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4295_r340_n80_relaunch_armed.done
echo R340_RELAUNCH_ARMED
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
REMOTE

echo "[p4295] $(date -u +%Y-%m-%dT%H:%M:%SZ) armed — poll chall readiness"
