#!/usr/bin/env bash
# p4175: R1025 chall :8003 OOM'd mid-n80 (util0.72@65536). Teacher TP4 :8000 +
# king :8001 still warm; GPUs5,6 free. Relaunch chall with expandable_segments +
# util0.65 + max_num_batched_tokens 4096, then fresh v4 n80. Never pkill -f.
# Do not touch teacher :8000 or king :8001.
set -euo pipefail
LOG=/root/logs/p4175_r926_chall_rearm_n80.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4175] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4175] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

source /root/venv/bin/activate
[[ -f /root/mine.env ]] && { set -a; source /root/mine.env; set +a; }
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}

MERGE=/tmp/r1025_merged
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
tag=r1025
port=8003
gpus=5,6

test -d "$MERGE" || { echo FATAL merge; exit 1; }
python3 - <<'PY'
import json
from pathlib import Path
out=Path("/tmp/r1025_merged")
assert (out/"config.json").exists(), "missing config"
wm=json.load(open(out/"model.safetensors.index.json"))["weight_map"]
missing=[s for s in set(wm.values()) if not (out/s).exists()]
assert not missing, missing
print("MERGE_OK nkeys",len(wm),"shards",len(set(wm.values())), flush=True)
PY

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { echo FATAL teacher; exit 1; }
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king; exit 1; }
kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4175] king=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo ERROR king not vera; exit 5; }

for gi in 5 6; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gi" | awk '{print $1+0}')
  echo "[p4175] gpu$gi used_mib=$used"
  [[ "$used" -lt 2048 ]] || { echo FATAL GPU$gi busy; nvidia-smi; exit 2; }
done

# Stop stale sim + chall only (exact PIDs / :8003 listeners). Never pkill -f.
[[ -f /root/logs/${tag}_sim_wvk7.pid ]] && stop_pid "$(cat /root/logs/${tag}_sim_wvk7.pid)" "stale sim"
[[ -f /root/logs/vllm_chall_${tag}.pid ]] && stop_pid "$(cat /root/logs/vllm_chall_${tag}.pid)" "stale chall pidfile"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" ":8003"
done < <(ss -lptn "sport = :8003" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
# Reap orphan compute on GPUs 5,6 only (protect teacher 0,1,3,4 + king 2)
python3 - <<'PY'
import os, signal, subprocess, time
want={5,6}
uu={}
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True).splitlines():
  idx,u=line.split(","); uu[u.strip()]=int(idx.strip())
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader"], text=True)
kill=set(); keep=set()
for line in apps.splitlines():
  parts=[p.strip() for p in line.split(",")]
  if len(parts)<2: continue
  pid=int(parts[0]); gi=uu.get(parts[1])
  if gi in (0,1,2,3,4): keep.add(pid)
  elif gi in want: kill.add(pid)
print(f"[p4175] reap={sorted(want)} kill={sorted(kill-keep)} keep={sorted(keep)}", flush=True)
for pid in sorted(kill-keep):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill-keep):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
PY
sleep 2
for gi in 5 6; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gi" | awk '{print $1+0}')
  [[ "$used" -lt 2048 ]] || { echo FATAL GPU$gi still busy used=$used; exit 2; }
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null

chall_log=/root/logs/vllm_chall_${tag}_p4175.log
pidf=/root/logs/vllm_chall_${tag}.pid
tcache=/root/.triton/cache/chall_${tag}
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r1025 /root/.triton/cache/chall_r973 /root/.triton/cache/chall_r944; do
  if [[ -d "$cand" ]] && find "$cand" -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
    echo "[p4175] seed triton from $cand"
    rm -rf "$tcache"; cp -a "$cand" "$tcache"; break
  fi
done
mkdir -p "$tcache"
: >"$chall_log"
echo "[p4175] launch chall :$port GPUs$gpus util=0.65 max_model_len=65536 batched=4096 expandable"
CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$tcache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
    --port "$port" --tensor-parallel-size 2 --max-model-len 65536 \
    --gpu-memory-utilization 0.65 --max-num-batched-tokens 4096 \
    --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton --additional-config '{"gdn_prefill_backend": "triton"}' \
    --enforce-eager >"$chall_log" 2>&1 &
echo $! >"$pidf"
chall_pid=$(cat "$pidf")
echo "[p4175] chall pid=$chall_pid"

ready=0
for i in $(seq 1 360); do
  curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { ready=1; echo "[p4175] CHALL_READY poll=$i"; break; }
  kill -0 "$chall_pid" 2>/dev/null || { echo FATAL chall died; tail -100 "$chall_log"; exit 1; }
  grep -q 'ValueError: There is no module' "$chall_log" 2>/dev/null && { echo FATAL layout; tail -40 "$chall_log"; exit 1; }
  grep -q 'ImportError:.*__triton_launcher' "$chall_log" 2>/dev/null && { echo FATAL Triton; tail -40 "$chall_log"; exit 1; }
  grep -q 'CUDA out of memory' "$chall_log" 2>/dev/null && { echo FATAL OOM at boot; tail -40 "$chall_log"; exit 1; }
  (( i % 12 == 0 )) && echo "[p4175] wait chall iter=$i last=$(tail -1 "$chall_log" 2>/dev/null | cut -c1-120)"
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL not ready; tail -100 "$chall_log"; exit 1; }

PROBE_OK=0
for i in $(seq 1 8); do
  if curl -sf -m 90 "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MERGE"'","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"temperature":0}' \
    >/tmp/r1025_probe_p4175.json 2>/dev/null; then
    PROBE_OK=1
    echo "[p4175] probe_ok poll=$i"
    break
  fi
  echo "[p4175] probe_retry=$i"
  sleep 5
done
[[ "$PROBE_OK" -eq 1 ]] || { echo FATAL probe failed; tail -80 "$chall_log"; exit 1; }

# Confirm TK still warm
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4175] TKC warm — arm n80"

bh=$(python3 -c 'import hashlib,time;print(hashlib.sha256(f"r1025-reign36-wvk7-p4175-{time.time()}".encode()).hexdigest())')
sim_out=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
prog=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
dec=/root/affine_data/${tag}_decision_reign36_wvk7.json
rm -f "$sim_out" "$prog" "$dec"
: >/root/logs/p4175_${tag}_chall_n80_wvk7.log
nohup env -u HF_TOKEN \
  HF_HOME="${HF_HOME}" PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/venv/bin/python3 "$SIM" \
  --teacher-repo "$TEACHER_REPO" --king-repo "$kid" --king-rev "$KING_REV" \
  --chall-repo "$MERGE" --chall-rev local --chall-port "$port" --n-turns 80 \
  --hotkey "local-r1025-reign36-wvk7-p4175" --block-hash "$bh" \
  --out "$sim_out" --progress-out "$prog" --save-artifact \
  >>/root/logs/p4175_${tag}_chall_n80_wvk7.log 2>&1 &
echo $! >/root/logs/${tag}_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/${tag}_n80_launched.p4175
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4175_r1025_chall_n80_armed.done
echo "[p4175] ARMED n80 pid=$(cat /root/logs/${tag}_sim_wvk7.pid) bh=${bh:0:16}"

SIM_PID=$(cat /root/logs/${tag}_sim_wvk7.pid)
for i in $(seq 1 120); do
  if [[ -f "$prog" ]]; then echo "[p4175] progress_tick $(cat "$prog")"; break; fi
  kill -0 "$SIM_PID" 2>/dev/null || { echo FATAL sim died; tail -80 /root/logs/p4175_${tag}_chall_n80_wvk7.log; exit 1; }
  # Early OOM detect on chall
  if grep -q 'CUDA out of memory' "$chall_log" 2>/dev/null; then
    echo FATAL chall OOM during n80; tail -40 "$chall_log"; exit 1
  fi
  sleep 5
done

nohup bash -c "
set -euo pipefail
SIM_PID=\$(cat /root/logs/${tag}_sim_wvk7.pid)
while kill -0 \"\$SIM_PID\" 2>/dev/null; do sleep 30; done
wait \"\$SIM_PID\" || true
if [[ -f $sim_out ]]; then
  /root/venv/bin/python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('$sim_out').read_text())
v=d.get('verdict') if isinstance(d.get('verdict'), dict) else {}
chal=(v.get('challenger') or {}) if isinstance(v, dict) else {}
dp=(v.get('duel_params') or {}) if isinstance(v, dict) else {}
margin = v.get('margin') if v else (d.get('margin') or d.get('mean_margin'))
se = v.get('se') if v else (d.get('se') or d.get('stderr'))
bar = max(2.0*float(se), 0.002) if se is not None else None
dec={
  'utc': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
  'hypo': 'R1025', 'contract': 'wvk7',
  'n_teacher_samples': dp.get('n_teacher_samples'), 'tau': dp.get('tau'),
  'king': 'reign36', 'margin': margin, 'se': se,
  'z': v.get('z') if v else d.get('z'),
  'n': v.get('n_paired_turns') if v else d.get('n'),
  'bar': bar, 'thought_median': chal.get('median_len_z'),
  'b_pass': chal.get('b_gate_pass_rate'),
  'wins': v.get('challenger_wins') if v else d.get('wins'),
  'note': 'p4175 chall rearm util0.65 expandable + fresh v4 n80 R1025 cryptoDev SoftCtx MidRank Midβ Ultra MidLR vs reign36',
  'hf_ok': False,
}
Path('$dec').write_text(json.dumps(dec, indent=2)+'\n')
print(json.dumps(dec, indent=2))
if dp.get('n_teacher_samples') != 3:
  raise SystemExit('FATAL k!=3')
PY
  date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1025_reign36_wvk7_pipeline.done
  echo '[p4175] SIM_DONE'
else
  echo FATAL missing sim result; exit 1
fi
" >/root/logs/p4175_r1025_decision_waiter.nohup 2>&1 &
echo $! >/root/logs/p4175_r1025_decision_waiter.pid
echo "[p4175] DONE launch $(date -u +%Y-%m-%dT%H:%M:%SZ)"
