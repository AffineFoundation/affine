#!/usr/bin/env bash
# p4270: R1127 MERGE done but prior TP2 chall died (KeyboardInterrupt during tokenizer
# load); GPUs 6,7 idle. Prefer TP1 util=0.85 on GPU6 (p4269 OOM lesson @0.90).
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1126:8002 / R1143 TRAIN 4,5.
set -euo pipefail

LOG=/root/logs/p4270_r1127_rearm_tp1.log
PIDF_OUTER=/root/logs/p4270_r1127_rearm_tp1.pid
MERGE=/tmp/r1127_merged
CHALL_PORT=8003
GPU=6
UTIL=0.85
TCACHE=/root/.triton/cache/chall_r1127_tp1
mkdir -p /root/logs /root/tmp /root/affine_data
: >"$LOG"
echo $$ >"$PIDF_OUTER"
log() { echo "[p4270-r1127] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "SIGTERM pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    log "SIGKILL pid=$pid ($why)"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

mapfile -t KILL_PIDS < <(python3 - <<'PY'
import os, subprocess
want_gpus = {6, 7}
kill = set()
me = os.getpid()
parent = os.getppid()

try:
    out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
except Exception:
    out = ""
for line in out.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if len(parts) < 2:
        continue
    pid_s, args = parts[0], parts[1]
    if not pid_s.isdigit():
        continue
    pid = int(pid_s)
    if pid in (me, parent):
        continue
    # Do NOT match this script's own basename (self-kill bug p4269).
    if "p4270_rearm_r1127_tp1" in args:
        continue
    if "lean_chall_n80_r339_gpus67_p4254.sh" in args:
        kill.add(pid)
    if "wait_r1127_merge_then_n80" in args:
        kill.add(pid)
    if "vllm serve" in args and "r1127_merged" in args:
        kill.add(pid)
    if "run_sim_duel.py" in args and "r1127" in args:
        kill.add(pid)

try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name", "--format=csv,noheader"],
        text=True,
    )
except Exception:
    smi = ""
try:
    idx = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True,
    )
except Exception:
    idx = ""
bus_to_i = {}
for line in idx.splitlines():
    bits = [b.strip() for b in line.split(",")]
    if len(bits) >= 2:
        try:
            bus_to_i[bits[1].lower()] = int(bits[0])
        except Exception:
            pass

for line in smi.splitlines():
    bits = [b.strip() for b in line.split(",")]
    if len(bits) < 3:
        continue
    bus, pid_s, name = bits[0], bits[1], bits[2]
    if not pid_s.isdigit():
        continue
    gi = bus_to_i.get(bus.lower())
    if gi in want_gpus:
        pid = int(pid_s)
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
        except Exception:
            cmd = ""
        if "train_dpo" in cmd or "train_online" in cmd:
            continue
        if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
            if "r1127_merged" not in cmd:
                continue
        kill.add(pid)

for pf in [
    "/root/logs/vllm_chall_r1127.pid",
    "/root/logs/vllm_chall_r1125.pid",
    "/root/logs/r1127_sim_wvk7.pid",
    "/root/logs/r1127_merge_then_n80.pid",
]:
    try:
        p = open(pf).read().strip()
        if p.isdigit():
            kill.add(int(p))
    except Exception:
        pass

extra = set()
for pid in list(kill):
    try:
        for cdir in os.listdir("/proc"):
            if not cdir.isdigit():
                continue
            try:
                ppid = open(f"/proc/{cdir}/stat").read().split()[3]
                if int(ppid) == pid:
                    extra.add(int(cdir))
            except Exception:
                pass
    except Exception:
        pass
kill |= extra
kill.discard(me)
kill.discard(parent)
print("\n".join(str(p) for p in sorted(kill)))
PY
)

log "exact-PID reap list: ${KILL_PIDS[*]:-none}"
for pid in "${KILL_PIDS[@]:-}"; do
  stop_pid "$pid" "r1127 chall tree"
done

for i in $(seq 1 45); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then
    log "GPUs 6,7 free poll=$i used_mib=$used"
    break
  fi
  sleep 2
done

[[ -f "$MERGE/config.json" ]] || { log "FATAL missing merge"; exit 1; }
n=$(ls "$MERGE"/model-*-of-*.safetensors 2>/dev/null | wc -l)
log "merge shards=$n util=$UTIL"

rm -rf "$TCACHE"
mkdir -p "$TCACHE"
for cand in /root/.triton/cache/king /root/.triton/cache/chall /root/.triton/cache/chall_r1126 /root/.triton/cache/chall_r1127; do
  if [[ -d "$cand" ]] && find "$cand" -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
    log "seed $TCACHE from $cand"
    cp -a "$cand/." "$TCACHE/"
    break
  fi
done

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export TMPDIR=/root/tmp
export TRITON_CACHE_DIR=$TCACHE
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_TOKEN

CHALL_LOG=/root/logs/vllm_chall_r1127_p4270_tp1.log
PIDF=/root/logs/vllm_chall_r1127.pid
: >"$CHALL_LOG"
nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
  --port "$CHALL_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN \
  --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' \
  --enforce-eager \
  >"$CHALL_LOG" 2>&1 &
echo $! >"$PIDF"
CHALL_PID=$(cat "$PIDF")
log "chall TP1 pid=$CHALL_PID port=$CHALL_PORT GPU=$GPU util=$UTIL"

ready=0
for i in $(seq 1 360); do
  if curl -sf -m 3 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null 2>&1; then
    log "CHALL_READY poll=$i"
    ready=1
    break
  fi
  if ! kill -0 "$CHALL_PID" 2>/dev/null; then
    log "FATAL chall died early; tail:"
    tail -40 "$CHALL_LOG" | tee -a "$LOG"
    exit 2
  fi
  if [[ $i -eq 90 || $i -eq 180 ]]; then
    u=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1+0}')
    log "health-wait i=$i gpu${GPU}_mib=$u"
    if [[ $i -ge 180 && "${u:-0}" -lt 4000 ]]; then
      log "FATAL VRAM stall; abort TP1 rearm"
      stop_pid "$CHALL_PID" "stall"
      exit 3
    fi
  fi
  sleep 5
done
[[ "$ready" == "1" ]] || { log "FATAL CHALL not ready after 30m"; exit 4; }

KING_ID=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "king id=$KING_ID"

SIM_OUT=/root/affine_data/r1127_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r1127_sim_progress_reign36_wvk7.json
DEC=/root/affine_data/r1127_decision_reign36_wvk7.json
rm -f "$SIM_OUT" "$PROG" "$DEC"
N80_LOG=/root/logs/p4270_r1127_chall_n80_wvk7.log
: >"$N80_LOG"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo zai-org/GLM-4.5-Air-FP8 \
  --king-repo "$KING_ID" \
  --king-rev 8e3f1695e058837ed80fec3238ff439fdc2d0f0e \
  --chall-repo "$MERGE" \
  --chall-rev local \
  --chall-port "$CHALL_PORT" \
  --n-turns 80 \
  --hotkey local-r1127-reign36-wvk7-p4270-tp1-u085 \
  --block-hash "$(python3 -c 'import hashlib,time; print(hashlib.sha256(("r1127-p4270-"+str(time.time())).encode()).hexdigest())')" \
  --out "$SIM_OUT" \
  --progress-out "$PROG" \
  --save-artifact \
  >"$N80_LOG" 2>&1 &
echo $! >/root/logs/r1127_sim_wvk7.pid
SIM_PID=$(cat /root/logs/r1127_sim_wvk7.pid)
log "n80 sim pid=$SIM_PID log=$N80_LOG"
echo "ARMED" >/root/logs/r1127_n80_launched.p4270
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4270_r1127_tp1_armed.done

# write decision when sim finishes (background waiter)
nohup bash -c '
set -euo pipefail
SIM_PID=$(cat /root/logs/r1127_sim_wvk7.pid)
while kill -0 "$SIM_PID" 2>/dev/null; do sleep 30; done
SIM_OUT=/root/affine_data/r1127_sim_result_reign36_wvk7.json
DEC=/root/affine_data/r1127_decision_reign36_wvk7.json
LOG=/root/logs/p4270_r1127_chall_n80_wvk7.log
if [[ -f "$SIM_OUT" ]]; then
  /root/venv/bin/python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
d=json.loads(Path("'"$SIM_OUT"'").read_text())
v=d.get("verdict") if isinstance(d.get("verdict"), dict) else {}
chal=(v.get("challenger") or {}) if isinstance(v, dict) else {}
dp=(v.get("duel_params") or {}) if isinstance(v, dict) else {}
margin = v.get("margin") if v else (d.get("margin") or d.get("mean_margin"))
se = v.get("se") if v else (d.get("se") or d.get("stderr"))
bar = None
try:
    if se is not None:
        bar = max(2.0 * float(se), 0.002)
except Exception:
    bar = None
dec={
  "utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
  "hypo": "R1127",
  "contract": "wvk7",
  "n_teacher_samples": dp.get("n_teacher_samples"),
  "tau": dp.get("tau"),
  "king": "reign36",
  "margin": margin,
  "se": se,
  "z": v.get("z") if v else d.get("z"),
  "n": v.get("n_paired_turns") if v else (d.get("n") or d.get("n_scored")),
  "bar": bar,
  "thought_median": chal.get("median_len_z"),
  "b_pass": chal.get("b_gate_pass_rate"),
  "wins": v.get("challenger_wins") if v else d.get("wins"),
  "note": "p4270 TP1 util0.85 :8003 GPU6; HiAlpha LoRank Midβ ShortCtx HyperSuperExtra ep4 HiLR (β=0.1 r=16 lr=2e-6 @6144); vs reign36 vera wvk7",
  "hf_ok": False,
  "raw_keys": sorted(d.keys())[:40],
}
Path("'"$DEC"'").write_text(json.dumps(dec, indent=2)+"\n")
print(json.dumps(dec, indent=2))
k = dp.get("n_teacher_samples")
if k != 3:
    raise SystemExit(f"FATAL duel_params.n_teacher_samples={k} (want 3)")
PY
fi
' >/root/logs/p4270_r1127_decision_waiter.nohup 2>&1 &

log "DONE re-arm TP1 util=$UTIL +n80"
