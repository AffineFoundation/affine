#!/usr/bin/env bash
# p4080: R955 MERGE_DONE idle on R338 → chall :8002 GPUs6,7 + v4 n80 vs reign36 (wvk=7 k=3 τ=0.03).
# Axis: vera Soft Mid Mid Soft HiAlpha HiRank MidBeta MidCtx UltraSuperExtra ep4×UltraLoLR
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R947 train on GPUs 4,5.
set -euo pipefail

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi

GPUS=6,7
CHALL_PORT=8002
export CUDA_VISIBLE_DEVICES=$GPUS

export HF_HOME=${HF_HOME:-/root/hf}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
_CU13="${_SITE}/nvidia/cu13"
if [[ -x "${_CU13}/bin/nvcc" && -f "${_CU13}/include/cuda_fp16.h" ]]; then
  export CUDA_HOME=${CUDA_HOME:-$_CU13}
  export CUDA_PATH=$CUDA_HOME
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
fi

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r955_merged
UTIL=${UTIL:-0.72}
LOG=/root/logs/p4080_r955_chall_n80_wvk7.log
CHALL_LOG=/root/logs/vllm_chall_r955_p4080.log
PIDF=/root/logs/vllm_chall_r955.pid
TCACHE=/root/.triton/cache/chall_r955
SIM_N80=/root/affine_data/r955_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r955_sim_progress_reign36_wvk7.json
SIM_DEC=/root/affine_data/r955_decision_reign36_wvk7.json
mkdir -p /root/logs /root/affine_data

: >"$LOG"
log() { echo "[p4080-r955] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1 why=${2:-}
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)" "$why pidf=$pidf"
  rm -f "$pidf"
}

hub_ok() {
  local path=$1 n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge 16 ]]
}

log "START R955 chall+v4-n80 R338 GPUs=$GPUS merge=$MERGE_DIR"
hub_ok "$MERGE_DIR" || { log "FATAL merge incomplete"; exit 1; }
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
log "reuse merge shards=$n"

cat >"$MERGE_DIR/README.md" <<'CARD'
# R955 merged challenger (local)

- **Base:** `vera6/affine-5g4yy75zuz-t6` @ `8e3f1695…` (live reign36 king)
- **Method:** Offline DPO on Reason duel pairs (teacher-side)
- **Axis:** HiAlpha HiRank MidBeta MidCtx UltraSuperExtra ep4 UltraLoLR
- **Knobs:** lr=5e-7, LoRA r=64 / α=128, β=0.1, max_len=8192, epochs=4, max_steps=28800
- **Hardware:** mine-r338 8×B200 GPUs 6,7 TP2 :8002; TK on same pod
- **Parent signal:** R935 HiRank Loβ MidCtx REFUTE ~0.065× → UltraExtra isolate
- **Experiment:** `mining/experiments/r955-vera-offline-dpo-hialpha-hirank-midbeta-midctx-ultrasuperextrasteps-ep4-ultralolr/`
CARD

for i in $(seq 1 90); do
  if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
    && curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    log "TEACHER+KING warm poll=$i"
    break
  fi
  sleep 2
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "pre-chall king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { log "FATAL king not reign36 vera ($kid)"; exit 5; }

stop_pidfile "$PIDF" "stale r955 chall"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r955_merged argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r955_merged/ && !/awk/ {print $1}')

GPUS="$GPUS" python3 - <<'PY'
import os, signal, subprocess, time
want = {int(x) for x in os.environ.get("GPUS", "6,7").split(",") if x.strip()}
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
idx_to_uuid = {}
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        idx_to_uuid[int(parts[0])] = parts[1]
uuids = {idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    text=True,
)
kill_set = set()
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
    if pid <= 1:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
    except Exception:
        cmd = ""
    if any(tok in cmd for tok in ["train_online_dpo", "train_dpo", "train_full", "train_reason_grpo", "r947", ":8003"]):
        print(f"[p4080-r955] SKIP sibling/train pid={pid}", flush=True)
        continue
    if any(tok in cmd for tok in ["GLM-4.5-Air", ":8000", ":8001", "vera6", "5g4yy75zuz"]):
        if "r955_merged" not in cmd:
            print(f"[p4080-r955] SKIP TK pid={pid}", flush=True)
            continue
    kill_set.add(pid)
print(f"[p4080-r955] reap gpu={sorted(want)} kill={sorted(kill_set)}", flush=True)
for pid in kill_set:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in kill_set:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
print("[p4080-r955] GPUs 6,7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then
    log "GPUs 6,7 free (poll $i used_mib=$used)"
    break
  fi
  sleep 2
done

_seed_src=""
for cand in /root/.triton/cache/chall_r935 /root/.triton/cache/chall_r937 /root/.triton/cache/king /root/.triton/cache/chall_r338 /root/.triton/cache/chall; do
  if [[ -d "$cand" ]]; then
    _n=$(find "$cand" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
    if [[ "${_n:-0}" -ge 1 ]]; then
      _seed_src=$cand
      break
    fi
  fi
done
if [[ -n "$_seed_src" ]]; then
  log "FORCE wipe+seed $TCACHE from $_seed_src"
  rm -rf "$TCACHE"
  mkdir -p "$(dirname "$TCACHE")"
  cp -a "$_seed_src" "$TCACHE"
else
  log "WARN no triton seed; empty $TCACHE"
  mkdir -p "$TCACHE"
fi
log "skip triton purge; keep seeded tree intact"

export CUDA_VISIBLE_DEVICES=$GPUS
export TRITON_CACHE_DIR=$TCACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
: >"$CHALL_LOG"
nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE_DIR" \
  --port "$CHALL_PORT" \
  --tensor-parallel-size 2 \
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
log "chall pid=$CHALL_PID port=$CHALL_PORT"

for i in $(seq 1 240); do
  if curl -sf -m 3 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null 2>&1; then
    log "CHALL_READY poll=$i"
    break
  fi
  if ! kill -0 "$CHALL_PID" 2>/dev/null; then
    log "FATAL chall died"; tail -n 100 "$CHALL_LOG"; exit 1
  fi
  if grep -q 'ImportError:.*__triton_launcher' "$CHALL_LOG" 2>/dev/null; then
    log "FATAL Triton ImportError"; tail -n 80 "$CHALL_LOG"; exit 1
  fi
  sleep 5
done
curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null

BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r955-reign36-wvk7-p4080-{time.time()}".encode()).hexdigest())
PY
)
kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "launch n80 vs king_id=$kid block_hash=${BLOCK_HASH:0:16}…"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$kid" \
  --king-rev "$KING_REV" \
  --chall-repo "$MERGE_DIR" \
  --chall-rev local \
  --chall-port "$CHALL_PORT" \
  --n-turns 80 \
  --hotkey local-r955-reign36-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" >/root/logs/r955_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r955_n80_launched.p4080
log "n80 pid=$SIM_PID — waiting"

while kill -0 "$SIM_PID" 2>/dev/null; do sleep 30; done
wait "$SIM_PID" || true

if [[ -f "$SIM_N80" ]]; then
  log "SIM_DONE $SIM_N80"
  /root/venv/bin/python3 - <<PY
import json
from pathlib import Path
d=json.loads(Path("$SIM_N80").read_text())
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
  "hypo": "R955",
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
  "note": "p4080 chall+v4-n80 R338 GPUs6,7 :8002; HiRank Midβ MidCtx UltraExtra; vs reign36 wvk7",
}
Path("$SIM_DEC").write_text(json.dumps(dec, indent=2)+"\n")
print(json.dumps(dec, indent=2))
k = dp.get("n_teacher_samples")
if k != 3:
    raise SystemExit(f"FATAL duel_params.n_teacher_samples={k} (want 3)")
PY
else
  log "FATAL missing sim result"
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r955_reign36_wvk7_pipeline.done
log "DONE R955 v4 n80 vs reign36 on R338 6,7"
