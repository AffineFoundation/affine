#!/usr/bin/env bash
# p3926: R838 MERGE_DONE sat idle → chall :8002 + v4 n80 on R252 GPUs 4,5 vs reign35 tammy (wvk=7 k=3 τ=0.03).
# Axis: r252 SoftCtx MidRank LoBeta MegaSuperExtra ep4×UltraLoLR (β=0.1 r=32 @12288 steps=19200)
# Never --no-save-original-format. Never pkill -f.
# Do not touch teacher 0,1 or king 2,3. Sibling R762 n80 uses 4,5 — do not touch.
set -euo pipefail

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi

GPUS=4,5
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
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

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

KING_REPO=tammyfritz/Affine-5hmwhnfbix-tammy2
KING_REV=7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r838_merged
export CUDA_VISIBLE_DEVICES=$GPUS
UTIL=${UTIL:-0.72}
LOG=/root/logs/p3926_r838_chall_n80_wvk7.log
CHALL_LOG=/root/logs/vllm_chall_r838_p3926.log
PIDF=/root/logs/vllm_chall_r838.pid
TCACHE=/root/.triton/cache/chall_r838
SIM_N80=/root/affine_data/r838_sim_result_reign35_wvk7.json
PROG=/root/affine_data/r838_sim_progress_reign35_wvk7.json
SIM_DEC=/root/affine_data/r838_decision_reign35_wvk7.json
mkdir -p /root/logs /root/affine_data

: >"$LOG"
log() { echo "[p3926-r838] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

stop_pid() {
  local pid=$1
  local why=${2:-}
  [[ -n "${pid:-}" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_pidfile() {
  local pidf=$1
  local why=${2:-}
  [[ -f "$pidf" ]] || return 0
  local pid
  pid=$(cat "$pidf" 2>/dev/null || true)
  stop_pid "$pid" "$why pidf=$pidf"
  rm -f "$pidf"
}

hub_ok() {
  local path=$1
  local n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge 16 ]]
}

log "START R838 chall+v4-n80 R252 GPUs=$GPUS merge=$MERGE_DIR vs king=$KING_REPO@$KING_REV"
hub_ok "$MERGE_DIR" || { log "FATAL merge incomplete"; exit 1; }
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
log "reuse merge shards=$n"

cat >"$MERGE_DIR/README.md" <<'CARD'
# R838 merged challenger (local)

- **Base:** `tammyfritz/Affine-5hmwhnfbix-tammy2` @ `7e5fd5f8…`
- **Method:** Offline DPO on Reason duel pairs (teacher-ref preferred)
- **Axis:** HiAlpha MidRank MidBeta ShortCtx MegaSuperExtra ep4 UltraLoLR
- **Knobs:** lr=5e-7, LoRA r=32 / α=128, β=0.1, max_len=12288, epochs=4, max_steps=19200
- **Hardware:** R252 8×B300 GPUs 4,5
- **Experiment:** `mining/experiments/r838-tammy-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-lolr/`
- **Parent signal:** R760 MidBeta Mega REFUTE ~−0.68×; R724/R734 MidBeta ladder
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
if ! echo "$kid" | grep -qiE 'tammyfritz|5hmwhnfbix|tammy2'; then
  log "ERROR king not reign35 tammy ($kid) — abort"
  exit 5
fi

for stale in /root/logs/vllm_chall_r743.pid /root/logs/vllm_chall_r742.pid /root/logs/vllm_chall_r734.pid "$PIDF"; do
  stop_pidfile "$stale" "stale chall"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r838_merged/ && !/awk/ {print $1}')

CHALL_PORT="$CHALL_PORT" GPUS="$GPUS" python3 - <<'PY' | tee -a "$LOG"
import os, signal, subprocess, time, re
port = os.environ.get("CHALL_PORT", "8002")
want = {int(x) for x in os.environ.get("GPUS", "4,5").split(",") if x.strip()}
try:
    out = subprocess.check_output(["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ""
pids = set(int(x) for x in re.findall(r"pid=(\d+)", out))
for pid in sorted(pids):
    print(f"[p3926-r838] kill :{port} listener pid={pid}", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in sorted(pids):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
    text=True,
)
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
gpu_pids = set()
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2 and parts[0] in uuids:
        try:
            gpu_pids.add(int(parts[1]))
        except ValueError:
            pass
kill_set = set()
for pid in gpu_pids:
    if pid <= 1:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
    except Exception:
        cmd = ""
    if "train_online_dpo" in cmd or "train_dpo" in cmd or "train_full" in cmd or "train_reason_grpo" in cmd:
        print(f"[p3926-r838] SKIP train pid={pid}", flush=True)
        continue
    if "r769" in cmd or "r769_merged" in cmd or "/root/r769" in cmd:
        print(f"[p3926-r838] SKIP sibling R759 pid={pid}", flush=True)
        continue
    if any(tok in cmd for tok in ["GLM-4.5-Air", ":8000", ":8001"]):
        if "r838_merged" not in cmd:
            print(f"[p3926-r838] SKIP TK pid={pid}", flush=True)
            continue
    kill_set.add(pid)
print(f"[p3926-r838] reap gpu={sorted(want)} kill={sorted(kill_set)}", flush=True)
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
print("[p3926-r838] chall GPUs reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then
    log "GPUs 4,5 free (poll $i used_mib=$used)"
    break
  fi
  sleep 2
done

_seed_src=""
for cand in /root/.triton/cache/king /root/.triton/cache/chall /root/.triton/cache/chall_r633 /root/.triton/cache/chall_r646 /root/.triton/cache/chall_r645; do
  if [[ -d "$cand" ]]; then
    _n=$(find "$cand" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
    if [[ "${_n:-0}" -ge 1 ]]; then
      _seed_src=$cand
      break
    fi
  fi
done
_pre_n=$(find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
_pre_sz=$(du -sm "$TCACHE" 2>/dev/null | awk '{print $1}' || echo 0)
if [[ "${_pre_n:-0}" -ge 1 && "${_pre_sz:-0}" -ge 50 ]]; then
  log "REUSE preseed $TCACHE n_so=$_pre_n size_mb=$_pre_sz — skip wipe"
else
  if [[ -n "$_seed_src" ]]; then
    log "FORCE wipe+seed $TCACHE from $_seed_src"
    rm -rf "$TCACHE"
    mkdir -p "$(dirname "$TCACHE")"
    cp -a "$_seed_src" "$TCACHE"
  else
    log "WARN no triton seed; empty $TCACHE"
    mkdir -p "$TCACHE"
  fi
fi
log "skip triton purge; keep seeded tree intact"

export CUDA_VISIBLE_DEVICES=$GPUS
export TRITON_CACHE_DIR=$TCACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_TOKEN
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
    log "FATAL chall died"; tail -n 100 "$CHALL_LOG" | tee -a "$LOG"; exit 1
  fi
  sleep 5
done
curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null

BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r838-reign35-wvk7-p3926-{time.time()}".encode()).hexdigest())
PY
)
log "launch n80 vs $KING_REPO block_hash=${BLOCK_HASH:0:16}… (HF_TOKEN unset; hub ids)"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$KING_REPO" \
  --king-rev "$KING_REV" \
  --chall-repo "$MERGE_DIR" \
  --chall-rev local \
  --chall-port "$CHALL_PORT" \
  --n-turns 80 \
  --hotkey local-r838-reign35-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" > /root/logs/r838_sim_wvk7.pid
log "n80 pid=$SIM_PID — waiting for result"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r838_n80_launched.p3926

while kill -0 "$SIM_PID" 2>/dev/null; do
  sleep 30
done
wait "$SIM_PID" || true

if [[ -f "$SIM_N80" ]]; then
  log "SIM_DONE $SIM_N80"
  /root/venv/bin/python3 - <<PY | tee -a "$LOG"
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
  "hypo": "R838",
  "contract": "wvk7",
  "n_teacher_samples": dp.get("n_teacher_samples"),
  "tau": dp.get("tau"),
  "king": "reign35",
  "margin": margin,
  "se": se,
  "z": v.get("z") if v else d.get("z"),
  "n": v.get("n_paired_turns") if v else (d.get("n") or d.get("n_scored")),
  "bar": bar,
  "thought_median": chal.get("median_len_z"),
  "b_pass": chal.get("b_gate_pass_rate"),
  "wins": v.get("challenger_wins") if v else d.get("wins"),
  "note": "p3926 chall+v4-n80 R252 4,5; r252 Short MidRank MidBeta MegaSuperExtra ep4×LoLR; vs reign35 tammy wvk7",
  "hf_ok": False,
  "raw_keys": sorted(d.keys())[:40],
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
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r838_reign35_wvk7_pipeline.done
log "DONE R838 v4 n80 vs reign35 on R252 6,7"
