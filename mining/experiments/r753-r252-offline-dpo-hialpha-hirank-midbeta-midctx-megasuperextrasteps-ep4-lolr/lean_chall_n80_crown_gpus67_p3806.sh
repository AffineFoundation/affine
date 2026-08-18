#!/usr/bin/env bash
# p3806: R753 MERGE_DONE → chall :8003 + v4 n80 on crown GPUs 6,7 vs reign35 tammy (wvk=7 k=3 τ=0.03).
# Axis: HiRank MidBeta MidCtx MegaSuperExtra ep4×LoLR (β=0.1 r=64 @8192 steps=19200; after R744 REFUTE).
# Never --no-save-original-format. Never pkill -f.
# Do not touch teacher 0,1 or king 2,3. Leave R758 TRAIN on GPUs 4,5 alone (do not touch :8002).
set -euo pipefail

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi

# Hard-pin AFTER mine.env (p3768 collision lesson)
GPUS=6,7
CHALL_PORT=8003
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
MERGE_DIR=/tmp/r753_merged
GPUS=${GPUS:-6,7}
export CUDA_VISIBLE_DEVICES=$GPUS
CHALL_PORT=${CHALL_PORT:-8003}
UTIL=${UTIL:-0.72}
LOG=/root/logs/p3806_r753_chall_n80_wvk7.log
CHALL_LOG=/root/logs/vllm_chall_r753_p3806.log
PIDF=/root/logs/vllm_chall_r753.pid
TCACHE=/root/.triton/cache/chall_r753
SIM_N80=/root/affine_data/r753_sim_result_reign35_wvk7.json
PROG=/root/affine_data/r753_sim_progress_reign35_wvk7.json
SIM_DEC=/root/affine_data/r753_decision_reign35_wvk7.json
mkdir -p /root/logs /root/affine_data

: >"$LOG"
log() { echo "[p3806-r753] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

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

log "START R753 chall+v4-n80 crown GPUs=$GPUS merge=$MERGE_DIR vs king=$KING_REPO@$KING_REV"
hub_ok "$MERGE_DIR" || { log "FATAL merge incomplete"; exit 1; }
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
log "reuse merge shards=$n"

cat >"$MERGE_DIR/README.md" <<'EOF'
---
license: apache-2.0
base_model: unconst/Affine-5czsc2fc98-r252-merged
tags:
  - affine
  - sn120
  - offline-dpo
  - reason-v4
---

# Affine-5czsc2fc98-r753-r252-odpo-hirank-midbeta-midctx-superextra-ep3-lolr-merged

## Training story

- **Base / parent king:** `unconst/Affine-5czsc2fc98-r252-merged` @ `b42d6245d77fe30885ea8a90387771e1bc465e0f` (SN120 reign-33).
- **Method:** Offline DPO on teacher-anchored Reason pairs. Optimized for Reason (teacher-side only).
- **Data:** duel-derived Reason preference pairs; MidCtx×HiRank MidBeta SuperExtra; experiment
  `mining/experiments/r753-r252-offline-dpo-hialpha-hirank-midbeta-midctx-superextrasteps-ep3-lolr/`.
- **Hyperparameters:** lr=`1e-6` (LoLR), LoRA r=`64` / α=`128`, β=`0.1`, max_len=`8192`, epochs=`4`, max_steps=`19200` (MegaSuperExtra).
- **Hardware:** train+merge+n80 on `mine-crown-1` (gentle-orbit-bd) GPUs **6,7** :8003 (local MERGE_DONE; R758 relay keeps 4,5).
- **Axis note:** HiRank MidBeta MidCtx SuperExtra (R737 MidCtx HiRank MidBeta HyperExtra REFUTE ~−0.90× → SuperExtra amplify). ≠ HyperExtra R737; ≠ UltraExtra R689; ≠ SoftCtx SuperExtra R715; ≠ Online; ≠ GRPO.
- **Decision rule:** paired margin > max(2·SE, δ=0.002) **and** median thought ≥80 **and** B pass ≥0.30 vs **reign35** tammyfritz (v4 k=3 τ=0.03).

This card is the training write-up required before any Stage-5 submit.
EOF

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

for stale in /root/logs/vllm_chall_r727.pid /root/logs/vllm_chall_r718.pid \
  /root/logs/vllm_chall_r696.pid /root/logs/vllm_chall_r698.pid \
  /root/logs/vllm_chall_r705.pid "$PIDF"; do
  stop_pidfile "$stale" "stale chall"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r(737|727|718)_merged/ && !/awk/ {print $1}')

CHALL_PORT="$CHALL_PORT" GPUS="$GPUS" python3 - <<'PY' | tee -a "$LOG"
import os, signal, subprocess, time, re
port = os.environ.get("CHALL_PORT", "8003")
want = {int(x) for x in os.environ.get("GPUS", "6,7").split(",") if x.strip()}
try:
    out = subprocess.check_output(["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ""
pids = set(int(x) for x in re.findall(r"pid=(\d+)", out))
for pid in sorted(pids):
    print(f"[p3806-r753] kill :{port} listener pid={pid}", flush=True)
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
        print(f"[p3806-r753] SKIP train pid={pid}", flush=True)
        continue
    if "r758" in cmd or "r758_merged" in cmd or ":8002" in cmd or "wait_r758" in cmd:
        print(f"[p3806-r753] SKIP R715 pid={pid}", flush=True)
        continue
    if any(tok in cmd for tok in ["GLM-4.5-Air", ":8000", ":8001", ":8002"]):
        if "r753_merged" not in cmd:
            print(f"[p3806-r753] SKIP TK/R715 pid={pid}", flush=True)
            continue
    kill_set.add(pid)
print(f"[p3806-r753] reap gpu={sorted(want)} kill={sorted(kill_set)}", flush=True)
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
print("[p3806-r753] chall GPUs reaped", flush=True)
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
for cand in /root/.triton/cache/chall_r727 /root/.triton/cache/chall_r718 \
  /root/.triton/cache/chall_r717 /root/.triton/cache/chall_r705 \
  /root/.triton/cache/chall_r698 /root/.triton/cache/chall \
  /root/.triton/cache/king; do
  _n=$(find "$cand" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
  if [[ "${_n:-0}" -ge 8 ]]; then
    _seed_src=$cand
    _n_star=$_n
    break
  fi
done
if [[ -z "$_seed_src" ]]; then
  log "ERROR no usable Triton seed"
  exit 1
fi
_pre_n=$(find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
_pre_sz=$(du -sm "$TCACHE" 2>/dev/null | awk '{print $1}' || true)
_pre_sz=${_pre_sz:-0}
if [[ "${_pre_n:-0}" -ge 25 && "${_pre_sz}" -ge 100 ]]; then
  log "REUSE preseed $TCACHE n_so=$_pre_n size_mb=$_pre_sz — skip wipe"
else
  log "FORCE wipe+seed $TCACHE from $_seed_src n_star=$_n_star"
  rm -rf "$TCACHE"
  mkdir -p "$(dirname "$TCACHE")"
  cp -a "$_seed_src" "$TCACHE"
  _n_cur=$(find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
  log "seeded chall n_star=${_n_cur:-0}"
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
print(hashlib.sha256(f"r753-reign35-wvk7-p3806-{time.time()}".encode()).hexdigest())
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
  --hotkey local-r753-reign35-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" > /root/logs/r753_sim_wvk7.pid
log "n80 pid=$SIM_PID — waiting for result"

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
  "hypo": "R753",
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
  "note": "p3806 chall+v4-n80 crown 6,7; HiRank MidBeta MidCtx SuperExtra ep3 LoLR; vs reign35 tammy wvk7",
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
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r753_reign35_wvk7_pipeline.done
log "DONE R753 v4 n80 vs reign35 on crown 6,7"
