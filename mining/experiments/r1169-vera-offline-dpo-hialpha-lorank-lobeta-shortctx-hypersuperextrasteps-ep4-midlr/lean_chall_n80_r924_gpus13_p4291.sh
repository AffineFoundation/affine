#!/usr/bin/env bash
# p4291: R1169 MERGE_DONE → chall :8003 + v4 n80 on R924 GPUs 1,3 vs reign36 vera (wvk=7 k=3 τ=0.03).
# Axis: vera Soft Mid Mid Soft HiAlpha LoRank Loβ ShortCtx HyperSuperExtra ep4 MidLR (β=0.02 r=16 lr=1e-6 @6144 steps=38400)
# Parent: R1154 ShortCtx LoRank Loβ Hyper UltraLoLR REFUTE m=-0.000395 ~-0.08× → MidLR isolate
# Never --no-save-original-format. Never pkill -f.
# Do not touch teacher:8000 / king:8001 / R1163 on :8002 GPUs4,5.
set -euo pipefail

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi

GPUS=1,3
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

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r1169_merged
export CUDA_VISIBLE_DEVICES=$GPUS
UTIL=${UTIL:-0.72}
LOG=/root/logs/p4291_r1169_chall_n80_wvk7.log
CHALL_LOG=/root/logs/vllm_chall_r1169_p4291.log
PIDF=/root/logs/vllm_chall_r1169.pid
TCACHE=/root/.triton/cache/chall_r1169
SIM_N80=/root/affine_data/r1169_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r1169_sim_progress_reign36_wvk7.json
SIM_DEC=/root/affine_data/r1169_decision_reign36_wvk7.json
mkdir -p /root/logs /root/affine_data

: >"$LOG"
log() { echo "[p4291-r1169] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

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

log "START R1169 chall+v4-n80 R924 GPUs=$GPUS merge=$MERGE_DIR vs king=$KING_REPO@$KING_REV"
hub_ok "$MERGE_DIR" || { log "FATAL merge incomplete"; exit 1; }
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
log "reuse merge shards=$n"

cat >"$MERGE_DIR/README.md" <<'CARD'
# R1169 merged challenger (local)

- **Base:** `vera6/affine-5g4yy75zuz-t6` @ `8e3f1695…` (live reign36 king)
- **Method:** Offline DPO on Reason duel pairs (teacher-side)
- **Axis:** HiAlpha LoRank MidBeta ShortCtx HyperSuperExtra ep4 HiLR (β=0.02 r=16 lr=2e-6 @12288 steps=38400)
- **Knobs:** lr=2e-6, LoRA r=32 / α=128, β=0.02, max_len=12288, epochs=4, max_steps=38400
- **Hardware:** mine-r924 8×B200 GPUs 1,3 TP2 :8003
- **Experiment:** `mining/experiments/r1169-vera-offline-dpo-hialpha-lorank-lobeta-shortctx-hypersuperextrasteps-ep4-midlr/`
- **Parent signal: R1154 ShortCtx LoRank Loβ Hyper UltraLoLR REFUTE m=-0.000395 ~-0.08× → MidLR isolate
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
if ! echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|affine-5g4yy75zuz-t6'; then
  log "ERROR king not reign36 vera ($kid) — abort"
  exit 5
fi

for stale in /root/logs/vllm_chall_r743.pid /root/logs/vllm_chall_r742.pid /root/logs/vllm_chall_r734.pid "$PIDF"; do
  stop_pidfile "$stale" "stale chall"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale chall argv"
done < <(ps -eo pid=,args= | awk 'index($0,"/tmp/r1169_merged") && /vllm serve/ && !/awk/ {print $1}')

CHALL_PORT="$CHALL_PORT" GPUS="$GPUS" python3 - <<'PY' | tee -a "$LOG"
import os, signal, subprocess, time, re
port = os.environ.get("CHALL_PORT", "8003")
want = {int(x) for x in os.environ.get("GPUS", "1,3").split(",") if x.strip()}
try:
    out = subprocess.check_output(["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ""
pids = set(int(x) for x in re.findall(r"pid=(\d+)", out))
for pid in sorted(pids):
    print(f"[p4291-r1169] kill :{port} listener pid={pid}", flush=True)
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
        print(f"[p4291-r1169] SKIP train pid={pid}", flush=True)
        continue
    if "r769" in cmd or "r769_merged" in cmd or "/root/r769" in cmd:
        print(f"[p4291-r1169] SKIP sibling train pid={pid}", flush=True)
        continue
    if any(tok in cmd for tok in ["GLM-4.5-Air", ":8000", ":8001"]):
        if "r1169_merged" not in cmd:
            print(f"[p4291-r1169] SKIP TK pid={pid}", flush=True)
            continue
    kill_set.add(pid)
print(f"[p4291-r1169] reap gpu={sorted(want)} kill={sorted(kill_set)}", flush=True)
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
print("[p4291-r1169] chall GPUs reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi -i 1,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then
    log "GPUs 1,3 free (poll $i used_mib=$used)"
    break
  fi
  sleep 2
done

# p4291: NEVER REUSE chall_r1169 — /v1/models can pass then first sample hits a
# half-written Triton launcher .so. Always wipe+seed from a known-good tree.
_seed_src=""
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r1143 /root/.triton/cache/chall_r954 /root/.triton/cache/chall_r949 /root/.triton/cache/chall_r941 /root/.triton/cache/chall_r939 /root/.triton/cache/chall_r924 /root/.triton/cache/chall; do
  if [[ -d "$cand" ]]; then
    _n=$(find "$cand" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
    if [[ "${_n:-0}" -ge 1 ]]; then
      _seed_src=$cand
      break
    fi
  fi
done
rm -rf "$TCACHE"
mkdir -p "$(dirname "$TCACHE")"
if [[ -n "$_seed_src" ]]; then
  log "p4291 FORCE wipe+seed $TCACHE from $_seed_src"
  cp -a "$_seed_src" "$TCACHE"
else
  log "WARN no triton seed; empty $TCACHE"
  mkdir -p "$TCACHE"
fi
chmod -R a+rX "$TCACHE" 2>/dev/null || true
_post_n=$(find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
log "triton seed n_so=$_post_n"

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

# p4291: probe one completion before n80 — catches Triton .so death /v1/models misses
log "probe sample before n80"
if ! CHALL_PORT="$CHALL_PORT" python3 - <<'PY'
import json, os, urllib.request
port = os.environ["CHALL_PORT"]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps({
        "model": "/tmp/r1169_merged",
        "prompt": "Next command:\n",
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode(),
    headers={"Content-Type": "application/json"},
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

BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r1169-reign36-wvk7-p4291-{time.time()}".encode()).hexdigest())
PY
)
# p4041: when king is local-path served, hub string 404s — pass /v1/models id
KING_ID=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "launch n80 vs king_id=$KING_ID block_hash=${BLOCK_HASH:0:16}… (HF_TOKEN unset)"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 /root/mining_src/s4-h2-merge/run_sim_duel.py \
  --teacher-repo "$TEACHER_REPO" \
  --king-repo "$KING_ID" \
  --king-rev "$KING_REV" \
  --chall-repo "$MERGE_DIR" \
  --chall-rev local \
  --chall-port "$CHALL_PORT" \
  --n-turns 80 \
  --hotkey local-r1169-reign36-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" > /root/logs/r1169_sim_wvk7.pid
log "n80 pid=$SIM_PID — waiting for result"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1169_n80_launched.p4291

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
  "hypo": "R1169",
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
  "note": "p4291 chall+v4-n80 R924 1,3 :8003; HiAlpha LoRank Loβ ShortCtx HyperSuperExtra ep4 MidLR (β=0.02 r=16 lr=1e-6 @6144 steps=38400); vs reign36 vera wvk7",
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
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1169_reign36_wvk7_pipeline.done
log "DONE R1169 v4 n80 vs reign36 on R924 1,3"
