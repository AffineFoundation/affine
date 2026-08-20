#!/usr/bin/env bash
# p4063: mine-r924 idle 8×H200 — cold TK + R930/R931 MERGE_DONE → dual v4 n80.
# Teacher :8000 GPU0 · king reign36 vera :8001 GPU2 · R930 chall :8002 GPUs4,5 · R931 chall :8003 GPUs6,7.
# Never pkill -f. Do not touch non-mine pods.
set -euo pipefail
LOG=/root/logs/p4063_r924_cold_tk_r930_r931.log
mkdir -p /root/logs /root/affine_data /root/.triton/cache/teacher /root/.triton/cache/king /root/.triton/cache/chall_r930 /root/.triton/cache/chall_r931
exec > >(tee -a "$LOG") 2>&1
echo "[p4063-r924] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
[[ -n "${HF_TOKEN:-}" ]] || { echo FATAL missing HF_TOKEN; exit 1; }

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
KING_LOCAL=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/${KING_REV}
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
TEACHER_PIDF=/root/logs/vllm_teacher.pid
KING_PIDF=/root/logs/vllm_king.pid
TEACHER_LOG=/root/logs/vllm_teacher.log
KING_LOG=/root/logs/vllm_king.log
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4063-r924] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
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
  local path=$1 min=${2:-1} n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge "$min" ]]
}

wait_ready() {
  local port=$1 name=$2 pidf=$3 logf=$4 ready=0
  for i in $(seq 1 720); do
    if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1; echo "[p4063-r924] ${name}_READY poll=$i"; break
    fi
    if [[ -f "$pidf" ]]; then
      local pid; pid=$(cat "$pidf")
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[p4063-r924] ERROR $name died"; tail -80 "$logf"; exit 1
      fi
    fi
    (( i % 12 == 0 )) && echo "[p4063-r924] wait $name :$port iter=$i last=$(tail -1 "$logf" 2>/dev/null | cut -c1-120)"
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || { echo "[p4063-r924] ERROR $name not ready"; tail -80 "$logf"; exit 1; }
}

for idx in 0 1 2 3 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | awk '{print $1+0}')
  echo "[p4063-r924] gpu$idx used_mib=$used"
  if [[ "$used" -ge 8192 ]]; then
    echo "[p4063-r924] FATAL GPU $idx busy used_mib=$used — abort"
    exit 2
  fi
done

hub_ok "$KING_LOCAL" 2 || { echo FATAL king incomplete at $KING_LOCAL; exit 3; }
hub_ok /tmp/r930_merged 16 || { echo FATAL r930 merge incomplete; exit 3; }
hub_ok /tmp/r931_merged 16 || { echo FATAL r931 merge incomplete; exit 3; }
[[ -f "$SIM" ]] || { echo FATAL missing $SIM; exit 3; }
echo "[p4063-r924] king shards=$(ls "$KING_LOCAL"/model-*-of-*.safetensors | wc -l) r930=$(ls /tmp/r930_merged/model-*-of-*.safetensors | wc -l) r931=$(ls /tmp/r931_merged/model-*-of-*.safetensors | wc -l)"

python3 - <<'PY'
import importlib, subprocess, sys
try:
    importlib.import_module("pyarrow")
    print("[p4063-r924] pyarrow ok")
except Exception:
    print("[p4063-r924] installing pyarrow via ensurepip+pip")
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
    importlib.import_module("pyarrow")
    print("[p4063-r924] pyarrow ok after install")
PY
python3 - <<'PY'
import sys
sys.path.insert(0, "/root/mining_src/affine_pkg")
from evalsrv.chat import THINK_OPEN
print("[p4063-r924] evalsrv THINK_OPEN ok", THINK_OPEN[:12])
PY

echo "[p4063-r924] DOWNLOAD teacher start $TEACHER_REPO"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
token=os.environ["HF_TOKEN"]
path=snapshot_download("zai-org/GLM-4.5-Air-FP8", token=token)
print("[p4063-r924] DOWNLOAD teacher done", path, flush=True)
open("/root/logs/teacher_dl.done","w").write(path+"\n")
PY

# Pin mine.env king → reign36 vera
python3 - <<PY
from pathlib import Path
p = Path("/root/mine.env")
text = p.read_text() if p.exists() else ""
lines = text.splitlines()
want = {
    "KING_REPO": "$KING_REPO",
    "KING_REV": "$KING_REV",
    "KING_LOCAL": "$KING_LOCAL",
    "KING_SERVED_NAME": "$KING_REPO",
    "TEACHER_REPO": "$TEACHER_REPO",
    "SKIP_LOCAL_TKC": "0",
}
seen=set(); out=[]
for line in lines:
    if line.startswith("export ") and "=" in line:
        key=line.split("=",1)[0].removeprefix("export ").strip()
        if key in want:
            out.append(f"export {key}={want[key]}"); seen.add(key); continue
    if "=" in line and not line.strip().startswith("#"):
        key=line.split("=",1)[0].strip()
        if key in want and not key.startswith("export "):
            out.append(f"{key}={want[key]}"); seen.add(key); continue
    out.append(line)
for key,val in want.items():
    if key not in seen:
        out.append(f"{key}={val}")
p.write_text("\n".join(out)+"\n")
print("[p4063-r924] mine.env KING→vera reign36")
PY

stop_pidfile "$TEACHER_PIDF" "stale teacher"
stop_pidfile "$KING_PIDF" "stale king"
for port in 8000 8001 8002 8003; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "listener :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
done
sleep 2

TEACHER_COMMON=(--tensor-parallel-size 1 --max-model-len 65536 --max-num-batched-tokens 8192
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton
  --additional-config '{"gdn_prefill_backend": "triton"}'
  --enforce-eager)

echo "[p4063-r924] launch teacher :8000 GPU0 TP=1"
: >"$TEACHER_LOG"
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/root/.triton/cache/teacher \
  nohup /root/venv/bin/vllm serve "$TEACHER_REPO" \
  --port 8000 --gpu-memory-utilization 0.90 \
  "${TEACHER_COMMON[@]}" --served-model-name "$TEACHER_REPO" >>"$TEACHER_LOG" 2>&1 &
echo $! >"$TEACHER_PIDF"
echo "[p4063-r924] teacher pid=$(cat "$TEACHER_PIDF")"
wait_ready 8000 teacher "$TEACHER_PIDF" "$TEACHER_LOG"

echo "[p4063-r924] launch king :8001 GPU2 TP=1 $KING_REPO@$KING_REV"
: >"$KING_LOG"
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/root/.triton/cache/king \
  nohup /root/venv/bin/vllm serve "$KING_LOCAL" \
  --port 8001 --gpu-memory-utilization 0.90 \
  "${TEACHER_COMMON[@]}" --served-model-name "$KING_REPO" >>"$KING_LOG" 2>&1 &
echo $! >"$KING_PIDF"
echo "[p4063-r924] king pid=$(cat "$KING_PIDF")"
wait_ready 8001 king "$KING_PIDF" "$KING_LOG"

kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4063-r924] king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo ERROR king not vera; exit 5; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r924_tk_ready_p4063.done

launch_chall_n80() {
  local tag=$1 merge=$2 gpus=$3 port=$4
  local chall_log=/root/logs/vllm_chall_${tag}_p4063.log
  local pidf=/root/logs/vllm_chall_${tag}.pid
  local tcache=/root/.triton/cache/chall_${tag}
  local sim_out=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
  local prog=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
  local sim_dec=/root/affine_data/${tag}_decision_reign36_wvk7.json
  local n80_log=/root/logs/p4063_${tag}_chall_n80_wvk7.log

  echo "[p4063-r924] START $tag chall :$port GPUs=$gpus merge=$merge"
  hub_ok "$merge" 16 || { echo FATAL $tag merge incomplete; exit 1; }

  # Seed triton from king if present
  if [[ -d /root/.triton/cache/king ]] && find /root/.triton/cache/king -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
    rm -rf "$tcache"
    cp -a /root/.triton/cache/king "$tcache"
    echo "[p4063-r924] $tag seeded triton from king"
  else
    mkdir -p "$tcache"
  fi

  stop_pidfile "$pidf" "stale $tag"
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "stale :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

  : >"$chall_log"
  CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$tcache \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$merge" \
      --port "$port" \
      --tensor-parallel-size 2 \
      --max-model-len 65536 \
      --gpu-memory-utilization 0.72 \
      --max-num-batched-tokens 8192 \
      --attention-backend FLASH_ATTN \
      --attention-config.use_trtllm_attention 0 \
      --compilation-config.pass_config.fuse_allreduce_rms false \
      --moe-backend triton \
      --additional-config '{"gdn_prefill_backend": "triton"}' \
      --enforce-eager \
      >"$chall_log" 2>&1 &
  echo $! >"$pidf"
  local chall_pid; chall_pid=$(cat "$pidf")
  echo "[p4063-r924] $tag chall pid=$chall_pid"

  local ready=0
  for i in $(seq 1 240); do
    if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1; echo "[p4063-r924] ${tag}_CHALL_READY poll=$i"; break
    fi
    if ! kill -0 "$chall_pid" 2>/dev/null; then
      echo "[p4063-r924] FATAL $tag chall died"; tail -100 "$chall_log"; exit 1
    fi
    if grep -q 'ImportError:.*__triton_launcher' "$chall_log" 2>/dev/null; then
      echo "[p4063-r924] FATAL $tag Triton ImportError"; tail -80 "$chall_log"; exit 1
    fi
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || { echo FATAL $tag chall not ready; tail -80 "$chall_log"; exit 1; }

  local bh
  bh=$(python3 - <<PY
import hashlib, time
print(hashlib.sha256(f"${tag}-reign36-wvk7-p4063-{time.time()}".encode()).hexdigest())
PY
  )
  rm -f "$sim_out" "$prog" "$sim_dec"
  : >"$n80_log"
  nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    HF_HOME="${HF_HOME}" \
    PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
    AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
    /root/venv/bin/python3 "$SIM" \
    --teacher-repo "$TEACHER_REPO" \
    --king-repo "$kid" \
    --king-rev "$KING_REV" \
    --chall-repo "$merge" \
    --chall-rev local \
    --chall-port "$port" \
    --n-turns 80 \
    --hotkey "local-${tag}-reign36-wvk7" \
    --block-hash "$bh" \
    --out "$sim_out" \
    --progress-out "$prog" \
    --save-artifact \
    >>"$n80_log" 2>&1 &
  echo $! >"/root/logs/${tag}_sim_wvk7.pid"
  date -u +%Y-%m-%dT%H:%M:%SZ >"/root/logs/${tag}_n80_launched.p4063"
  echo "[p4063-r924] $tag n80 pid=$(cat /root/logs/${tag}_sim_wvk7.pid) bh=${bh:0:16}…"
}

# Launch both challengers (sequential load to avoid RAM spike), then both n80s run concurrent.
launch_chall_n80 r930 /tmp/r930_merged 4,5 8002
launch_chall_n80 r931 /tmp/r931_merged 6,7 8003

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4063_r924_cold_tk_r930_r931_armed.done
echo "[p4063-r924] ARMED dual n80 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[p4063-r924] waiting on sim pids (non-blocking outer ends after launch — waiters stay)"
# Keep process alive until both sims finish so outer nohup ownership is clear.
for tag in r930 r931; do
  pidf=/root/logs/${tag}_sim_wvk7.pid
  if [[ -f "$pidf" ]]; then
    pid=$(cat "$pidf")
    echo "[p4063-r924] wait $tag sim pid=$pid"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    wait "$pid" 2>/dev/null || true
    echo "[p4063-r924] $tag sim exited"
    if [[ -f /root/affine_data/${tag}_sim_result_reign36_wvk7.json ]]; then
      /root/venv/bin/python3 - <<PY
import json
from pathlib import Path
tag="$tag"
d=json.loads(Path(f"/root/affine_data/{tag}_sim_result_reign36_wvk7.json").read_text())
v=d.get("verdict") if isinstance(d.get("verdict"), dict) else {}
chal=(v.get("challenger") or {}) if isinstance(v, dict) else {}
dp=(v.get("duel_params") or {}) if isinstance(v, dict) else {}
margin = v.get("margin") if v else (d.get("margin") or d.get("mean_margin"))
se = v.get("se") if v else (d.get("se") or d.get("stderr"))
bar = max(2.0 * float(se), 0.002) if se is not None else None
dec={
  "utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
  "hypo": tag.upper(),
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
  "note": f"p4063 cold-TK R924 dual n80 {tag} vs reign36 wvk7",
}
Path(f"/root/affine_data/{tag}_decision_reign36_wvk7.json").write_text(json.dumps(dec, indent=2)+"\n")
print(json.dumps(dec, indent=2))
if dp.get("n_teacher_samples") != 3:
    raise SystemExit(f"FATAL {tag} n_teacher_samples={dp.get('n_teacher_samples')}")
PY
    fi
  fi
done
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4063_r924_r930_r931_pipeline.done
echo "[p4063-r924] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
