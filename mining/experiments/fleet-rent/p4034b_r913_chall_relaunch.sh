#!/usr/bin/env bash
# p4034b: R913 chall died (triton .so missing in chall_r913 cache).
# Re-seed Triton from king, restart chall :8002 GPUs 4,5, relaunch n80.
# Do NOT touch R912 (:8003 / GPUs 6,7) or teacher/king. Never pkill -f.
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi
export HF_HOME=${HF_HOME:-/root/hf}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_TOKEN

LOG=/root/logs/p4034b_r913_chall_relaunch.log
: >"$LOG"
log() { echo "[p4034b-r913] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

GPUS=4,5
CHALL_PORT=8002
MERGE_DIR=/tmp/r913_merged
TCACHE=/root/.triton/cache/chall_r913
CHALL_LOG=/root/logs/vllm_chall_r913_p4034b.log
PIDF=/root/logs/vllm_chall_r913.pid
UTIL=0.72
export CUDA_VISIBLE_DEVICES=$GPUS

# Guard: R912 must stay up
curl -sf -m 5 http://127.0.0.1:8003/v1/models >/dev/null || { log "FATAL R912 :8003 down — abort"; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { log "FATAL teacher down"; exit 1; }
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { log "FATAL king down"; exit 1; }
hub_n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${hub_n:-0}" -ge 16 ]] || { log "FATAL merge incomplete n=$hub_n"; exit 1; }

# Kill only stale R913 chall by pidfile (not pkill -f)
if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [[ "${old:-}" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    log "stop old chall pid=$old"
    kill "$old" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$old" 2>/dev/null || break; sleep 1; done
    kill -9 "$old" 2>/dev/null || true
  fi
  rm -f "$PIDF"
fi
# Also reap any orphan engine on GPUs 4,5 that still holds memory (by CUDA_VISIBLE pid map)
python3 - <<'PY'
import os, signal, time
want={4,5}
kill_set=set()
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try:
        env=open(f"/proc/{pid}/environ","rb").read().split(b"\0")
        cvd=None
        for e in env:
            if e.startswith(b"CUDA_VISIBLE_DEVICES="):
                cvd=e.split(b"=",1)[1].decode()
                break
        if cvd is None: continue
        gpus={int(x) for x in cvd.split(",") if x.strip().isdigit()}
        if not (gpus & want): continue
        cmd=open(f"/proc/{pid}/cmdline","rb").read().decode(errors="ignore")
        # never touch teacher/king ports or r912
        if ":8000" in cmd or ":8001" in cmd or ":8003" in cmd or "r912" in cmd:
            continue
        if "vllm" in cmd or "EngineCore" in cmd or "Worker_TP" in cmd:
            kill_set.add(int(pid))
    except Exception:
        pass
print(f"[p4034b] reap gpus4,5 kill={sorted(kill_set)}", flush=True)
for pid in kill_set:
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in kill_set:
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then
    log "GPUs 4,5 free used_mib=$used"
    break
  fi
  sleep 2
done

# Force reseed Triton from king (broken .so was the failure mode)
_seed=""
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r912 /root/.triton/cache/chall; do
  if [[ -d "$cand" ]]; then
    n=$(find "$cand" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 1 ]]; then _seed=$cand; break; fi
  fi
done
[[ -n "$_seed" ]] || { log "FATAL no triton seed"; exit 1; }
log "FORCE wipe+seed $TCACHE from $_seed"
rm -rf "$TCACHE"
mkdir -p "$(dirname "$TCACHE")"
cp -a "$_seed" "$TCACHE"
n_so=$(find "$TCACHE" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
log "seeded n_so=$n_so"

export TRITON_CACHE_DIR=$TCACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPUS
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
log "chall pid=$CHALL_PID :$CHALL_PORT"

for i in $(seq 1 240); do
  if curl -sf -m 3 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null 2>&1; then
    log "CHALL_READY poll=$i"
    break
  fi
  if ! kill -0 "$CHALL_PID" 2>/dev/null; then
    log "FATAL chall died"; tail -n 80 "$CHALL_LOG" | tee -a "$LOG"; exit 1
  fi
  sleep 5
done
curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null

# Relaunch n80 only
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
SIM_OUT=/root/affine_data/r913_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r913_sim_progress_reign36_wvk7.json
DEC=/root/affine_data/r913_decision_reign36_wvk7.json
SLOG=/root/logs/p4034b_r913_n80_wvk7.log
rm -f "$SIM_OUT" "$PROG" "$DEC"
bh=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r913-reign36-wvk7-p4034b-{time.time()}".encode()).hexdigest())
PY
)
: >"$SLOG"
log "launch r913 n80 bh=${bh:0:16}…"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" \
  PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 "$SIM" \
  --teacher-repo zai-org/GLM-4.5-Air-FP8 \
  --king-repo vera6/affine-5g4yy75zuz-t6 \
  --king-rev 8e3f1695e058837ed80fec3238ff439fdc2d0f0e \
  --chall-repo "$MERGE_DIR" \
  --chall-rev local \
  --chall-port "$CHALL_PORT" \
  --n-turns 80 \
  --hotkey local-r913-reign36-wvk7-p4034b \
  --block-hash "$bh" \
  --out "$SIM_OUT" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$SLOG" 2>&1 &
echo $! >/root/logs/r913_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r913_n80_launched.p4034b
log "n80 pid=$(cat /root/logs/r913_sim_wvk7.pid)"
log "DONE R913 chall+n80 relaunched (R912 untouched)"
