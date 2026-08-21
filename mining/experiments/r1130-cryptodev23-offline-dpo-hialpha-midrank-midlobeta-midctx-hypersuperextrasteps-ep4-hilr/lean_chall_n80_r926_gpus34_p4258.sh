#!/usr/bin/env bash
# p4258: R1130 MERGE_DONE → chall :8002 GPUs3,4 + v4 n80 vs reign36.
# Axis: cryptoDev23 MidCtx MidRank MidLoβ Hyper HiLR (β=0.05 r=32 α=128 lr=2e-6 @8192 steps=38400)
# Never pkill -f. Do not touch teacher :8000 or king :8001.
set -euo pipefail
LOG=/root/logs/p4258_r926_r1130_chall_n80.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4258-r1130] $(date -u +%Y-%m-%dT%H:%M:%SZ) start — serve HF-layout merge + v4 n80"

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}

MERGE=/tmp/r1130_merged
SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8

python3 - <<'PY'
import json
from pathlib import Path
out=Path("/tmp/r1130_merged")
assert (out/"config.json").exists(), "missing config"
wm=json.load(open(out/"model.safetensors.index.json"))["weight_map"]
missing=[s for s in set(wm.values()) if not (out/s).exists()]
assert not missing, missing
print("MERGE_OK nkeys",len(wm),"shards",len(set(wm.values())), flush=True)
PY

for gi in 3 4; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gi" | awk '{print $1+0}')
  echo "[p4258-r1130] gpu$gi used_mib=$used"
  [[ "$used" -lt 2048 ]] || { echo FATAL GPU$gi busy; exit 2; }
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { echo FATAL teacher; exit 2; }
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null || { echo FATAL king; exit 2; }
kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4258-r1130] king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo ERROR king not vera; exit 5; }

stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
[[ -f /root/logs/vllm_chall_r1130.pid ]] && stop_pid "$(cat /root/logs/vllm_chall_r1130.pid)"
[[ -f /root/logs/vllm_chall_r1098.pid ]] && stop_pid "$(cat /root/logs/vllm_chall_r1098.pid)"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
sleep 2

gpus=3,4; port=8002; tag=r1130
chall_log=/root/logs/vllm_chall_${tag}_p4258.log
pidf=/root/logs/vllm_chall_${tag}.pid
tcache=/root/.triton/cache/chall_${tag}
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r1098 /root/.triton/cache/chall_r1051 /root/.triton/cache/chall_r1060; do
  if [[ -d "$cand" ]] && find "$cand" -name '__triton_launcher*.so' 2>/dev/null | grep -q .; then
    echo "[p4258-r1130] seed triton from $cand"
    rm -rf "$tcache"; cp -a "$cand" "$tcache"; break
  fi
done
mkdir -p "$tcache"
: >"$chall_log"
CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$tcache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$MERGE" \
    --port "$port" --tensor-parallel-size 2 --max-model-len 65536 \
    --gpu-memory-utilization 0.72 --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton --additional-config '{"gdn_prefill_backend": "triton"}' \
    --enforce-eager >"$chall_log" 2>&1 &
echo $! >"$pidf"
chall_pid=$(cat "$pidf")
echo "[p4258-r1130] chall pid=$chall_pid port=$port util=0.72"
ready=0
for i in $(seq 1 360); do
  curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { ready=1; echo "[p4258-r1130] CHALL_READY poll=$i"; break; }
  kill -0 "$chall_pid" 2>/dev/null || { echo FATAL chall died; tail -80 "$chall_log"; exit 1; }
  grep -q 'ValueError: There is no module' "$chall_log" 2>/dev/null && { echo FATAL layout; tail -40 "$chall_log"; exit 1; }
  grep -q 'ImportError:.*__triton_launcher' "$chall_log" 2>/dev/null && { echo FATAL Triton; tail -40 "$chall_log"; exit 1; }
  (( i % 12 == 0 )) && echo "[p4258-r1130] wait chall iter=$i last=$(tail -1 "$chall_log" 2>/dev/null | cut -c1-120)"
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo FATAL not ready; tail -80 "$chall_log"; exit 1; }

PROBE_OK=0
for i in $(seq 1 6); do
  if curl -sf -m 60 "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MERGE"'","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"temperature":0}' \
    >/tmp/r1130_probe.json 2>/dev/null; then
    PROBE_OK=1
    echo "[p4258-r1130] probe_ok poll=$i"
    break
  fi
  echo "[p4258-r1130] probe_retry=$i"
  sleep 5
done
[[ "$PROBE_OK" -eq 1 ]] || { echo FATAL probe failed; tail -80 "$chall_log"; exit 1; }

bh=$(python3 -c 'import hashlib,time;print(hashlib.sha256(f"r1130-reign36-wvk7-p4258-{time.time()}".encode()).hexdigest())')
sim_out=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
prog=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
dec=/root/affine_data/${tag}_decision_reign36_wvk7.json
rm -f "$sim_out" "$prog" "$dec"
nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME="${HF_HOME}" PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
  AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
  /root/venv/bin/python3 "$SIM" \
  --teacher-repo "$TEACHER_REPO" --king-repo "$kid" --king-rev "$KING_REV" \
  --chall-repo "$MERGE" --chall-rev local --chall-port "$port" --n-turns 80 \
  --hotkey "local-r1130-reign36-wvk7-p4258" --block-hash "$bh" \
  --out "$sim_out" --progress-out "$prog" --save-artifact \
  >>/root/logs/p4258_${tag}_chall_n80_wvk7.log 2>&1 &
echo $! >/root/logs/${tag}_sim_wvk7.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/${tag}_n80_launched.p4258
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4258_r1130_chall_n80_armed.done
echo "[p4258-r1130] ARMED n80 pid=$(cat /root/logs/${tag}_sim_wvk7.pid) block_hash=${bh:0:16}…"

SIM_PID=$(cat /root/logs/${tag}_sim_wvk7.pid)
while kill -0 "$SIM_PID" 2>/dev/null; do sleep 30; done
wait "$SIM_PID" || true

if [[ -f "$sim_out" ]]; then
  echo "[p4258-r1130] SIM_DONE $sim_out"
  /root/venv/bin/python3 - <<PY
import json
from pathlib import Path
d=json.loads(Path("$sim_out").read_text())
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
  "hypo": "R1130",
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
  "note": "p4258 chall+v4-n80 R926 GPUs3,4; cryptoDev MidCtx MidRank MidLoβ Hyper HiLR; vs reign36 vera wvk7",
  "hf_ok": False,
}
Path("$dec").write_text(json.dumps(dec, indent=2)+"\\n")
print(json.dumps(dec, indent=2))
k = dp.get("n_teacher_samples")
if k != 3:
    raise SystemExit(f"FATAL duel_params.n_teacher_samples={k} (want 3)")
PY
else
  echo FATAL missing sim result
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1130_reign36_wvk7_pipeline.done
echo "[p4258-r1130] DONE R1130 v4 n80 vs reign36 on R926 3,4"
