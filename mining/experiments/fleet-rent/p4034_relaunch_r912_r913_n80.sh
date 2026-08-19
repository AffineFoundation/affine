#!/usr/bin/env bash
# p4034: crown challs READY but n80 died — missing s4-h2-merge/run_sim_duel.py.
# Upload already done; sync corpus + relaunch both v4 n80s; keep challs.
# Never pkill -f. Do not touch teacher/king/chall vLLM.
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
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p /root/logs /root/affine_data

LOG=/root/logs/p4034_relaunch_r912_r913_n80.log
: >"$LOG"
log() { echo "[p4034-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SIM=/root/mining_src/s4-h2-merge/run_sim_duel.py
test -f "$SIM" || { log "FATAL missing $SIM"; exit 1; }

for p in 8000 8001 8002 8003; do
  curl -sf -m 5 "http://127.0.0.1:${p}/v1/models" >/dev/null \
    || { log "FATAL port $p not ready"; exit 1; }
done
log "TKC READY :8000/:8001/:8002/:8003"

# Corpus sync (schema v2) — fail-closed if empty after
if [[ ! -s /root/affine_data/turns_index.parquet && ! -s /root/affine_data/turns.jsonl ]]; then
  log "corpus sync start"
  bash /root/mining_src/s3-duel-sim/sync_corpus.sh >>"$LOG" 2>&1 \
    || log "WARN sync_corpus rc=$? — run_sim refresh may recover"
fi
if [[ -s /root/affine_data/turns_index.parquet ]]; then
  log "corpus OK turns_index.parquet $(du -h /root/affine_data/turns_index.parquet | awk '{print $1}')"
elif [[ -s /root/affine_data/turns.jsonl ]]; then
  log "corpus OK turns.jsonl lines=$(wc -l </root/affine_data/turns.jsonl)"
else
  log "FATAL no corpus after sync"
  exit 1
fi

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8

launch_one() {
  local tag=$1 port=$2 merge=$3
  local sim_out=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
  local prog=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
  local dec=/root/affine_data/${tag}_decision_reign36_wvk7.json
  local slog=/root/logs/p4034_${tag}_n80_wvk7.log
  local pidf=/root/logs/${tag}_sim_wvk7.pid
  rm -f "$sim_out" "$prog" "$dec"
  local bh
  bh=$(python3 - <<PY
import hashlib, time
print(hashlib.sha256(f"${tag}-reign36-wvk7-p4034-{time.time()}".encode()).hexdigest())
PY
)
  : >"$slog"
  log "launch ${tag} n80 chall=:${port} merge=$merge bh=${bh:0:16}…"
  nohup env -u HF_TOKEN -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    HF_HOME="${HF_HOME}" \
    PYTHONPATH="/root/mining_src/affine_pkg:${PYTHONPATH:-}" \
    AFFINE_DATA_DIR="${AFFINE_DATA_DIR}" \
    /root/venv/bin/python3 "$SIM" \
    --teacher-repo "$TEACHER_REPO" \
    --king-repo "$KING_REPO" \
    --king-rev "$KING_REV" \
    --chall-repo "$merge" \
    --chall-rev local \
    --chall-port "$port" \
    --n-turns 80 \
    --hotkey "local-${tag}-reign36-wvk7-p4034" \
    --block-hash "$bh" \
    --out "$sim_out" \
    --progress-out "$prog" \
    --save-artifact \
    >>"$slog" 2>&1 &
  local spid=$!
  echo "$spid" >"$pidf"
  date -u +%Y-%m-%dT%H:%M:%SZ >"/root/logs/${tag}_n80_launched.p4034"
  log "${tag} n80 pid=$spid → $sim_out"
}

# R913 :8002 GPUs4,5 · R912 :8003 GPUs6,7
launch_one r913 8002 /tmp/r913_merged
launch_one r912 8003 /tmp/r912_merged

# Waiter: poll both → write decision JSON (same schema as lean_chall)
nohup bash -c '
set -euo pipefail
LOG=/root/logs/p4034_n80_waiter.log
log(){ echo "[p4034-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
: >"$LOG"
decide() {
  local tag=$1
  local sim=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
  local dec=/root/affine_data/${tag}_decision_reign36_wvk7.json
  /root/venv/bin/python3 - <<PY
import json
from pathlib import Path
d=json.loads(Path("$sim").read_text())
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
  "hypo": "$tag".upper(),
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
  "note": "p4034 relaunch after missing run_sim_duel.py; vs reign36 vera wvk7",
  "hf_ok": False,
  "raw_keys": sorted(d.keys())[:40],
}
Path("$dec").write_text(json.dumps(dec, indent=2)+"\n")
print(json.dumps(dec, indent=2))
k = dp.get("n_teacher_samples")
if k != 3:
    raise SystemExit(f"FATAL duel_params.n_teacher_samples={k} (want 3)")
PY
}
for i in $(seq 1 240); do
  ok=0
  for tag in r912 r913; do
    sim=/root/affine_data/${tag}_sim_result_reign36_wvk7.json
    pidf=/root/logs/${tag}_sim_wvk7.pid
    pid=$(cat "$pidf" 2>/dev/null || true)
    if [[ -f "$sim" ]]; then
      if [[ ! -f /root/affine_data/${tag}_decision_reign36_wvk7.json ]]; then
        log "SIM_DONE $tag — write decision"
        decide "$tag" | tee -a "$LOG"
        date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/${tag}_reign36_wvk7_pipeline.done
      fi
      ok=$((ok+1))
    elif [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
      log "WARN $tag sim dead without result (pid was $pid)"
      ok=$((ok+1))  # stop waiting forever
    fi
  done
  if [[ "$ok" -ge 2 ]]; then
    log "both resolved"
    break
  fi
  if (( i % 6 == 0 )); then
    for tag in r912 r913; do
      p=/root/affine_data/${tag}_sim_progress_reign36_wvk7.json
      [[ -f "$p" ]] && log "prog $tag $(cat "$p" | tr -d "\n" | head -c 200)"
    done
  fi
  sleep 30
done
log "waiter exit"
' >/root/logs/p4034_n80_waiter.outer.nohup 2>&1 &
echo $! >/root/logs/p4034_n80_waiter.outer.pid
log "waiter pid=$(cat /root/logs/p4034_n80_waiter.outer.pid)"
log "DONE relaunch armed"
