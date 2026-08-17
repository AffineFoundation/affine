#!/usr/bin/env bash
# p3665: R596 warm-chall v4 re-sim vs reign34 (wvk=7, k=3, τ=0.03).
# Prior p3588c was k=1 — advisory only after fork. Never pkill -f.
set -euo pipefail

source /root/venv/bin/activate
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export HF_HOME=${HF_HOME:-/root/hf}
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

KING_REPO=cryptoDev23/Affine-5Dku3dYp9j-hk8161
KING_REV=55b7ffe003d078a8a131673f677b2584548a502e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
MERGE_DIR=/tmp/r596_merged
CHALL_PORT=8002
LOG=/root/logs/p3665_r596_n80_reign34_wvk7.log
SIM_N80=/root/affine_data/r596_sim_result_reign34_wvk7.json
PROG=/root/affine_data/r596_sim_progress_reign34_wvk7.json
SIM_DEC=/root/affine_data/r596_decision_reign34_wvk7.json
mkdir -p /root/logs /root/affine_data
: >"$LOG"
log() { echo "[p3665-r596-wvk7] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

# Confirm live v4 knobs from pod toml
python3 - <<'PY' | tee -a "$LOG"
import tomllib
from pathlib import Path
raw = tomllib.loads(Path("/root/mining_src/affine_pkg/affine.toml").read_text())
d, s = raw["duel"], raw["subnet"]
assert s["weight_version_key"] == 7, s["weight_version_key"]
assert d["n_teacher_samples"] == 3, d["n_teacher_samples"]
assert abs(float(d["tau"]) - 0.03) < 1e-9, d["tau"]
print("toml_ok wvk=7 k=3 tau=0.03")
PY

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null
kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
cid=$(curl -sf -m 3 "http://127.0.0.1:${CHALL_PORT}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "pre-n80 king=$kid chall=$cid"
if [[ "$kid" != *5Dku3dYp9j* && "$kid" != *hk8161* ]]; then
  log "FATAL king not reign34 ($kid)"
  exit 5
fi
test -f "$MERGE_DIR/config.json"
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
[[ "$n" -ge 16 ]] || { log "FATAL merge shards=$n"; exit 1; }

BLOCK_HASH=$(python3 - <<'PY'
import hashlib, time
print(hashlib.sha256(f"r596-reign34-wvk7-p3665-{time.time()}".encode()).hexdigest())
PY
)
log "launch v4 n80 vs $KING_REPO@$KING_REV block_hash=${BLOCK_HASH:0:16}…"
rm -f "$SIM_N80" "$PROG" "$SIM_DEC" /root/logs/r596_reign34_wvk7_n80.done

if [[ -f /root/logs/r596_sim_wvk7.pid ]]; then
  old=$(cat /root/logs/r596_sim_wvk7.pid 2>/dev/null || true)
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    log "stop prior wvk7 sim pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
  rm -f /root/logs/r596_sim_wvk7.pid
fi

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
  --hotkey local-r596-reign34-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" > /root/logs/r596_sim_wvk7.pid
log "n80 pid=$SIM_PID — waiting (k=3 ≈3× teacher work)"

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
  "hypo": "R596",
  "vs": "reign34",
  "contract": "wvk7",
  "n_teacher_samples": dp.get("n_teacher_samples"),
  "tau": dp.get("tau"),
  "margin": margin,
  "se": se,
  "z": v.get("z") if v else d.get("z"),
  "n": v.get("n_paired_turns") if v else (d.get("n") or d.get("n_scored")),
  "bar": bar,
  "thought_median": chal.get("median_len_z"),
  "b_pass": chal.get("b_gate_pass_rate"),
  "wins": v.get("challenger_wins") if v else d.get("wins"),
  "ranking_formula": v.get("ranking_formula"),
  "note": "p3665 R596 warm chall v4 re-sim (k=3 tau=0.03) vs reign34; prior p3588c was k=1",
}
Path("$SIM_DEC").write_text(json.dumps(dec, indent=2)+"\n")
print(json.dumps(dec, indent=2))
# fail closed if still stamped k=1
k = dp.get("n_teacher_samples")
if k != 3:
    raise SystemExit(f"FATAL duel_params.n_teacher_samples={k} (want 3)")
PY
  date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r596_reign34_wvk7_n80.done
else
  log "FATAL missing sim result"
  exit 1
fi
log "DONE R596 v4 n80 vs reign34"
