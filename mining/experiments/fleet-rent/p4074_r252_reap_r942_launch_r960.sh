#!/usr/bin/env bash
# p4074: R942 REFUTE ~0.48× → exact-PID reap chall:8002 → R960 MidCtx MidLoβ Ultra TRAIN on GPUs 4,5
set -euo pipefail
log() { echo "[p4074-arm] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
STOP() {
  local pid=$1 why=$2
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "stop pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
ARMED=/root/logs/p4074_r960_armed.done
if [[ -f "$ARMED" ]]; then log "already armed $(cat $ARMED) — exit"; exit 0; fi

# stamp R942 decision if result exists
if [[ -f /root/affine_data/r942_sim_result_reign36_wvk7.json ]]; then
  /root/venv/bin/python3 - <<'PY'
import json,time
from pathlib import Path
d=json.loads(Path("/root/affine_data/r942_sim_result_reign36_wvk7.json").read_text())
v=d.get("verdict") if isinstance(d.get("verdict"), dict) else {}
chal=(v.get("challenger") or {}) if isinstance(v, dict) else {}
dp=(v.get("duel_params") or {}) if isinstance(v, dict) else {}
margin = v.get("margin") if v else (d.get("margin") or d.get("mean_margin"))
se = v.get("se") if v else (d.get("se") or d.get("stderr"))
bar = max(2.0 * float(se), 0.002) if se is not None else None
dec={
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo": "R942", "contract": "wvk7",
  "n_teacher_samples": dp.get("n_teacher_samples"), "tau": dp.get("tau"),
  "king": "reign36", "margin": margin, "se": se,
  "z": v.get("z") if v else d.get("z"),
  "n": v.get("n_paired_turns") if v else (d.get("n") or d.get("n_scored")),
  "bar": bar, "thought_median": chal.get("median_len_z"),
  "b_pass": chal.get("b_gate_pass_rate"),
  "verdict": "REFUTE",
  "note": "p4074 R942 SoftCtx MidLoβ UltraExtra ~0.48× → R960 MidCtx isolate",
}
Path("/root/affine_data/r942_decision_reign36_wvk7.json").write_text(json.dumps(dec, indent=2)+"\n")
print(json.dumps(dec, indent=2))
PY
fi

for pf in /root/logs/vllm_chall_r942.pid /root/logs/r942_sim_wvk7.pid \
          /root/logs/p4073_r942_lean_outer.pid /root/logs/p4055_r942_lean_outer.pid; do
  [[ -f "$pf" ]] || continue
  STOP "$(cat "$pf" 2>/dev/null || true)" "pidfile $pf"
  rm -f "$pf"
done

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  STOP "$pid" "r942 :8002 listener"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -qE 'r942_merged|local-r942-reign36|vllm_chall_r942' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|vera6/affine' && continue
  STOP "$pid" "r942 chall/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r942_merged|run_sim_duel.py.*r942|chall_r942/ && !/awk/ {print $1}')

for i in $(seq 1 60); do
  used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  log "VRAM45=$used45 iter=$i"
  [[ "$used45" -lt 8192 ]] && break
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
    echo "$cmd" | grep -qE 'r942_merged|VLLM::Worker' || continue
    echo "$cmd" | grep -qE 'GLM-4.5|8000|8001|vera6' && continue
    STOP "$pid" "r942 residual VRAM"
  done < <(ps -eo pid=,args= | awk '/r942_merged|VLLM::Worker_TP/ && !/awk|GLM-4.5|vera6/ {print $1}')
  sleep 2
done
used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
log "post-reap VRAM45=$used45"

EXP=r960-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-ultralolr
chmod +x /root/mining_src/$EXP/*.sh
mkdir -p /root/r960
[[ -s /root/r960/dpo_duel_reason.jsonl ]] || cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r960/dpo_duel_reason.jsonl
test -s /root/r960/dpo_duel_reason.jsonl

nohup bash /root/mining_src/$EXP/lean_train_r252_gpus45_p4074.sh >/root/logs/p4074_r960_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4074_r960_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r960_train_then_merge_p4074.sh >/root/logs/r960_wait_outer.nohup 2>&1 &
echo $! >/root/logs/r960_wait_outer.pid

sleep 12
log "r960_train.pid=$(cat /root/logs/r960_train.pid 2>/dev/null || echo missing)"
log "lean_outer=$(cat /root/logs/p4074_r960_lean_outer.pid) wait=$(cat /root/logs/r960_wait_outer.pid)"
tail -n 25 /root/logs/r960_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ss -ltnp | grep -E ':800[01]' || true
date -u +%Y-%m-%dT%H:%M:%SZ >"$ARMED"
log "ARMED $(cat $ARMED)"
