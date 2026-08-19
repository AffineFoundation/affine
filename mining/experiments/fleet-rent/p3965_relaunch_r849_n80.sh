#!/usr/bin/env bash
# p3965: R849 n80 died on corpus-manifest rename race with R848.
# Challenger :8003 still warm — relaunch sim only (do not touch R848 / T+K).
set -euo pipefail
LOG=/root/logs/p3965_r849_n80_relaunch.log
SIM_N80=/root/affine_data/r849_sim_result_reign36_wvk7.json
PROG=/root/affine_data/r849_sim_progress_reign36_wvk7.json
SIM_DEC=/root/affine_data/r849_sim_decision_reign36_wvk7.json
MERGE_DIR=/tmp/r849_merged
CHALL_PORT=8003
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
AFFINE_DATA_DIR=/root/affine_data
HF_HOME=${HF_HOME:-/root/hf}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[p3965-r849] $(ts) $*" | tee -a "$LOG"; }

if [[ -f /root/logs/r849_sim_wvk7.pid ]]; then
  opid=$(cat /root/logs/r849_sim_wvk7.pid || true)
  if [[ -n "${opid:-}" ]] && kill -0 "$opid" 2>/dev/null; then
    log "R849 sim already live pid=$opid — exit"
    exit 0
  fi
fi

# Do not disturb R848
if ! kill -0 20779 2>/dev/null; then
  log "WARN R848 sim pid 20779 not alive (may have finished)"
fi
if ! kill -0 17948 2>/dev/null; then
  log "FATAL chall :8003 pid 17948 dead"
  exit 1
fi
if ! curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null; then
  log "FATAL chall :${CHALL_PORT} not serving"
  exit 1
fi

# Fresh slice hash for relaunch (prior n80 never scored)
BLOCK_HASH=$(python3 - <<'PY'
import hashlib
print(hashlib.sha256(b"r849-reign36-wvk7-p3965-relaunch-corpus-race").hexdigest())
PY
)

# Drop stale lock if orphaned (>120s, no holder)
if [[ -f "${AFFINE_DATA_DIR}/.corpus_sync.lock" ]]; then
  age=$(( $(date +%s) - $(stat -c %Y "${AFFINE_DATA_DIR}/.corpus_sync.lock") ))
  log "corpus lock age=${age}s"
  if [[ $age -gt 120 ]] && ! lsof "${AFFINE_DATA_DIR}/.corpus_sync.lock" >/dev/null 2>&1; then
    log "removing stale corpus lock"
    rm -f "${AFFINE_DATA_DIR}/.corpus_sync.lock"
  else
    for i in $(seq 1 20); do
      [[ -f "${AFFINE_DATA_DIR}/.corpus_sync.lock" ]] || break
      sleep 3
    done
  fi
fi

rm -f "$SIM_N80" "$PROG" "$SIM_DEC" \
  /root/logs/r849_n80_launched.p3956 \
  /root/logs/r849_reign36_wvk7_pipeline.done

log "launch n80 vs $KING_REPO block_hash=${BLOCK_HASH:0:16}… chall=:${CHALL_PORT}"
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
  --hotkey local-r849-reign36-wvk7 \
  --block-hash "$BLOCK_HASH" \
  --out "$SIM_N80" \
  --progress-out "$PROG" \
  --save-artifact \
  >>"$LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" > /root/logs/r849_sim_wvk7.pid
ts > /root/logs/r849_n80_launched.p3965
log "n80 pid=$SIM_PID — arming waiter"

nohup bash -c "
SIM_PID=$SIM_PID
SIM_N80=$SIM_N80
SIM_DEC=$SIM_DEC
LOG=$LOG
while kill -0 \$SIM_PID 2>/dev/null; do sleep 30; done
wait \$SIM_PID || true
if [[ -f \$SIM_N80 ]]; then
  /root/venv/bin/python3 - <<PY
import json
from pathlib import Path
d=json.loads(Path('$SIM_N80').read_text())
v=d.get('verdict') if isinstance(d.get('verdict'), dict) else {}
chal=(v.get('challenger') or {}) if isinstance(v, dict) else {}
dp=(v.get('duel_params') or {}) if isinstance(v, dict) else {}
margin = v.get('margin') if v else (d.get('margin') or d.get('mean_margin'))
se = v.get('se') if v else (d.get('se') or d.get('stderr'))
bar = max(2.0 * float(se), 0.002) if se is not None else None
dec={
  'utc': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
  'hypo': 'R849',
  'contract': 'wvk7',
  'pass': 'p3965',
  'n_teacher_samples': dp.get('n_teacher_samples'),
  'tau': dp.get('tau'),
  'king': 'reign36',
  'margin': margin,
  'se': se,
  'z': v.get('z') if v else d.get('z'),
  'n': v.get('n_paired_turns') if v else (d.get('n') or d.get('n_scored')),
  'bar': bar,
  'thought_median': chal.get('median_len_z'),
  'b_pass': chal.get('b_gate_pass_rate'),
  'wins': v.get('challenger_wins') if v else d.get('wins'),
  'note': 'p3965 relaunch after corpus-manifest race; chall :8003 reused; vs reign36 vera wvk7',
}
Path('$SIM_DEC').write_text(json.dumps(dec, indent=2)+'\n')
print(json.dumps(dec, indent=2))
if dp.get('n_teacher_samples') != 3:
  raise SystemExit('FATAL k!=3')
PY
  echo \"[p3965-r849] \$(date -u +%Y-%m-%dT%H:%M:%SZ) SIM_DONE\" >>\$LOG
  date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r849_reign36_wvk7_pipeline.done
else
  echo \"[p3965-r849] \$(date -u +%Y-%m-%dT%H:%M:%SZ) FATAL missing sim result\" >>\$LOG
fi
" >>/root/logs/p3965_r849_n80_waiter.outer.log 2>&1 &
echo $! > /root/logs/p3965_r849_n80_waiter.outer.pid
log "waiter pid=$(cat /root/logs/p3965_r849_n80_waiter.outer.pid)"

sleep 12
if kill -0 "$SIM_PID" 2>/dev/null; then
  log "STILL_ALIVE pid=$SIM_PID"
else
  log "DIED_EARLY — tail log"
  tail -60 "$LOG" || true
  exit 1
fi
tail -30 "$LOG" || true
