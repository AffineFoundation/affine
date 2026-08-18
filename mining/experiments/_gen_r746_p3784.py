#!/usr/bin/env python3
"""Generate R746 after R730 REFUTE (p3784)."""
from __future__ import annotations

import shutil
from pathlib import Path

root = Path("/home/const/subnet120/mining/experiments")
parent = root / "r730-r252-offline-dpo-hialpha-hirank-midbeta-shortctx-superextrasteps-ep3-lolr"
dest = root / "r746-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-superextrasteps-ep3-lolr"
BASE = (
    "/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/"
    "snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f"
)
name = dest.name

if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(parent, dest)

for f in list(dest.glob("*")):
    if f.name.startswith(("lean_chall", "lean_train", "wait_", "start_", "p378")):
        f.unlink()

for ms in list(dest.glob("lean_merge_*.sh")):
    t = (
        ms.read_text()
        .replace("/tmp/r730_merged", "/tmp/r746_merged")
        .replace("/root/r730/", "/root/r746/")
        .replace("r730_", "r746_")
        .replace(parent.name, name)
    )
    ms.write_text(t)

merge = list(dest.glob("lean_merge_*.sh"))[0].name

(dest / "start_r746.sh").write_text(
    f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.local/bin:${{PATH}}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${{HF_HOME:-/root/hf}}
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-6,7}}
export PYTHONPATH=/root/mining_src/affine_pkg:${{PYTHONPATH:-}}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${{SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}}
OUT=${{OUT:-/root/r746}}
DATA=${{DATA:-$OUT/dpo_duel_reason.jsonl}}
BASE={BASE}
TRAIN_DIR=$OUT/train
LOG=${{LOG:-/root/logs/r746_train.nohup}}
LR=${{R746_LR:-1e-6}}; LORA_R=${{R746_LORA_R:-64}}; LORA_ALPHA=${{R746_LORA_ALPHA:-128}}
BETA=${{R746_BETA:-0.02}}; MAX_STEPS=${{R746_MAX_STEPS:-14400}}; MAX_LEN=${{R746_MAX_LEN:-6144}}; EPOCHS=${{R746_EPOCHS:-3}}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"; test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r746] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \\
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r746_train.pid >"$OUT/train.pid"
python3 -c "
import json, time
from pathlib import Path
meta = {{
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'axis': 'r252_offline_dpo_hialpha_hirank_lobeta_shortctx_superextrasteps_ep3_lolr',
  'base': '$BASE', 'data': '$DATA', 'examples': $n,
  'lr': '$LR', 'lora_r': $LORA_R, 'lora_alpha': $LORA_ALPHA, 'beta': $BETA,
  'max_steps': $MAX_STEPS, 'max_len': $MAX_LEN, 'epochs': $EPOCHS,
  'gpus': '$CUDA_VISIBLE_DEVICES',
  'pid': int(Path('/root/logs/r746_train.pid').read_text().strip()),
  'parent_signal': 'R730 Short HiRank MidBeta SuperExtra REFUTE ~-0.57x → LoBeta sibling; ≠ MidBeta SuperExtra R730; ≠ HyperExtra Short HiRank LoBeta R743; ≠ Online / ≠ GRPO',
  'decision_rule': 'Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign35 (v4 k=3)',
}}
Path('$OUT/train_meta.json').write_text(json.dumps(meta, indent=2)+'\\n')
Path('/root/affine_data/r746_train_launched.json').write_text(json.dumps(meta, indent=2)+'\\n')
print(json.dumps(meta, indent=2))
"
echo "[r746] TRAIN_ARMED pid=$(cat /root/logs/r746_train.pid)"
"""
)

(dest / "lean_train_golden_gpus67_p3784.sh").write_text(
    f"""#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/r746_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${{PYTHONPATH:-}}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r746-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE={BASE}
EXP={name}
mkdir -p /root/r746 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r746/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r730/dpo_duel_reason.jsonl ]]; then cp -f /root/r730/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r746-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{{s+=$1}} END{{print s+0}}')
  echo "[r746-lean] wait used=$used"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{{s+=$1}} END{{print s+0}}')
[[ "$used" -lt 8192 ]] || {{ echo FATAL busy; exit 1; }}
rm -rf /root/r746/train; mkdir -p /root/r746/train; : >/root/logs/r746_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE CUDA_VISIBLE_DEVICES=6,7 R746_LR=1e-6 R746_MAX_STEPS=14400 R746_LORA_R=64 R746_LORA_ALPHA=128 R746_BETA=0.02 R746_MAX_LEN=6144 R746_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r746 DATA=/root/r746/dpo_duel_reason.jsonl LOG=/root/logs/r746_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r746.sh
echo "[r746-lean] TRAIN launched pid=$(cat /root/logs/r746_train.pid)"
"""
)

(dest / "wait_r746_train_then_merge_p3784.sh").write_text(
    f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[p3784-r746-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }}
TRAIN_PID_FILE=/root/logs/r746_train.pid
ADAPTER=/root/r746/train/adapter
MERGE_SCRIPT=/root/mining_src/{name}/{merge}
LAUNCHED=/root/logs/r746_merge_launched.p3784
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && {{ log already; exit 0; }}
log armed
while true; do
  train_alive=0
  if [[ -f "$TRAIN_PID_FILE" ]]; then
    tpid=$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then train_alive=1; fi
  fi
  adapter_ok=0
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] && adapter_ok=1
  if [[ "$train_alive" -eq 0 && "$adapter_ok" -eq 1 ]]; then
    log TRAIN_DONE
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r746_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3784_r746_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3784_r746_merge.outer.pid
    exit 0
  fi
  sleep 30
done
"""
)

(dest / "p3784_reap_r730_launch_r746_golden.sh").write_text(
    f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[p3784-reap-r730] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }}
stop_pid() {{
  local pid=$1
  [[ -n "${{pid:-}}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}}
stop_pidfile() {{
  local pidf=$1
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)"
  rm -f "$pidf"
}}
log START
stop_pidfile /root/logs/r730_sim_wvk7.pid
stop_pidfile /root/logs/p3783_r730_lean_outer.pid
stop_pidfile /root/logs/vllm_chall_r730.pid
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\\/tmp\\/r730_merged/ && !/awk/ {{print $1}}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{{s+=$1}} END{{print s+0}}')
  [[ "${{used:-999}}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{{s+=$1}} END{{print s+0}}')
[[ "${{used:-999}}" -lt 8192 ]] || {{ log FATAL; exit 1; }}
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r730_refute_reaped.p3784
python3 - <<'PY'
import json, time
from pathlib import Path
v = json.load(open('/root/affine_data/r730_sim_result_reign35_wvk7.json'))['verdict']
m, se, z, n = v['margin'], v['se'], v['z'], v['n_paired_turns']
bar = max(2.0*se, 0.002)
dec = {{
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'exp': 'R730', 'verdict': 'REFUTE',
  'margin': m, 'se': se, 'z': z, 'n': n, 'bar': bar,
  'xbar': (m/bar) if bar else None,
  'thought_median': v['challenger']['median_len_z'],
  'b_pass': v['challenger']['b_gate_pass_rate'],
  'k': v['duel_params']['n_teacher_samples'],
  'tau': v['duel_params']['tau'],
  'king': 'tammyfritz/Affine-5hmwhnfbix-tammy2',
  'next': 'R746 TRAIN on GPUs 6,7',
}}
Path('/root/affine_data/r730_decision_reign35_wvk7.json').write_text(json.dumps(dec, indent=2)+'\\n')
print(json.dumps(dec, indent=2))
PY
nohup bash /root/mining_src/{name}/lean_train_golden_gpus67_p3784.sh >/root/logs/p3784_r746_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3784_r746_lean_outer.pid
sleep 2
nohup bash /root/mining_src/{name}/wait_r746_train_then_merge_p3784.sh >/root/logs/p3784_r746_wait.nohup 2>&1 &
echo $! >/root/logs/p3784_r746_wait.pid
log R746 outer=$(cat /root/logs/p3784_r746_lean_outer.pid) wait=$(cat /root/logs/p3784_r746_wait.pid)
"""
)

result = """# R730 result — REFUTE v4 (p3784)

- **margin** = −0.003333 · **SE** = 0.002948 · **z** = −1.130 · **n** = 76
- **bar** = max(2·SE, δ) = 0.005897 · **×bar** ≈ **−0.57×**
- thought median = 169.5 ✓ · B pass = 0.385 ✓ · k=3 τ=0.03 ✓
- next = **R746** Short HiRank LoBeta SuperExtra TRAIN on golden 6,7
"""
(dest / "result.md").write_text(result)
(parent / "result.md").write_text(result)
(dest / "plan.md").write_text(
    "# R746 Short HiRank LoBeta SuperExtra — sibling of R730 MidBeta REFUTE ~-0.57x\n"
)
for sh in dest.glob("*.sh"):
    sh.chmod(0o755)
print("OK", dest, len(list(dest.iterdir())))
