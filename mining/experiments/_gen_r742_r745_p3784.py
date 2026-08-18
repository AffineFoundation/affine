#!/usr/bin/env python3
"""Generate R742-R745 experiment packs for p3784 (local only)."""
from __future__ import annotations

import shutil
from pathlib import Path

root = Path("/home/const/subnet120/mining/experiments")
BASE = (
    "/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/"
    "snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f"
)

SPECS = [
    dict(
        rid="r742",
        RID="R742",
        name="r742-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-hyperextrasteps-ep3-lolr",
        parent="r735-r252-offline-dpo-hialpha-midrank-hibeta-shortctx-hyperextrasteps-ep3-lolr",
        pod="r252",
        gpus="6,7",
        lr="1e-6",
        lora_r=64,
        lora_a=128,
        beta=0.3,
        steps=10800,
        maxlen=6144,
        epochs=3,
        axis="r252_offline_dpo_hialpha_hirank_hibeta_shortctx_hyperextrasteps_ep3_lolr",
        parent_signal=(
            "R735 Short MidRank HiBeta HyperExtra REFUTE ~-0.08x near-parity → HiRank sibling; "
            "≠ MidRank HyperExtra R735; ≠ UltraExtra Short HiRank HiBeta R679; "
            "≠ MidCtx HiRank HiBeta HyperExtra R694; ≠ Online / ≠ GRPO"
        ),
        data_fallbacks=["r735", "r725", "r683"],
        refute_of="r735",
        outer_pid="p3781_r735",
    ),
    dict(
        rid="r743",
        RID="R743",
        name="r743-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-hyperextrasteps-ep3-lolr",
        parent="r734-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-hyperextrasteps-ep3-lolr",
        pod="r252",
        gpus="4,5",
        lr="1e-6",
        lora_r=64,
        lora_a=128,
        beta=0.02,
        steps=10800,
        maxlen=6144,
        epochs=3,
        axis="r252_offline_dpo_hialpha_hirank_lobeta_shortctx_hyperextrasteps_ep3_lolr",
        parent_signal=(
            "R734 Short MidRank LoBeta HyperExtra REFUTE ~-1.14x → HiRank sibling; "
            "≠ MidRank HyperExtra R734; ≠ UltraExtra Short HiRank LoBeta R686; "
            "≠ MidCtx HiRank LoBeta HyperExtra R707; ≠ Online / ≠ GRPO"
        ),
        data_fallbacks=["r734", "r724", "r696"],
        refute_of="r734",
        outer_pid="p3781_r734",
    ),
    dict(
        rid="r744",
        RID="R744",
        name="r744-r252-offline-dpo-hialpha-hirank-midbeta-midctx-superextrasteps-ep3-lolr",
        parent="r737-r252-offline-dpo-hialpha-hirank-midbeta-midctx-hyperextrasteps-ep3-lolr",
        pod="crown",
        gpus="6,7",
        lr="1e-6",
        lora_r=64,
        lora_a=128,
        beta=0.1,
        steps=14400,
        maxlen=8192,
        epochs=3,
        axis="r252_offline_dpo_hialpha_hirank_midbeta_midctx_superextrasteps_ep3_lolr",
        parent_signal=(
            "R737 MidCtx HiRank MidBeta HyperExtra REFUTE ~-0.90x → SuperExtra amplify; "
            "≠ HyperExtra R737; ≠ UltraExtra MidCtx HiRank MidBeta R689; "
            "≠ SoftCtx HiRank MidBeta SuperExtra R715; ≠ Online / ≠ GRPO"
        ),
        data_fallbacks=["r737", "r727", "r689"],
        refute_of="r737",
        outer_pid="p3782_r737",
    ),
    dict(
        rid="r745",
        RID="R745",
        name="r745-r252-offline-dpo-hialpha-hirank-hibeta-midctx-superextrasteps-ep3-lolr",
        parent="r729-r252-offline-dpo-hialpha-midrank-hibeta-midctx-superextrasteps-ep3-lolr",
        pod="golden",
        gpus="4,5",
        lr="1e-6",
        lora_r=64,
        lora_a=128,
        beta=0.3,
        steps=14400,
        maxlen=8192,
        epochs=3,
        axis="r252_offline_dpo_hialpha_hirank_hibeta_midctx_superextrasteps_ep3_lolr",
        parent_signal=(
            "R729 MidCtx MidRank HiBeta SuperExtra REFUTE ~-0.49x → HiRank sibling; "
            "≠ MidRank SuperExtra R729; ≠ UltraExtra MidCtx MidRank HiBeta R684; "
            "≠ MidCtx HiRank HiBeta HyperExtra R694; ≠ Online / ≠ GRPO"
        ),
        data_fallbacks=["r729", "r719", "r684"],
        refute_of="r729",
        outer_pid="p3782_r729",
    ),
]


def main() -> None:
    for s in SPECS:
        dest = root / s["name"]
        dest.mkdir(parents=True, exist_ok=True)
        parent = root / s["parent"]
        for f in ("train_dpo.py", "dpo_duel_reason.jsonl"):
            src = parent / f
            if src.exists():
                shutil.copy2(src, dest / f)

        rid, RID = s["rid"], s["RID"]
        g0, g1 = s["gpus"].split(",")
        envp = RID
        ps = s["parent_signal"].replace("'", "\\'")

        start = f"""#!/usr/bin/env bash
# {RID}: {s['axis']}
set -euo pipefail
export PATH="/root/.local/bin:${{PATH}}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${{HF_HOME:-/root/hf}}
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{s['gpus']}}}
export PYTHONPATH=/root/mining_src/affine_pkg:${{PYTHONPATH:-}}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${{SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}}
OUT=${{OUT:-/root/{rid}}}
DATA=${{DATA:-$OUT/dpo_duel_reason.jsonl}}
BASE={BASE}
TRAIN_DIR=$OUT/train
LOG=${{LOG:-/root/logs/{rid}_train.nohup}}
LR=${{{envp}_LR:-{s['lr']}}}
LORA_R=${{{envp}_LORA_R:-{s['lora_r']}}}
LORA_ALPHA=${{{envp}_LORA_ALPHA:-{s['lora_a']}}}
BETA=${{{envp}_BETA:-{s['beta']}}}
MAX_STEPS=${{{envp}_MAX_STEPS:-{s['steps']}}}
MAX_LEN=${{{envp}_MAX_LEN:-{s['maxlen']}}}
EPOCHS=${{{envp}_EPOCHS:-{s['epochs']}}}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *r252-merged*|*5czsc2fc98*) ;; *) echo "FATAL bad BASE"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[{rid}] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \\
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \\
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \\
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \\
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/{rid}_train.pid >"$OUT/train.pid"
python3 -c "
import json, time
from pathlib import Path
meta = {{
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'axis': '{s['axis']}',
  'base': '$BASE', 'data': '$DATA', 'examples': $n,
  'lr': '$LR', 'lora_r': $LORA_R, 'lora_alpha': $LORA_ALPHA, 'beta': $BETA,
  'max_steps': $MAX_STEPS, 'max_len': $MAX_LEN, 'epochs': $EPOCHS,
  'gpus': '$CUDA_VISIBLE_DEVICES',
  'pid': int(Path('/root/logs/{rid}_train.pid').read_text().strip()),
  'parent_signal': '{ps}',
  'decision_rule': 'Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign35 (v4 k=3)',
}}
Path('$OUT/train_meta.json').write_text(json.dumps(meta, indent=2)+'\\n')
Path('/root/affine_data/{rid}_train_launched.json').write_text(json.dumps(meta, indent=2)+'\\n')
print(json.dumps(meta, indent=2))
"
echo "[{rid}] TRAIN_ARMED pid=$(cat /root/logs/{rid}_train.pid)"
"""
        (dest / f"start_{rid}.sh").write_text(start)

        fb_lines = []
        for d in s["data_fallbacks"]:
            fb_lines.append(
                f'  elif [[ -s /root/{d}/dpo_duel_reason.jsonl ]]; then\n'
                f'    cp -f /root/{d}/dpo_duel_reason.jsonl "$DATA"'
            )
        fb = "\n".join(fb_lines)
        lean_name = f"lean_train_{s['pod']}_gpus{g0}{g1}_p3784.sh"
        lean = f"""#!/usr/bin/env bash
# p3784: {RID} on {s['pod']} GPUs {s['gpus']} after {s['refute_of'].upper()} REFUTE
set -euo pipefail
exec >/root/logs/{rid}_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${{PYTHONPATH:-}}
export CUDA_VISIBLE_DEVICES={s['gpus']} SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[{rid}-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs {s['gpus']}"
BASE={BASE}
EXP={s['name']}
mkdir -p /root/{rid} /root/logs /root/affine_data \\
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \\
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/{rid}/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
{fb}
  else
    echo "FATAL missing data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[{rid}-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/{rid}_train.pid ]]; then
  old=$(cat /root/logs/{rid}_train.pid 2>/dev/null || true)
  if [[ -n "${{old:-}}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL {rid} already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {s['gpus']} | awk '{{s+=$1}} END{{print s+0}}')
  echo "[{rid}-lean] wait VRAM{g0}+{g1} used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {s['gpus']} | awk '{{s+=$1}} END{{print s+0}}')
[[ "$used" -lt 8192 ]] || {{ echo "FATAL GPUs {s['gpus']} busy"; exit 1; }}
rm -rf /root/{rid}/train; mkdir -p /root/{rid}/train; : >/root/logs/{rid}_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES={s['gpus']} {envp}_LR={s['lr']} {envp}_MAX_STEPS={s['steps']} {envp}_LORA_R={s['lora_r']} {envp}_LORA_ALPHA={s['lora_a']} {envp}_BETA={s['beta']} {envp}_MAX_LEN={s['maxlen']} {envp}_EPOCHS={s['epochs']}
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/{rid} DATA=/root/{rid}/dpo_duel_reason.jsonl LOG=/root/logs/{rid}_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_{rid}.sh
echo "[{rid}-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/{rid}_train.pid) BASE=$BASE"
"""
        (dest / lean_name).write_text(lean)

        merges = list(parent.glob("lean_merge_*.sh"))
        if merges:
            shutil.copy2(merges[0], dest / merges[0].name)
            merge_script = f"/root/mining_src/{s['name']}/{merges[0].name}"
            txt = (dest / merges[0].name).read_text()
            old_rid = s["refute_of"]
            txt2 = (
                txt.replace(f"/tmp/{old_rid}_merged", f"/tmp/{rid}_merged")
                .replace(f"/root/{old_rid}/", f"/root/{rid}/")
                .replace(f"{old_rid}_", f"{rid}_")
                .replace(s["parent"], s["name"])
            )
            (dest / merges[0].name).write_text(txt2)
        else:
            merge_script = f"/root/mining_src/{s['name']}/lean_merge_{s['pod']}_gpus{g0}{g1}_p3784.sh"
            (dest / f"lean_merge_{s['pod']}_gpus{g0}{g1}_p3784.sh").write_text(
                '#!/usr/bin/env bash\necho "FATAL no merge script"; exit 1\n'
            )

        wait = f"""#!/usr/bin/env bash
# p3784: wait {RID} TRAIN_DONE → merge on GPUs {s['gpus']} — never pkill -f.
set -euo pipefail
log() {{ echo "[p3784-{rid}-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }}
TRAIN_PID_FILE=/root/logs/{rid}_train.pid
ADAPTER=/root/{rid}/train/adapter
MERGE_SCRIPT={merge_script}
LAUNCHED=/root/logs/{rid}_merge_launched.p3784
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && {{ log "already launched"; exit 0; }}
log "armed wait {RID} → merge GPUs {s['gpus']}"
while true; do
  train_alive=0
  if [[ -f "$TRAIN_PID_FILE" ]]; then
    tpid=$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then train_alive=1; fi
  fi
  adapter_ok=0
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] && adapter_ok=1
  if [[ "$train_alive" -eq 0 && "$adapter_ok" -eq 1 ]]; then
    log "TRAIN_DONE — launch merge"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/{rid}_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3784_{rid}_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3784_{rid}_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/p3784_{rid}_merge.outer.pid)"
    exit 0
  fi
  step=$(grep -o '"step": [0-9]*' /root/logs/{rid}_train.nohup 2>/dev/null | tail -1 | awk '{{print $2}}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${{step:-?}}"
  sleep 30
done
"""
        (dest / f"wait_{rid}_train_then_merge_p3784.sh").write_text(wait)

        refute = s["refute_of"]
        REFU = refute.upper()
        reap = f"""#!/usr/bin/env bash
# p3784: {REFU} REFUTE → reap chall GPUs {s['gpus']} → launch {RID} TRAIN + wait→merge. Never pkill -f.
set -euo pipefail
log() {{ echo "[p3784-reap-{refute}] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }}
stop_pid() {{
  local pid=$1; local why=${{2:-}}
  [[ -n "${{pid:-}}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}}
stop_pidfile() {{
  local pidf=$1; local why=${{2:-}}
  [[ -f "$pidf" ]] || return 0
  local pid; pid=$(cat "$pidf" 2>/dev/null || true)
  stop_pid "$pid" "$why pidf=$pidf"
  rm -f "$pidf"
}}
log "START reap {REFU} chall; keep /tmp/{refute}_merged"
stop_pidfile /root/logs/{refute}_sim_wvk7.pid "{refute} sim"
stop_pidfile /root/logs/{s['outer_pid']}_lean_outer.pid "{refute} outer"
stop_pidfile /root/logs/vllm_chall_{refute}.pid "{refute} vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "{refute} argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\\/tmp\\/{refute}_merged/ && !/awk/ {{print $1}}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i {s['gpus']} --query-gpu=memory.used --format=csv,noheader,nounits | awk '{{s+=$1}} END{{print s+0}}')
  log "wait free {s['gpus']} used_mib=$used iter=$i"
  [[ "${{used:-999999}}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i {s['gpus']} --query-gpu=memory.used --format=csv,noheader,nounits | awk '{{s+=$1}} END{{print s+0}}')
[[ "${{used:-999999}}" -lt 8192 ]] || {{ log "FATAL GPUs {s['gpus']} still busy"; exit 1; }}
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/{refute}_refute_reaped.p3784
python3 - <<'PY'
import json, time
from pathlib import Path
v = json.load(open('/root/affine_data/{refute}_sim_result_reign35_wvk7.json'))['verdict']
m, se, z, n = v['margin'], v['se'], v['z'], v['n_paired_turns']
bar = max(2.0*se, 0.002)
dec = {{
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'exp': '{REFU}',
  'verdict': 'REFUTE',
  'margin': m, 'se': se, 'z': z, 'n': n, 'bar': bar,
  'xbar': (m/bar) if bar else None,
  'thought_median': v['challenger']['median_len_z'],
  'b_pass': v['challenger']['b_gate_pass_rate'],
  'k': v['duel_params']['n_teacher_samples'],
  'tau': v['duel_params']['tau'],
  'king': 'tammyfritz/Affine-5hmwhnfbix-tammy2',
  'next': '{RID} TRAIN on GPUs {s['gpus']}',
}}
Path('/root/affine_data/{refute}_decision_reign35_wvk7.json').write_text(json.dumps(dec, indent=2)+'\\n')
print(json.dumps(dec, indent=2))
PY
log "launch {RID} TRAIN"
nohup bash /root/mining_src/{s['name']}/{lean_name} \\
  >/root/logs/p3784_{rid}_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3784_{rid}_lean_outer.pid
sleep 2
nohup bash /root/mining_src/{s['name']}/wait_{rid}_train_then_merge_p3784.sh \\
  >/root/logs/p3784_{rid}_wait.nohup 2>&1 &
echo $! >/root/logs/p3784_{rid}_wait.pid
log "{RID} lean outer=$(cat /root/logs/p3784_{rid}_lean_outer.pid) wait=$(cat /root/logs/p3784_{rid}_wait.pid)"
"""
        (dest / f"p3784_reap_{refute}_launch_{rid}_{s['pod']}.sh").write_text(reap)

        (dest / "plan.md").write_text(
            f"# {RID} — {s['axis']}\n"
            f"Parent: {s['parent_signal']}\n"
            f"Knobs: β={s['beta']} r={s['lora_r']} α={s['lora_a']} lr={s['lr']} "
            f"@{s['maxlen']} steps={s['steps']} ep={s['epochs']}\n"
            f"Pod/GPUs: {s['pod']} {s['gpus']}\n"
            "Decision: Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) "
            "AND thought≥80 AND B≥0.30 vs reign35.\n"
        )
        for sh in dest.glob("*.sh"):
            sh.chmod(0o755)
        print("OK", s["name"], "nfiles", len(list(dest.iterdir())))


if __name__ == "__main__":
    main()
