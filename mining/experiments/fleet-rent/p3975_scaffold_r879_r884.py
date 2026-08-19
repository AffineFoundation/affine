#!/usr/bin/env python3
"""Scaffold R879–R884 marsplan Soft Mid Mid Soft UltraLoLR knob isolates for R337/R338 idle fill."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/home/const/subnet120/mining")
SRC_JSONL = (
    ROOT
    / "experiments/r863-marsplan-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl"
)
SRC_TRAIN = (
    ROOT
    / "experiments/r863-marsplan-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/train_dpo.py"
)
BASE = (
    "/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/"
    "snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46"
)
STAMP = "p3975"

# rid, suffix, gpus, host_tag, lr, lora_r, lora_alpha, beta, max_steps, max_len, epochs, axis, signal
AXES = [
    (
        "879",
        "lorank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr",
        "2,3",
        "r337",
        "5e-7",
        16,
        64,
        "0.1",
        19200,
        12288,
        4,
        "marsplan_offline_dpo_hialpha_lorank_midbeta_softctx_megasuperextrasteps_ep4_ultralolr",
        "R863 MidRank SoftCtx Midβ → LoRank r=16 α=64 isolate; ≠ MidRank R863 / ≠ Online / ≠ GRPO",
    ),
    (
        "880",
        "midrank-midbeta-softctx-megasuperextrasteps-ep4-lolr",
        "4,5",
        "r337",
        "1e-6",
        32,
        128,
        "0.1",
        19200,
        12288,
        4,
        "marsplan_offline_dpo_hialpha_midrank_midbeta_softctx_megasuperextrasteps_ep4_lolr",
        "R863 UltraLoLR 5e-7 SoftCtx MidRank Midβ → LoLR 1e-6 isolate; ≠ UltraLoLR R863 / ≠ Online / ≠ GRPO",
    ),
    (
        "881",
        "midrank-midbeta-softctx-ultramegasuperextrasteps-ep4-ultralolr",
        "6,7",
        "r337",
        "5e-7",
        32,
        128,
        "0.1",
        28800,
        12288,
        4,
        "marsplan_offline_dpo_hialpha_midrank_midbeta_softctx_ultramegasuperextrasteps_ep4_ultralolr",
        "R863 MegaSuper 19200 SoftCtx MidRank Midβ → UltraMega 28800 isolate; ≠ 19200 R863 / ≠ Online / ≠ GRPO",
    ),
    (
        "882",
        "midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr",
        "2,3",
        "r338",
        "5e-7",
        32,
        128,
        "0.05",
        19200,
        12288,
        4,
        "marsplan_offline_dpo_hialpha_midrank_midlobeta_softctx_megasuperextrasteps_ep4_ultralolr",
        "R863 Midβ=0.1 / R864 Loβ=0.02 SoftCtx MidRank → MidLoβ=0.05 isolate; ≠ Online / ≠ GRPO",
    ),
    (
        "883",
        "midrank-midbeta-softctx-megasuperextrasteps-ep5-ultralolr",
        "4,5",
        "r338",
        "5e-7",
        32,
        128,
        "0.1",
        19200,
        12288,
        5,
        "marsplan_offline_dpo_hialpha_midrank_midbeta_softctx_megasuperextrasteps_ep5_ultralolr",
        "R863 ep=4 SoftCtx MidRank Midβ → ep=5 isolate; ≠ ep4 R863 / ≠ Online / ≠ GRPO",
    ),
    (
        "884",
        "megaalpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr",
        "6,7",
        "r338",
        "5e-7",
        32,
        256,
        "0.1",
        19200,
        12288,
        4,
        "marsplan_offline_dpo_megaalpha_midrank_midbeta_softctx_megasuperextrasteps_ep4_ultralolr",
        "R863 α=128 SoftCtx MidRank Midβ → MegaAlpha α=256 isolate; ≠ HiAlpha128 R863 / ≠ Online / ≠ GRPO",
    ),
]


def main() -> None:
    for (
        rid,
        suffix,
        gpus,
        host_tag,
        lr,
        lora_r,
        lora_alpha,
        beta,
        max_steps,
        max_len,
        epochs,
        axis,
        signal,
    ) in AXES:
        dirname = f"r{rid}-marsplan-offline-dpo-hialpha-{suffix}"
        d = ROOT / "experiments" / dirname
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC_JSONL, d / "dpo_duel_reason.jsonl")
        shutil.copy2(SRC_TRAIN, d / "train_dpo.py")
        g0, g1 = gpus.split(",")
        merge_name = f"lean_merge_r{rid}_gpus{g0}{g1}_{STAMP}.sh"

        start = f"""#!/usr/bin/env bash
# R{rid}: marsplan Soft Mid Mid Soft ({axis}) {STAMP} {host_tag} idle fill
set -euo pipefail
export PATH="/root/.local/bin:${{PATH}}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${{HF_HOME:-/root/hf}}
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{gpus}}}
export PYTHONPATH=/root/mining_src/affine_pkg:${{PYTHONPATH:-}}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${{SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}}
OUT=${{OUT:-/root/r{rid}}}
DATA=${{DATA:-$OUT/dpo_duel_reason.jsonl}}
BASE={BASE}
TRAIN_DIR=$OUT/train
LOG=${{LOG:-/root/logs/r{rid}_train.nohup}}
LR=${{R{rid}_LR:-{lr}}}
LORA_R=${{R{rid}_LORA_R:-{lora_r}}}
LORA_ALPHA=${{R{rid}_LORA_ALPHA:-{lora_alpha}}}
BETA=${{R{rid}_BETA:-{beta}}}
MAX_STEPS=${{R{rid}_MAX_STEPS:-{max_steps}}}
MAX_LEN=${{R{rid}_MAX_LEN:-{max_len}}}
EPOCHS=${{R{rid}_EPOCHS:-{epochs}}}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *marsplan0624*|*5gedzafcvg*) ;; *) echo "FATAL bad BASE"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r{rid}] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R α=$LORA_ALPHA β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \\
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \\
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \\
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \\
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r{rid}_train.pid >"$OUT/train.pid"
python3 -c "
import json, time
from pathlib import Path
meta = {{
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'axis': '{axis}',
  'base': '$BASE', 'data': '$DATA', 'examples': $n,
  'lr': '$LR', 'lora_r': $LORA_R, 'lora_alpha': $LORA_ALPHA, 'beta': $BETA,
  'max_steps': $MAX_STEPS, 'max_len': $MAX_LEN, 'epochs': $EPOCHS,
  'gpus': '$CUDA_VISIBLE_DEVICES',
  'pid': int(Path('/root/logs/r{rid}_train.pid').read_text().strip()),
  'parent_signal': {signal!r},
  'decision_rule': 'Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign36 vera (v4 k=3 tau=0.03)',
  'n80_path': 'host-relay → lunar TKC after merge',
}}
Path('$OUT/train_meta.json').write_text(json.dumps(meta, indent=2)+'\\n')
Path('/root/affine_data/r{rid}_train_launched.json').write_text(json.dumps(meta, indent=2)+'\\n')
print(json.dumps(meta, indent=2))
"
echo "[r{rid}] TRAIN_ARMED pid=$(cat /root/logs/r{rid}_train.pid)"
"""
        (d / f"start_r{rid}.sh").write_text(start)

        wait = f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[{STAMP}-r{rid}-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }}
TRAIN_PID_FILE=/root/logs/r{rid}_train.pid
ADAPTER=/root/r{rid}/train/adapter
MERGE_SCRIPT=/root/mining_src/{dirname}/{merge_name}
LAUNCHED=/root/logs/r{rid}_merge_launched.{STAMP}
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && {{ log "already launched"; exit 0; }}
log "armed wait r{rid} → merge GPUs {gpus}"
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
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r{rid}_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/{STAMP}_r{rid}_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/{STAMP}_r{rid}_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/{STAMP}_r{rid}_merge.outer.pid)"
    exit 0
  fi
  step=$(grep -o '"step": [0-9]*' /root/logs/r{rid}_train.nohup 2>/dev/null | tail -1 | awk '{{print $2}}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${{step:-?}}"
  sleep 30
done
"""
        (d / f"wait_r{rid}_train_then_merge_{STAMP}.sh").write_text(wait)

        merge = f"""#!/usr/bin/env bash
# {STAMP}: r{rid} TRAIN_DONE → LoRA merge on GPUs {gpus}
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${{HF_HOME:-/root/hf}} PYTHONPATH=/root/mining_src/affine_pkg:${{PYTHONPATH:-}}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
BASE={BASE}
ADAPTER=/root/r{rid}/train/adapter
MERGE_DIR=/tmp/r{rid}_merged
GPUS=${{GPUS:-{gpus}}}
export CUDA_VISIBLE_DEVICES={gpus}
LOG=/root/logs/{STAMP}_r{rid}_merge.log
mkdir -p /root/logs /root/affine_data /root/r{rid}/train
: >"$LOG"
log() {{ echo "[{STAMP}-r{rid}] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }}
hub_ok() {{ local n; n=$(ls "$1"/model-*-of-*.safetensors 2>/dev/null | wc -l || true); [[ -f "$1/config.json" && "${{n:-0}}" -ge 16 ]]; }}
log "START merge GPUs=$GPUS"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r{rid}/train.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r{rid}/train/train.done
rm -rf "$MERGE_DIR"
/root/venv/bin/python3 /root/mining_src/s4-h1-sft/merge_lora.py \\
  --base "$BASE" --adapter "$ADAPTER" --out "$MERGE_DIR" \\
  --device-map auto --max-shard-size 5GB | tee -a "$LOG"
hub_ok "$MERGE_DIR"
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
du -sh "$MERGE_DIR" | tee -a "$LOG"
log "MERGE_DONE shards=$n — stamp for host-relay→lunar n80"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r{rid}_merge.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r{rid}_scp_needed.{STAMP}
"""
        (d / merge_name).write_text(merge)
        for name in (f"start_r{rid}.sh", f"wait_r{rid}_train_then_merge_{STAMP}.sh", merge_name):
            (d / name).chmod(0o755)
        print(f"OK {dirname}")


if __name__ == "__main__":
    main()
