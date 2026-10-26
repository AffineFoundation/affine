#!/usr/bin/env bash
# One-shot recovery: finish the interrupted v1 crown (D retrain OOM'd
# without grad-ckpt). King_r1 is already installed; this retrains D from the
# archive, gates, publishes v1, fixes reign state, relaunches the driver.
set -uo pipefail
W=/dshare/koth
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
note() { echo "$(date -u +%FT%TZ) [eval] $*" >> "$W/status.log"; }

pkill -f "koth_eva[l].py"; sleep 3

# 1. held gate pairs from the NEW king (server is up)
$PY - <<'PY'
import sys, json
sys.path.insert(0, "/root/work")
from transformers import AutoTokenizer
from gad_common import both_text, load_rollout_cache, load_turns
from trackM_common import render_prefix, sample_model

turns = load_turns("/root/work/turns.jsonl.gz")
t_held = load_rollout_cache("/dshare/gad/teacher_heldout.jsonl")
held_tids = sorted(t for t in t_held if t in turns)
m_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")
cands, vr, _ = sample_model("http://127.0.0.1:8001", "king_r1", m_tok,
                            turns, held_tids, 1, 0.9, 1792)
n = 0
with open("/dshare/koth/held_pairs_v1.jsonl", "w") as fh:
    for tid in held_tids:
        cs = cands.get(tid)
        if not cs or not cs[0]["valid"]:
            continue
        r = t_held[tid][0]
        fh.write(json.dumps({
            "turn_id": tid,
            "prefix_text": render_prefix(turns[tid], 3584 * 6),
            "ref": both_text(r["z"], r["y"]),
            "mine": both_text(cs[0]["z"], cs[0]["y"])}) + "\n")
        n += 1
print(f"held_pairs_v1={n} valid_rate={vr:.3f}", flush=True)
PY

# 2. DDP retrain with grad-ckpt on GPUs 2,3,4,5 (sampling servers paused)
docker stop vllm_8001 vllm_8003 >/dev/null 2>&1
note "CROWN v1 (recovery): D retrain restarted with --grad-ckpt (qwen3_5 gated-delta OOMs without it); EVALS PAUSED"
T0=$(date +%s)
rm -rf "$W/d_versions/v1"
CUDA_VISIBLE_DEVICES=2,3,4,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/trainenv/bin/torchrun --nproc-per-node=4 /root/work/d_train.py \
  --base Qwen/Qwen3.8-27B --pairs "$W/d_pairs_v1.jsonl" \
  --lora-out "$W/d_versions/v1" --grad-ckpt \
  --max-steps 150 --batch 2 --accum 1 --lr 1e-5 --max-len 3584 --seed 1 \
  > "$W/d_train_v1.log" 2>&1
RC=$?
WALL=$(( $(date +%s) - T0 ))
docker start vllm_8001 vllm_8003 >/dev/null 2>&1
if [ $RC -ne 0 ]; then
  note "CROWN v1 recovery D-train FAILED again rc=$RC wall=${WALL}s -- keeping v0f"
  exit 1
fi
note "CROWN v1 D-train done: wall=${WALL}s (DDPx4 + grad-ckpt)"

# 3. wait for sampling servers, re-pin king adapter
for p in 8001 8003; do
  for i in $(seq 1 80); do
    curl -sf -m 5 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && break
    sleep 10
  done
done
curl -sf -m 600 -X POST http://127.0.0.1:8001/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name":"king_r1","lora_path":"/dshare/koth/kings/king_r1"}' >/dev/null

# 4. gate (effect guard + held band) + publish v1
cp "$W/held_pairs_v1.jsonl" "$W/held_pairs.jsonl.v1bak" 2>/dev/null
$PY - <<'PY'
import sys, json, shutil
sys.path.insert(0, "/root/work")
shutil.copy("/dshare/koth/held_pairs.jsonl", "/dshare/koth/held_pairs_v0.jsonl")
shutil.copy("/dshare/koth/held_pairs_v1.jsonl", "/dshare/koth/held_pairs.jsonl")
PY
$PY /root/work/publish_v0.py --ver v1 || exit 1

# 5. reign state surgery + relaunch driver
$PY - <<'PY'
import json, time
p = "/dshare/koth/state.json"
s = json.load(open(p))
s.update({"reign": 1, "dver": "v1", "reign_turns": 0,
          "reign_t0": time.time(),
          "ratchet": {"n": 0, "f": 0.0, "logged": False},
          "agree": {"n": 0, "ex": 0.0, "tok": 0.0}})
json.dump(s, open(p, "w"))
print("state: reign=1 dver=v1 king=", s["king_name"])
PY
F=koth_eval
nohup $PY /root/work/${F}.py > "$W/eval_driver.log" 2>&1 < /dev/null &
disown
sleep 15
pgrep -f "${F}[.]py" >/dev/null && note "eval driver relaunched under v1, reign 1"

# 6. benches for the crowned king (background)
nohup bash /root/work/bench_king.sh king_r1 /dshare/koth/kings/king_r1 1 \
  > "$W/bench_r1.log" 2>&1 < /dev/null &
disown
note "bench launched for king_r1 (GPUs 6,7, background)"
echo CROWN_V1_COMPLETE
