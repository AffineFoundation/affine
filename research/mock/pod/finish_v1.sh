#!/usr/bin/env bash
# Corrected v1 crown completion: runs AFTER d_train_v1 finishes.
# Order matters: servers up -> KING ADAPTER RE-PINNED -> held pairs from the
# real king_r1 -> gate -> publish -> state surgery -> driver + bench.
set -uo pipefail
W=/dshare/koth
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
note() { echo "$(date -u +%FT%TZ) [eval] $*" >> "$W/status.log"; }

# 0. wait for training to finish
for i in $(seq 1 240); do
  grep -q D_TRAIN_DONE "$W/d_train_v1.log" 2>/dev/null && break
  sleep 15
done
grep -q D_TRAIN_DONE "$W/d_train_v1.log" || { note "finish_v1: train never finished"; exit 1; }
WALL_LINE=$(grep -oE "D_TRAIN_DONE.*" "$W/d_train_v1.log" | tail -1)
note "CROWN v1 D-train complete (DDPx4+grad-ckpt): $WALL_LINE"

# 1. sampling servers up + king adapter re-pinned
docker start vllm_8001 vllm_8003 >/dev/null 2>&1
for p in 8001 8003; do
  for i in $(seq 1 80); do
    curl -sf -m 5 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && break
    sleep 10
  done
done
curl -sf -m 600 -X POST http://127.0.0.1:8001/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name":"king_r1","lora_path":"/dshare/koth/kings/king_r1"}' >/dev/null \
  && note "king_r1 adapter re-pinned on :8001"

# 2. held gate pairs from the REAL king_r1
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
assert n >= 100, "held pair generation failed again"
PY
[ $? -ne 0 ] && { note "finish_v1: held-pair generation FAILED"; exit 1; }

# 3. gate on the v1 held set + publish (guard inside publish_v0.py)
cp "$W/held_pairs.jsonl" "$W/held_pairs_v0.jsonl" 2>/dev/null || true
cp "$W/held_pairs_v1.jsonl" "$W/held_pairs.jsonl"
$PY /root/work/publish_v0.py --ver v1 || { note "finish_v1: publish REFUSED/failed"; exit 1; }

# 4. reign state surgery + relaunch driver
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
pgrep -f "${F}[.]py" >/dev/null || {
  nohup $PY /root/work/${F}.py > "$W/eval_driver.log" 2>&1 < /dev/null &
  disown
}
sleep 15
pgrep -f "${F}[.]py" >/dev/null && note "eval driver relaunched: reign 1 under v1, king=king_r1"

# 5. benches for the crowned king (background, GPUs 6,7)
nohup bash /root/work/bench_king.sh king_r1 /dshare/koth/kings/king_r1 1 \
  > "$W/bench_r1.log" 2>&1 < /dev/null &
disown
note "bench launched for king_r1 (proxy 16 then full 150, GPUs 6,7)"
echo FINISH_V1_COMPLETE
