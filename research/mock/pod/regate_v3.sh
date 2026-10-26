#!/usr/bin/env bash
# Deploy logprob-exact judge scoring (variant-summed, soft mass gate) and
# re-gate judge v3 under it. Runs on the eval box.
set -uo pipefail
W=/dshare/koth
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
note() { echo "$(date -u +%FT%TZ) [eval] $*" >> "$W/status.log"; }

F=koth_eval
pkill -f "${F}[.]py"
sleep 3
note "eval driver PAUSED for SCORING-CHANGE deploy (direct A/B logit reading, variant-summed; answer-first HARD gate replaced by soft mass>=0.5 gate)"

# 1. v2 agreement check: legacy top-20 parse-read vs new exact variant-summed
$PY - <<'PY'
import sys, json, math, time, requests
sys.path.insert(0, "/root/work")
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from judge_common import fit_ids
from trackM_common import Judge

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
held = [json.loads(x) for x in open("/dshare/koth/held_pairs.jsonl")][:50]
URL = "http://127.0.0.1:8000"

def legacy_one(ids):
    """The pre-fix top-20 read: unconstrained call, sum A/B in top logprobs,
    0.5 when neither appears (the 'parse'-style path being retired)."""
    r = requests.post(f"{URL}/v1/completions", json={
        "model": "d_v2", "prompt": ids, "max_tokens": 1,
        "temperature": 0.0, "logprobs": 20}, timeout=300)
    r.raise_for_status()
    lp = r.json()["choices"][0].get("logprobs") or {}
    tops = (lp.get("top_logprobs") or [{}])[0] or {}
    pa = pb = 0.0
    for t, v in tops.items():
        s = t.strip()
        if s == "A": pa += math.exp(v)
        elif s == "B": pb += math.exp(v)
    if pa + pb <= 0:
        return None
    return pa / (pa + pb)

def legacy_pair(p):
    a = legacy_one(fit_ids(tok, p["prefix_text"], p["ref"], p["mine"], 3584))
    b = legacy_one(fit_ids(tok, p["prefix_text"], p["mine"], p["ref"], 3584))
    if a is None or b is None:
        return None
    return 0.5 * (a + (1.0 - b))

with ThreadPoolExecutor(max_workers=16) as ex:
    leg = list(ex.map(legacy_pair, held))
j = Judge(URL, "d_v2", tok, 3584)
new = j.score_pairs(held, workers=16)

pairs = [(l, n["p_teacher"], n["mass"]) for l, n in zip(leg, new)
         if l is not None and n]
diffs = [abs(l - p) for l, p, _ in pairs]
agree = sum(1 for l, p, _ in pairs if (l > 0.5) == (p > 0.5)) / len(pairs)
mass = sum(m for _, _, m in pairs) / len(pairs)
line = (f"SCORING CHANGE: judge scoring is now DIRECT LOGIT READING of A/B "
        f"token variants (summed, incl leading-space forms), score="
        f"P(A)/(P(A)+P(B)), both orders averaged; answer-first hard gate "
        f"retired in favor of soft mass gate (refuse < 0.5). v2 parse-vs-"
        f"logit agreement on {len(pairs)} held pairs: mean|dp|="
        f"{sum(diffs)/len(diffs):.4f} max|dp|={max(diffs):.4f} "
        f"verdict_agree={agree:.3f} mean_ab_mass={mass:.3f} "
        f"legacy_misses={sum(1 for l in leg if l is None)}")
with open("/dshare/koth/status.log", "a") as fh:
    fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
             + " [eval] " + line + "\n")
print(line)
PY
[ $? -ne 0 ] && { note "SCORING CHANGE agreement check FAILED; aborting deploy"; exit 1; }

# 2. fresh held pairs from king_r3 for the v3 gate
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
cands, vr, _ = sample_model("http://127.0.0.1:8001", "king_r3", m_tok,
                            turns, held_tids, 1, 0.9, 1792)
n = 0
with open("/dshare/koth/held_pairs_v3.jsonl", "w") as fh:
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
print(f"held_pairs_v3={n} valid_rate={vr:.3f}")
assert n >= 100
PY
[ $? -ne 0 ] && { note "v3 re-gate: held-pair generation FAILED"; exit 1; }

# 3. re-gate v3 under the new scoring (publish_v0.py has effect guard +
# held band + soft mass gate)
cp "$W/held_pairs.jsonl" "$W/held_pairs_prev.jsonl" 2>/dev/null || true
cp "$W/held_pairs_v3.jsonl" "$W/held_pairs.jsonl"
if $PY /root/work/publish_v0.py --ver v3; then
  note "v3 re-gate PASSED under logit scoring -- reign 3 begins under v3"
  $PY - <<'PY'
import json, time
p = "/dshare/koth/state.json"
s = json.load(open(p))
s.update({"reign": 3, "dver": "v3", "reign_turns": 0,
          "reign_t0": time.time(),
          "ratchet": {"n": 0, "f": 0.0, "logged": False},
          "agree": {"n": 0, "ex": 0.0, "tok": 0.0}})
# active duel was measured under v2 -- reset stats, keep the challenger
if s.get("chal"):
    s["duel"] = {"n": 0, "kf": 0.0, "cf": 0.0, "sd": 0.0, "sd2": 0.0, "ck": 0}
json.dump(s, open(p, "w"))
print("state: reign=3 dver=v3 king=", s["king_name"],
      "chal kept:", (s.get("chal") or {}).get("sub"))
PY
  note "active duel stats RESET (v2-measured fools not comparable under v3); challenger kept for a fresh duel"
else
  note "v3 re-gate REFUSED under logit scoring too -- keeping v2; next crown retrains with hardened recipe"
fi

# 4. relaunch driver
nohup $PY /root/work/${F}.py > "$W/eval_driver.log" 2>&1 < /dev/null &
disown
sleep 15
pgrep -f "${F}[.]py" >/dev/null && note "eval driver RESUMED (logit scoring live)"
echo REGATE_DONE
