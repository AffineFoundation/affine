#!/bin/bash
# Full D_tau2 force-only chain on e11. Log: /root/logs/tau2f_chain.log
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root
export HF_HOME=/root/hf
mkdir -p /root/logs /root/hf /root/results
source /root/.env.hf

bash /root/scripts/pod_bootstrap_e9_tau2n.sh 2>&1 | tee /root/logs/bootstrap.log
# drop CI from download wait — bootstrap already has 8 kings; we serve 6

bash /root/scripts/pod_serve_tau2f.sh 2>&1 | tee /root/logs/serve.log

source /root/venv/bin/activate
cd /root
# Force-only: pin y to gold tool JSON; sample miner z only.
# Reuse native teacher ref cache when present.
python -m harness.runner \
  --turns /root/data/turns_tau2_native.jsonl \
  --miners king-genesis:8001 king-II:8002 king-XCIX:8003 king-VIII:8004 king-XL:8005 king-XLVI:8006 \
  --n-turns 80 \
  --force-y-from-ref \
  --out /root/results/ekings_tau2f.jsonl \
  --ref-cache /root/results/ref_tau2_native.jsonl \
  --concurrency 16 \
  2>&1 | tee /root/logs/runner_tau2f.log

python /root/scripts/analyze_tau2_prog.py \
  --pairs /root/results/ekings_tau2f.jsonl \
  --out /root/results/tau2_prog_force_table.txt \
  2>&1 | tee /root/logs/analyze_tau2f.log || true

# Fallback analyze if CLI flags not present
python - <<'PY'
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy import stats
sys.path.insert(0, "/root")
from harness.score import gate_pass, rank_term
from harness.config import KING_BENCH

TAU2 = json.loads(Path("/root/results/tau2_mean_all.json").read_text())
by = defaultdict(list)
for line in open("/root/results/ekings_tau2f.jsonl"):
    r = json.loads(line)
    if r.get("valid") and "pairs" in r:
        by[r["miner"]].extend(r["pairs"])
rows = []
for suf, swe in KING_BENCH.items():
    m = f"king-{suf}"
    ps = by.get(m)
    if not ps or suf not in TAU2:
        continue
    mix = st.mean(rank_term(p) for p in ps)
    gate = st.mean(1.0 if gate_pass(p) else 0.0 for p in ps)
    zlen = st.mean(len(p.get("z_a") or "") for p in ps)
    rows.append((suf, mix, gate, zlen, swe, TAU2[suf]))
rows.sort(key=lambda x: -x[5])
lines = ["D_tau2 FORCE-ONLY programmability probe\n"]
lines.append(f"{'king':8} {'S':>8} {'gate':>5} {'zlen':>6} {'swe':>6} {'tau2':>6}")
print(lines[-1])
for suf, mix, gate, zlen, swe, t2 in rows:
    line = f"{suf:8} {mix:+8.4f} {gate:4.0%} {zlen:6.0f} {swe:6.1f} {t2:6.3f}"
    print(line); lines.append(line)
if len(rows) >= 4:
    s, swe, t2 = [r[1] for r in rows], [r[4] for r in rows], [r[5] for r in rows]
    rs, ps = stats.spearmanr(s, swe)
    rt, pt = stats.spearmanr(s, t2)
    lines.append(f"\nSpearman S vs swe  = {rs:+.3f} (p={ps:.3g})")
    lines.append(f"Spearman S vs tau2 = {rt:+.3f} (p={pt:.3g})")
    print(lines[-2]); print(lines[-1])
    if rt > 0.5 and abs(rs) < 0.35:
        v = "SUPPORTED"
    elif rt < -0.5:
        v = "ANTI-isomorphism (force-only did not rescue)"
    else:
        v = "inconclusive"
    lines.append(f"VERDICT: {v}")
    print(lines[-1])
Path("/root/results/tau2_prog_force_table.txt").write_text("\n".join(lines) + "\n")
print("WROTE tau2_prog_force_table.txt")
PY

echo CHAIN_DONE
