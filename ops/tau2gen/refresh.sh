#!/bin/bash
# ops/tau2gen/refresh.sh <epoch> [airline_n retail_n telecom_n]
#
# Generate a fresh tau2-gen task set for fold epoch <epoch> (seed = epoch), run
# Alan's fidelity + leakage reports, export every domain into Affine's
# affine_tau2_v1 record format (scripts/export_affine.py, our PR to his repo)
# and package the result under
#   rollouts/envs/affine_tau2_gen_v1/affine_tau2_gen_v1/data/e<epoch>/
# ready for `ops/king-datagen/deploy_pods.sh` + a `data_epoch` bump in the
# `[source.affine_tau2_gen]` stanza. Nothing here touches the pods or D.
#
# Layout expected (box): ~/tau2gen/tau2-gen = clone of unarbos/tau2-gen (fork
# of catoneone/tau2-gen) on its `affine` branch = Alan's main + our three PRs
# (user-sim pass-through, export_affine, composition guard), with
# upstream/tau2-bench + its .venv + patches applied. TAU2GEN_ROOT overrides.
#
# Decontamination (fold worker, 2026-09-21 13:13 UTC): the persona changes how
# the customer talks, not what the agent repairs, so a task that shares a
# benchmark's (intent, fault composition) under another persona is an
# overlap (`bench_panel_overlap`, 6/66 telecom king rollouts at epoch 62).
# The telecom generator runs with --exclude-benchmark-compositions AND the
# exporter drops any survivor whose persona-blind key matches a `base` id.
#
# Reward bases: airline [DB, COMMUNICATE] with communicate_info required on
# every no-write task; retail [DB] (NL_ASSERTION dropped: no LLM judge);
# telecom as generated (ENV_ASSERTION, tau2's own basis).
set -euo pipefail
EPOCH=${1:?fold epoch (seed)}; N_AIR=${2:-400}; N_RET=${3:-200}; N_TEL=${4:-200}
ROOT=${TAU2GEN_ROOT:-$HOME/tau2gen/tau2-gen}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DST="$REPO/rollouts/envs/affine_tau2_gen_v1/affine_tau2_gen_v1/data/e$EPOCH"
P="$ROOT/upstream/tau2-bench/.venv/bin/python"
OUT="$ROOT/out/e$EPOCH"
cd "$ROOT"
[ -x "$P" ] || { echo "no tau2-bench venv at $P"; exit 1; }
[ -f scripts/export_affine.py ] && grep -q "exclude-benchmark-compositions" domains/telecom/gen.py || { echo "checkout lacks our PRs — git checkout affine (unarbos/tau2-gen integration branch)"; exit 1; }
echo "== generating epoch $EPOCH (seed $EPOCH): airline $N_AIR, retail $N_RET, telecom $N_TEL"
$P domains/airline/gen.py --n "$N_AIR" --seed "$EPOCH" --out "$OUT/airline" 2>&1 | grep -E "generated|leakage|wrote|rror" || true
$P domains/retail/gen.py  --n "$N_RET" --seed "$EPOCH" --out "$OUT/retail"  2>&1 | grep -E "generated|leakage|wrote|rror" || true
$P domains/telecom/gen.py --n "$N_TEL" --seed "$EPOCH" --exclude-benchmark-compositions --out "$OUT/telecom" 2>&1 | grep -E "generated|leakage|wrote|rror" || true
for d in airline retail telecom; do [ -f "$OUT/$d/tasks_tau2.json" ] || { echo "generation failed for $d"; exit 1; }; done
echo "== fidelity reports"
$P fidelity/report.py --gen-tasks "$OUT/airline/tasks.jsonl" --domain airline --bench-tasks --out "$OUT/airline/fidelity_report.md" >/dev/null 2>&1
$P fidelity/report.py --gen-tasks "$OUT/retail/tasks.jsonl"  --domain retail  --bench-tasks --out "$OUT/retail/fidelity_report.md"  >/dev/null 2>&1
$P fidelity/report.py --gen-tasks "$OUT/telecom/tasks.jsonl" --domain telecom --out "$OUT/telecom/fidelity_report.md" >/dev/null 2>&1
grep -h "Result:" "$OUT"/*/fidelity_report.md
grep -q "Result: \*\*PASS\*\*" "$OUT/airline/fidelity_report.md" && grep -q "Result: \*\*PASS\*\*" "$OUT/retail/fidelity_report.md" && grep -q "Result: \*\*PASS\*\*" "$OUT/telecom/fidelity_report.md" || { echo "leakage report not PASS on every domain"; exit 1; }
echo "== exports"
$P scripts/export_affine.py --tasks "$OUT/airline/tasks_tau2.json" --domain airline --manifest "$OUT/airline/manifest.json" --reward-basis DB,COMMUNICATE --require-communicate-info --drop-held-out-compositions --out "$OUT/airline/affine_tasks.json" 2>&1 | grep -E "wrote|refusing"
$P scripts/export_affine.py --tasks "$OUT/retail/tasks_tau2.json"  --domain retail  --manifest "$OUT/retail/manifest.json"  --reward-basis DB --drop-held-out-compositions --out "$OUT/retail/affine_tasks.json" 2>&1 | grep -E "wrote|refusing"
$P scripts/export_affine.py --tasks "$OUT/telecom/tasks_tau2.json" --domain telecom --manifest "$OUT/telecom/manifest.json" --drop-held-out-compositions --out "$OUT/telecom/affine_tasks.json" 2>&1 | grep -E "wrote|refusing"
echo "== packaging -> $DST"
mkdir -p "$DST"
for d in airline retail telecom; do
  gzip -9 -c "$OUT/$d/affine_tasks.json" > "$DST/$d.json.gz"
  gzip -9 -c "$OUT/$d/meta.jsonl" > "$DST/$d.meta.jsonl.gz"
  cp "$OUT/$d/manifest.json" "$DST/$d.manifest.json"
  cp "$OUT/$d/fidelity_report.md" "$DST/$d.fidelity_report.md"
done
# held-out ids for the fold's decontamination list: tau2 `base` split per domain
$P - "$DST/bench_task_ids.json" <<'PY'
import json, sys
from tau2.run import load_tasks
out = {}
for d in ("airline", "retail", "telecom"):
    ids = sorted(t.id for t in load_tasks(task_set_name=d, task_split_name="base"))
    out[d] = ids
out["note"] = "tau2-bench `base` split ids = the kingboard tau2-airline / tau2-retail / tau2-telecom cards; the generator excludes their (intent, composition, persona) triples and export_affine refuses any id collision"
json.dump(out, open(sys.argv[1], "w"), indent=0)
print({k: len(v) for k, v in out.items() if k != "note"})
PY
du -sh "$DST"; ls "$DST"
echo "done: set [source.affine_tau2_gen] extra_flags data-epoch to $EPOCH (or the taskset DEFAULT_EPOCH), then deploy_pods.sh --restart --all"
