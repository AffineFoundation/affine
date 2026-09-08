#!/usr/bin/env bash
# wvk 10 -> 11 T0 cutover: action dialects + trace-first corpus on data.affine.io.
#
# This script does NOT run on its own and must never be started by an agent.
# It executes only under an explicit, dated operator directive:
#
#     AFFINE_T0_DIRECTIVE=2026-09-DD ops/t0/t0_cutover.sh
#
# Order (each step is idempotent; rerun after fixing a failure):
#   0. preconditions   pod publishing traces, eval box idle, clean contract file
#   1. epoch 14        ops/corpus_build.py --init  -> data.affine.io/corpus/manifest.json
#   2. contract        affine.toml: wvk 11, allowed_action_kinds, manifest_key ->
#                      corpus/manifest.json (corpus_base_url already moved at stage 1,
#                      ops/t0/stage1_go_live.sh; ops/t0/wvk11_T0.patch is the preview),
#                      delete [fold_mix], drop the dashboard banner, llms.txt notice -> history
#   3. deploy          validator restart, eval pod redeploy between duels, daily fold under pm2
#   4. notices         Discord post (epoch announce is automatic; fork post text printed here)
#   5. Hippius         read-only by construction (no writer left); retire after 30 days
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
# Validator snapshot env, not .env alone: redeploy_pods.py writes the pod
# .eval_env from the process env and .env lacks HF_TOKEN / AFFINE_EVAL_TOKEN.
# shellcheck disable=SC1091
source ops/t0/validator_env.sh

T0="${AFFINE_T0_DIRECTIVE:-}"
if ! [[ "$T0" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "refusing: AFFINE_T0_DIRECTIVE=YYYY-MM-DD (the explicit dated operator directive) is required" >&2
  exit 2
fi
# Notice posted 2026-09-02 said "not before 2026-09-09". On 2026-09-03 the
# operator moved T0 to 2026-09-04 ("I need to ship this tomorrow"); the date
# change was re-noticed the same evening (Discord + llms.txt + release notes).
if [[ "$T0" < "2026-09-04" ]]; then
  echo "refusing: earliest noticed T0 is 2026-09-04; directive date $T0 is earlier" >&2
  exit 2
fi
step() { printf '\n==== [T0 %s] %s\n' "$T0" "$*"; }

# -- 0. preconditions ----------------------------------------------------------
step "0 preconditions"
python - <<'PY'
import json, sys, time, httpx
from datetime import datetime, timezone
m = httpx.get("https://data.affine.io/traces/manifest.json", timeout=60).json()
age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(m["published_at"])).total_seconds() / 3600
print(f"traces manifest: {m['n_chunks']} chunks / {m['n_rollouts']} rollouts, published {age_h:.1f}h ago")
if age_h > 6:
    sys.exit("datagen pod has not published traces in 6h -- check /root/logs/rollouts.log on the pod")
r = httpx.get("https://data.affine.io/corpus/manifest.json", timeout=60)
print("production corpus manifest:", "absent (fresh --init)" if r.status_code == 404 else f"present (epoch {r.json()['corpus_epoch']}, --init will be skipped)")
PY
if ! git diff --quiet -- affine/affine.toml; then
  echo "affine/affine.toml has uncommitted changes; commit or stash them first" >&2; exit 1
fi
grep -q '^weight_version_key = 10$' affine/affine.toml || { echo "contract is not at wvk 10; refusing" >&2; exit 1; }

# -- 1. epoch 14 on data.affine.io ---------------------------------------------
step "1 publish epoch 14 (schema 3, all three dialects, traces only)"
# --no-legacy (decided 2026-09-03 from the staging rehearsal): the v2 epochs
# 1-13 are NOT imported. Importing them made the mix waterfill count 60k
# legacy turns (coding 67%, python-heavy, GLM-era teacher) and admit zero new
# coding/terminal rollouts -- the multi-language / multi-harness data could
# not enter D until math and tool_use (both exhausted) caught up. From the
# traces alone D takes the [mix] targets on day one. The v2 history stays
# byte-identical at data.affine.io/turns/** and is chained via prev_manifest.
# Set AFFINE_T0_IMPORT_LEGACY=1 to keep the old behaviour.
# Normally pre-published the night before (2026-09-03/04: --init --no-legacy
# --no-announce, ~25 min) so this run is contract + deploy only; the epoch
# announce is then posted by the fold in step 2b. Nothing reads
# corpus/manifest.json until the toml manifest_key moves in step 2.
if curl -sf -o /dev/null https://data.affine.io/corpus/manifest.json; then
  echo "corpus manifest already published; skipping --init"
else
  LEGACY_FLAG="--no-legacy"; [[ "${AFFINE_T0_IMPORT_LEGACY:-}" == "1" ]] && LEGACY_FLAG=""
  python ops/corpus_build.py --init $LEGACY_FLAG --ignore-fold-mix --allowed-kinds bash,tool_call,boxed --no-announce
fi
python - <<'PY'
import httpx
m = httpx.get("https://data.affine.io/corpus/manifest.json", timeout=60).json()
assert m["schema_version"] == 3 and m["view_spec"] == "duel_turns@v4", m
assert set(m["allowed_action_kinds"]) == {"bash", "tool_call", "boxed"}, m["allowed_action_kinds"]
print(f"epoch {m['corpus_epoch']}: {m['index']['n_turns']:,} turns, {len(m['shards'])} shards")
PY

# -- 2. contract + notices in the tree -----------------------------------------
step "2 contract (wvk 11, dialects, schema-3 manifest_key), [fold_mix], banner, llms.txt"
grep -q '^corpus_base_url = "https://data.affine.io"$' affine/affine.toml \
  || { echo "corpus_base_url is not data.affine.io -- run ops/t0/stage1_go_live.sh first" >&2; exit 1; }
python ops/t0/t0_toml_edits.py --apply "$T0"
grep -q '^weight_version_key = 11$' affine/affine.toml
grep -q '^manifest_key = "corpus/manifest.json"$' affine/affine.toml
grep -q '^allowed_action_kinds = \["bash", "tool_call", "boxed"\]$' affine/affine.toml
python - "$T0" <<'PY'
import re, sys
from pathlib import Path
t0 = sys.argv[1]
# [fold_mix] held the fold on the bash-only mix during the notice period.
p = Path("rollouts/rollouts/sources.toml"); s = p.read_text()
s2 = re.sub(r"(#[^\n]*\n)*\[fold_mix\]\n(?:[a-z0-9_]+ = [^\n]*\n)*\n?", "", s, count=1)
assert "[fold_mix]" not in s2, "fold_mix block not removed"
p.write_text(s2); print("removed [fold_mix]")
# Dashboard banner: static HTML, remove the whole fork-notice div. Already
# gone since 2026-09-03 (operator took all banners down; notice lives in
# llms.txt + Discord) -- idempotent either way.
p = Path("affine/website/index.html"); s = p.read_text()
if 'id="fork-notice"' in s:
    s2 = re.sub(r'\s*<!-- Fork notice \(wvk 10 -> 11.*?<div class="fork-notice" id="fork-notice".*?</a>\s*</div>\s*</div>\n',
                "\n", s, count=1, flags=re.S)
    assert 'id="fork-notice"' not in s2, "banner not removed"
    p.write_text(s2); print("removed dashboard fork banner")
else:
    print("dashboard fork banner already absent")
# llms.txt: the notice becomes history.
p = Path("affine/scripts/build_llms_txt.py"); s = p.read_text()
s2 = s.replace("## Upcoming fork: wvk 11 — action dialects (notice posted 2026-09-02)",
               f"## Fork history: wvk 11 — action dialects + data.affine.io (notice 2026-09-02, effective {t0})")
s2 = s2.replace("- Upcoming fork: wvk 11 — action dialects (`tool_call`, `boxed` join \\",
                "- Fork history: wvk 11 — action dialects (`tool_call`, `boxed` join \\")
s2 = re.sub(r"\*\*When\.\*\* 2026-09-04, 18:00 UTC .*?\(`check_dialects` in `code/evalsrv/dueling\.py`\)\.",
            f"**Effective {t0}.** `weight_version_key = 11` in `affine.toml`; "
            "`[dataset].allowed_action_kinds = [\"bash\", \"tool_call\", \"boxed\"]`; "
            "D served from `https://data.affine.io/corpus/manifest.json` (schema_version 3).",
            s2, count=1, flags=re.S)
assert s2 != s, "llms.txt notice text not found"
p.write_text(s2); print("llms.txt: Upcoming fork -> Fork history")
PY
(cd affine && python scripts/build_llms_txt.py)
git add affine/affine.toml rollouts/rollouts/sources.toml affine/website affine/scripts/build_llms_txt.py
git commit -m "wvk 11 T0 ($T0): action dialects + schema-3 trace-first corpus D

Explicit dated operator directive $T0. Forward-only: reign stands,
min_submission_block unchanged. allowed_action_kinds admits tool_call and
boxed; manifest_key -> corpus/manifest.json on data.affine.io (schema 3 view
duel_turns@v4 over rollout traces; v2 history byte-identical under
data.affine.io/turns/**). Removes the notice banner and [fold_mix]."
echo "committed; push when the deploy below is green"

# -- 2b. fold under the new contract --------------------------------------------
step "2b fold: announce the pre-published epoch, fold traces since, all dialects"
# The toml now admits all three kinds and [fold_mix] is gone, so this is the
# first fold on the T0 contract: it posts the still-unannounced init epoch,
# then folds every trace chunk published since the pre-publish into the next
# epoch (skips below MIN_NEW_TURNS; the daily pm2 job continues from here).
python ops/corpus_build.py

# -- 3. deploy -----------------------------------------------------------------
step "3 deploy: validator, eval pod (between duels), daily fold"
pm2 restart affine-validator --update-env
python - <<'PY'
import json, os, time, httpx
# Wait for the eval pod to be idle so the redeploy does not kill a duel.
st = json.load(open("affine/state/state.json"))
host, port = st["eval_machine"]["ssh"].split("@")[1].split(" -p ")
import subprocess
for _ in range(120):
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-p", port.strip(), f"root@{host}",
                          "source /root/affine/.eval_env; curl -s -H \"X-Affine-Token: $AFFINE_EVAL_TOKEN\" localhost:9000/health"],
                         capture_output=True, text=True).stdout
    try:
        busy = json.loads(out).get("busy")
    except Exception:
        busy = None
    print("eval pod busy =", busy)
    if busy is False:
        break
    time.sleep(60)
else:
    raise SystemExit("eval pod stayed busy for 2h; rerun step 3 later")
PY
# Validator paused for the redeploy: its health loop would otherwise
# soft-restart the OLD bootstrap while the tar uploads (stage-1 race,
# 2026-09-03). A duel dispatched in the idle->stop gap is recovered from
# in_flight on start (requeued at front, uncounted).
pm2 stop affine-validator
(cd affine && python scripts/redeploy_pods.py)
pm2 start affine-validator --update-env
pm2 delete affine-corpus-refresh >/dev/null 2>&1 || true
pm2 start affine/scripts/ecosystem.config.js --only affine-corpus-refresh
pm2 save
python - <<'PY'
import json, subprocess, time
st = json.load(open("affine/state/state.json"))
host, port = st["eval_machine"]["ssh"].split("@")[1].split(" -p ")
for _ in range(60):
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-p", port.strip(), f"root@{host}",
                          "source /root/affine/.eval_env; curl -s -H \"X-Affine-Token: $AFFINE_EVAL_TOKEN\" localhost:9000/health"],
                         capture_output=True, text=True).stdout
    try:
        h = json.loads(out); c = h.get("corpus") or {}
    except Exception:
        h, c = {}, {}
    if h.get("ok") and c.get("schema_version") == 3 and c.get("ready"):
        print("eval pod synced schema 3:", {k: c[k] for k in ("corpus_epoch", "view_spec", "manifest_sha256")})
        break
    time.sleep(30)
else:
    raise SystemExit("eval pod did not come back on schema 3 within 30 min -- check bootstrap.log on the pod")
PY

# -- 4. notices ----------------------------------------------------------------
step "4 notices (epoch announce already posted by corpus_build; fork post below)"
cat <<EOF
Post to Discord (#announcements), then remove nothing else -- llms.txt + toml carry the record:

**wvk 10 → 11 is live ($T0).** Four changes, one fork, forward-only (reign stands, min_submission_block unchanged):
1. **Action dialects.** D now admits \`<tool_call>…</tool_call>\` (tool use / search) and \`\\boxed{…}\` (math) turns next to \`\`\`bash. Same min(R,G) score; each turn's system prompt states its format. A model that cannot emit a dialect forfeits those turns.
2. **Corrected prefixes.** Every turn's prefix is now the exact root→parent path of the message graph the model saw — harnesses that compact or rewrite history are represented faithfully (no phantom linear history).
3. **More harnesses on the same tasks.** Coding/terminal tasks are generated under mini-swe (bash fence), verifiers' native bash tool, and pi — three prompt styles over the same tasks so "can drive a shell" is what transfers.
4. **D becomes a schema-3 view over full rollout traces** on https://data.affine.io: \`corpus/manifest.json\` (view \`duel_turns@v4\`), \`traces/manifest.json\` (the full envelopes incl. reasoning_content). D is rebuilt from the traces alone — the pre-fork turns (epochs 1–13, python-heavy, old-teacher era) are **not** carried into the new slice population; they stay byte-identical at \`data.affine.io/turns/**\` for verdict replay. Mix at the published targets from day one (coding 0.50 / terminal 0.25 / math 0.10 / tool_use 0.10 / nl2repo 0.05, by turn count).
Spec + query recipes: https://affine.io/llms.txt (§ Fork history, § Turn corpus D).
EOF

# -- 5. Hippius ----------------------------------------------------------------
step "5 Hippius affine-sn120/turns/** is read-only by construction"
echo "No writer remains: ops/datagen_refresh.py is retired (pm2 entry now runs ops/corpus_build.py)"
echo "and the pod no longer stages turns. Retire the prefix after $(date -d "$T0 + 30 days" +%F) once"
echo "no verdict replay references s3.hippius.com (all slice.manifest_sha256 resolve on data.affine.io/turns/)."
echo
echo "T0 complete. git push when ready."
