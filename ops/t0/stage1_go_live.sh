#!/usr/bin/env bash
# Stage 1 of the September release: private R2 mining go-live + corpus D
# served from data.affine.io. NO scoring change, NO weight_version_key bump
# (same schema-2 manifest, same sha; admission mechanics only). Stage 2 is
# the wvk 11 fork on 2026-09-09 (ops/t0/t0_cutover.sh).
#
# This script does NOT run on its own and must never be started by an agent.
# It executes only under an explicit, dated operator directive:
#
#     AFFINE_STAGE1_DIRECTIVE=2026-09-03 ops/t0/stage1_go_live.sh
#
# Order (each step is idempotent; rerun after fixing a failure):
#   0. preconditions   snapshot secrets, data.affine.io serves the live sha,
#                      datagen publishing, pods reachable, finney tip
#   1. tree            affine.toml ([submission.r2].enabled=true,
#                      hf_cutover_block=<tip>, corpus_base_url=data.affine.io),
#                      website/app.js MANIFEST_URL, llms.txt notice text,
#                      rebuild llms.txt, secret-scan, commit
#   2. pods FIRST      redeploy eval/bench/chat between duels so .eval_env
#                      carries AFFINE_EVAL_R2_* before any r2:// challenger
#                      can be enqueued; verify /health corpus sha unchanged
#   3. validator       pm2 restart affine-validator + affine-dash; verify
#                      contract.json shows submission_r2 + data.affine.io
#   4. notices         Discord post text + the two pending operator replies
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
# Validator snapshot env (HF_TOKEN, AFFINE_EVAL_TOKEN, R2/CF keys) — see the
# header of validator_env.sh for why `.env` alone is not enough here.
# shellcheck disable=SC1091
source ops/t0/validator_env.sh

D="${AFFINE_STAGE1_DIRECTIVE:-}"
if ! [[ "$D" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "refusing: AFFINE_STAGE1_DIRECTIVE=YYYY-MM-DD (the explicit dated operator directive) is required" >&2
  exit 2
fi
if [[ "$D" < "2026-09-03" ]]; then
  echo "refusing: go-live was announced for 2026-09-03; directive date $D is earlier" >&2
  exit 2
fi
step() { printf '\n==== [stage1 %s] %s\n' "$D" "$*"; }
EVAL_HEALTH() { curl -sf -H "X-Affine-Token: $AFFINE_EVAL_TOKEN" http://127.0.0.1:9000/health; }

# -- 0. preconditions ----------------------------------------------------------
step "0 preconditions"
for k in HF_TOKEN AFFINE_EVAL_TOKEN CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_TOKEN \
         R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ENDPOINT \
         AFFINE_EVAL_R2_ACCESS_KEY_ID AFFINE_EVAL_R2_SECRET_ACCESS_KEY \
         AFFINE_MAILBOX_SIGNING_SEED; do
  [[ -n "${!k:-}" ]] || { echo "missing secret in validator snapshot: $k" >&2; exit 1; }
done
echo "secrets: ok"
grep -q '^enabled = false$' affine/affine.toml || echo "note: [submission.r2].enabled already flipped (rerun?)"
grep -q '^weight_version_key = 10$' affine/affine.toml || { echo "contract is not at wvk 10; refusing" >&2; exit 1; }
grep -q '^manifest_key = "turns/manifest.json"$' affine/affine.toml || { echo "manifest_key is not turns/manifest.json; stage 1 keeps the schema-2 manifest" >&2; exit 1; }
TIP="$(python - <<'PY'
import bittensor as bt
print(int(bt.Subtensor(network="finney").block))
PY
)"
echo "finney tip: $TIP"
python - <<'PY'
import hashlib, json, sys, tempfile, httpx
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "affine")
from evalsrv.corpus import CorpusSync
a = httpx.get("https://data.affine.io/turns/manifest.json", timeout=60).content
b = httpx.get("https://s3.hippius.com/affine-sn120/turns/manifest.json", timeout=60).content
if a != b:
    sys.exit("data.affine.io/turns/manifest.json differs from the Hippius pointer -- re-mirror before flipping")
sha = hashlib.sha256(a).hexdigest()
h = httpx.get("http://127.0.0.1:9000/health", timeout=20,
              headers={"X-Affine-Token": __import__("os").environ["AFFINE_EVAL_TOKEN"]}).json()
live = (h.get("corpus") or {}).get("manifest_sha256")
if live != sha:
    sys.exit(f"eval pod is on manifest {live} but data.affine.io serves {sha}")
cs = CorpusSync("https://data.affine.io", "turns/manifest.json", Path(tempfile.mkdtemp()), lazy_chunks=True)
if not cs.refresh() or cs.stale:
    sys.exit("CorpusSync against data.affine.io failed verification")
print(f"corpus: data.affine.io == Hippius == eval pod, sha {sha[:16]}, epoch {cs.manifest['corpus_epoch']} (schema {cs.schema_version})")
m = httpx.get("https://data.affine.io/traces/manifest.json", timeout=60).json()
age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(m["published_at"])).total_seconds() / 3600
print(f"traces: {m['n_chunks']} chunks / {m['n_rollouts']} rollouts, published {age_h:.1f}h ago")
if age_h > 6:
    sys.exit("datagen pod has not published traces in 6h")
for url in ("https://models.affine.io/", "https://dash.affine.io/"):
    r = httpx.get(url, timeout=30)
    if r.status_code not in (200, 404):
        sys.exit(f"{url} -> HTTP {r.status_code}")
print("domains: models.affine.io / dash.affine.io answer")
PY

# -- 1. tree -------------------------------------------------------------------
step "1 tree: affine.toml, app.js, llms.txt notice, rebuild, commit"
python - "$D" "$TIP" <<'PY'
import re, sys
from pathlib import Path
d, tip = sys.argv[1], sys.argv[2]

p = Path("affine/affine.toml"); s = p.read_text()
if "\nenabled = false\n" in s:
    s = s.replace("[submission.r2]\nenabled = false\n",
                  f"[submission.r2]\n# {d}: LIVE (explicit dated operator directive). affine1 (HF) reveals\n"
                  f"# above hf_cutover_block are dropped at intake; queued HF entries below it\n"
                  f"# still duel.\nenabled = true\n", 1)
    s = re.sub(r"^hf_cutover_block = -1$", f"hf_cutover_block = {tip}", s, count=1, flags=re.M)
    assert f"hf_cutover_block = {tip}" in s, "hf_cutover_block not set"
old = 'corpus_base_url = "https://s3.hippius.com/affine-sn120"\nmanifest_key = "turns/manifest.json"\n'
if old in s:
    s = s.replace(old,
        f'# {d}: data LOCATION only (explicit dated operator directive). Same\n'
        f'# schema-2 manifest, same sha, turns/** a byte-identical copy of the\n'
        f'# Hippius history; Hippius affine-sn120/turns/** is read-only from this\n'
        f'# date. Not a scoring change -- no weight_version_key event. The\n'
        f'# schema-3 manifest_key (corpus/manifest.json) lands at the wvk 11 T0.\n'
        f'corpus_base_url = "https://data.affine.io"\n'
        f'manifest_key = "turns/manifest.json"\n', 1)
    s = s.replace("# At T0 corpus_base_url/manifest_key move to\n"
                  "# https://data.affine.io + corpus/manifest.json (schema_version 3); the\n"
                  "# Hippius keys stay resolvable under data.affine.io/turns/** for replay.",
                  "# At T0 manifest_key moves to corpus/manifest.json (schema_version 3); the\n"
                  "# Hippius keys stay resolvable under data.affine.io/turns/** for replay.", 1)
assert 'corpus_base_url = "https://data.affine.io"' in s and "enabled = true" in s
p.write_text(s); print("affine.toml: [submission.r2] live, corpus_base_url -> data.affine.io")

p = Path("affine/website/app.js"); s = p.read_text()
s2 = s.replace('const MANIFEST_URL = "https://s3.hippius.com/affine-sn120/turns/manifest.json";',
               'const MANIFEST_URL = "https://data.affine.io/turns/manifest.json";', 1)
p.write_text(s2); print("app.js: MANIFEST_URL ->", "data.affine.io" if s2 != s else "(already)")

p = Path("affine/scripts/build_llms_txt.py"); s = p.read_text()
s2 = s.replace(
    "**Bundled with the fork: D moves to `https://data.affine.io` as a view over \\\n"
    "full rollout traces (schema_version 3).** Today D is cut on the datagen pod \\",
    f"**Bundled with the fork: D becomes a view over full rollout traces \\\n"
    f"(schema_version 3).** D is already served from `https://data.affine.io` \\\n"
    f"(since {d}, same schema-2 manifest and sha as before -- a location move, \\\n"
    f"not a fork). Today D is cut on the datagen pod \\", 1)
if s2 != s:
    p.write_text(s2); print("llms.txt builder: fork notice reworded (location already moved)")
else:
    print("llms.txt builder: notice already reworded")
PY
(cd affine && python scripts/build_llms_txt.py)
grep -q "https://data.affine.io/turns/manifest.json" affine/website/llms.txt
grep -q "Status: LIVE" affine/website/llms.txt
! grep -q "s3.hippius.com/affine-sn120/turns" affine/website/llms.txt || { echo "llms.txt still links Hippius turns/" >&2; exit 1; }

git add AGENTS.md affine/.gitignore affine/ARCHITECTURE.md affine/affine.toml \
        affine/affine affine/evalsrv affine/scripts affine/website affine/datagen \
        ops/.gitignore ops/corpus_build.py ops/t0 rollouts
# Secret scan on what is about to be committed (fail closed): no live secret
# VALUE from the validator env and no token-shaped literal may be staged.
python - <<'PY'
import os, re, subprocess, sys
diff = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True).stdout
added = "\n".join(l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
hits = []
for k, v in os.environ.items():
    if re.search(r"TOKEN|SECRET|SEED|KEY|PASS", k) and len(v) >= 12 and v in added:
        hits.append(k)
for pat in (r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}"):
    if re.search(pat, added):
        hits.append(pat)
if hits:
    sys.exit(f"possible secret in staged diff: {hits} -- inspect before committing")
print("secret scan: clean")
PY
git commit -q -m "release stage 1 ($D): private R2 mining live; corpus D served from data.affine.io

Explicit dated operator directive $D. [submission.r2].enabled = true with
hf_cutover_block = $TIP (affine1 HF reveals above it are dropped at intake).
[dataset].corpus_base_url -> https://data.affine.io: same schema-2 manifest,
same sha, byte-identical turns/**; Hippius turns/** read-only from today.
Admission + data location only: no scoring change, weight_version_key stays 10.
Ships the trace-first corpus code (schema-3 capable evalsrv, view/trace
modules, ops/corpus_build.py) ahead of the wvk 11 T0 on 2026-09-09." \
  && echo "committed $(git rev-parse --short HEAD); push when the deploy below is green"

# -- 2. pods FIRST --------------------------------------------------------------
step "2 pods: redeploy eval (between duels), bench, chat with AFFINE_EVAL_R2_*"
for _ in $(seq 1 180); do
  busy="$(EVAL_HEALTH | python -c 'import json,sys; print(json.load(sys.stdin).get("busy"))' 2>/dev/null || echo unknown)"
  echo "eval pod busy = $busy"
  [[ "$busy" == "False" ]] && break
  sleep 60
done
[[ "$busy" == "False" ]] || { echo "eval pod stayed busy for 3h; rerun from step 2 later" >&2; exit 1; }
(cd affine && python scripts/redeploy_pods.py --all)
for _ in $(seq 1 60); do
  if h="$(EVAL_HEALTH 2>/dev/null)" && python - "$h" <<'PY'
import json, sys
h = json.loads(sys.argv[1]); c = h.get("corpus") or {}
ok = h.get("ok") and c.get("ready") and not c.get("stale") \
     and c.get("corpus_base_url", "https://data.affine.io") == "https://data.affine.io"
print("eval /health:", {k: c.get(k) for k in ("manifest_sha256", "corpus_epoch", "schema_version", "corpus_base_url", "stale")})
sys.exit(0 if ok else 1)
PY
  then break; fi
  sleep 30
done
python - <<'PY'
import json, re, subprocess, sys
sys.path.insert(0, "affine")
from affine.provisioner import _ssh_run
st = json.load(open("affine/state/state.json"))
for role in ("eval", "bench", "chat"):
    ssh = (st.get(f"{role}_machine") or {}).get("ssh")
    if not ssh:
        print(f"{role}: no machine"); continue
    p = _ssh_run(ssh, "grep -c '^export AFFINE_EVAL_R2_' /root/affine/.eval_env; grep -c '^export HF_TOKEN=' /root/affine/.eval_env", timeout=60)
    r2n, hfn = (p.stdout or "0\n0").split()[:2]
    print(f"{role}: AFFINE_EVAL_R2_* lines={r2n} HF_TOKEN={hfn}")
    if role == "eval" and (int(r2n) < 3 or int(hfn) != 1):
        sys.exit("eval pod .eval_env is not complete -- do NOT restart the validator with enabled=true")
PY

# -- 3. validator + dash -------------------------------------------------------
step "3 validator: restart, verify AccessController + contract"
pm2 restart affine-validator --update-env
pm2 restart affine-dash --update-env
sleep 45
pm2 logs affine-validator --nostream --lines 80 | grep -Ei "AccessController|submission.r2|registrations|hf_cutover|Traceback|SystemExit" || true
python - "$TIP" <<'PY'
import json, sys, time, httpx
tip = int(sys.argv[1])
for _ in range(20):
    try:
        c = httpx.get("https://affine.io/api/v1/contract", timeout=20).json()
        s = httpx.get("https://affine.io/api/v1/snapshot", timeout=20).json()
    except Exception as e:
        print("dash not ready:", e); time.sleep(15); continue
    r2 = (s.get("submission_r2") or {})
    ds = (c.get("dataset") or {})
    print("contract dataset:", ds.get("corpus_base_url"), ds.get("manifest_key"))
    print("snapshot submission_r2:", {k: r2.get(k) for k in ("enabled", "validator_identity", "hf_cutover_block")})
    if r2.get("enabled") and int(r2.get("hf_cutover_block", -1)) == tip and ds.get("corpus_base_url") == "https://data.affine.io":
        print("validator + dash: LIVE"); break
    time.sleep(15)
else:
    sys.exit("contract/snapshot did not reflect stage 1 within 5 min -- check pm2 logs affine-validator")
PY

# -- 4. notices ----------------------------------------------------------------
step "4 notices"
cat <<EOF
Post to Discord (#announcements):

**Private R2 mining is live ($D).** Two changes, no scoring change, weight_version_key stays 10:
1. **Submissions are private.** Commit \`affine2|activate|<hotkey>|<sig>\`, fetch your sealed credentials from https://dash.affine.io/mailbox/..., upload with \`submit.py upload\`, commit \`affine2|ready|...\`. Nobody but the validator reads your bucket; only a crowned model is copied to https://models.affine.io. Hotkeys must be **Ed25519** (the mailbox is a NaCl sealed box; sr25519 keys cannot open it). HF \`affine1|…\` reveals above block $TIP are dropped at intake; already-queued HF entries still duel.
2. **Corpus D is served from https://data.affine.io** (same manifest, same sha, byte-identical turns/**; Hippius turns/** is read-only from today). Update your download scripts; nothing about the slice or the score changed.
Checklist + client: https://affine.io/llms.txt (§ Submit checklist) — \`submit.py hotkey / check / submit / status\`.
Reminder: the wvk 11 fork (tool_call + boxed dialects, schema-3 view of D) follows on 2026-09-09 as noticed.

Pending operator replies in the announcement thread:
- "why not sr25519?" -> bittensor.Keypair.encrypt/decrypt (the NaCl sealed box) is ed25519-only; it raises "encrypt/decrypt is only supported for ed25519 keypairs". The mailbox has to be openable by the hotkey and nobody else, so the hotkey must be Ed25519.
- "models should be private forever" -> losers are never published (private prefix expires after the retention window). A crowned model is copied to models.affine.io so the reign can be replayed and audited by anyone; making the king private too is a policy decision, not a code toggle -- say so and take it to the operator.
EOF
echo
echo "stage 1 complete. git push when ready. Watch 24h: first affine2 activation end to end, pods stale=false, fold cron stays stopped until T0."
