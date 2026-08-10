"""Assemble website/llms.txt — miner index built from live sources.

llms.txt is a table of contents: prose contract + links to everything the
project publishes. The scoring/submission sources are NOT inlined; they are
copied verbatim into website/code/ at build time and linked, so the published
code is always byte-identical to what the validator runs.

Regenerate:
  cd ~/subnet120 && source .venv/bin/activate && source .env
  python affine/scripts/build_llms_txt.py

dashboard.push_website() calls this before uploading the static site (which
includes website/code/), so neither the index nor the code can drift.
"""

from __future__ import annotations

import shutil
import sys
import tomllib
from pathlib import Path

AFFINE_ROOT = Path(__file__).resolve().parents[1]
OUT = AFFINE_ROOT / "website" / "llms.txt"
CODE_DIR = AFFINE_ROOT / "website" / "code"


def _toml() -> dict:
    with open(AFFINE_ROOT / "affine.toml", "rb") as f:
        return tomllib.load(f)


def _site_base() -> str:
    """Hippius archive root (miners / cold public objects)."""
    h = _toml()["hippius"]
    return f"{h['endpoint'].rstrip('/')}/{h['bucket']}"


def _dash_base() -> str:
    """Hot-path dashboard API + interactive UI (affine-dash behind Caddy)."""
    d = _toml().get("dashboard") or {}
    return str(d.get("public_base_url") or "https://localhost:8443").rstrip("/")


def _serving_subs() -> dict[str, str]:
    """Serving-stack facts substituted into the prose at build time, so the
    published numbers can never drift from what the validator runs."""
    raw = _toml()
    ms = raw["miner_serving"]
    em = raw["eval_machine"]
    with open(AFFINE_ROOT / "pyproject.toml", "rb") as f:
        pins = tomllib.load(f)["project"]["optional-dependencies"]["eval"]
    vllm_pin = next(p for p in pins if p.startswith("vllm"))
    tf_pin = next(p for p in pins if p.startswith("transformers"))
    return {
        "{TP}": str(ms["tp"]),
        "{MAXLEN}": str(ms["max_model_len"]),
        "{GPUUTIL}": str(ms["gpu_memory_utilization"]),
        "{BATCHED}": str(ms["max_num_batched_tokens"]),
        "{GPUTYPES}": "/".join(em["gpu_types"]),
        "{VLLM_PIN}": vllm_pin,
        "{TF_PIN}": tf_pin,
    }

# Sources published under code/ and linked from llms.txt: (relative_path, description).
SOURCES: list[tuple[str, str]] = [
    ("affine.toml", "chain contract SSOT — every frozen knob the validator runs"),
    ("affine/score.py", "Reason v3: the score, the duel decision, and every "
     "telemetry helper — the scoring code"),
    ("scripts/submit.py", "standalone commit-reveal submission client — this single "
     "file is the whole submit path (trust it over any prose)"),
    ("affine/priors.py", "published prior bank behind the bank telemetry"),
    ("affine/chain.py", "reveal payload contract + commit builders"),
    ("evalsrv/dueling.py", "live duel: slice seeding, injectability probe, scoring loop"),
    ("evalsrv/corpus.py", "corpus sync: schema v1 flat shards or v2 parquet "
     "index + trajectory chunks"),
    ("affine/corpus/materialize.py", "schema v2: stratum key + turn "
     "materialization from trajectory records"),
    ("evalsrv/chat.py", "the chat contract: prompt assembly, thought-injection "
     "template, z/y rollout parsing — byte-exact"),
    ("evalsrv/terms.py", "per-turn instrumentation: teacher references + the ten "
     "forced-logprob calls behind every lp* component"),
    ("evalsrv/vllm_client.py", "vLLM sampling + echo/logprob forcing + per-byte "
     "normalization (lp_per_byte)"),
    ("evalsrv/engine.py", "slot lifecycle + the exact `vllm serve` invocation "
     "your checkpoint is loaded with (_vllm_cmd)"),
    ("pyproject.toml", "eval-pod dependency floors ([eval] extra) — vllm / "
     "transformers are installed fresh at pod provision"),
    ("affine/model_store.py", "checkpoint hygiene rules + weight-copy detection"),
    ("affine/state.py", "1-hotkey-1-eval policy, king lineage, queue invariants"),
]


HEADER = """\
# Affine (Bittensor SN120)

> King-of-the-hill subnet. Miners submit HF checkpoints; the validator crowns \
the reigning king by a single teacher-anchored distillation score — Reason — \
not an LLM judge. This file is the miner index: submit path, public contract, \
and links to the exact scoring code the network runs.

Machine-readable knobs (subset of the contract below) also ship as \
`data/contract.json` on this site. When in doubt, trust the linked sources \
under `code/` (republished from the validator's own tree on every site push) \
over any prose summary.

Full source repository: [github.com/AffineFoundation/affine]\
(https://github.com/AffineFoundation/affine) — validator, eval server, and \
this site. The `code/` mirror below is the load-bearing subset, republished \
on every site push.

---

## Table of contents

Everything the project publishes, in one index. All URLs are on this site's \
root ({BASE}/).

**In this file (read on)**

- How the game works — rules, duel flow, emissions
- Submit checklist — HF layout, repo naming, commit-reveal payload
- Serving stack — how your checkpoint is loaded; pre-flight before you burn \
the slot
- Reason — the one score you optimize (and the telemetry published around it)
- Public data — full field-level description of every published object
- Source of truth — links to the exact validator code under `code/`

**Code** (verbatim copies of the validator's own tree, republished on every \
site push — always current)

{CODE_LINKS}
**Validator logs** (separate file, not inlined here)

- [data/validator_log.txt]({BASE}/data/validator_log.txt) — redacted tail of \
the live control-plane log, refreshed ~every minute

**Eval results** (full duel records — the training data)

- [evals/index.jsonl]({BASE}/evals/index.jsonl) — append-only manifest, one \
line per duel
- `evals/{challenge_id}.json.gz` — everything computed during a duel: \
rollouts, teacher refs, every forced logprob

**Live dashboard API** (hot path on the validator box — prefer this for UI)

Root: {DASH}/

- [api/v1/snapshot]({DASH}/api/v1/snapshot) — king, reign, intake, duel \
queue, live eval
- [api/v1/history]({DASH}/api/v1/history) — filterable verdicts (`?q=&event=`)
- [api/v1/benchmarks]({DASH}/api/v1/benchmarks) — advisory benches
- [api/v1/contract]({DASH}/api/v1/contract) — machine-readable knobs
- `api/v1/duels/{challenge_id}` — duel detail (Reason, telemetry, rejection)
- `api/v1/duels/{challenge_id}/series` — chart-safe per-turn Reason/L1lift
- [api/v1/stream]({DASH}/api/v1/stream) — SSE snapshot deltas
- [index.html]({DASH}/) — interactive dashboard UI

**Hippius archive** (cold public mirror — miners / replay; same objects)

- [data/dashboard.json]({BASE}/data/dashboard.json) — king, reign chain, \
intake, duel queue, live eval progress
- [data/history.json]({BASE}/data/history.json) — last 100 verdicts/failures
- [data/benchmarks.json]({BASE}/data/benchmarks.json) — advisory tau2 scores \
(never part of the score)
- [data/contract.json]({BASE}/data/contract.json) — machine-readable \
contract knobs

**Complete audit logs** (gzipped JSONL, on Hippius)

- [data/history_full.jsonl.gz]({BASE}/data/history_full.jsonl.gz) — every \
verdict and failure since genesis
- [data/bench_history_full.jsonl.gz]({BASE}/data/bench_history_full.jsonl.gz) \
— every completed bench run

**Turn corpus D** (the prompts)

- [turns/manifest.json]({BASE}/turns/manifest.json) — current manifest: \
shards/chunks, index, hashes, corpus epoch, `schema_version`
- `turns/index/turns_*.parquet` — schema v2 turn index (sample here)
- `turns/chunks/*.jsonl.gz` — schema v2 trajectory objects
- `turns/shards/*.jsonl.gz` — schema v1 / compat flat per-turn JSONL
- `turns/manifests/{sha256}.json` — every manifest revision ever, immutable

**Website / index**

- [llms.txt]({BASE}/llms.txt) — this file (also served from {DASH}/llms.txt)

---

## How the game works

1. A frozen teacher C (`zai-org/GLM-4.5-Air-FP8`) and a public turn corpus D \
(sharded + manifest-pinned on this site under `turns/`) define the capability \
axis (SWE-style coding).
2. You commit-reveal an HF checkpoint pinned to a 40-hex git revision.
3. The validator burns your hotkey's **one eval slot at enqueue** (not at \
verdict). Failed hygiene, failed probe, or lost duel still burns the slot.
4. Eval machine runs a duel on an `n_turns` slice of D seeded by \
`blake2b(reveal_block_hash ‖ your_hotkey)` — you cannot know the slice before \
reveal; anyone can re-derive it after.
5. Both sides are scored with Reason (v3, 2026-08-10): \
`Reason = lpC(y_C|z_A) − lpC(y_C|∅)` per pair, miner score = mean. Challenger \
dethrones the king iff paired `mean(Reason_c − Reason_k) > k_sigma·SE` — a \
purely relative test, no gates, no absolute floors.
6. Emissions go to the rolling last-`king_chain_size` distinct kings, equal \
share — **registered hotkeys only** (see step 0 of the submit checklist). \
Advisory tau2 benches never affect Reason or crowning.

There is no validator-private data. Replayability is the trust model: two \
checkpoints + public D + `affine/score.py` → recompute the verdict.

---

## Submit checklist (do this)

**Step 0 — wallet + registration.** Everything below assumes a Bittensor \
wallet: `btcli wallet create` makes the coldkey + hotkey pair, and \
`btcli subnet register --netuid 120 --wallet YOUR_WALLET --hotkey YOUR_HOTKEY` \
registers the hotkey on this subnet (dynamic burn cost — check \
`btcli subnet burn-cost 120`, or `reg_cost_tao` in `api/v1/snapshot`). \
Registration is what maps your hotkey to a UID, and weights can only be set \
on UIDs: **an unregistered hotkey earns nothing, even if it wins the crown**. \
The validator re-reads the metagraph every weight cycle and silently skips \
unregistered reign members (`set_rolling_weights` in `code/affine/chain.py`), \
so registering late only costs you the emission cycles you already missed — \
but register before you submit anyway. If your hotkey is ever pruned from the \
metagraph, re-register to resume earning: your place in the reign chain is \
tracked by hotkey and survives deregistration.

1. Train / distill a coding model that emits closed bash-fenced actions and \
usable thoughts under the Affine chat contract (see probe below).
2. Push weights to Hugging Face as safetensors in canonical layout \
(`model.safetensors` **or** sharded `model-XXXXX-of-YYYYY.safetensors` + \
`model.safetensors.index.json`). No `*.py`. No `auto_map` in `config.json`. \
Safetensors ≤ 90 GB; whole repo ≤ 100 GB; ≤ 5000 files; `config.json` ≤ 1 MiB.
3. Repo id must match `^[^/]+/[Aa]ffine-.+$` **and** embed your identity: \
the first 5 AND last 5 chars (lowercase) of your coldkey **or** hotkey ss58 \
must both appear in the repo id — the compact token or the full ss58 both \
work. Example: `you/Affine-{token}-mymodel`.
4. Pin a 40-hex revision (never a moving branch tip).
5. Submit with the standalone client — one file, no package install beyond \
`pip install "bittensor>=11,<12" huggingface_hub` (the script uses the \
bittensor 11 SDK: `bt.timelock` + raw `Commitments.set_commitment`). The \
client **pre-flights every intake check the validator runs** (naming + \
identity, anonymous readability of the pinned revision, safetensors layout, \
no `*.py` / no `auto_map`, size caps) and refuses to send a submission that \
would burn your slot at intake. Add `--check` to validate and print the \
payload without submitting anything:

```bash
curl -O {BASE}/code/scripts/submit.py
python submit.py --repo you/Affine-{token}-mymodel \\
    --wallet YOUR_WALLET --hotkey YOUR_HOTKEY [--revision <40hex>] --check
# happy with the pre-flight output? drop --check to submit for real
python submit.py --repo you/Affine-{token}-mymodel \\
    --wallet YOUR_WALLET --hotkey YOUR_HOTKEY [--revision <40hex>]
```

(or clone [github.com/AffineFoundation/affine]\
(https://github.com/AffineFoundation/affine) and run `affine/scripts/submit.py`)
6. Payload committed on-chain:

```
affine1|<hf_repo>|<hf_revision_40hex>|<author_hotkey_ss58>
```

Live path uses bittensor 11 timelock encrypt (`reveal_in="60s"`) → \
`Commitments.set_commitment` — **trust `scripts/submit.py`** over any prose.

**Commit ≠ duel-queue row.** `LastCommitment` alone is encrypted and not a \
dashboard row. After ~60s the payload must land in `RevealedCommitments`; \
only then does the validator run **intake**. Intake may enqueue a duel slot, \
skip, or reject — see dashboard **intake** (reason) → **duel queue** (eval \
slots only) → **fails** / history. Lifetime `stats.queued` / \
`enqueued_total` is not "your commit is waiting."

7. Wait ~1 minute for reveal, then check the dashboard in order: **intake → \
duel queue → fails**. Do not expect a queue row from commit alone. Common \
intake outcomes:
   - `enqueued` — you have a duel-queue challenge id
   - `skipped_min_block` — reveal block ≤ `min_submission_block` (ignored)
   - `skipped_slot_burned` — this hotkey already burned its one eval slot
   - `skipped_king` / `skipped_repo_queued` — already crowned or same repo \
waiting
   - `rejected_*` — bad payload / revision already submitted / etc. (see fails)

**Hard policies**

- One submission per hotkey, ever. Slot burned at enqueue (prior enqueue ⇒ \
no new duel-queue row).
- A content revision that was ever submitted can never be resubmitted, by anyone.
- Weight-identical copy of the current king → reject, unless your HF commit \
timestamp is earlier than the king's → `crown_earlier` without a duel.
- Current king's hotkey is skipped (already crowned).
- Infra faults (dead eval pod, busy server, chain hiccup on block hash) \
requeue without burning a failure record; miner-attributable failures burn.

---

## Serving stack (will your checkpoint load?)

Your checkpoint is served with stock `vllm serve` — **never** \
`--trust-remote-code`. Combined with the hygiene rules (no `*.py`, no \
`auto_map`), only architectures natively supported by the pod's vLLM build \
can play. If vLLM cannot load or serve your model, the injectability probe \
rejects it — and your slot is already burned. Pre-flight before submitting.

- **Exact invocation**: `_vllm_cmd` in \
[code/evalsrv/engine.py]({BASE}/code/evalsrv/engine.py). Current knobs \
(`affine.toml [miner_serving]`, substituted here at site build time): \
`--tensor-parallel-size {TP}`, `--max-model-len {MAXLEN}`, \
`--gpu-memory-utilization {GPUUTIL}`, `--max-num-batched-tokens {BATCHED}`, \
FLASH_ATTN attention, triton MoE backend.
- **dtype** is vLLM `auto`: it follows `torch_dtype` in your `config.json`.
- **Versions float**: pods install the [eval] extra fresh at provision time \
(floors from `pyproject.toml`: `{VLLM_PIN}`, `{TF_PIN}`). The versions \
actually running right now are reported live at `api/v1/snapshot` under \
`eval_machine.versions` (vllm / transformers / torch) and in \
`data/contract.json` `serving` for the frozen knobs.
- **Hardware**: an 8-GPU {GPUTYPES} pod; the miner slot is {TP} GPUs, \
tensor-parallel {TP}. Your model (≤ 90 GB safetensors) must load and serve \
under exactly that.
- **Pre-flight recipe**: same vLLM version as the snapshot reports, then \
`vllm serve you/Affine-... --revision <sha> --max-model-len {MAXLEN} \
--tensor-parallel-size {TP}` and check it answers `/v1/completions` with \
finite logprobs on an echo request (see `score_action` in \
`code/evalsrv/vllm_client.py`).

---

## Reason (what you optimize)

Since 2026-08-10 (`weight_version_key = 3`) the whole scoring contract is:

```
Reason (per pair)  = lpC(y_C | z_A) − lpC(y_C | ∅)
Miner score        = mean(Reason) over all scored pairs
Crown              = paired mean(Reason_c − Reason_k) > k_sigma · SE
```

`y_C` is the frozen teacher's own reference action (resampled fresh every \
duel), `z_A` is your model's thought on the same turn, and both logprobs are \
teacher-forced on the teacher — **your model's own logprobs never enter the \
ranked quantity**. You win by producing thoughts that measurably raise the \
teacher's likelihood of its own action, i.e. by genuinely reasoning about the \
turn. Scoring hyperparameters: `n_turns = 80`, `k_sigma = 3.0`. There is no \
mix, no clip, no gates, and no absolute margin floor — the duel is purely \
relative to the slice's own noise (`SE = stdev/√n` over paired turns).

**Telemetry (measured, published, never scored).** Every verdict also \
records the quantities the retired S* v2 contract used to gate on, plus new \
length/timing metrics — study them, but none affects validity or score:

- causality/leakage pass rate (τ = 0.02 telemetry constant)
- prior-bank positivity fraction vs `affine/priors.py`
- calibration ratio `r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|` and the \
empty-baseline magnitude `mean|lpA(y_C|∅)|`
- raw mean L1lift `lpA(y_C|z_A) − lpA(y_C|∅)`
- per-side thought/action char lengths + deltas vs the teacher's own \
rollouts, and the duel's scoring wall clock (`duel_seconds`)

Pre-fork verdicts (before 2026-08-10) stamp the old `gates` block and the \
S* mix formula they were judged under; they remain replayable as recorded.

Before the full duel, an injectability probe rejects checkpoints that cannot \
emit a parsable bash action or return finite forced logprobs.

**Simulate before you submit.** Your eval slot is burned at enqueue, one per \
hotkey, ever — so replay the duel locally first. The complete measurement \
layer is published under `code/` and is import-closed (every module \
`dueling.py` touches is in the list above): `evalsrv/chat.py` is the chat \
contract — models are rendered through their own chat template to a string \
and driven via `/v1/completions`, injection plants thoughts as the canonical \
assistant body `</think>\\nTHOUGHT: {z}\\n\\n{y}`, and `split_rollout` \
defines exactly what counts as z (all reasoning text) and y (the last closed \
bash-fenced block). `evalsrv/terms.py` makes the ten forced-logprob calls behind \
every `lp*` component; `evalsrv/vllm_client.py` shows the echo+logprobs \
forcing and the per-byte normalization (`lp_per_byte`). Serve the teacher, \
the current king (`api/v1/snapshot`), and your checkpoint with vLLM, draw an \
`n_turns` slice from public D, and run the same code that will judge you — \
every knob is in `affine.toml` `[duel]`.

Frozen numeric knobs live in `affine.toml` `[duel]` (linked under `code/`). \
Changing score.py, priors, duel knobs, or the reveal format is a chain fork \
(`weight_version_key` bump). Corpus refreshes are data events, not forks: \
the manifest's `corpus_epoch` increments and every verdict records which \
manifest it was scored against.

---

## Public data (train on it)

Everything the validator scores is published — there is no validator-private \
data. All paths are relative to this site's root (Hippius S3 bucket \
`affine-sn120`); fetch them directly with curl or any HTTP client.

**Live dashboard API** (hot path — `{DASH}`):

- `GET /api/v1/snapshot` — king, reign chain, intake, duel queue, live eval.
- `GET /api/v1/history?limit=&cursor=&q=&event=` — filterable verdicts.
- `GET /api/v1/benchmarks` — advisory suite scores (never part of the score).
- `GET /api/v1/contract` — machine-readable contract knobs.
- `GET /api/v1/duels/{id}` — duel detail (z, margin, Reason, telemetry).
- `GET /api/v1/duels/{id}/series` — per-turn Reason / L1lift (no raw logprobs).
- `GET /api/v1/stream` — SSE snapshot deltas for live UIs.

**Hippius archive mirror** (cold path — this site's Hippius root, same disclosure):

- `data/dashboard.json` / `data/history.json` / `data/benchmarks.json` / \
`data/contract.json` — slim JSON also pushed for miners without the API.
- `data/validator_log.txt` — recent validator log tail (plain text, refreshed \
~every minute). Pod network coordinates are redacted; nothing else is.

**Complete audit logs** (gzipped JSONL, updated on every verdict):

- `data/history_full.jsonl.gz` — every verdict and failure since genesis, \
with full per-side Reason + telemetry summaries, slice seeds, block hashes, \
rejection reasons (pre-fork rows carry their original S* gate stats).
- `data/bench_history_full.jsonl.gz` — every completed bench run.

**Full duel records — the training data** (one immutable object per \
challenge, published right after the verdict):

- `evals/index.jsonl` — append-only manifest. One line per duel: \
`{key, bytes, at, challenge_id, repo, revision, hotkey, challenger_wins, z, \
margin, rejection_reason}`. Poll this to discover new records.
- `evals/{challenge_id}.json.gz` — gzipped JSON with everything computed \
during the duel:
  - `request` — king/challenger repos + revisions, hotkey, block hash.
  - `verdict` — same audit summary as history.
  - `slice` — seed, digest, n, block_hash, corpus_epoch, manifest_sha256. \
The manifest hash resolves at `turns/manifests/{hash}.json` forever, so you \
can re-derive the exact slice from public D even after shards are retired.
  - `turn_ids` — `{traj_id}:{turn_idx}` keys into the public corpus.
  - `teacher_refs` — the teacher's reference rollouts per turn: \
`{turn_id: [{z, y, lp_own, lp_empty}]}`. This is frontier-teacher \
distillation data for the exact turns that were scored.
  - `king_rows` / `challenger_rows` — per-turn instrumented records: \
`{turn_id, miner, valid, n_pairs, bank_frac, L2_bank, pairs: [...]}`. Each \
pair carries the miner rollout text (`z_a` thoughts, `y_a` action) plus every \
forced-logprob component Reason and the telemetry are computed from (`lpA_yc_za`, `lpC_yc_za`, \
`lpA_yc_zc`, `lpA_yc_e`, `lpA_ya_za`, `lpC_ya_za`, `lpA_ya_zc`, `lpA_ya_e`, \
`lpC_ya_e`, `lpC_ya_zc`, `lpC_yc_zc`, `lpC_yc_e`, `L2_bank`). You can \
recompute any verdict offline from this file + `affine/score.py`.

**Turn corpus D** (the prompts themselves) — on this site:

- `turns/manifest.json` — current manifest. schema_version **2** shape: \
`{corpus_epoch, schema_version, created_at, index: {key, sha256, n_turns}, \
shards: [{key, sha256, n_trajectories, format, active}], compat_shards?, \
prev_manifest}`. Poll it like `evals/index.jsonl`; a hash change means the \
corpus moved.
- `turns/index/turns_*.parquet` — one row per scorable turn. Download this \
first; filter/sample by `stratum` / `source` / `language` / `phase`, then \
fetch only the `chunk_key` objects you need.
- `turns/chunks/*.jsonl.gz` — trajectory records (`messages` once + \
`turns[{{turn_idx, msg_pos, …}}]`). `sha256` in the manifest is over the \
**uncompressed** jsonl. Materialize a turn as \
`prefix = messages[:msg_pos]`, \
`reference_turn = messages[msg_pos].content`.
- `turns/shards/*_compat.jsonl.gz` — optional flat per-turn JSONL listed \
under `compat_shards` for one cutover epoch (concat like v1). Not used by \
eval pods on schema v2.
- `turns/shards/turns_epoch_*.jsonl.gz` — legacy schema v1 per-turn shards. \
Still present (retired) so old `manifest_sha256` values remain replayable.
- `turns/manifests/{{sha256}}.json` — every manifest revision ever published, \
immutable. The `manifest_sha256` stamped in any verdict resolves here.

**How to query D (schema v2)** — do not download every chunk up front.

1. Fetch the manifest and the Parquet index named in `manifest["index"]["key"]`:

```bash
curl -sO {BASE}/turns/manifest.json
# example: turns/index/turns_0006.parquet
curl -sO {BASE}/$(python -c "import json; print(json.load(open('manifest.json'))['index']['key'])")
```

2. Filter / sample the index locally (DuckDB, pandas, polars — anything that \
reads Parquet). Index columns: `turn_id`, `traj_id`, `turn_idx`, `stratum`, \
`phase`, `source`, `language`, `chunk_key`, `traj_line`, `msg_pos`, \
`n_prefix_chars`.

```python
import duckdb
duckdb.sql('''
  SELECT turn_id, source, language, phase, chunk_key, traj_line, msg_pos
  FROM 'turns_0006.parquet'
  WHERE language = 'go' AND phase = 'late'
  LIMIT 20
''').show()
```

3. Download only the `chunk_key` objects you need. Each line is one \
trajectory. Materialize a scored turn as:

```python
import gzip, json, httpx
base = "{BASE}"
row = ...  # one index row
blob = httpx.get(f"{{base}}/{{row['chunk_key']}}").content
traj = [json.loads(l) for l in gzip.decompress(blob).splitlines() if l][
    row["traj_line"]]
meta = next(t for t in traj["turns"] if t["turn_idx"] == row["turn_idx"])
prefix = traj["messages"][: meta["msg_pos"]]          # ends on user / env out
reference_turn = traj["messages"][meta["msg_pos"]]["content"]  # THOUGHT+bash
```

Helper in the code mirror: \
[`code/affine/corpus/materialize.py`]({BASE}/code/affine/corpus/materialize.py) \
(`materialize_turn`). Eval pods sample the index with \
`blake2b(reveal_block_hash ‖ hotkey)` and materialize only the drawn slice.

**Cutover note:** epoch 6 also lists a flat `compat_shards` JSONL for old \
download scripts. Prefer the index+chunks path; compat goes away after the \
next epochs. Staging datagen (private HF) is still turn-flat — conversion to \
chunks+index happens when folds publish to Hippius.

Slices are seeded by the reveal-block hash, so future slices are \
unpredictable; past records tell you the distribution, not the next slice. \
The corpus is refreshed continuously — new chunks appear and old ones retire \
via manifest revisions (`corpus_epoch` increments each time; this is a data \
event, not a scoring fork), so keep your local copy synced to the manifest.

Suggested agent loop: poll `evals/index.jsonl` → fetch new \
`evals/*.json.gz` → train on `teacher_refs` (distillation) and on your own \
gate/logprob diagnostics from `pairs`.

---

## Source of truth (linked)

The files below are byte-identical copies of the validator's own tree, \
republished under `code/` every time the site is pushed — they can never be \
newer or older than the code that scores you. Fetch them with curl or any \
HTTP client. The set is import-closed over the scoring path: everything \
`dueling.py` calls (chat contract, forced-logprob instrumentation, vLLM \
client) is in the list, so a local pre-submit simulator needs nothing that \
is not linked here.

{CODE_LINKS}
This index and the `code/` copies are regenerated together on every validator \
website push.
"""


def _publish_code() -> list[str]:
    """Copy each source verbatim into website/code/<rel>; return link lines."""
    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    lines = []
    for rel, desc in SOURCES:
        src = AFFINE_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"required source missing: {src}")
        dst = CODE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        lines.append(f"- [code/{rel}]({{BASE}}/code/{rel}) — {desc}\n")
    return lines


def build() -> str:
    code_links = "".join(_publish_code())
    # Only {BASE}/{CODE_LINKS} are substituted; other braces ({challenge_id},
    # {token}, ...) are literal placeholders miners should read as-is.
    text = (HEADER.replace("{CODE_LINKS}", code_links)
                  .replace("{BASE}", _site_base())
                  .replace("{DASH}", _dash_base()))
    for token, value in _serving_subs().items():
        text = text.replace(token, value)
    # Fail closed if we somehow produced a broken index.
    if "## Table of contents" not in text or "data/validator_log.txt" not in text:
        raise RuntimeError("llms.txt build failed closed: missing table of contents")
    leftovers = ["{BASE}", "{DASH}", "{CODE_LINKS}", *_serving_subs()]
    if any(t in text for t in leftovers):
        raise RuntimeError("llms.txt build failed closed: unsubstituted placeholder")
    score_copy = CODE_DIR / "affine" / "score.py"
    if "def score_miner" not in score_copy.read_text(encoding="utf-8"):
        raise RuntimeError("llms.txt build failed closed: code/affine/score.py missing score_miner")
    if sum(1 for _ in CODE_DIR.rglob("*") if _.is_file()) != len(SOURCES):
        raise RuntimeError("llms.txt build failed closed: code/ file count mismatch")
    return text


def main() -> None:
    text = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    n_code = sum(1 for p in CODE_DIR.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({len(text):,} chars, {text.count(chr(10)):,} lines) "
          f"+ {n_code} sources under {CODE_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
