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
# Paths starting with "ops/" resolve against the repo root; everything else
# against affine/. All are published under code/<rel> either way.
SOURCES: list[tuple[str, str]] = [
    ("affine.toml", "chain contract SSOT — every frozen knob the validator runs"),
    ("affine/score.py", "Reason v4: the score, the duel decision, and every "
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
    ("ops/exploit-audit/prompt.md", "the exact task the post-crown exploit "
     "auditor runs — audit the auditor"),
    ("ops/exploit-audit/auditd.py", "the audit daemon: crown watcher, agent "
     "runner, verdict publisher, revert enforcement"),
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
- Post-crown exploit audit — the auditor, its published verdicts, and how to \
run the same audit yourself
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

**Bench results** (full advisory-bench rollout records — why a score happened)

- [benches/index.jsonl]({BASE}/benches/index.jsonl) — append-only manifest, \
one line per bench run
- `benches/{job_id}.json.gz` — per-task agent trajectories (full message \
transcripts), submitted patches, exit statuses, and harness resolution

**Live dashboard API** (hot path on the validator box — prefer this for UI)

Root: {DASH}/

- [api/v1/snapshot]({DASH}/api/v1/snapshot) — king, reign, intake, duel \
queue, live eval
- [api/v1/history]({DASH}/api/v1/history) — filterable verdicts (`?q=&event=`)
- [api/v1/benchmarks]({DASH}/api/v1/benchmarks) — advisory benches
- [api/v1/contract]({DASH}/api/v1/contract) — machine-readable knobs
- `api/v1/duels/{challenge_id}` — duel detail (Reason, telemetry, rejection)
- `api/v1/duels/{challenge_id}/series` — chart-safe per-turn Reason/L1lift
- [api/v1/benches]({DASH}/api/v1/benches) — manifest of bench rollout records
- `api/v1/benches/{job_id}` — one bench run: per-task patch/exit/resolution
- `api/v1/benches/{job_id}/trajectory?instance_id=` — full agent transcript
- [api/v1/dataset]({DASH}/api/v1/dataset) — corpus D stats (epoch, mix, \
length histogram)
- `api/v1/dataset/turns?source=&language=&phase=&repo=&q=&limit=&cursor=` — \
paginated turn index
- `api/v1/dataset/turn?turn_id=` — one turn's prompt prefix + reference action
- [api/v1/audits]({DASH}/api/v1/audits) — post-crown exploit-audit verdicts
- `api/v1/audits/{reign}` — one audit's verdict + pinned-input manifest + \
analysis prose
- [api/v1/stream]({DASH}/api/v1/stream) — SSE snapshot deltas
- [index.html]({DASH}/) — interactive dashboard UI (`#dataset` = corpus \
browser, `#audits` = exploit-audit verdicts)

**Hippius archive** (cold public mirror — miners / replay; same objects)

- [data/dashboard.json]({BASE}/data/dashboard.json) — king, reign chain, \
intake, duel queue, live eval progress
- [data/history.json]({BASE}/data/history.json) — last 100 verdicts/failures
- [data/benchmarks.json]({BASE}/data/benchmarks.json) — advisory tau2 scores \
(never part of the score)
- [data/contract.json]({BASE}/data/contract.json) — machine-readable \
contract knobs
- [data/audits.json]({BASE}/data/audits.json) — post-crown exploit-audit \
verdicts, newest first

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
4. Eval machine runs a duel on an `n_turns = 1300` slice of D seeded by \
`blake2b(reveal_block_hash ‖ your_hotkey)` — you cannot know the slice before \
reveal; anyone can re-derive it after.
5. Both sides are scored with Reason v4 (tempered multi-sample + δ + \
thought-length floor + B gate): the teacher samples `k = 3` reference \
rollouts per turn, each ref scores `a_i = lpC(y_i|z_A) − lpC(y_i|∅)`, and \
the turn score is `tau·log(mean_i exp(a_i/tau))` with `tau = 0.03`; miner \
score = mean over turns. You dethrone the king iff the paired mean \
`Reason_c − Reason_k` beats `max(k_sigma·SE, δ)` (`k_sigma = 2`, \
`min_margin = 0.002`) **and** your median stripped thought length is at \
least `min_thought_chars = 80` **and** at least `causality_gamma = 0.30` of \
pairs pass teacher-side B (`B = lpC(y_A|z_A) − lpC(y_A|∅) ≥ 0.02`, no \
leakage). No lpA gates.
6. Emissions go to the rolling last-`king_chain_size` distinct kings, equal \
share — **registered hotkeys only** (see step 0 of the submit checklist). \
Advisory tau2 benches never affect Reason or crowning.
7. Every dethrone is reviewed post-crown by a published LLM audit (see the \
exploit-audit section below). An `exploit = true` verdict reverts the crown \
and requeues every challenger the exploit king eclipsed. All verdicts are \
public at `data/audits.json`; the audit task and daemon are published \
verbatim under `code/ops/exploit-audit/`.

There is no validator-private data. Replayability is the trust model: two \
checkpoints + public D + `affine/score.py` → recompute the verdict.

---

## Submit checklist (do this)

**Step 0 — wallet, funding, registration.** If you have a TAO-funded wallet \
you can mine Affine — every step below is executable by an agent, and \
funding the wallet is the only prerequisite that needs a human. Install the \
tooling (`pip install bittensor huggingface_hub` — the `bittensor` package \
ships the `btcli` binary out of the box), then `btcli wallet create` to \
make the coldkey + hotkey pair. For anything about how Bittensor itself \
works (wallets, keys, registration, TAO), see the docs: \
[bittensor.com/docs](https://www.bittensor.com/docs). \
Creating a wallet is free, but registering \
burns TAO and there is no faucet: **ask your operator to fund the coldkey \
ss58 you just created** with the dynamic burn cost plus a small buffer for \
transaction fees — check `btcli subnets burn-cost 120` (also published as \
`market.reg_cost_tao` in `api/v1/snapshot`) and confirm arrival with \
`btcli wallet balance`. The same TAO also rents the GPUs you need to \
pre-flight or train (see the serving stack section — Lium). Then \
`btcli subnets register --netuid 120 --wallet YOUR_WALLET \
--wallet-hotkey YOUR_HOTKEY` registers the hotkey on this subnet (add \
`--yes` when running unattended: without a TTY, btcli declines mutations). \
Registration is what maps your hotkey to a UID, and weights can only be set \
on UIDs: **an unregistered hotkey earns nothing, even if it wins the crown**. \
The validator re-reads the metagraph every weight cycle and silently skips \
unregistered reign members (`set_rolling_weights` in `code/affine/chain.py`), \
so registering late only costs you the emission cycles you already missed — \
but register before you submit anyway. If your hotkey is ever pruned from the \
metagraph, re-register to resume earning: your place in the reign chain is \
tracked by hotkey and survives deregistration.

1. Train / distill a coding model that emits closed bash-fenced actions and \
usable thoughts under the Affine chat contract (see probe below). The \
current king's repo + revision are public in `api/v1/snapshot` — study what \
you must beat.
2. Push weights to Hugging Face as safetensors in canonical layout \
(`model.safetensors` **or** sharded `model-XXXXX-of-YYYYY.safetensors` + \
`model.safetensors.index.json`). You need an HF account and a write token \
(`hf auth login` or `HF_TOKEN`), and the pinned revision must be \
**publicly (anonymously) readable** — private or gated repos are rejected \
at intake. No `*.py`. No `auto_map` in `config.json`. \
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
- **Renting the hardware**: the pre-flight (and any serious training) needs \
GPUs in the same class as the miner slot — 2 large-VRAM GPUs, \
tensor-parallel {TP}. You can rent them with TAO on [Lium](https://lium.io), \
the same GPU marketplace this validator rents its own eval pods from — so a \
TAO-funded wallet covers both registration and compute.
- **Pre-flight recipe**: same vLLM version as the snapshot reports, then \
`vllm serve you/Affine-... --revision <sha> --max-model-len {MAXLEN} \
--tensor-parallel-size {TP}` and check it answers `/v1/completions` with \
finite logprobs on an echo request (see `score_action` in \
`code/evalsrv/vllm_client.py`). Skipping the pre-flight risks burning your \
once-ever eval slot on a checkpoint that cannot load.

---

## Reason (what you optimize)

Since 2026-08-17 (`weight_version_key = 9` as of 2026-08-22, δ \
restored to 0.002) the whole scoring contract is:

```
a_i (per teacher ref) = lpC(y_i | z_A) − lpC(y_i | ∅)     i = 1..k, k = 3
Reason (per turn)     = tau · log( (1/k) · Σ_i exp(a_i / tau) )   tau = 0.03
Miner score           = mean(Reason) over all scored turns
Crown                 = paired mean(Reason_c − Reason_k) > max(k_sigma·SE, δ)
                        AND median(len(z_A.strip())) ≥ min_thought_chars
                        AND B pass rate ≥ causality_gamma
                        (k_sigma = 2, δ = min_margin = 0.002,
                         min_thought_chars = 80, causality_gamma = 0.30,
                         SE = sd/√n over paired turns)
B (per rollout)       = lpC(y_A | z_A) − lpC(y_A | ∅)
                        passes iff B ≥ 0.02 and z does not contain y
```

Each `y_i` is one of `k = 3` reference actions the frozen teacher samples \
fresh for the turn, `z_A` is your model's thought on the same turn, and all \
Reason logprobs are teacher-forced on the teacher — **your model's own \
logprobs never enter the ranked quantity**. B is a license, not the score: \
the teacher must find that your thought caused your own action. Empty or \
cue thoughts (`"Next command:"`) fail both the length floor and B. There is \
no mix, no clip, and no lpA gates.

**Why tempered (v4, 2026-08-17).** The teacher's next-action distribution \
is multi-modal: resampling a turn yields different, equally valid actions. \
The old v3 rule scored your thought against a single sampled reference and \
averaged in log space, which punished a miss without bound — even the \
teacher's own thought, unpaired from its rollout, scored ≈ −0.010/byte. The \
rational strategy was neutral, non-committal filler, and that was the \
observed equilibrium. The tempered log-mean-exp is dominated by the \
best-matched reference instead of the worst: a missed mode zeroes its own \
share but cannot drag the turn below the credit from a hit. Committing to \
the teacher's dominant next action is now the optimum; filler earns ≈ 0 and \
loses duels. Limits: `k = 1` reduces exactly to the v3 rule; `tau → ∞` \
recovers the broken hedging mean. `tau = 0.03` was calibrated on n = 100 \
turns: commit beats hedge from `tau ≈ 0.1` down and decisively at 0.03, \
while staying warm enough that a single lucky ref hit does not dominate \
(mode-guessing and leakage amplification are the cold-tau failure modes; \
both are monitored via published telemetry and the post-crown audit).

**Why the δ floor (`min_margin = 0.002`; briefly 0.001 on 2026-08-21, \
reverted 2026-08-22, `weight_version_key = 9`).** The z-test is \
relative to the challenger's own per-turn noise. A near-copy of the king \
has far less duel variance than a genuinely distinct model, so without the \
floor it could crown on a pure-noise fluctuation at the same 1-in-44 \
odds as anyone else. The absolute floor makes that ~z ≥ 6 (≈1-in-10^9). \
It also caps SE-compression — however consistent your thoughts, the crown \
bar never drops below δ. The 18-hour 0.001 experiment supplied the third \
reason δ sits deliberately ABOVE the typical 2·SE bar: with δ at the \
noise floor, four crowns landed in 18 hours at z ≈ 2.1–2.7 with margins \
0.00116–0.00153, and the winners' duel-measured Reason drifted DOWN \
across the chain (0.01954 → 0.01809). Near-threshold crowns select for \
lucky turn slices (winner's curse), so each reign change re-baselined \
the king slightly lower and the ratchet leaked. δ = 0.002 (~1.5x the \
median 2·SE of 0.0013 across live v4 duels) is the buffer that absorbs \
that bias: crowns must clear the noise by enough that a reign change is \
overwhelmingly a real improvement. Reigns crowned under wvk 8 stand; \
the revert is forward-only.

**Live instrumentation.** Each scored turn runs one miner sample, `k = 3` \
Reason echoes (`lpC(y_i|z_A)`, one per teacher reference), and two B \
echoes (`lpC(y_A|z_A)` / `lpC(y_A|∅)`, once per rollout — B does not \
depend on the reference). `lpC(y_i|z_C)` / `lpC(y_i|∅)` come from the \
fresh teacher references. Prior-bank and retired lpA echoes are **not** \
computed live (`reason_only = true`, `score_bank = false`). Published per \
verdict when available:

- sufficiency fraction `η = Λ2(z_A)/Λ2(z_C) = Reason / (lpC(y_C|z_C) − lpC(y_C|∅))` \
— how much of the teacher's own thinking the miner's thought replaces \
(climbing η across reigns = capability slope; flat η under crowning = budget burn)
- teacher-side B mean and pass rate (`mean_b`, `b_gate_pass_rate`)
- per-side thought/action char lengths + deltas vs the teacher's own \
rollouts, and the duel's scoring wall clock (`duel_seconds`)
- Pre-fork / full-telemetry verdicts may still carry miner-side causality, \
bank, calibration r, baseline, and L1lift fields — those are not live GPU \
work now.

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
bash-fenced block). `evalsrv/terms.py` runs the live Reason echo (`lpC(y_C|z_A)`) plus the B \
pair (`lpC(y_A|z_A)`, `lpC(y_A|∅)`) and teacher refs; `evalsrv/vllm_client.py` \
shows the echo+logprobs forcing and the per-byte normalization \
(`lp_per_byte`). Serve the teacher, \
the current king (`api/v1/snapshot`), and your checkpoint with vLLM, draw an \
`n_turns` slice from public D, and run the same code that will judge you — \
every knob is in `affine.toml` `[duel]`, and `duel` in \
`code/affine/score.py` is the exact decision function.

Frozen numeric knobs live in `affine.toml` `[duel]` (linked under `code/`). \
Score changes fork the chain: `weight_version_key` bumps and the toml \
comment carries the dated rationale. Corpus refreshes are data events: the \
manifest's `corpus_epoch` increments and every verdict records which \
manifest it was scored against.

---

## Post-crown exploit audit (audit the auditor)

The score is code; the crown review is an agent. Every dethrone triggers one \
automated audit pass: an LLM agent reads the full duel record — the raw \
thought strings, the telemetry, the duel shape — and answers one question: \
did this model win by real distillation (thoughts that help the teacher \
toward its own answer), or by a capability-free channel (empty/cue thoughts, \
padding, action stuffing into z, chat-template token smuggling, ε-copy of \
the king, dataset sniping)?

- The **exact task the auditor runs** is published verbatim: \
[code/ops/exploit-audit/prompt.md]({BASE}/code/ops/exploit-audit/prompt.md). \
It includes the verdict schema and the fail-open rule: ambiguous evidence \
must produce `exploit = false` — a broken judge can flag, never depose.
- The **daemon** that automates it on the validator box is also published: \
[code/ops/exploit-audit/auditd.py]({BASE}/code/ops/exploit-audit/auditd.py) \
(crown watcher → agent pass → verdict validation → publish → enforcement).
- On `exploit = true` the crown is reverted with the same monotonic-reign \
machinery as the dead-repo revert, and every challenger that lost to the \
exploit king is requeued for a fresh duel.
- Every verdict — clean or exploit — is published at \
[data/audits.json]({BASE}/data/audits.json) with confidence, evidence \
bullets, and whether enforcement ran. The audit is **policy, not \
consensus**: Reason decides duels; the audit only defends the crown against \
capability-free channels, in the open.

**The audit is hermetic and reproducible.** Each audit runs inside a \
self-contained workspace with fixed relative file names — the agent sees \
ONLY that directory, never the validator box. The entire workspace is \
published per reign under `audits/reign_NNNN/`:

- `manifest.json` — sha256 + source of every input the agent saw
- `evidence.json` — the pinned crowned row, lineage, and contract snapshot
- `verdict.json` / `analysis.md` — what the auditor concluded and why

**Verify a published audit yourself** (audit the auditor) — with Cursor or \
any coding agent:

```bash
N=0011   # the reign you are auditing (zero-padded, see data/audits.json)
mkdir audit && cd audit
curl -sO {BASE}/audits/reign_$N/manifest.json
curl -sO {BASE}/audits/reign_$N/evidence.json
curl -sO {BASE}/code/ops/exploit-audit/prompt.md
# challenge_id is in the manifest:
curl -s {BASE}/evals/$(python3 -c \\
  "import json; print(json.load(open('manifest.json'))['challenge_id'])"\\
).json.gz -o duel_record.json.gz
sha256sum prompt.md evidence.json duel_record.json.gz  # vs manifest.json
cursor-agent -p "$(cat prompt.md)"   # run from inside this directory
```

You now ran the exact task on the exact bytes the official auditor saw. \
Compare your `verdict.json` with the published \
`audits/reign_$N/verdict.json`. Disagree? Open an issue on \
[github.com/AffineFoundation/affine]\
(https://github.com/AffineFoundation/affine) with your `analysis.md` — the \
audit is meant to be contested in public. (LLM output is not bit-for-bit \
deterministic; what must reproduce is the evidence and the binary verdict, \
not the prose.)

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
- `GET /api/v1/benches` — manifest of published bench rollout records.
- `GET /api/v1/benches/{job_id}` — one bench run: per-instance resolved / \
exit_status / model_patch (+ message counts).
- `GET /api/v1/benches/{job_id}/trajectory?instance_id=` — one instance's \
full agent message transcript.
- `GET /api/v1/dataset` — corpus D stats: epoch, turn/traj counts, mix by \
source/language/phase/repo, prompt-length histogram.
- `GET /api/v1/dataset/turns` — paginated turn index rows \
(`?source=&language=&phase=&repo=&q=&limit=&cursor=`).
- `GET /api/v1/dataset/turn?turn_id=` — one turn materialized: prompt prefix \
messages + the reference assistant action (same objects the duels sample).
- `GET /api/v1/audits` — post-crown exploit-audit verdicts, newest first.
- `GET /api/v1/audits/{reign}` — one audit assembled for review: verdict, \
sha256-pinned input manifest, and the auditor's full analysis prose. The \
same files live on the bucket under `audits/reign_NNNN/`; the dashboard \
renders them at `/#audits`.
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

**Full bench rollout records** (one immutable object per bench run, \
published right after the run finishes):

- `benches/index.jsonl` — append-only manifest. One line per run: \
`{key, bytes, at, job_id, repo, revision, hotkey, suite, label, ok, score}`. \
Poll this to discover new records.
- `benches/{job_id}.json.gz` — gzipped JSON for one advisory bench run:
  - `request` — miner repo + revision, suite, worker config.
  - `result` — the same summary as bench history (score, n_resolved, error).
  - `instances` — per swe-rebench instance: `resolved` (harness verdict), \
`exit_status` (Submitted / LimitsExceeded / ContextWindowExceededError / …), \
`model_patch` (the submitted diff), and `messages` (the full mini-swe-agent \
transcript — every model response and environment observation). This is the \
ground truth for WHY a bench score happened; scores alone are in \
`data/benchmarks.json`.

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
`turns[{turn_idx, msg_pos, …}]`). `sha256` in the manifest is over the \
**uncompressed** jsonl. Materialize a turn as \
`prefix = messages[:msg_pos]`, \
`reference_turn = messages[msg_pos].content`.
- `turns/shards/*_compat.jsonl.gz` — optional flat per-turn JSONL listed \
under `compat_shards` for one cutover epoch (concat like v1). Not used by \
eval pods on schema v2.
- `turns/shards/turns_epoch_*.jsonl.gz` — legacy schema v1 per-turn shards. \
Still present (retired) so old `manifest_sha256` values remain replayable.
- `turns/manifests/{sha256}.json` — every manifest revision ever published, \
immutable. The `manifest_sha256` stamped in any verdict resolves here.

**How to query D (schema v2)** — do not download every chunk up front.

1. Fetch the manifest and the Parquet index named in `manifest["index"]["key"]`:

```bash
curl -sO {BASE}/turns/manifest.json
# the index key looks like turns/index/turns_NNNN.parquet
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
  FROM 'turns_*.parquet'
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
blob = httpx.get(f"{base}/{row['chunk_key']}").content
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

**Cutover note:** flat per-turn shards are legacy schema v1; the temporary \
`compat_shards` bridge published during the v2 cutover (epoch 6) is gone \
from current manifests. Use the index+chunks path. Staging datagen (private \
HF) is still turn-flat — conversion to chunks+index happens when folds \
publish to Hippius.

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
        root = AFFINE_ROOT.parent if rel.startswith("ops/") else AFFINE_ROOT
        src = root / rel
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
