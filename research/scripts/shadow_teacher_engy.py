"""Shadow-teacher re-scoring of the RT-7 live panel via the engy API.

Question: is the live-board inversion (Spearman(S, swe) = -0.37 under the
GLM-4.5-Air teacher the miners optimized against) a property of THAT teacher
(quirk mining) or of teacher-anchored scoring itself (shared bias)?

Method: take the stored duel artifacts for the n=29 RT-7 panel (miner
rollouts z_a/y_a are frozen in state.wvk9/evals/chal-*.json.gz), re-score
them under independent teachers served by engy, and re-run the same
correlation against the same swe_lite vector.

Teachers (all verified to support echo+logprobs and to match their HF
tokenizer):
  deepseek-v4-flash-0731  different lab        (numerics jitter ~±0.05/token)
  qwen3.8-27b             different lab; the CURRENT live teacher
  glm-5.2                 same lab as the era teacher, newer generation

Readout:
  scores transfer to glm-5.2 but not the others -> lab-level quirk mining
  scores collapse under all shadows            -> version overfitting
  scores transfer everywhere, corr stays neg   -> shared bias; multi-teacher
                                                  will not fix RT-7

Scoring is the frozen min(R,G) v5 imported from affine.score (no math
duplication); prompts are rendered byte-exactly through evalsrv.chat with
each shadow teacher's own chat template. Echoes use engy's
`logprob_start_len` span scoring with token-id prompts.

Usage (from research/):
  python scripts/shadow_teacher_engy.py --smoke
  python scripts/shadow_teacher_engy.py --turns 40 \
      --teachers deepseek-v4-flash-0731,qwen3.8-27b,glm-5.2
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import random
import statistics as st
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))  # evalsrv package lives there

from affine.score import (  # noqa: E402
    DEFAULT_BAND_C, DEFAULT_BAND_FLOOR, DEFAULT_TEMPER_TAU,
    centered_reason, grounding, turn_min_rg,
)
from evalsrv.chat import (  # noqa: E402
    force_text, gen_prompt, get_tokenizer, split_rollout, thought_text,
)

log = logging.getLogger("shadow_teacher")

ENGY_BASE = "https://api.engy.ai/v1"
ECHO_MAX_TOKENS = 1024          # engy per-call scored-span cap
EVALS_DIR = REPO / "affine/state.wvk9/evals"
CORPUS_DIR = REPO / "affine/state.wvk9/corpus_cache"
PANEL_JSON = REPO / "research/results/rt7_live_isomorphism.json"

# Duel knobs mirrored from affine/affine.toml [duel] (sampling side only).
K_REFS = 3
TEMPERATURE = 0.8
MAX_ROLLOUT_TOKENS = 1024 + 768  # max_thought_tokens + max_action_tokens

TEACHERS = {
    "deepseek-v4-flash-0731": "deepseek-ai/DeepSeek-V4-Flash",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "glm-5.2": "zai-org/GLM-5.2",
}
# DeepSeek V4 repos ship no chat_template; the V3.1 template renders cleanly
# on the V4-Flash tokenizer (all special tokens are single ids in its vocab).
FALLBACK_TEMPLATE_REPO = "deepseek-ai/DeepSeek-V3.1"


def ensure_chat_template(tok_repo: str):
    """get_tokenizer with a fallback chat template installed if missing.

    get_tokenizer is lru_cached, so mutating the returned instance sticks for
    every later call through evalsrv.chat (gen_prompt/force_text/...).
    """
    # NB: chat.py calls get_tokenizer(repo, revision) positionally; the lru
    # cache keys on the exact call shape, so mirror it here or the template
    # lands on a different cached instance.
    tok = get_tokenizer(tok_repo, None)
    if tok.chat_template is None:
        with urllib.request.urlopen(
                f"https://huggingface.co/{FALLBACK_TEMPLATE_REPO}"
                "/resolve/main/tokenizer_config.json", timeout=60) as r:
            tok.chat_template = json.load(r)["chat_template"]
        log.info("installed %s chat template on %s",
                 FALLBACK_TEMPLATE_REPO, tok_repo)
    return tok


def load_engy_key() -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("ENGY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("ENGY key not found in .env")


# -- engy client -------------------------------------------------------------------

class EngyClient:
    def __init__(self, key: str, concurrency: int):
        self.http = httpx.AsyncClient(
            base_url=ENGY_BASE,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(240.0, connect=15.0))
        self.sem = asyncio.Semaphore(concurrency)
        self.tokens_in = 0
        self.tokens_out = 0

    async def close(self) -> None:
        await self.http.aclose()

    async def _post(self, payload: dict) -> dict:
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.http.post("/completions", json=payload)
                    if r.status_code == 400:
                        raise FatalRequestError(r.text[:300])
                    r.raise_for_status()
                    d = r.json()
                    u = d.get("usage") or {}
                    self.tokens_in += u.get("prompt_tokens", 0)
                    self.tokens_out += u.get("completion_tokens", 0)
                    return d
                except FatalRequestError:
                    raise
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(3 * 2 ** attempt)
        raise RuntimeError("unreachable")

    async def sample(self, model: str, prompt_ids: list[int],
                     max_tokens: int, temperature: float) -> str:
        d = await self._post({
            "model": model, "prompt": prompt_ids,
            "max_tokens": max_tokens, "temperature": temperature,
        })
        return d["choices"][0]["text"]

    async def echo_span(self, model: str, tokenizer, full: str,
                        span_start_char: int, span_bytes: int) -> dict:
        """Mean logprob per byte of full[span_start_char:] (production math).

        Mirrors VllmModel._echo_span: token boundary from the tokenizer's
        offset mapping; engy scores from `logprob_start_len` so cost scales
        with the span. Long spans are scored in <=ECHO_MAX_TOKENS windows by
        truncating the token-id prompt at each window end (logprobs of a
        token depend only on earlier tokens, so windows compose exactly).
        """
        enc = tokenizer(full, add_special_tokens=False,
                        return_offsets_mapping=True)
        ids = enc["input_ids"]
        n_prompt = sum(1 for s, _ in enc["offset_mapping"]
                       if s < span_start_char)
        span_lps: list[float] = []
        start = n_prompt
        while start < len(ids):
            end = min(start + ECHO_MAX_TOKENS - 1, len(ids))
            # Engy returns logprobs for prompt positions [start_len, end)
            # only (no generated token), and the FIRST returned position is
            # always None (vLLM prompt-logprobs convention). Request one
            # token early and drop that sacrificial first entry, otherwise
            # the first span token's logprob is lost.
            req_start = max(0, start - 1)
            d = await self._post({
                "model": model, "prompt": ids[:end], "max_tokens": 1,
                "temperature": 0, "echo": True, "logprobs": 1,
                "logprob_start_len": req_start,
            })
            lp = d["choices"][0]["logprobs"]["token_logprobs"]
            window = lp[start - req_start:]
            expect = end - start
            got = [x for x in window if x is not None]
            if len(got) != expect:
                raise RuntimeError(
                    f"echo window misaligned: expected {expect} lps, "
                    f"got {len(got)} (start={start} end={end})")
            span_lps.extend(got)
            start = end
        n_bytes = max(span_bytes, 1)
        return {
            "sum_lp": sum(span_lps),
            "n_tokens": len(span_lps),
            "lp_per_byte": sum(span_lps) / n_bytes if span_lps else 0.0,
        }


class FatalRequestError(Exception):
    """400s are deterministic for this prompt; do not retry."""


# -- data loading ------------------------------------------------------------------

def load_panel() -> list[dict]:
    rows = json.load(open(PANEL_JSON))["rows"]
    return sorted(rows, key=lambda r: -r["s"])


def artifact_for(repo: str) -> Path | None:
    """Latest local duel artifact whose challenger is `repo`."""
    best: tuple[str, Path] | None = None
    for p in sorted(EVALS_DIR.glob("chal-*.json.gz")):
        try:
            d = json.load(gzip.open(p))
        except Exception:
            continue
        req = d.get("request", {})
        if (req.get("challenger_repo") or "").lower() != repo:
            continue
        rows = d.get("challenger_rows") or []
        if any(r.get("pairs") and r["pairs"][0].get("z_a") for r in rows):
            best = (p.name, p)
    return best[1] if best else None


def rollouts_from_artifact(path: Path) -> dict[str, tuple[str, str]]:
    """turn_id -> (z_a, y_a) for the challenger side."""
    d = json.load(gzip.open(path))
    out: dict[str, tuple[str, str]] = {}
    for row in d.get("challenger_rows") or []:
        pairs = row.get("pairs") or []
        if not pairs or not row.get("turn_id"):
            continue
        z_a, y_a = pairs[0].get("z_a"), pairs[0].get("y_a")
        if z_a and y_a:
            out[row["turn_id"]] = (z_a, y_a)
    return out


def build_corpus():
    """CorpusSync against the local cache (lazy chunk fetch from the bucket)."""
    from affine.config import load_config
    from evalsrv.corpus import CorpusSync
    cfg = load_config()
    return CorpusSync(cfg.dataset.corpus_base_url, cfg.dataset.manifest_key,
                      CORPUS_DIR, lazy_chunks=True)


# -- scoring -----------------------------------------------------------------------

@dataclass
class TurnResult:
    turn_id: str
    score: float
    r_leg: float
    g_leg: float
    n_refs: int


@dataclass
class Progress:
    done_calls: int = 0
    t0: float = field(default_factory=time.time)


async def score_turn(client: EngyClient, model: str, tokenizer,
                     tok_repo: str, prefix: list[dict],
                     z_a: str, y_a: str, turn_id: str) -> TurnResult | None:
    """One turn under one shadow teacher: k refs + the min(R,G) echo set."""
    gen = gen_prompt(tok_repo, None, prefix)
    gen_ids = tokenizer(gen, add_special_tokens=False)["input_ids"]

    texts = await asyncio.gather(*[
        client.sample(model, gen_ids, MAX_ROLLOUT_TOKENS, TEMPERATURE)
        for _ in range(K_REFS)
    ])
    refs = [(z, y) for z, y in (split_rollout(t) for t in texts) if y]
    if len(refs) < 2:  # band needs >=2 reference thoughts
        return None

    async def action_echo(thoughts: str, action: str) -> float:
        full = force_text(tok_repo, None, prefix, thoughts, action)
        r = await client.echo_span(model, tokenizer, full,
                                   len(full) - len(action),
                                   len(action.encode()))
        return r["lp_per_byte"]

    async def thought_echo(thoughts: str) -> float:
        full = thought_text(tok_repo, None, prefix, thoughts)
        r = await client.echo_span(model, tokenizer, full,
                                   len(full) - len(thoughts),
                                   len(thoughts.encode()))
        return r["lp_per_byte"]

    k = len(refs)
    res = await asyncio.gather(
        *[action_echo(z_a, y_i) for _, y_i in refs],        # lpC(y_i|z_A)
        *[action_echo("", y_i) for _, y_i in refs],          # lpC(y_i|∅)
        *[thought_echo(z_i) for z_i, _ in refs],             # t_i
        thought_echo(z_a),                                   # m
    )
    pairs = []
    for i in range(k):
        pairs.append({
            "lpC_yc_za": res[i],
            "lpC_yc_e": res[k + i],
            "lpC_zc_x": res[2 * k + i],
            "lpC_za_x": res[3 * k],
            "z_a": z_a,
            "y_a": y_a,
        })
    score = turn_min_rg(pairs, DEFAULT_TEMPER_TAU,
                        DEFAULT_BAND_C, DEFAULT_BAND_FLOOR)
    return TurnResult(
        turn_id=turn_id,
        score=score,
        r_leg=centered_reason(pairs, DEFAULT_TEMPER_TAU),
        g_leg=grounding(pairs, DEFAULT_BAND_C, DEFAULT_BAND_FLOOR),
        n_refs=k,
    )


async def score_model_teacher(client: EngyClient, corpus, model: str,
                              tok_repo: str, repo: str,
                              turns: list[tuple[str, list[dict], str, str]],
                              out_path: Path, turn_conc: int) -> dict:
    """All selected turns for one (panel model, shadow teacher)."""
    tokenizer = ensure_chat_template(tok_repo)
    sem = asyncio.Semaphore(turn_conc)
    results: list[TurnResult] = []
    errors: list[str] = []

    async def one(turn_id: str, prefix: list[dict], z_a: str, y_a: str):
        async with sem:
            try:
                r = await score_turn(client, model, tokenizer, tok_repo,
                                     prefix, z_a, y_a, turn_id)
                if r is not None:
                    results.append(r)
            except FatalRequestError as e:
                errors.append(f"{turn_id}: {e}")
            except Exception as e:  # noqa: BLE001 — survey job, keep going
                errors.append(f"{turn_id}: {type(e).__name__}: {e}")

    await asyncio.gather(*[one(*t) for t in turns])

    rec = {
        "repo": repo,
        "teacher": model,
        "n_turns_scored": len(results),
        "n_errors": len(errors),
        "score": st.mean(r.score for r in results) if results else None,
        "mean_r_leg": st.mean(r.r_leg for r in results) if results else None,
        "mean_g_leg": st.mean(r.g_leg for r in results) if results else None,
        "g_bind_frac": (st.mean(1.0 if r.g_leg < r.r_leg else 0.0
                                for r in results) if results else None),
        "turns": [vars(r) for r in results],
        "errors": errors[:10],
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return rec


# -- stats (mirrors rt7_live_isomorphism.py) ----------------------------------------

def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra)
           * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def permutation_p(a, b, observed, trials=100_000, seed=0):
    rng = random.Random(seed)
    shuffled = list(b)
    hits = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        if abs(spearman(a, shuffled)) >= abs(observed):
            hits += 1
    return (hits + 1) / (trials + 1)


# -- main --------------------------------------------------------------------------

async def run(args) -> None:
    key = load_engy_key()
    client = EngyClient(key, concurrency=args.concurrency)
    corpus = build_corpus()
    if not corpus.ready:
        raise SystemExit("corpus cache not ready; run evalsrv corpus sync")

    index_by_tid = {r["turn_id"]: r for r in corpus.load_index_rows()}
    panel = load_panel()
    if args.models:
        panel = panel[: args.models]
    teachers = {m: TEACHERS[m] for m in args.teachers.split(",")}

    out_path = Path(args.out).with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.fresh:
        out_path.unlink(missing_ok=True)
    elif out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["repo"], r["teacher"]))

    for row in panel:
        repo = row["repo"]
        art = artifact_for(repo)
        if art is None:
            log.warning("no artifact for %s; skipping", repo)
            continue
        rollouts = rollouts_from_artifact(art)
        tids = [t for t in sorted(rollouts) if t in index_by_tid]
        rng = random.Random(f"shadow:{repo}")
        rng.shuffle(tids)
        tids = tids[: args.turns]
        if not tids:
            log.warning("no corpus-covered turns for %s; skipping", repo)
            continue
        turn_rows = corpus.materialize_turns([index_by_tid[t] for t in tids])
        turns = [(tid, tr["prefix"], *rollouts[tid])
                 for tid, tr in zip(tids, turn_rows)]

        for model, tok_repo in teachers.items():
            if (repo, model) in done:
                continue
            t0 = time.time()
            rec = await score_model_teacher(
                client, corpus, model, tok_repo, repo, turns,
                out_path, args.turn_concurrency)
            log.info("%s x %s: score=%s over %d turns (%d err) in %.0fs "
                     "[tokens in=%.1fM out=%.1fM]",
                     repo[:40], model, rec["score"], rec["n_turns_scored"],
                     rec["n_errors"], time.time() - t0,
                     client.tokens_in / 1e6, client.tokens_out / 1e6)

    await client.close()
    summarize(out_path, Path(args.out), panel)


def summarize(jsonl_path: Path, out_base: Path, panel: list[dict]) -> None:
    swe = {r["repo"]: r["swe"] for r in panel}
    s_live = {r["repo"]: r["s"] for r in panel}
    by_teacher: dict[str, dict[str, float]] = {}
    for line in jsonl_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("score") is not None:
            by_teacher.setdefault(r["teacher"], {})[r["repo"]] = r["score"]

    res = {"n_panel": len(panel), "teachers": {}}
    lines = ["Shadow-teacher re-scoring of the RT-7 live panel (engy)",
             "=" * 64]
    live_repos = [r["repo"] for r in panel]
    rho_live = spearman([s_live[r] for r in live_repos],
                        [swe[r] for r in live_repos])
    lines.append(f"live teacher (stored S):  Spearman(S, swe) = {rho_live:+.3f}"
                 f"  (n={len(live_repos)})")
    for teacher, scores in sorted(by_teacher.items()):
        repos = [r for r in live_repos if r in scores]
        if len(repos) < 5:
            lines.append(f"{teacher}: only {len(repos)} scored, skipping stats")
            continue
        sv = [scores[r] for r in repos]
        wv = [swe[r] for r in repos]
        lv = [s_live[r] for r in repos]
        rho_swe = spearman(sv, wv)
        p_swe = permutation_p(sv, wv, rho_swe)
        rho_transfer = spearman(sv, lv)
        res["teachers"][teacher] = {
            "n": len(repos),
            "spearman_shadow_swe": rho_swe,
            "p_shadow_swe": p_swe,
            "spearman_shadow_liveS": rho_transfer,
            "scores": {r: scores[r] for r in repos},
        }
        lines.append(
            f"{teacher:24s} n={len(repos):2d}  "
            f"Spearman(S_shadow, swe) = {rho_swe:+.3f} (p={p_swe:.4f})  "
            f"Spearman(S_shadow, S_live) = {rho_transfer:+.3f}")

    out_base.with_suffix(".json").write_text(json.dumps(res, indent=2))
    out_base.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers",
                    default="deepseek-v4-flash-0731,qwen3.8-27b,glm-5.2")
    ap.add_argument("--models", type=int, default=0,
                    help="limit to top-N panel models (0 = all)")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="max in-flight engy requests")
    ap.add_argument("--turn-concurrency", type=int, default=3,
                    help="turns scored in parallel per model")
    ap.add_argument("--out", default="results/shadow_teacher_engy")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoint jsonl")
    ap.add_argument("--smoke", action="store_true",
                    help="2 models x 5 turns x deepseek only")
    args = ap.parse_args()
    if args.smoke:
        args.models, args.turns = 2, 5
        args.teachers = "deepseek-v4-flash-0731"
        args.out = "results/shadow_teacher_engy_smoke"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
