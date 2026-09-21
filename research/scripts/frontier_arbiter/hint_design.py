"""Hint design + informed TEACHER vs informed KING as the A-leg reference
sampler (store doc §3c, probe W2) — 2026-09-21.

Builds on D7 (privileged_refs.py: 150 turns, stored blind teacher refs, king
rollouts) and W1 (hindsight_refs.py: forward trajectory + note plumbing;
its trace mirror under /tmp/fa + /tmp/vr is reused read-only).

Rank on A only, anchored to a HINDSIGHT-INFORMED model:

    b_j(y)     = [lpC(y | x, z^p_j) − lpC(y | x, ∅)] · bytes(y)          (nats, a_norm 1)
    A_priv(y)  = τ · log mean_j exp(b_j / τ)                             (τ = 0.03)
    z^p_j      ~ INFORMED sampler (teacher qwen3.8-27b | king reign 20) given x + a private hint

Q1 hint design (glm-5.3, T0; 429 -> glm-5.3-flash, stamped), three variants per turn
written from the forward trajectory + outcome, one JSON call per turn:
  h_ground   grounded attention nudge, <= 60 words, may only point at things ALREADY in the prefix;
             every identifier / path / number / quoted string must occur in the prefix (programmatic);
             no commands, no future facts.
  h_causal   causal hindsight, <= 60 words, what went wrong LATER and why; future facts allowed,
             no commands / code / paths.
  h_answer   answer leak (control): names the decision to take in words, no code.

Q2 informed teacher vs informed king: k=3 refs per variant x sampler (T0.8, 4096 cap) with the
hint appended as a final user message; k=3 BLIND king refs once; echo grid under the teacher.

    python hint_design.py hints   [--turns 100]
    python hint_design.py sample  [--sampler teacher|king|both] [--concurrency N]
    python hint_design.py judge   [--n 40]
    python hint_design.py echo    [--start N --turns M --concurrency N]
    python hint_design.py report

Terms (one line each):
  x / z / y          turn prefix / thought / action (dialect action span).
  blind teacher      the k=3 stored teacher refs of the wvk-22 verdict (z_C^i, y_C^i) ~ T(.|x).
  p_none             D7's fresh blind TEACHER refs on the same turns — the teacher's sampling-noise floor.
  y_K / z_K          the stored king rollout of the verdict (one blind king sample at duel time).
  blind king         k=3 fresh king refs sampled here without a note ~ K(.|x); vs y_K = the king's noise floor.
  informed refs      k=3 refs of sampler S given x + hint h: ~ S(.|x,h); own = their actions, LOO under the other two.
  change             a ref action norm-different from all blind actions of the SAME sampler AND best token-Jaccard < .5.
  turn changed       >= 2 of the 3 informed refs changed (the proposed fold-time "decision state" filter).
  grounding rate     share of a hint's word tokens (>= 3 chars, alnum) that occur in the prefix text.
  fwd-leak           a >= 20-char whitespace-normalised window of the hint occurs in the FUTURE trajectory and not in the prefix.
  leak               a >= 30-char window of the hint occurs in an informed ref's z / y.
  A_blind (stored)   the same summed A under the blind teacher thoughts, from the verdict's vLLM echoes.
  A_blind_king       A under the fresh blind KING thoughts (Engy echo).
  matched estimator  external candidates are scored as mean_j LME over the 2 thoughts i != j, like the LOO own value.
  hz                 paired z over turns of a per-turn difference (mean / SE).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (  # noqa: E402
    REPO, TEACHER_ENGY, Engy, TeacherEcho, agree, exact, jaccard, lme, norm_action,
    read_jsonl, append_jsonl, write_jsonl,
)
from privileged_refs import (  # noqa: E402
    K, TAU, REF_TEMPERATURE, REF_MAX_TOKENS,
    TURNS as D7_TURNS, SAMPLES as D7_SAMPLES, JUDGE as D7_JUDGE,
    trace_index, load_envelope, render_transcript, parse_reply, leaks,
)
from hindsight_refs import (  # noqa: E402
    HIND_PATH_RE, JUDGE_SYSTEM, LABELS, forward_part, render_forward, generic_action, repeat_last,
)

RESULTS = REPO / "research" / "results" / "frontier_arbiter" / "hint_design"
HINTS = RESULTS / "hints.jsonl"
SAMPLES = RESULTS / "samples.jsonl"
JUDGE = RESULTS / "judge.jsonl"
ECHOES = RESULTS / "echoes.jsonl"
COST = RESULTS / "cost.jsonl"
BUDGET_USD = 22.0
N_TURNS = 100            # first 100 of D7's 150 (D7 order is a seeded shuffle); budget-bound

VARIANTS = ("h_ground", "h_causal", "h_answer")
SAMPLERS = ("teacher", "king")
KING_BLIND = ("blind", "blind2")   # two independent k=3 blind king draws: blind2-vs-blind = the king's 3-vs-3 noise floor
FRONTIER = "glm-5.3"
FRONTIER_FALLBACK = "glm-5.3-flash"
JUDGE_MODEL = "glm-5.3-flash"
FWD_CAP = 40_000          # forward part p50 ~15k chars (W1); the cap only trims the long tail
PREFIX_CAP = 90_000       # = D7's MAX_PREFIX_CHARS, so no turn is elided
MAX_WORDS_HARD = 70       # "<= 60 words" with ~15% slack; the share <= 60 is reported
NOTE_HEAD = "Privileged note (the assistant cannot see this at test time): "

KING_BASE = "http://167.150.153.180:3000/v1"      # ops/king-datagen/state/state.json, pod king-dg-50e3c081ab1c-0e6d
KING_MODEL = "king-50e3c081ab1c"
KING_KEY = "e85415c9ddde54e95bf3aa0388e1f9fac275f460a203d747"
KING_REIGN = 20

CODE_LIKE_RE = re.compile(r"[{}]|\w\(\)|\w\(\w|==|=>|->|\$\(|\$\w|<<|>>|\b\w+\.\w+\(|\|\||&&|(?<!\w)\|(?!\w)|(?<=\s)-{1,2}[A-Za-z]\w*\b")
# grounded-nudge tokens that must occur in the prefix
QUOTED_RE = re.compile(r"[\"'“‘]([^\"'”’]{2,80})[\"'”’]")
IDENT_RE = re.compile(r"(?<![\w/.-])(?:[A-Za-z_][\w]*(?:[._/-][\w]+)+|[a-z]+[A-Z]\w+|[A-Z][a-z]+[A-Z]\w*|[A-Z][A-Z0-9_]{2,}|_\w+|\w+_)(?![\w/.-])")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
PROSE_HYPHEN_RE = re.compile(r"[A-Za-z]+(?:-[a-z]+)+")      # re-read, hand-applying: prose, not identifiers
WORD_RE = re.compile(r"[A-Za-z0-9]{3,}")
HINT_MAX_TOKENS = 10000    # glm-5.3 thinks 10-25k chars before the JSON; 6000 hit the cap on 2/3 smoke turns
COMMON_CAPS = {"THE", "AND", "NOT", "YOU", "API", "URL", "CLI", "JSON", "HTTP", "TODO", "OK"}


# ------------------------------------------------------------------ cost ledger
def log_cost(stage: str, engy: Engy, note: str = "") -> None:
    append_jsonl(COST, {"at": time.time(), "stage": stage, "cost_usd": engy.cost_usd,
                        "usage": engy.usage, "note": note})
    tot = total_cost()
    print(f"  [$] {stage}: this run ${engy.cost_usd:.3f} | total so far ${tot:.2f} {note}", flush=True)
    if tot > BUDGET_USD:
        raise SystemExit(f"budget exceeded: ${tot:.2f} > ${BUDGET_USD}")


def total_cost() -> float:
    groups: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in read_jsonl(COST):
        groups[(r["stage"], str(r.get("note", "")).split("/")[-1])].append(r["cost_usd"])
    total = 0.0
    for vals in groups.values():
        run_max = 0.0
        for c in vals:
            if c < run_max:
                total += run_max
                run_max = 0.0
            run_max = max(run_max, c)
        total += run_max
    return total


def load_turns(n: int = N_TURNS) -> list[dict]:
    return read_jsonl(D7_TURNS)[:n]


# ------------------------------------------------------------------ king client
class King(Engy):
    """The sitting king's vLLM endpoint, RAW `/completions` on the chat-template
    prompt (rendered locally with the king's own tokenizer, fetched from
    models.affine.io). The box's qwen3_xml tool parser strips `<tool_call>`
    blocks from chat-completions content when no `tools` are sent (verified:
    77/84 tool_call-turn replies came back as bare prose), so the raw path is
    the only faithful one. Reply = "<latent>\\n</think>\\n\\n<visible>"; a reply
    without `</think>` is an unclosed thought (forfeit under the live rule).
    Free: cost stays 0."""

    def __init__(self, concurrency: int = 12, timeout: float = 1200.0, retries: int = 4):
        super().__init__(key="none", concurrency=concurrency, timeout=timeout, retries=retries)
        self.cli = httpx.AsyncClient(base_url=KING_BASE, timeout=timeout,
                                     headers={"Authorization": f"Bearer {KING_KEY}"})
        self.tok = king_tokenizer()

    def prompt(self, messages: list[dict]) -> str:
        p = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not p.rstrip().endswith(common.THINK_OPEN):
            p = p + common.THINK_OPEN + "\n"
        return p

    async def chat(self, model: str, messages: list[dict], temperature: float = 0.8,
                   max_tokens: int = REF_MAX_TOKENS, **extra) -> dict:
        payload = {"model": KING_MODEL, "prompt": self.prompt(messages), "max_tokens": max_tokens,
                   "temperature": temperature, **extra}
        d = await self._post("/completions", payload, KING_MODEL)
        self._tally(KING_MODEL, d)
        ch = d["choices"][0]
        raw = ch.get("text") or ""
        closed = common.THINK_CLOSE in raw
        if closed:
            reasoning, _, content = raw.partition(common.THINK_CLOSE)
        else:
            reasoning, content = raw, ""
        return {"reasoning": reasoning.strip("\n"), "content": content.lstrip("\n"), "tool_calls": [],
                "finish": ch.get("finish_reason"), "usage": d.get("usage"), "cost_usd": 0.0,
                "worker": None, "model": d.get("model") or KING_MODEL, "raw": raw, "think_closed": closed}


KING_TOK_DIR = Path("/tmp/king_tok")
KING_TOK_FILES = ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json")


def king_tokenizer():
    from transformers import AutoTokenizer  # heavy import kept lazy for CLI help
    KING_TOK_DIR.mkdir(parents=True, exist_ok=True)
    digest = "50e3c081ab1c45bae37f3b1677728f06168ed38c6a3a3ac1b7711b702b8cbd12"
    with httpx.Client(timeout=300) as c:
        for f in KING_TOK_FILES:
            p = KING_TOK_DIR / f
            if not p.exists():
                r = c.get(f"https://models.affine.io/models/sha256/{digest}/{f}")
                r.raise_for_status()
                p.write_bytes(r.content)
    return AutoTokenizer.from_pretrained(str(KING_TOK_DIR))


async def king_reachable(timeout_s: float = 20.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout_s, headers={"Authorization": f"Bearer {KING_KEY}"}) as c:
            r = await c.get(KING_BASE + "/models")
            return r.status_code == 200 and KING_MODEL in r.text
    except httpx.HTTPError:
        return False


# ------------------------------------------------------------------ hint validation
def prefix_text(t: dict) -> str:
    return re.sub(r"\s+", " ", "\n".join(m["content"] for m in t["prefix"]))


def future_text(fp: dict) -> str:
    return re.sub(r"\s+", " ", fp["step"] + "\n" + "\n".join(m["text"] for m in fp["fwd"]))


def ground_tokens(hint: str) -> list[str]:
    """Identifiers / paths / numbers / quoted strings that a grounded nudge must
    have taken from the prefix."""
    toks: list[str] = []
    toks += [q.strip() for q in QUOTED_RE.findall(hint)]
    toks += [x for x in IDENT_RE.findall(hint)
             if not (x.isupper() and x in COMMON_CAPS) and not PROSE_HYPHEN_RE.fullmatch(x)]
    toks += [x for x in NUMBER_RE.findall(hint) if not re.fullmatch(r"[1-9]|1\d|20|30|40|50|60|100", x)]  # small counts / ordinals are prose
    return sorted(set(toks))


def ungrounded(hint: str, prefix: str) -> list[str]:
    pl = prefix.lower()
    return [x for x in ground_tokens(hint) if re.sub(r"\s+", " ", x).lower() not in pl]


def grounding_rate(hint: str, prefix: str) -> float | None:
    words = [w.lower() for w in WORD_RE.findall(hint)]
    if not words:
        return None
    pl = prefix.lower()
    return sum(1.0 for w in words if w in pl) / len(words)


def fwd_leak_windows(hint: str, fut: str, pre: str, n: int = 20) -> list[str]:
    """Windows of the hint that occur in the FUTURE trajectory but not in the prefix."""
    hn = re.sub(r"\s+", " ", hint)
    out = []
    for s in range(0, max(1, len(hn) - n + 1), 2):
        w = hn[s:s + n]
        if len(w) == n and w in fut and w not in pre and not (out and w[:-2] in out[-1]):
            out.append(w)
    return out


def fwd_leak(hint: str, fut: str, pre: str, n: int = 20) -> bool:
    return bool(fwd_leak_windows(hint, fut, pre, n))


def validate(variant: str, hint: str, prefix: str) -> str | None:
    if not hint or not hint.strip():
        return "empty"
    if "`" in hint:
        return "backtick"
    if len(hint.split()) > MAX_WORDS_HARD:
        return "too_long"
    if CODE_LIKE_RE.search(hint):
        return "code_like"
    if variant == "h_causal" and HIND_PATH_RE.search(hint):
        return "path"
    if variant == "h_ground":
        bad = ungrounded(hint, prefix)
        if bad:
            return "ungrounded"
    return None


# ------------------------------------------------------------------ stage: hints
HINT_SYSTEM = (
    "You write PRIVATE notes for an agent at step t of a coding / terminal / tool task. You see the transcript "
    "up to step t, what the agent then did, and the final grade. Return ONE JSON object with exactly the keys "
    "h_ground, h_causal, h_answer — each a string of AT MOST 60 words, plain prose, one paragraph, no lists, "
    "no backticks, no code, no literal command text.\n"
    "h_ground — GROUNDED ATTENTION NUDGE: point only at things ALREADY VISIBLE in the transcript so far that "
    "turned out to matter later (e.g. 're-read the output of your third command: the error line names the real "
    "cause', 'you have not yet looked at the test that exercises this function'). Every identifier, filename, "
    "number or quoted string you mention must appear verbatim in the transcript so far. Do NOT mention anything "
    "that happened after step t, do NOT say what to do next, no commands.\n"
    "h_causal — CAUSAL HINDSIGHT: state what went wrong LATER and why (for a solved run: what later proved "
    "decisive and why). You may name future facts (which test failed, what the assertion was about, what was "
    "overlooked) but no commands, no code, no file paths or filenames, no identifiers.\n"
    "h_answer — ANSWER LEAK (control): name in plain words the decision the agent should take at THIS step given "
    "hindsight (solved run: the next step it actually took, if that was right; failed run: what it should have done "
    "instead). Be concrete about the decision, but still no code and no literal command text.\n"
    "Output ONLY the JSON object."
)
REJECT_MSG = {
    "empty": "the field was empty",
    "backtick": "it contained a backtick",
    "too_long": "it exceeded 60 words",
    "code_like": "it contained code / a command / a flag / an operator",
    "path": "it contained a file path or filename (not allowed in h_causal)",
    "ungrounded": "it mentioned tokens that do not occur in the transcript so far: {bad}",
}


def parse_json_obj(txt: str) -> dict | None:
    """The last JSON object in the reply that has one of our keys (glm-5.3
    sometimes writes prose or a stray brace before the object)."""
    txt = (txt or "").strip()
    dec = json.JSONDecoder()
    best = None
    for m in re.finditer(r"\{", txt):
        try:
            d, _ = dec.raw_decode(txt, m.start())
        except ValueError:
            continue
        if isinstance(d, dict) and any(k in d for k in VARIANTS):
            best = d
    return best


async def frontier_chat(engy: Engy, engy_fb: Engy, msgs: list[dict], model: str = FRONTIER) -> tuple[dict, str]:
    """glm-5.3 at T0; 429 -> glm-5.3-flash (stamped). A reply whose thinking ate
    the whole token budget (finish=length, empty content) is re-asked once on
    glm-5.3-flash and stamped '<flash>@think_cap'."""
    if model == FRONTIER:
        try:
            r = await engy.chat(FRONTIER, msgs, temperature=0.0, max_tokens=HINT_MAX_TOKENS)
            if not (r["finish"] == "length" and not (r["content"] or "").strip()):
                return r, FRONTIER
            model = FRONTIER_FALLBACK + "@think_cap"
        except RuntimeError as ex:
            if "429" not in str(ex):
                raise
            model = FRONTIER_FALLBACK
    r = await engy_fb.chat(FRONTIER_FALLBACK, msgs, temperature=0.0, max_tokens=HINT_MAX_TOKENS)
    return r, model


async def gen_hints(turns: list[dict], engy: Engy, engy_fb: Engy) -> None:
    done = {r["turn_id"] for r in read_jsonl(HINTS)}
    todo = [t for t in turns if t["turn_id"] not in done]
    print(f"hints: {len(todo)} turns to do")
    tidx = trace_index()

    async def one(t: dict) -> None:
        env = load_envelope(*tidx[t["rollout_id"]])
        fp = forward_part(env["trace"], t["turn_idx"])
        outcome = t["down"]["outcome"]
        pre, fut = prefix_text(t), future_text(fp)
        rec = {"turn_id": t["turn_id"], "outcome": outcome, "n_fwd": fp["n_fwd"], "fwd_chars": fp["fwd_chars"],
               "attempts": [], "hints": {}, "model": {}, "reject": {}}
        user = (f"Transcript so far — the agent must now produce its next step (dialect: {t['dialect']}):\n\n"
                f"{render_transcript(t['prefix'], PREFIX_CAP)}\n\n--- END OF TRANSCRIPT SO FAR (step t) ---\n\n"
                f"What the agent did from step t onward, and how it was graded:\n\n{render_forward(fp, outcome)}\n\n"
                f"--- END ---\nWrite the JSON object with h_ground, h_causal, h_answer now.")
        msgs = [{"role": "system", "content": HINT_SYSTEM}, {"role": "user", "content": user}]
        pending = set(VARIANTS)
        model = FRONTIER
        for attempt in range(3):
            try:
                r, model = await frontier_chat(engy, engy_fb, msgs, FRONTIER if model == FRONTIER else FRONTIER_FALLBACK)
            except Exception as ex:  # noqa: BLE001
                rec["attempts"].append({"error": repr(ex)[:200]})
                break
            obj = parse_json_obj(r["content"])
            att = {"model": model, "usage": r["usage"], "cost_usd": r["cost_usd"], "finish": r["finish"],
                   "reasoning_chars": len(r["reasoning"] or ""), "raw": (r["content"] or "")[:1500], "reject": {}}
            if obj is None:
                att["reject"] = {v: "no_json" for v in pending}
                rec["attempts"].append(att)
                if attempt < 2:
                    msgs = msgs + [{"role": "assistant", "content": r["content"] or ""},
                                   {"role": "user", "content": "That was not a JSON object. Output ONLY the JSON object with the keys "
                                                               + ", ".join(sorted(pending)) + "."}]
                continue
            fixes = []
            for v in sorted(pending):
                hint = obj.get(v)
                hint = hint.strip() if isinstance(hint, str) else ""
                why = validate(v, hint, pre)
                att["reject"][v] = why
                if why is None:
                    rec["hints"][v] = hint
                    rec["model"][v] = model
                else:
                    bad = ungrounded(hint, pre) if why == "ungrounded" else []
                    fixes.append(f"{v}: rejected because {REJECT_MSG[why].format(bad=', '.join(repr(b) for b in bad[:8]))}")
                    rec["reject"][v] = why
            rec["attempts"].append(att)
            pending = {v for v in pending if v not in rec["hints"]}
            if not pending:
                break
            msgs = msgs + [{"role": "assistant", "content": r["content"] or ""},
                           {"role": "user", "content": "Some fields were rejected:\n" + "\n".join(fixes)
                            + "\nRewrite ONLY these fields (same rules: <= 60 words, plain prose, no backticks / code / commands / flags; "
                              "h_ground may only use identifiers, numbers and quoted strings that occur verbatim in the transcript so far; "
                              "h_causal names no paths or identifiers). Output ONLY a JSON object with keys " + ", ".join(sorted(pending)) + "."}]
        for v in VARIANTS:
            h = rec["hints"].get(v)
            rec.setdefault("metrics", {})[v] = {
                "words": len(h.split()) if h else None,
                "grounding": grounding_rate(h, pre) if h else None,
                "fwd_leak": fwd_leak(h, fut, pre) if h else None,
                "ungrounded_tokens": ungrounded(h, pre) if h else None,
            }
        append_jsonl(HINTS, rec)

    sem = asyncio.Semaphore(16)
    n_done = 0

    async def guarded(t: dict) -> None:
        nonlocal n_done
        async with sem:
            await one(t)
        n_done += 1
        if n_done % 16 == 0 or n_done == len(todo):
            log_cost("hints", engy, f"{n_done}/{len(todo)}")
            if engy_fb.cost_usd:
                log_cost("hints_fb", engy_fb, f"{n_done}/{len(todo)}")

    await asyncio.gather(*[guarded(t) for t in todo])


def cmd_hints(args: argparse.Namespace) -> None:
    turns = load_turns(args.turns)
    asyncio.run(gen_hints(turns, Engy(concurrency=16, retries=3), Engy(concurrency=16)))
    hs = read_jsonl(HINTS)
    for v in VARIANTS:
        ok = sum(1 for h in hs if h["hints"].get(v))
        rej = collections.Counter(a["reject"].get(v) for h in hs for a in h["attempts"] if a.get("reject", {}).get(v))
        print(f"{v}: ok {ok}/{len(hs)} rejections {dict(rej)}")


# ------------------------------------------------------------------ stage: sample
def with_note(prefix: list[dict], p: str | None) -> list[dict]:
    if not p:
        return prefix
    msgs = [dict(m) for m in prefix]
    msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + NOTE_HEAD + p
    return msgs


def sample_jobs(turns: list[dict], hints: dict[str, dict], samplers: list[str]) -> list[tuple]:
    done = {(r["turn_id"], r["sampler"], r["variant"], r["i"]) for r in read_jsonl(SAMPLES) if "error" not in r}
    jobs = []
    for t in turns:
        h = hints.get(t["turn_id"]) or {}
        for s in samplers:
            variants = list(VARIANTS) + (list(KING_BLIND) if s == "king" else [])
            for v in variants:
                p = None if v in KING_BLIND else h.get("hints", {}).get(v)
                if v not in KING_BLIND and not p:
                    continue
                for i in range(K):
                    if (t["turn_id"], s, v, i) not in done:
                        jobs.append((t, s, v, i, p))
    return jobs


async def sample_all(turns: list[dict], hints: dict[str, dict], samplers: list[str],
                     engy: Engy, king: King | None) -> None:
    jobs = sample_jobs(turns, hints, samplers)
    print(f"sampling {len(jobs)} refs: " + str(collections.Counter(j[1] for j in jobs)))
    n_done = 0
    t0 = time.time()

    async def one(t, s, v, i, p):
        nonlocal n_done
        msgs = with_note(t["prefix"], p)
        cli = king if s == "king" else engy
        model = KING_MODEL if s == "king" else TEACHER_ENGY
        try:
            r = await cli.chat(model, msgs, temperature=REF_TEMPERATURE, max_tokens=REF_MAX_TOKENS)
        except Exception as ex:  # noqa: BLE001
            append_jsonl(SAMPLES, {"turn_id": t["turn_id"], "sampler": s, "variant": v, "i": i, "error": repr(ex)[:300]})
            n_done += 1
            return
        parsed = parse_reply(r, t["dialect"])
        if s == "king" and not r.get("think_closed", True):     # raw path: no </think> = unclosed thought = forfeit
            parsed.update({"z": r["raw"], "y": "", "parsed": False, "kind_used": None, "think_closed": False})
        append_jsonl(SAMPLES, {"turn_id": t["turn_id"], "sampler": s, "variant": v, "i": i, "note": p, "model": r["model"],
                               "reasoning": r["reasoning"], "content": r["content"], "tool_calls": r["tool_calls"],
                               "finish": r["finish"], "usage": r["usage"], "cost_usd": r["cost_usd"],
                               **{k: parsed[k] for k in ("z", "y", "parsed", "kind_used", "think_closed", "repaired")}})
        n_done += 1
        if n_done % 50 == 0:
            el = time.time() - t0
            print(f"  sample {n_done}/{len(jobs)} {el/60:.1f} min, eta {(len(jobs)-n_done)*el/n_done/60:.1f} min", flush=True)
            log_cost("sample", engy, f"{n_done}/{len(jobs)}")

    # king jobs interleaved with teacher jobs; both clients bound their own concurrency
    await asyncio.gather(*[one(*j) for j in jobs])
    log_cost("sample", engy, f"{len(jobs)}/{len(jobs)}")


def cmd_sample(args: argparse.Namespace) -> None:
    turns = load_turns(args.turns)
    hints = {h["turn_id"]: h for h in read_jsonl(HINTS)}
    samplers = list(SAMPLERS) if args.sampler == "both" else [args.sampler]
    king = None
    if "king" in samplers:
        t0 = time.time()
        while not asyncio.run(king_reachable()):
            if time.time() - t0 > args.king_wait_min * 60:
                print(f"king endpoint {KING_BASE} unreachable for {args.king_wait_min} min — continuing with the teacher only", flush=True)
                samplers = [s for s in samplers if s != "king"]
                break
            print("king endpoint unreachable, retrying in 60 s", flush=True)
            time.sleep(60)
        else:
            king = King(concurrency=args.king_concurrency)
    asyncio.run(sample_all(turns, hints, samplers, Engy(concurrency=args.concurrency), king))


# ------------------------------------------------------------------ stage: judge
def ref_sets(tid: str, samples: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """(sampler, variant) -> parsed refs sorted by i."""
    out: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for s in samples:
        if s["turn_id"] == tid and s.get("parsed"):
            out[(s["sampler"], s["variant"])].append({"i": s["i"], "z": s["z"], "y": s["y"]})
    for k in out:
        out[k].sort(key=lambda r: r["i"])
    return out


def blind_actions(t: dict, sampler: str, sets: dict) -> list[str]:
    if sampler == "teacher":
        return [r["y"] for r in t["refs"]]
    return [r["y"] for r in sets.get(("king", "blind"), [])]


async def judge_all(turns: list[dict], engy: Engy, n_turns: int) -> None:
    samples = [s for s in read_jsonl(SAMPLES) if s.get("parsed")]
    done = {(r["turn_id"], r["sampler"], r["variant"], r["pair"]) for r in read_jsonl(JUDGE) if r.get("label")}
    jobs = []
    for t in turns[:n_turns]:
        tid = t["turn_id"]
        sets = ref_sets(tid, samples)
        tail = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in t["prefix"][-3:])
        if len(tail) > 12_000:
            tail = tail[-12_000:]
        yk = t["king"]["pairs"][0]["y_a"]
        for s in SAMPLERS:
            blind = blind_actions(t, s, sets)
            cells = list(VARIANTS) + (list(KING_BLIND) if s == "king" else [])
            for v in cells:
                refs = sets.get((s, v), [])
                for pi in range(2):
                    if (tid, s, v, pi) in done:
                        continue
                    if v == "blind":                       # king noise floor (1-sample): fresh blind king ref vs y_K
                        if pi < len(refs):
                            jobs.append((tid, s, v, pi, tail, yk, refs[pi]["y"]))
                    elif pi < len(refs) and pi < len(blind):   # blind2 vs blind = the king's 3-vs-3 floor; informed vs blind otherwise
                        jobs.append((tid, s, v, pi, tail, blind[pi], refs[pi]["y"]))
    print(f"judge: {len(jobs)} comparisons")

    async def one(tid, s, v, pi, tail, ya, yb):
        user = (f"Recent transcript context:\n{tail}\n\n--- Candidate action A (reference) ---\n{ya[:3000]}\n\n"
                f"--- Candidate action B (alternative) ---\n{yb[:3000]}\n\nOne word: SAME, EXPLORE, ACT_FINISH or DIFFERENT_FIX?")
        try:
            r = await engy.chat(JUDGE_MODEL, [{"role": "system", "content": JUDGE_SYSTEM},
                                              {"role": "user", "content": user}], temperature=0.0, max_tokens=2000)
            txt = (r["content"] or "").strip().upper().replace("-", "_").replace(" ", "_")
            label = next((l for l in LABELS if txt.startswith(l)), None)
            if label is None:
                words = re.findall(r"\b(SAME|EXPLORE|ACT_FINISH|DIFFERENT_FIX)\b",
                                   ((r["content"] or "") + " " + (r["reasoning"] or "")).upper().replace("-", "_"))
                label = words[-1] if words else None
            append_jsonl(JUDGE, {"turn_id": tid, "sampler": s, "variant": v, "pair": pi, "label": label,
                                 "same": (label == "SAME") if label else None, "cost_usd": r["cost_usd"]})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(JUDGE, {"turn_id": tid, "sampler": s, "variant": v, "pair": pi, "label": None, "same": None,
                                 "error": repr(ex)[:200]})

    for s in range(0, len(jobs), 32):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + 32]])
        log_cost("judge", engy, f"{min(s + 32, len(jobs))}/{len(jobs)}")


def cmd_judge(args: argparse.Namespace) -> None:
    asyncio.run(judge_all(load_turns(args.turns), Engy(concurrency=16), args.n))


# ------------------------------------------------------------------ stage: echo
def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


CELL_TAG = {("teacher", "h_ground"): "tg", ("teacher", "h_causal"): "tc", ("teacher", "h_answer"): "ta",
            ("king", "h_ground"): "kg", ("king", "h_causal"): "kc", ("king", "h_answer"): "ka",
            ("king", "blind"): "kb"}


def candidates(t: dict, sets: dict, tools: list[dict]) -> dict[str, str]:
    c = {"king": t["king"]["pairs"][0]["y_a"]}
    for i, r in enumerate(t["refs"]):
        c[f"bo{i}"] = r["y"]
    for cell, tag in CELL_TAG.items():
        for r in sets.get(cell, []):
            c[f"{tag}o{r['i']}"] = r["y"]
    g = generic_action(t, tools)
    if g:
        c["gen"] = g
    rl = repeat_last(t)
    if rl:
        c["rep"] = rl
    return c


def contexts(t: dict, sets: dict) -> dict[str, str]:
    c = {"e": ""}
    for cell, tag in CELL_TAG.items():
        for r in sets.get(cell, []):
            c[f"{tag}z{r['i']}"] = r["z"]
    return c


def grid(cands: dict[str, str], ctxs: dict[str, str]) -> list[tuple[str, str]]:
    """(cand, ctx) pairs: every candidate under ∅; under an informed / blind-king
    thought set only the shared candidates + that set's own actions."""
    shared = [c for c in cands if c == "king" or c.startswith("bo") or c.startswith("kbo") or c in ("gen", "rep")]
    out = [(c, "e") for c in cands]
    for xid in ctxs:
        if xid == "e":
            continue
        tag = xid[:2]
        own = [c for c in cands if c.startswith(f"{tag}o")]
        for c in dict.fromkeys(shared + own):
            out.append((c, xid))
    return out


def read_echoes() -> dict[str, dict]:
    out = {}
    for p in sorted(RESULTS.glob("echoes*.jsonl")):
        for r in read_jsonl(p):
            if "error" not in r:
                out[r["key"]] = r
    return out


# importing hindsight_refs already installs its per-prefix memo on common.gen_prompt
assert common.gen_prompt.__name__ == "_gen_prompt_memo"


async def echo_all(turns: list[dict], engy: Engy, start: int) -> None:
    te = TeacherEcho(engy)
    samples = read_jsonl(SAMPLES)
    done = set(read_echoes())
    out_path = ECHOES if start == 0 else RESULTS / f"echoes.part{start}.jsonl"
    tidx = trace_index()
    jobs = []
    for t in turns:
        tid = t["turn_id"]
        sets = ref_sets(tid, samples)
        tools = load_envelope(*tidx[t["rollout_id"]])["trace"].get("tools") or []
        cands, ctxs = candidates(t, sets, tools), contexts(t, sets)
        seen = set()
        for cid, xid in grid(cands, ctxs):
            key = f"{tid}|{h(ctxs[xid])}|{h(cands[cid])}"
            if key in done or key in seen:
                continue
            seen.add(key)
            jobs.append((key, t["prefix"], ctxs[xid], cands[cid], cid, xid))
    print(f"echo: {len(jobs)} action echoes to run ({len(jobs)/max(1,len(turns)):.0f}/turn)")

    async def one(key, prefix, z, y, cid, xid):
        try:
            r = await te.lp_action(prefix, z, y)
            append_jsonl(out_path, {"key": key, "cand": cid, "ctx": xid, **r})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(out_path, {"key": key, "cand": cid, "ctx": xid, "error": repr(ex)[:300]})

    step = 96
    for s in range(0, len(jobs), step):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + step]])
        log_cost("echo", engy, f"{min(s + step, len(jobs))}/{len(jobs)}")


def cmd_echo(args: argparse.Namespace) -> None:
    turns = load_turns(args.turns)[args.start:]
    turns = turns[: args.n] if args.n else turns
    asyncio.run(echo_all(turns, Engy(concurrency=args.concurrency), args.start))


# ------------------------------------------------------------------ stage: report
def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return st.mean(v) if v else None


def _paired(diffs: list) -> dict:
    d = [x for x in diffs if x is not None and math.isfinite(x)]
    if len(d) < 3:
        return {"n": len(d), "mean": _mean(d), "se": None, "z": None}
    se = st.stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "mean": st.mean(d), "se": se, "z": (st.mean(d) / se) if se > 0 else None,
            "frac_pos": sum(1 for x in d if x > 0) / len(d)}


def _f(x, w=6, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    if isinstance(x, dict):
        return _f(x.get("z"), w, 2)
    return f"{x:{w}.{p}f}"


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def _spearman(xs, ys):
    return _corr(_rank(xs), _rank(ys))


class TurnEchoes:
    def __init__(self, tid: str, cands: dict[str, str], ctxs: dict[str, str], echoes: dict[str, dict]):
        self.tid, self.cands, self.ctxs, self.e = tid, cands, ctxs, echoes

    def lp(self, cid: str, xid: str) -> dict | None:
        return self.e.get(f"{self.tid}|{h(self.ctxs[xid])}|{h(self.cands[cid])}")

    def lift(self, cid: str, xid: str) -> float | None:
        a, e = self.lp(cid, xid), self.lp(cid, "e")
        if a is None or e is None:
            return None
        return (a["lp_per_byte"] - e["lp_per_byte"]) * a["n_bytes"]

    def A(self, cid: str, thought_ids: list[str], exclude: str | None) -> float | None:
        ids = [x for x in thought_ids if x != exclude]
        if exclude is not None:
            b = [self.lift(cid, x) for x in ids]
            return lme(b, TAU) if b and all(v is not None for v in b) else None
        vals = []
        for j in thought_ids:
            b = [self.lift(cid, x) for x in thought_ids if x != j]
            if not b or any(v is None for v in b):
                return None
            vals.append(lme(b, TAU))
        return st.mean(vals)


def change_metrics(ys: list[str], blind: list[str], kind: str) -> dict:
    if not ys or not blind:
        return {"change": None, "turn_changed": None, "any_changed": None, "exact": None, "jac": None}
    ch = [1.0 if (not exact(y, blind, kind) and agree(y, blind) < 0.5) else 0.0 for y in ys]
    return {"change": st.mean(ch), "turn_changed": sum(ch) >= 2, "any_changed": sum(ch) >= 1,
            "exact": st.mean(1.0 if exact(y, blind, kind) else 0.0 for y in ys),
            "jac": st.mean(agree(y, blind) for y in ys)}


def cmd_report(args: argparse.Namespace) -> None:
    turns = load_turns(args.turns)
    hints = {x["turn_id"]: x for x in read_jsonl(HINTS)}
    samples_all = read_jsonl(SAMPLES)
    samples = [s for s in samples_all if "error" not in s]
    echoes = read_echoes()
    echo_errors = sum(1 for p in RESULTS.glob("echoes*.jsonl") for r in read_jsonl(p) if "error" in r)
    judge = [j for j in read_jsonl(JUDGE) if j.get("label")]
    d7_ctrl = collections.defaultdict(list)
    for s in read_jsonl(D7_SAMPLES):
        if "error" not in s and s.get("variant") == "p_none" and s.get("parsed"):
            d7_ctrl[s["turn_id"]].append(s["y"])
    d7_judge = read_jsonl(D7_JUDGE)
    tidx = trace_index()

    rows = []
    for t in turns:
        tid, kind = t["turn_id"], t["dialect"]
        sets = ref_sets(tid, samples)
        env = load_envelope(*tidx[t["rollout_id"]])
        tools = env["trace"].get("tools") or []
        cands, ctxs = candidates(t, sets, tools), contexts(t, sets)
        E = TurnEchoes(tid, cands, ctxs, echoes)
        yk = t["king"]["pairs"][0]["y_a"]
        blind_t = [r["y"] for r in t["refs"]]
        blind_k = [r["y"] for r in sets.get(("king", "blind"), [])]
        hrec = hints.get(tid) or {"hints": {}, "metrics": {}, "model": {}, "attempts": []}
        fp = forward_part(env["trace"], t["turn_idx"])
        pre, fut = prefix_text(t), future_text(fp)
        first = (hrec.get("attempts") or [{}])[0].get("reject") or {}
        row = {"turn_id": tid, "dialect": kind, "outcome": t["down"]["outcome"], "harness": t["harness"],
               "hints": {}, "cells": {}, "floor": {}}
        for v in VARIANTS:
            hv = hrec["hints"].get(v)
            row["hints"][v] = {"ok": bool(hv), "model": hrec["model"].get(v), "first_reject": first.get(v),
                               **(hrec.get("metrics", {}).get(v) or {}),
                               "fwd_leak20_windows": fwd_leak_windows(hv, fut, pre, 20) if hv else [],
                               "fwd_leak30": fwd_leak(hv, fut, pre, 30) if hv else None}
        # ---- noise floors
        row["floor"]["teacher"] = change_metrics(d7_ctrl.get(tid, []), blind_t, kind)              # fresh blind teacher vs stored blind (3 vs 3)
        row["floor"]["king"] = change_metrics([r["y"] for r in sets.get(("king", "blind2"), [])], blind_k, kind)   # blind2 vs blind (3 vs 3)
        row["floor"]["king_yK"] = change_metrics(blind_k, [yk], kind)                               # fresh blind king vs the single stored y_K
        row["floor"]["king_vs_teacher"] = change_metrics(blind_k, blind_t, kind)                    # blind king vs blind teacher
        row["floor"]["yK_vs_teacher"] = change_metrics([yk], blind_t, kind)
        # ---- stored A_blind (vLLM echoes of the verdict), summed nats
        bk = [(p["lpC_ya_zc"] - p["lpC_ya_e"]) * p["n_bytes_ya"] for p in t["king"]["pairs"]]
        row["A_blind_king_stored"] = st.mean(lme([b for m, b in enumerate(bk) if m != j], TAU) for j in range(K)) if len(bk) == K else None
        abo = []
        for j in range(K):
            b = [(t["refs"][j]["lp_cross"][i] - t["refs"][j]["lp_empty"]) * t["refs"][j]["n_bytes_y"] for i in range(K) if i != j]
            abo.append(lme(b, TAU))
        row["A_blind_bown_stored"] = st.mean(abo)
        row["A_blind_bown_stored_vals"] = abo
        # ---- blind KING thought set (Engy): A_blind_king(y)
        KB = [f"kbz{r['i']}" for r in sets.get(("king", "blind"), [])]
        row["kb"] = {"k": len(KB)}
        if len(KB) >= 2:
            own = [E.A(f"kbo{r['i']}", KB, f"kbz{r['i']}") for r in sets[("king", "blind")]]
            row["kb"]["A_own"] = _mean(own) if all(v is not None for v in own) else None
            row["kb"]["A_own_vals"] = own
            for c in ("king", "gen", "rep"):
                row["kb"][f"A_{c}"] = E.A(c, KB, None) if c in cands else None
            bo = [E.A(f"bo{j}", KB, None) for j in range(K)]
            row["kb"]["A_bown"] = _mean(bo) if all(v is not None for v in bo) else None
            row["kb"]["A_bown_vals"] = bo
        # ---- informed cells
        for (s, v), tag in CELL_TAG.items():
            if v == "blind":
                continue
            refs = sets.get((s, v), [])
            raw = [x for x in samples if x["turn_id"] == tid and x["sampler"] == s and x["variant"] == v]
            if not raw:
                continue
            p = raw[0].get("note") or ""
            ys = [r["y"] for r in refs]
            blind = blind_t if s == "teacher" else blind_k
            cell = {"n_sampled": len(raw), "n_parsed": len(refs),
                    "cap_hits": sum(1 for x in raw if x["finish"] == "length"),
                    "think_closed": sum(1 for x in raw if x.get("think_closed")),
                    "len_z": [len(r["z"]) for r in refs], "len_y": [len(r["y"]) for r in refs],
                    **change_metrics(ys, blind, kind)}
            # vs the OTHER sampler's blind set, and vs y_K
            cell["vs_yK"] = change_metrics(ys, [yk], kind)
            cell["vs_blind_teacher"] = change_metrics(ys, blind_t, kind)
            cell["leak_z"] = _mean([1.0 if leaks(p, r["z"]) else 0.0 for r in refs])
            cell["leak_y"] = _mean([1.0 if leaks(p, r["y"]) else 0.0 for r in refs])
            if len(ys) >= 2:
                cell["self_exact"] = st.mean(1.0 if norm_action(ys[a], kind) == norm_action(ys[b], kind) else 0.0
                                             for a in range(len(ys)) for b in range(a + 1, len(ys)))
            Z = [f"{tag}z{r['i']}" for r in refs]
            cell["k"] = len(Z)
            if len(Z) >= 2:
                own = [E.A(f"{tag}o{r['i']}", Z, f"{tag}z{r['i']}") for r in refs]
                L = {"own_vals": own, "own": _mean(own) if all(v is not None for v in own) else None}
                bo = [E.A(f"bo{j}", Z, None) for j in range(K)]
                L["bown_vals"] = bo
                L["bown"] = _mean(bo) if all(v is not None for v in bo) else None
                L["bown_max"] = max(bo) if all(v is not None for v in bo) else None
                kbo = [E.A(f"kbo{r['i']}", Z, None) for r in sets.get(("king", "blind"), [])]
                L["kbown"] = _mean(kbo) if kbo and all(v is not None for v in kbo) else None
                for c in ("king", "gen", "rep"):
                    L[c] = E.A(c, Z, None) if c in cands else None
                B = [E.lift(f"{tag}o{r['i']}", f"{tag}z{r['i']}") for r in refs]
                nb = [E.lp(f"{tag}o{r['i']}", "e") for r in refs]
                L["B_pass"] = _mean([1.0 if (b is not None and x is not None and b / x["n_bytes"] >= 0.02) else 0.0
                                     for b, x in zip(B, nb)]) if all(b is not None for b in B) else None
                cell["A"] = L
            row["cells"][f"{s}:{v}"] = cell
        rows.append(row)
    write_jsonl(RESULTS / "turn_metrics.jsonl", rows)

    # ------------------------------------------------------------ aggregate
    def agg(rs: list[dict]) -> dict:
        out: dict = {"n_turns": len(rs)}
        # hints
        out["hints"] = {}
        for v in VARIANTS:
            hs = [r["hints"][v] for r in rs]
            ok = [x for x in hs if x["ok"]]
            out["hints"][v] = {"ok": len(ok), "n": len(hs), "words_mean": _mean([x["words"] for x in ok]),
                               "words_p50": st.median([x["words"] for x in ok]) if ok else None,
                               "le60": _mean([1.0 if x["words"] <= 60 else 0.0 for x in ok]),
                               "grounding": _mean([x["grounding"] for x in ok]),
                               "fwd_leak": _mean([1.0 if x["fwd_leak"] else 0.0 for x in ok]),
                               "fwd_leak30": _mean([1.0 if x["fwd_leak30"] else 0.0 for x in ok]),
                               "first_reject": _mean([1.0 if x.get("first_reject") else 0.0 for x in hs]),
                               "first_reject_kinds": dict(collections.Counter(x.get("first_reject") for x in hs if x.get("first_reject"))),
                               "leak_examples": [w for x in ok for w in x["fwd_leak20_windows"][:1]][:6],
                               "models": dict(collections.Counter(x["model"] for x in ok))}
        # floors
        out["floor"] = {}
        for fk in ("teacher", "king", "king_yK", "king_vs_teacher", "yK_vs_teacher"):
            fl = [r["floor"][fk] for r in rs if r["floor"][fk]["change"] is not None]
            out["floor"][fk] = {"n": len(fl), "change": _mean([x["change"] for x in fl]),
                                "turn_changed": _mean([1.0 if x["turn_changed"] else 0.0 for x in fl]),
                                "any_changed": _mean([1.0 if x["any_changed"] else 0.0 for x in fl]),
                                "exact": _mean([x["exact"] for x in fl]), "jac": _mean([x["jac"] for x in fl])}
        tids = {r["turn_id"] for r in rs}
        jj = [j for j in d7_judge if j["turn_id"] in tids and j.get("same") is not None and j["variant"] == "p_none"]
        out["floor"]["teacher"]["judge_same"] = st.mean(1.0 if j["same"] else 0.0 for j in jj) if jj else None
        out["floor"]["teacher"]["judge_n"] = len(jj)
        for fk, v in (("king", "blind2"), ("king_yK", "blind")):
            jk = [j for j in judge if j["turn_id"] in tids and j["sampler"] == "king" and j["variant"] == v]
            out["floor"][fk]["judge_same"] = st.mean(1.0 if j["same"] else 0.0 for j in jk) if jk else None
            out["floor"][fk]["judge_n"] = len(jk)
        # cells
        out["cells"] = {}
        for (s, v), tag in CELL_TAG.items():
            if v == "blind":
                continue
            key = f"{s}:{v}"
            cs = [(r, r["cells"][key]) for r in rs if key in r["cells"]]
            if not cs:
                continue
            g: dict = {"n_turns": len(cs)}
            ns = max(1, sum(c["n_sampled"] for _, c in cs))
            g["parse_rate"] = sum(c["n_parsed"] for _, c in cs) / ns
            g["cap_rate"] = sum(c["cap_hits"] for _, c in cs) / ns
            g["think_close_rate"] = sum(c["think_closed"] for _, c in cs) / ns
            g["yield2"] = _mean([1.0 if c["n_parsed"] >= 2 else 0.0 for _, c in cs])
            g["len_z_p50"] = st.median([l for _, c in cs for l in c["len_z"]]) if any(c["len_z"] for _, c in cs) else None
            g["len_y_p50"] = st.median([l for _, c in cs for l in c["len_y"]]) if any(c["len_y"] for _, c in cs) else None
            for m in ("change", "exact", "jac", "leak_z", "leak_y", "self_exact"):
                g[m] = _mean([c.get(m) for _, c in cs])
            g["turn_changed"] = _mean([1.0 if c["turn_changed"] else 0.0 for _, c in cs if c["turn_changed"] is not None])
            g["any_changed"] = _mean([1.0 if c["any_changed"] else 0.0 for _, c in cs if c["any_changed"] is not None])
            g["change_vs_yK"] = _mean([c["vs_yK"]["change"] for _, c in cs])
            g["change_vs_blind_teacher"] = _mean([c["vs_blind_teacher"]["change"] for _, c in cs])
            fl = "teacher" if s == "teacher" else "king"
            g["change_up_vs_floor"] = _paired([c["change"] - r["floor"][fl]["change"] for r, c in cs
                                               if c["change"] is not None and r["floor"][fl]["change"] is not None])
            jj = [j for j in judge if j["turn_id"] in tids and j["sampler"] == s and j["variant"] == v]
            g["judge_n"] = len(jj)
            g["judge_same"] = st.mean(1.0 if j["label"] == "SAME" else 0.0 for j in jj) if jj else None
            g["judge_taxonomy"] = dict(collections.Counter(j["label"] for j in jj))
            # decision filter x outcome
            ch = [(r["outcome"], c["turn_changed"]) for r, c in cs if c["turn_changed"] is not None]
            n_ch = sum(1 for _, x in ch if x)
            g["filter"] = {"n": len(ch), "changed": n_ch,
                           "p_failed_given_changed": (sum(1 for o, x in ch if x and o == "failed") / n_ch) if n_ch else None,
                           "p_changed_given_failed": _mean([1.0 if x else 0.0 for o, x in ch if o == "failed"]),
                           "p_changed_given_solved": _mean([1.0 if x else 0.0 for o, x in ch if o == "solved"])}
            # A_priv headroom — complete turns only (own, blind-own and y_K all echoed under the cell's thoughts)
            La = [(r, c["A"]) for r, c in cs if c.get("A") and all(c["A"].get(k) is not None for k in ("own", "bown", "king"))]
            A: dict = {"n": len(La)}
            for cnd in ("own", "bown", "kbown", "king", "gen", "rep"):
                A[cnd] = _mean([L.get(cnd) for _, L in La])
            A["B_pass"] = _mean([L.get("B_pass") for _, L in La])

            def pz(f):
                return _paired([f(r, L) for r, L in La])

            def d(a, b):
                return lambda r, L: (L[a] - L[b]) if (L.get(a) is not None and L.get(b) is not None) else None
            A["hz_own_bown"] = pz(d("own", "bown"))
            A["hz_bown_kbown"] = pz(d("bown", "kbown"))
            A["hz_kbown_king"] = pz(d("kbown", "king"))
            A["hz_bown_king"] = pz(d("bown", "king"))
            A["hz_own_king"] = pz(d("own", "king"))
            A["hz_king_gen"] = pz(d("king", "gen"))
            A["hz_king_rep"] = pz(d("king", "rep"))
            A["hz_bown_gen"] = pz(d("bown", "gen"))
            A["hz_own_kbown"] = pz(d("own", "kbown"))
            # separation teacher-vs-king (bown − king) under stored A_blind on the same turns
            A["hz_bown_king_stored"] = pz(lambda r, L: (r["A_blind_bown_stored"] - r["A_blind_king_stored"])
                                          if r["A_blind_king_stored"] is not None else None)
            A["hz_bown_king_kblind"] = pz(lambda r, L: (r["kb"]["A_bown"] - r["kb"]["A_king"])
                                          if r["kb"].get("A_bown") is not None and r["kb"].get("A_king") is not None else None)
            # incumbency
            A["yK_ge_bown_mean"] = _mean([1.0 if L["king"] >= L["bown"] else 0.0 for _, L in La if L.get("king") is not None and L.get("bown") is not None])
            A["yK_ge_bown_max"] = _mean([1.0 if L["king"] >= L["bown_max"] else 0.0 for _, L in La if L.get("king") is not None and L.get("bown_max") is not None])
            A["yK_ge_bown_mean_stored"] = _mean([1.0 if r["A_blind_king_stored"] >= r["A_blind_bown_stored"] else 0.0 for r, _ in La if r["A_blind_king_stored"] is not None])
            A["yK_ge_bown_mean_kblind"] = _mean([1.0 if r["kb"]["A_king"] >= r["kb"]["A_bown"] else 0.0 for r, _ in La
                                                 if r["kb"].get("A_king") is not None and r["kb"].get("A_bown") is not None])
            # A_priv vs A_blind_king correlation over candidates, split by whether the informed king changed
            if s == "king":
                for split in ("unchanged", "changed", "all"):
                    xs, ys_, per_turn = [], [], []
                    for r, L in La:
                        c = r["cells"][key]
                        if split == "unchanged" and c["turn_changed"] is not False:
                            continue
                        if split == "changed" and c["turn_changed"] is not True:
                            continue
                        kb = r["kb"]
                        pts = [(L[cn], kb.get(f"A_{cn}")) for cn in ("king", "bown", "gen", "rep")
                               if L.get(cn) is not None and kb.get(f"A_{cn}") is not None]
                        if len(pts) >= 3:
                            per_turn.append(_corr([a for a, _ in pts], [b for _, b in pts]))
                        mA = st.mean(a for a, _ in pts) if pts else 0.0
                        mB = st.mean(b for _, b in pts) if pts else 0.0
                        xs += [a - mA for a, _ in pts]
                        ys_ += [b - mB for _, b in pts]
                    A[f"corr_priv_vs_kblind_{split}"] = {"n_turns": len(per_turn), "n_pts": len(xs),
                                                         "pearson_centered": _corr(xs, ys_), "spearman_centered": _spearman(xs, ys_) if len(xs) >= 3 else None,
                                                         "per_turn_mean": _mean(per_turn)}
                    # raw (uncentered) correlation of the king's own action value
                    kk = [(L["king"], r["kb"]["A_king"]) for r, L in La
                          if L.get("king") is not None and r["kb"].get("A_king") is not None
                          and (split == "all" or (split == "unchanged") == (r["cells"][key]["turn_changed"] is False))]
                    A[f"corr_priv_vs_kblind_{split}"]["pearson_yK_raw"] = _corr([a for a, _ in kk], [b for _, b in kk])
                    A[f"corr_priv_vs_kblind_{split}"]["n_yK"] = len(kk)
            g["A"] = A
            out["cells"][key] = g
        # blind king set summary
        kb = [r["kb"] for r in rs if r["kb"].get("A_own") is not None]
        out["kb"] = {"n": len(kb), "A_own": _mean([x["A_own"] for x in kb]), "A_bown": _mean([x.get("A_bown") for x in kb]),
                     "A_king": _mean([x.get("A_king") for x in kb]), "A_gen": _mean([x.get("A_gen") for x in kb]), "A_rep": _mean([x.get("A_rep") for x in kb]),
                     "hz_bown_king": _paired([x["A_bown"] - x["A_king"] for x in kb if x.get("A_bown") is not None and x.get("A_king") is not None]),
                     "hz_king_own": _paired([x["A_king"] - x["A_own"] for x in kb if x.get("A_king") is not None]),
                     "hz_own_bown": _paired([x["A_own"] - x["A_bown"] for x in kb if x.get("A_bown") is not None]),
                     "A_blind_bown_stored": _mean([r["A_blind_bown_stored"] for r in rs]),
                     "A_blind_king_stored": _mean([r["A_blind_king_stored"] for r in rs]),
                     "hz_bown_king_stored": _paired([r["A_blind_bown_stored"] - r["A_blind_king_stored"] for r in rs if r["A_blind_king_stored"] is not None])}
        return out

    rep = {"n_turns": len(rows), "dialects": dict(collections.Counter(r["dialect"] for r in rows)),
           "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
           "sample_errors": len([s for s in samples_all if "error" in s]), "echo_errors": echo_errors, "n_echoes": len(echoes),
           "cost_usd_total": total_cost(), "king": {"base": KING_BASE, "model": KING_MODEL, "reign": KING_REIGN}}
    hs = list(hints.values())
    att = [a for x in hs for a in x["attempts"]]
    rep["hint_gen"] = {"n": len(hs), "attempts": len(att), "errors": sum(1 for a in att if a.get("error")),
                       "rejections": {v: dict(collections.Counter(a["reject"].get(v) for a in att if a.get("reject", {}).get(v))) for v in VARIANTS},
                       "usd_per_turn": sum((a.get("cost_usd") or 0) for a in att) / max(1, len(hs)),
                       "prompt_tok_mean": _mean([(a.get("usage") or {}).get("prompt_tokens") for a in att if a.get("usage")])}
    rep["pooled"] = agg(rows)
    rep["by_outcome"] = {k: agg([r for r in rows if r["outcome"] == k]) for k in ("solved", "failed")}
    rep["by_dialect"] = {k: agg([r for r in rows if r["dialect"] == k]) for k in ("bash", "tool_call", "terminus_json")}
    (RESULTS / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print_report(rep)


def print_report(rep: dict) -> None:
    L: list[str] = []
    P = L.append
    P("Hint design + informed teacher vs informed king as A-leg reference sampler — probe W2 report")
    P(f"turns {rep['n_turns']} {rep['dialects']} outcomes {rep['outcomes']}; sample errors {rep['sample_errors']}, echo errors {rep['echo_errors']}, "
      f"echoes {rep['n_echoes']}; king {rep['king']['model']} (reign {rep['king']['reign']}) at {rep['king']['base']}; $ spent (Engy, all stages) {rep['cost_usd_total']:.2f}")
    hg = rep["hint_gen"]
    P(f"hint generation: {hg['n']} turns, {hg['attempts']} glm calls ({hg['errors']} errors), ${hg['usd_per_turn']:.4f}/turn, prompt tok mean {_f(hg['prompt_tok_mean'],7,0)}; "
      f"rejections per attempt {hg['rejections']}")
    P("")
    P("Terms: ok = hint passed validation (h_ground: every identifier/path/number/quoted string occurs in the prefix; h_causal: no path/filename; all: no code/command/flag/backtick, <= 70 words hard);")
    P("       words = hint length; <=60 = share within the 60-word spec; ground = share of hint word tokens (>=3 chars) found in the prefix; fwdleak = a >=20-char window of the hint occurs in the FUTURE trajectory but not in the prefix;")
    P("       change = share of informed refs norm-different from ALL blind refs of the SAME sampler AND Jaccard < .5 (teacher blind = 3 stored refs; king blind = 3 fresh blind king refs);")
    P("       turnchg = >= 2/3 refs changed (the decision-state filter); any = >= 1/3; judge = glm-5.3-flash SAME rate (ref i vs blind ref i, i=0,1; first 40 turns); floors: teacher = D7 fresh blind (3) vs stored blind (3); king (3v3) = second fresh blind king set vs first (the like-for-like floor); king(vs yK) = fresh blind king vs the single stored y_K (3v1, inflated);")
    P("       CAVEAT own (LOO): the 3 informed refs read the SAME hint, so a sibling thought predicts 'own' far better than any blind action — own-b overstates the headroom a blind miner could reach; b-yK / kb-yK are the miner-reachable separations;")
    P("       leakZ/leakY = >=30-char hint substring in the ref's thought / action; A = summed action leg (nats) under the cell's 3 informed thoughts: own (LOO), bown = blind teacher actions, kbown = blind king ref actions,")
    P("       yK = stored king action, gen = ls -la, rep = previous action re-issued; hz = paired z over turns; stored = A_blind from the verdict's echoes; kblind = A under the fresh blind KING thoughts.")
    for label, blk in [("POOLED", rep["pooled"])] + [(f"OUTCOME {k}", v) for k, v in rep["by_outcome"].items()] + \
            [(f"DIALECT {k}", v) for k, v in rep["by_dialect"].items()]:
        if not blk["n_turns"]:
            continue
        P("")
        P(f"== {label} (n turns {blk['n_turns']}) ==")
        P("-- (Q1) hints")
        P(f"{'variant':<9} {'ok':>7} {'rej1st':>6} {'words':>5} {'p50':>4} {'<=60':>5} {'ground':>6} {'fwdlk20':>7} {'fwdlk30':>7}  1st-attempt rejections | models")
        for v, g in blk["hints"].items():
            P(f"{v:<9} {g['ok']:>3}/{g['n']:<3} {_f(g['first_reject'],6,2)} {_f(g['words_mean'],5,1)} {_f(g['words_p50'],4,0)} {_f(g['le60'],5,2)} {_f(g['grounding'],6,3)} {_f(g['fwd_leak'],7,3)} {_f(g['fwd_leak30'],7,3)}  {g['first_reject_kinds']} | {g['models']}")
        if label == "POOLED":
            for v, g in blk["hints"].items():
                P(f"   {v} fwd-leak-20 window examples: " + " | ".join(repr(w) for w in g["leak_examples"]))
        P("-- (Q2.1/2/5) hint-responsiveness by variant x sampler (+ noise floors), leak, decision filter")
        P(f"{'cell':<17} {'n':>3} {'parse':>5} {'cap':>5} {'change':>6} {'turnchg':>7} {'any':>5} {'exact':>5} {'jac':>5} {'chg-fl z':>8} {'judge':>10} {'EXPL':>4} {'ACTF':>4} {'DIFF':>4} | {'vs yK':>6} {'vs bT':>6} | {'leakZ':>5} {'leakY':>5} | {'P(f|chg)':>8} {'P(chg|f)':>8} {'P(chg|s)':>8}")
        fl = blk["floor"]
        P(f"{'floor teacher':<17} {fl['teacher']['n']:>3} {'':>5} {'':>5} {_f(fl['teacher']['change'])} {_f(fl['teacher']['turn_changed'],7)} {_f(fl['teacher']['any_changed'],5,2)} {_f(fl['teacher']['exact'],5,2)} {_f(fl['teacher']['jac'],5,2)} {'':>8} {_f(fl['teacher'].get('judge_same'))}/{fl['teacher'].get('judge_n',0):<3}")
        P(f"{'floor king (3v3)':<17} {fl['king']['n']:>3} {'':>5} {'':>5} {_f(fl['king']['change'])} {_f(fl['king']['turn_changed'],7)} {_f(fl['king']['any_changed'],5,2)} {_f(fl['king']['exact'],5,2)} {_f(fl['king']['jac'],5,2)} {'':>8} {_f(fl['king'].get('judge_same'))}/{fl['king'].get('judge_n',0):<3}")
        P(f"{'floor king(vs yK)':<17} {fl['king_yK']['n']:>3} {'':>5} {'':>5} {_f(fl['king_yK']['change'])} {_f(fl['king_yK']['turn_changed'],7)} {_f(fl['king_yK']['any_changed'],5,2)} {_f(fl['king_yK']['exact'],5,2)} {_f(fl['king_yK']['jac'],5,2)} {'':>8} {_f(fl['king_yK'].get('judge_same'))}/{fl['king_yK'].get('judge_n',0):<3}")
        P(f"{'blind king vs bT':<17} {fl['king_vs_teacher']['n']:>3} {'':>5} {'':>5} {_f(fl['king_vs_teacher']['change'])} {_f(fl['king_vs_teacher']['turn_changed'],7)} {_f(fl['king_vs_teacher']['any_changed'],5,2)} {_f(fl['king_vs_teacher']['exact'],5,2)} {_f(fl['king_vs_teacher']['jac'],5,2)}")
        P(f"{'yK vs bT':<17} {fl['yK_vs_teacher']['n']:>3} {'':>5} {'':>5} {_f(fl['yK_vs_teacher']['change'])} {'':>7} {'':>5} {_f(fl['yK_vs_teacher']['exact'],5,2)} {_f(fl['yK_vs_teacher']['jac'],5,2)}")
        for key, g in blk["cells"].items():
            tx = g["judge_taxonomy"]; f_ = g["filter"]
            P(f"{key:<17} {g['n_turns']:>3} {_f(g['parse_rate'],5,2)} {_f(g['cap_rate'],5,2)} {_f(g['change'])} {_f(g['turn_changed'],7)} {_f(g['any_changed'],5,2)} {_f(g['exact'],5,2)} {_f(g['jac'],5,2)} {_f(g['change_up_vs_floor'],8)} "
              f"{_f(g['judge_same'])}/{g['judge_n']:<3} {tx.get('EXPLORE',0):>4} {tx.get('ACT_FINISH',0):>4} {tx.get('DIFFERENT_FIX',0):>4} | {_f(g['change_vs_yK'])} {_f(g['change_vs_blind_teacher'])} | {_f(g['leak_z'],5,2)} {_f(g['leak_y'],5,2)} | "
              f"{_f(f_['p_failed_given_changed'],8)} {_f(f_['p_changed_given_failed'],8)} {_f(f_['p_changed_given_solved'],8)}")
        P("-- (Q2.3) A_priv headroom: candidate actions under the cell's informed thoughts (nats, summed; matched 2-thought estimator)")
        P(f"{'cell':<17} {'n':>3} {'A_own':>7} {'A_bown':>7} {'A_kbown':>7} {'A_yK':>7} {'A_gen':>7} {'A_rep':>7} {'Bpass':>5} | {'own-b':>6} {'b-kb':>6} {'kb-yK':>6} {'b-yK':>6} {'own-yK':>6} {'yK-gen':>6} {'yK-rep':>6} {'own-kb':>6} | {'b-yK stored':>11} {'b-yK kblind':>11} | {'yK>=b':>5} {'yK>=bmax':>8} {'stored':>6} {'kblind':>6}")
        for key, g in blk["cells"].items():
            A = g["A"]
            P(f"{key:<17} {A['n']:>3} {_f(A['own'],7,2)} {_f(A['bown'],7,2)} {_f(A['kbown'],7,2)} {_f(A['king'],7,2)} {_f(A['gen'],7,2)} {_f(A['rep'],7,2)} {_f(A['B_pass'],5,2)} | "
              f"{_f(A['hz_own_bown'])} {_f(A['hz_bown_kbown'])} {_f(A['hz_kbown_king'])} {_f(A['hz_bown_king'])} {_f(A['hz_own_king'])} {_f(A['hz_king_gen'])} {_f(A['hz_king_rep'])} {_f(A['hz_own_kbown'])} | "
              f"{_f(A['hz_bown_king_stored'],11)} {_f(A['hz_bown_king_kblind'],11)} | {_f(A['yK_ge_bown_mean'],5,2)} {_f(A['yK_ge_bown_max'],8,2)} {_f(A['yK_ge_bown_mean_stored'],6,2)} {_f(A['yK_ge_bown_mean_kblind'],6,2)}")
        kb = blk["kb"]
        P(f"blind KING thoughts (A_blind_king, n {kb['n']}): own {_f(kb['A_own'],7,2)} bown {_f(kb['A_bown'],7,2)} yK {_f(kb['A_king'],7,2)} gen {_f(kb['A_gen'],7,2)} rep {_f(kb['A_rep'],7,2)} | hz bown−yK {_f(kb['hz_bown_king'])} hz yK−own {_f(kb['hz_king_own'])} hz own−bown {_f(kb['hz_own_bown'])} | "
          f"stored A_blind(teacher): bown {_f(kb['A_blind_bown_stored'],7,2)} yK {_f(kb['A_blind_king_stored'],7,2)} hz bown−yK {_f(kb['hz_bown_king_stored'])}")
        kcells = {k: g for k, g in blk["cells"].items() if k.startswith("king:")}
        if kcells:
            P("-- (Q2.4) incumbency: corr(A_priv_king(y), A_blind_king(y)) over candidates {yK, bown x3, gen, rep} (turn-centred), split by whether the informed king changed its action")
            P(f"{'cell':<17} {'split':<9} {'turns':>5} {'pts':>4} {'pearson':>7} {'spearman':>8} {'per-turn':>8} {'yK raw r':>8} {'n_yK':>4}")
            for key, g in kcells.items():
                for split in ("unchanged", "changed", "all"):
                    c = g["A"].get(f"corr_priv_vs_kblind_{split}")
                    if c:
                        P(f"{key:<17} {split:<9} {c['n_turns']:>5} {c['n_pts']:>4} {_f(c['pearson_centered'],7)} {_f(c['spearman_centered'],8)} {_f(c['per_turn_mean'],8)} {_f(c['pearson_yK_raw'],8)} {c['n_yK']:>4}")
    txt = "\n".join(L) + "\n"
    (RESULTS / "report.txt").write_text(txt)
    print(txt)
    print(f"-> {RESULTS / 'report.txt'} / report.json / turn_metrics.jsonl")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=N_TURNS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("hints"); a.set_defaults(fn=cmd_hints)
    b = sub.add_parser("sample")
    b.add_argument("--sampler", choices=("teacher", "king", "both"), default="both")
    b.add_argument("--concurrency", type=int, default=24)
    b.add_argument("--king-concurrency", type=int, default=12)
    b.add_argument("--king-wait-min", type=float, default=30.0)
    b.set_defaults(fn=cmd_sample)
    c = sub.add_parser("judge"); c.add_argument("--n", type=int, default=40); c.set_defaults(fn=cmd_judge)
    d = sub.add_parser("echo")
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--n", type=int, default=0)
    d.add_argument("--concurrency", type=int, default=24)
    d.set_defaults(fn=cmd_echo)
    e = sub.add_parser("report"); e.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
