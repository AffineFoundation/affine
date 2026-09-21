"""Hindsight-privileged references read by the ACTION leg (design S3 expanded,
store doc §3c, probe W1) — 2026-09-21.

Builds on D7 (privileged_refs.py, its 150 turns + stored blind refs + p_none
fresh-blind refs). New: two HINDSIGHT notes written from the source
trajectory's FORWARD part (what the agent did after this turn + the final
grade), teacher refs sampled with the note, and an A leg that reads the
informed refs' THOUGHTS as echo context:

    b_i(y)   = [lpC(y | x, z_i) − lpC(y | x, ∅)] · bytes(y) / a_norm      (a_norm = 1, nats)
    A_S(y)   = τ · log mean_{i∈S} exp(b_i / τ)                            (τ = 0.03)
    A_blind  = S = the 3 stored blind teacher thoughts;  A_priv = S = the informed thoughts

    python hindsight_refs.py notes                  # p_hind (glm-5.3, 429 -> glm-5.3-flash) + p_err (free)
    python hindsight_refs.py sample                 # k=3 teacher refs per variant (qwen3.8-27b T0.8)
    python hindsight_refs.py judge  [--n 60]        # glm-5.3-flash taxonomy judge (SAME/EXPLORE/ACT_FINISH/DIFFERENT_FIX)
    python hindsight_refs.py echo   [--start N]     # candidate-action x thought-context echo grid
    python hindsight_refs.py report

Terms (one line each):
  x / z / y          turn prefix / thought / action (dialect action span).
  blind refs         the k=3 stored teacher refs of the wvk-22 verdict, ~T(.|x) (thoughts z_C^i, actions y_C^i).
  p_none             D7's fresh blind refs — the sampling-noise floor for action change.
  p_hind             glm-5.3 hindsight post-mortem (<=80 words) written from the forward trajectory + final grade.
  p_err              the first error / failing test / traceback in the next 5 observations (<=600 chars), or the outcome line.
  informed refs      k=3 teacher refs sampled with the note appended to the last user message, ~T(.|x,p).
  informed-own       the informed refs' own actions y^p_j, scored leave-one-out under the other two informed thoughts.
  blind-own          the blind refs' own actions y_C^j, LOO under the other two blind thoughts (A_blind) /
                     under the informed thoughts (A_priv), matched 2-thought estimator.
  king               the stored king rollout (z_a, y_a) of the verdict.
  generic            `ls -la` in the turn's dialect (bash / tool_call bash tool or a zero-arg list tool / terminus_json).
  repeat-last        the previous assistant action in the prefix, re-issued verbatim.
  matched estimator  every external candidate is scored as mean_j LME over the 2 thoughts i != j, like the LOO own value.
  z_A                (A − μ_A) / σ_A, μ_A = the turn's blind-own mean, σ_A = pooled within-turn sd of the 3 blind-own
                     values per dialect (blind-own = 0 by construction; informed-own above 0 = headroom).
  R_blind            centred tempered LME (τ .03, per byte) of a_i = lpC(y_C^i|x,z) − lpC(y_C^i|x,∅) over the blind refs,
                     2-ref LOO-matched; z_R standardised like z_A.
  V4                 min(z_R_blind, z_A_priv) — think like the teacher, act like the informed teacher.
  V5                 A_priv − A_blind — how much more the informed teacher likes the action than the blind one.
  leak               a >=30-char substring of the note appears in the ref's z or y.
  change             ref action norm-exact-different from all 3 blind refs AND best token-Jaccard < 0.5.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (  # noqa: E402
    REPO, TEACHER_ENGY, Engy, TeacherEcho, agree, clme, exact, jaccard, lme,
    norm_action, read_jsonl, append_jsonl, write_jsonl,
)
from privileged_refs import (  # noqa: E402
    K, TAU, REF_TEMPERATURE, REF_MAX_TOKENS, NOTE_HEAD, PATH_RE,
    RESULTS as D7_RESULTS, TURNS as D7_TURNS, SAMPLES as D7_SAMPLES, JUDGE as D7_JUDGE,
    trace_index, load_envelope, render_transcript, with_note, parse_reply, leaks, loo_R,
)
from affine import dialects  # noqa: E402
from affine.corpus.trace import sampled_paths, message_text  # noqa: E402

RESULTS = REPO / "research" / "results" / "frontier_arbiter" / "hindsight"
NOTES = RESULTS / "notes.jsonl"
SAMPLES = RESULTS / "samples.jsonl"
JUDGE = RESULTS / "judge.jsonl"
ECHOES = RESULTS / "echoes.jsonl"
COST = RESULTS / "cost.jsonl"
BUDGET_USD = 18.0

VARIANTS = ("p_hind", "p_err")
FRONTIER = "glm-5.3"
FRONTIER_FALLBACK = "glm-5.3-flash"
JUDGE_MODEL = "glm-5.3-flash"
FWD_CAP = 60_000          # chars of the forward trajectory shown to the note writer
PREFIX_CAP = 100_000
ERR_OBS = 5
ERR_CHARS = 600
MAX_NOTE_WORDS = 100      # "<= 80 words" with 25% slack, like D7's 75 for 60
# a path: starts with / or ./, or has >= 2 slashes, or a slash-joined name with an
# extension, or a bare filename with a code/config extension ("matching/filtering" is prose)
HIND_PATH_RE = re.compile(
    r"(?<![\w])(?:\.{1,2})?/[\w.-]+(?:/[\w.-]+)*|[\w.-]+(?:/[\w.-]+){2,}|[\w-]+/[\w-]+\.\w{1,5}\b|"
    r"\b[\w-]+\.(?:py|js|ts|tsx|go|rs|java|json|toml|yaml|yml|md|txt|sh|c|h|cpp|rb|php|cs|sql|cfg|ini|lock)\b")
ERR_RE = re.compile(
    r"(Traceback \(most recent call last\)|command not found|No such file or directory|"
    r"\bFAILED\b|\bFAIL:|AssertionError|(?<!except )(?<!raise )(?<!import )\b\w*Error:|(?<!except )(?<!raise )\b\w*Exception:|panic:|\berror:|\[ERROR\]|ERROR:|"
    r"[Ss]yntax error|Permission denied|returncode>[1-9]\d*</returncode>|exit code [1-9]|"
    r"\d+ failed|\bfailing\b|Tests? failed|BUILD FAILURE|Segmentation fault)")


# ------------------------------------------------------------------ cost ledger
def log_cost(stage: str, engy: Engy, note: str = "") -> None:
    append_jsonl(COST, {"at": time.time(), "stage": stage, "cost_usd": engy.cost_usd,
                        "usage": engy.usage, "note": note})
    tot = total_cost()
    print(f"  [$] {stage}: this run ${engy.cost_usd:.3f} | total so far ${tot:.2f} {note}", flush=True)
    if tot > BUDGET_USD:
        raise SystemExit(f"budget exceeded: ${tot:.2f} > ${BUDGET_USD}")


def total_cost() -> float:
    """Sum of the max cumulative cost per (stage, run); a cumulative value
    dropping within a group marks a fresh process."""
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


# ------------------------------------------------------------------ forward trajectory
def _key(m: dict) -> tuple:
    return (m.get("role"), m.get("content"))


def forward_part(trace: dict, turn_idx: int) -> dict:
    """The recorded reply at `turn_idx` and everything after it on the longest
    sampled path that extends this turn's prefix (the branch the agent kept)."""
    paths = sampled_paths(trace)
    me = paths[turn_idx]
    n = len(me)
    best = me
    for pj in paths:
        if len(pj) > len(best) and all(_key(a) == _key(b) for a, b in zip(pj[:n], me)):
            best = pj
    fwd = best[n:]
    step = message_text(me[-1].get("content")) or ""
    msgs = [{"role": m.get("role"), "text": message_text(m.get("content")) or ""} for m in fwd]
    rewards = trace.get("rewards") or {}
    solved = rewards.get("solved") or rewards.get("correct") or rewards.get("passed_fraction") or {}
    score = solved.get("score") if isinstance(solved, dict) else solved
    return {"step": step, "fwd": msgs, "rewards": rewards, "metrics": trace.get("metrics") or {},
            "score": score, "stop": trace.get("stop_condition"), "n_fwd": len(msgs),
            "fwd_chars": sum(len(m["text"]) for m in msgs)}


def render_forward(fp: dict, outcome: str, cap: int = FWD_CAP) -> str:
    parts = [f"[ASSISTANT — the step the agent actually took here]\n{fp['step']}"]
    parts += [f"[{m['role'].upper()}]\n{m['text']}" for m in fp["fwd"]]
    s = "\n\n".join(parts)
    if len(s) > cap:
        s = s[: cap // 2] + "\n\n[... middle of the later trajectory elided ...]\n\n" + s[-cap // 2:]
    grade = json.dumps({"rewards": fp["rewards"], "metrics": fp["metrics"], "stop_condition": fp["stop"]})[:600]
    return s + f"\n\n[FINAL GRADE]\nThe task was finally {outcome.upper()}. Grader record: {grade}"


def err_note(fp: dict, outcome: str) -> tuple[str, str]:
    """(note, kind): first error-ish text within the next ERR_OBS observations."""
    obs = [m["text"] for m in fp["fwd"] if m["role"] in ("user", "tool")][:ERR_OBS]
    for o in obs:
        m = ERR_RE.search(o)
        if m:
            start = o.rfind("\n", 0, m.start()) + 1
            snippet = o[start:start + ERR_CHARS].strip()
            return (f"Hindsight: within the next few steps the environment returned this error / failure:\n"
                    f"<<<\n{snippet}\n>>>\nThe trajectory eventually {outcome.upper()} the task."), "error"
    return f"no error appeared later; the trajectory eventually {outcome.upper()} the task", "none"


# ------------------------------------------------------------------ stage: notes
HIND_SYSTEM = (
    "You are writing a private note for an agent at step t of this task. You can see what the agent did "
    "after this step and how the task was finally graded (below). In <= 80 words, tell the agent what "
    "matters at THIS step to end up solving the task: what to verify, what to avoid, what the later "
    "evidence showed. Do NOT give commands, code, file paths or quote the transcript. Plain prose, one "
    "paragraph, no lists, no backticks."
)
HIND_STRICT = (
    "Rejected ({why}; your note had {n_words} words, the hard limit is 80). Rewrite the note: plain prose, "
    "STRICTLY under 80 words, absolutely no commands, code, file paths, filenames, slashes, identifiers or "
    "backticks, and do not copy any phrase from the transcript — describe what matters in your own words."
)


def traj_text(t: dict, fp: dict) -> str:
    s = "\n".join(m["content"] for m in t["prefix"]) + "\n" + fp["step"] + "\n" + "\n".join(m["text"] for m in fp["fwd"])
    return re.sub(r"\s+", " ", s)


def validate_hind(note: str, traj: str) -> str | None:
    if not note or not note.strip():
        return "empty"
    if "`" in note:
        return "backtick"
    if HIND_PATH_RE.search(note):
        return "path"
    if len(note.split()) > MAX_NOTE_WORDS:
        return "too_long"
    if re.search(r"[{}]|\w\(\)|\w\(\w|==|=>|->|\$\(|\$\w|<<|>>|\b\w+\.\w+\(", note):
        return "code_like"
    if leaks(note, traj):
        return "quote"
    return None


async def frontier_chat(engy_main: Engy, engy_fb: Engy, msgs: list[dict]) -> tuple[dict, str]:
    try:
        r = await engy_main.chat(FRONTIER, msgs, temperature=0.0, max_tokens=6000)
        return r, FRONTIER
    except RuntimeError as ex:
        if "429" not in str(ex):
            raise
    r = await engy_fb.chat(FRONTIER_FALLBACK, msgs, temperature=0.0, max_tokens=6000)
    return r, FRONTIER_FALLBACK


async def gen_notes(turns: list[dict], engy: Engy, engy_fb: Engy) -> None:
    done = {r["turn_id"] for r in read_jsonl(NOTES) if r.get("p_err") is not None}
    todo = [t for t in turns if t["turn_id"] not in done]
    print(f"notes: {len(todo)} turns to do")
    tidx = trace_index()

    async def one(t: dict) -> None:
        env = load_envelope(*tidx[t["rollout_id"]])
        fp = forward_part(env["trace"], t["turn_idx"])
        outcome = t["down"]["outcome"]
        rec = {"turn_id": t["turn_id"], "outcome": outcome, "n_fwd": fp["n_fwd"], "fwd_chars": fp["fwd_chars"],
               "attempts": []}
        rec["p_err"], rec["p_err_kind"] = err_note(fp, outcome)
        traj = traj_text(t, fp)
        user = (f"Transcript so far — the agent must now produce its next step (dialect: {t['dialect']}):\n\n"
                f"{render_transcript(t['prefix'], PREFIX_CAP)}\n\n--- END OF TRANSCRIPT SO FAR (step t) ---\n\n"
                f"What the agent did from step t onward, and how it was graded:\n\n{render_forward(fp, outcome)}\n\n"
                f"--- END ---\nWrite the private note for step t now (<= 80 words, no commands/paths/code/backticks, "
                f"no quotes from the transcript).")
        msgs = [{"role": "system", "content": HIND_SYSTEM}, {"role": "user", "content": user}]
        content_attempts = 0
        for attempt in range(3):          # one content retry (stricter prompt); an empty reply (thinking ate the budget) is re-asked as is
            try:
                r, model = await frontier_chat(engy, engy_fb, msgs)
            except Exception as ex:  # noqa: BLE001
                rec["attempts"].append({"error": repr(ex)[:200]})
                break
            note = (r["content"] or "").strip()
            why = validate_hind(note, traj)
            rec["attempts"].append({"note": note, "reject": why, "model": model, "usage": r["usage"],
                                    "cost_usd": r["cost_usd"], "finish": r["finish"],
                                    "reasoning_chars": len(r["reasoning"] or "")})
            if why is None:
                rec["p_hind"] = note
                rec["p_hind_model"] = model
                break
            if why == "empty":
                continue
            content_attempts += 1
            if content_attempts >= 2:
                break
            msgs = msgs + [{"role": "assistant", "content": note},
                           {"role": "user", "content": HIND_STRICT.format(why=why, n_words=len(note.split()))}]
        append_jsonl(NOTES, rec)

    # worker pool (not batched gather): one slow 40k-token post-mortem must not gate 15 others
    sem = asyncio.Semaphore(16)
    n_done = 0

    async def guarded(t: dict) -> None:
        nonlocal n_done
        async with sem:
            await one(t)
        n_done += 1
        if n_done % 16 == 0 or n_done == len(todo):
            log_cost("notes", engy, f"{n_done}/{len(todo)}")
            if engy_fb.cost_usd:
                log_cost("notes_fb", engy_fb, f"{n_done}/{len(todo)}")

    await asyncio.gather(*[guarded(t) for t in todo])


def cmd_notes(args: argparse.Namespace) -> None:
    turns = read_jsonl(D7_TURNS)
    engy = Engy(concurrency=16, retries=3)
    engy_fb = Engy(concurrency=16)
    asyncio.run(gen_notes(turns, engy, engy_fb))
    notes = read_jsonl(NOTES)
    ok = sum(1 for n in notes if n.get("p_hind"))
    rej = collections.Counter(a.get("reject") for n in notes for a in n["attempts"] if a.get("reject"))
    models = collections.Counter(n.get("p_hind_model") for n in notes if n.get("p_hind"))
    print(f"p_hind ok {ok}/{len(notes)}; rejections {dict(rej)}; models {dict(models)}; "
          f"p_err kinds {dict(collections.Counter(n['p_err_kind'] for n in notes))}")


# ------------------------------------------------------------------ stage: sample
def note_text(t: dict, variant: str, notes: dict[str, dict]) -> str | None:
    n = notes.get(t["turn_id"])
    if not n:
        return None
    if variant == "p_hind":
        return n.get("p_hind") or None
    if variant == "p_err":
        return n.get("p_err") or None
    raise ValueError(variant)


async def sample_all(turns: list[dict], engy: Engy, notes: dict[str, dict]) -> None:
    done = {(r["turn_id"], r["variant"], r["i"]) for r in read_jsonl(SAMPLES) if "error" not in r}
    jobs = []
    for t in turns:
        for v in VARIANTS:
            p = note_text(t, v, notes)
            if p is None:
                continue
            for i in range(K):
                if (t["turn_id"], v, i) not in done:
                    jobs.append((t, v, i, p))
    print(f"sampling {len(jobs)} teacher refs")

    async def one(t, v, i, p):
        msgs = with_note(t["prefix"], p)
        try:
            r = await engy.chat(TEACHER_ENGY, msgs, temperature=REF_TEMPERATURE, max_tokens=REF_MAX_TOKENS)
        except Exception as ex:  # noqa: BLE001
            append_jsonl(SAMPLES, {"turn_id": t["turn_id"], "variant": v, "i": i, "error": repr(ex)[:300]})
            return
        parsed = parse_reply(r, t["dialect"])
        append_jsonl(SAMPLES, {"turn_id": t["turn_id"], "variant": v, "i": i, "note": p,
                               "reasoning": r["reasoning"], "content": r["content"], "tool_calls": r["tool_calls"],
                               "finish": r["finish"], "usage": r["usage"], "cost_usd": r["cost_usd"],
                               **{k: parsed[k] for k in ("z", "y", "parsed", "kind_used", "think_closed", "repaired")}})

    step = 48
    for s in range(0, len(jobs), step):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + step]])
        log_cost("sample", engy, f"{min(s + step, len(jobs))}/{len(jobs)}")


def cmd_sample(args: argparse.Namespace) -> None:
    turns = read_jsonl(D7_TURNS)
    notes = {n["turn_id"]: n for n in read_jsonl(NOTES)}
    asyncio.run(sample_all(turns, Engy(concurrency=args.concurrency), notes))


# ------------------------------------------------------------------ stage: judge (taxonomy)
JUDGE_SYSTEM = (
    "You compare two candidate next actions of a coding/terminal/tool agent at the same point of a transcript. "
    "Candidate A is the reference; candidate B is the alternative. Classify B relative to A with exactly one word:\n"
    "SAME — same decision (same intent / same next step, e.g. both inspect the same thing, both apply the same fix, both finish), even if the text differs;\n"
    "EXPLORE — B inspects / gathers information (reads, lists, searches, tests before changing anything) where A acts, or B inspects something different from A;\n"
    "ACT_FINISH — B applies a change, runs the decisive test/verification, or finishes/submits where A still explores or waits;\n"
    "DIFFERENT_FIX — both act, but B applies a different change, answer or approach than A.\n"
    "Reply with exactly one of: SAME, EXPLORE, ACT_FINISH, DIFFERENT_FIX."
)
LABELS = ("SAME", "EXPLORE", "ACT_FINISH", "DIFFERENT_FIX")


async def judge_all(turns: list[dict], engy: Engy, n_turns: int) -> None:
    samples = [s for s in read_jsonl(SAMPLES) if s.get("parsed")]
    done = {(r["turn_id"], r["variant"], r["pair"]) for r in read_jsonl(JUDGE) if r.get("label")}
    by_turn = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in samples:
        by_turn[s["turn_id"]][s["variant"]].append(s)
    jobs = []
    for t in turns[:n_turns]:
        blind = t["refs"]
        tail = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in t["prefix"][-3:])
        if len(tail) > 12_000:
            tail = tail[-12_000:]
        for v in VARIANTS:
            refs = sorted(by_turn[t["turn_id"]].get(v, []), key=lambda s: s["i"])
            for pi in range(min(2, len(refs), len(blind))):
                if (t["turn_id"], v, pi) not in done:
                    jobs.append((t["turn_id"], v, pi, tail, blind[pi]["y"], refs[pi]["y"]))
    print(f"judge: {len(jobs)} comparisons")

    async def one(tid, v, pi, tail, ya, yb):
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
            append_jsonl(JUDGE, {"turn_id": tid, "variant": v, "pair": pi, "label": label,
                                 "same": (label == "SAME") if label else None, "raw": txt[:40], "cost_usd": r["cost_usd"]})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(JUDGE, {"turn_id": tid, "variant": v, "pair": pi, "label": None, "same": None,
                                 "error": repr(ex)[:200]})

    for s in range(0, len(jobs), 32):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + 32]])
        log_cost("judge", engy, f"{min(s + 32, len(jobs))}/{len(jobs)}")


def cmd_judge(args: argparse.Namespace) -> None:
    asyncio.run(judge_all(read_jsonl(D7_TURNS), Engy(concurrency=16), args.n))


# ------------------------------------------------------------------ stage: echo
GENERIC_BASH = "```bash\nls -la\n```"
GENERIC_TERMINUS = json.dumps({
    "analysis": "Checking the current state of the working directory before continuing.",
    "plan": "List the files in the current directory.",
    "commands": [{"keystrokes": "ls -la\n", "duration": 1.0}]}, indent=2)


def generic_action(t: dict, tools: list[dict]) -> str | None:
    kind = t["dialect"]
    if kind == "bash":
        return GENERIC_BASH
    if kind == "terminus_json":
        return GENERIC_TERMINUS
    names = []
    for x in tools or []:
        f = x.get("function") or x
        req = ((f.get("parameters") or {}).get("required")) or []
        names.append((f.get("name") or "", len(req)))
    if any(n == "bash" for n, _ in names):
        return "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
    zero_list = [n for n, r in names if r == 0 and "list" in n]
    if zero_list:
        return f"<tool_call>\n<function={zero_list[0]}>\n</function>\n</tool_call>"
    return None


def repeat_last(t: dict) -> str | None:
    last = [m for m in t["prefix"] if m["role"] == "assistant"]
    if not last:
        return None
    _, y = dialects.split_action(last[-1]["content"], t["dialect"])
    return y or None


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def candidates(t: dict, sets: dict[str, list[dict]], tools: list[dict]) -> dict[str, str]:
    """candidate id -> action text."""
    c = {"king": t["king"]["pairs"][0]["y_a"]}
    for i, r in enumerate(t["refs"]):
        c[f"bo{i}"] = r["y"]
    for v, tag in (("p_hind", "ho"), ("p_err", "ro")):
        for r in sets.get(v, []):
            c[f"{tag}{r['i']}"] = r["y"]
    g = generic_action(t, tools)
    if g:
        c["gen"] = g
    rl = repeat_last(t)
    if rl:
        c["rep"] = rl
    return c


def contexts(t: dict, sets: dict[str, list[dict]]) -> dict[str, str]:
    """context id -> thought text (the echo's <think> body)."""
    c = {"e": "", "kz": t["king"]["pairs"][0]["z_a"]}     # kz: king thought, echoed only under the blind actions (R_blind(king))
    for i, r in enumerate(t["refs"]):
        c[f"b{i}"] = r["z"]
    for v, tag in (("p_hind", "h"), ("p_err", "r")):
        for r in sets.get(v, []):
            c[f"{tag}{r['i']}"] = r["z"]
    return c


def ref_sets(tid: str, samples: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for s in samples:
        if s["turn_id"] == tid and s.get("parsed"):
            out[s["variant"]].append({"i": s["i"], "z": s["z"], "y": s["y"]})
    return out


def read_echoes() -> dict[str, dict]:
    out = {}
    for p in sorted(RESULTS.glob("echoes*.jsonl")):
        for r in read_jsonl(p):
            if "error" not in r:
                out[r["key"]] = r
    return out


_prompt_cache: dict[tuple, str] = {}
_gen_prompt_raw = common.gen_prompt


def _gen_prompt_memo(prefix_messages: list[dict]) -> str:
    key = tuple((m["role"], m["content"]) for m in prefix_messages)
    kh = hash(key)
    if kh not in _prompt_cache:
        if len(_prompt_cache) > 8:
            _prompt_cache.pop(next(iter(_prompt_cache)))
        _prompt_cache[kh] = _gen_prompt_raw(prefix_messages)
    return _prompt_cache[kh]


common.gen_prompt = _gen_prompt_memo


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
        cands = candidates(t, sets, tools)
        ctxs = contexts(t, sets)
        seen = set()
        for cid, y in cands.items():
            for xid, z in ctxs.items():
                if xid == "kz" and not cid.startswith("bo"):
                    continue
                key = f"{tid}|{h(z)}|{h(y)}"
                if key in done or key in seen:
                    continue
                seen.add(key)
                jobs.append((key, t["prefix"], z, y, cid, xid))
    print(f"echo: {len(jobs)} action echoes to run")

    async def one(key, prefix, z, y, cid, xid):
        try:
            r = await te.lp_action(prefix, z, y)
            append_jsonl(out_path, {"key": key, "cand": cid, "ctx": xid, **r})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(out_path, {"key": key, "cand": cid, "ctx": xid, "error": repr(ex)[:300]})

    # worker pool (not gather batches): a batch would wait on its slowest long-prefix echo
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    n_done = 0

    async def worker():
        nonlocal n_done
        while True:
            try:
                j = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await one(*j)
            n_done += 1
            if n_done % 96 == 0 or n_done == len(jobs):
                log_cost("echo", engy, f"{n_done}/{len(jobs)} start={start}")

    await asyncio.gather(*[worker() for _ in range(engy.sem._value)])


def cmd_echo(args: argparse.Namespace) -> None:
    turns = read_jsonl(D7_TURNS)[args.start:]
    turns = turns[: args.turns] if args.turns else turns
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
    return {"n": len(d), "mean": st.mean(d), "se": se, "z": (st.mean(d) / se) if se > 0 else None}


def _f(x, w=6, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    if isinstance(x, dict):
        return _f(x.get("z"), w, 2)
    return f"{x:{w}.{p}f}"


def a_leg(b: list[float], tau: float = TAU) -> float | None:
    return lme(b, tau) if b else None


class TurnEchoes:
    """Lift lookups for one turn from the Engy echo grid."""

    def __init__(self, tid: str, cands: dict[str, str], ctxs: dict[str, str], echoes: dict[str, dict]):
        self.tid, self.cands, self.ctxs, self.e = tid, cands, ctxs, echoes

    def lp(self, cid: str, xid: str) -> dict | None:
        return self.e.get(f"{self.tid}|{h(self.ctxs[xid])}|{h(self.cands[cid])}")

    def lift(self, cid: str, xid: str, summed: bool) -> float | None:
        a, e = self.lp(cid, xid), self.lp(cid, "e")
        if a is None or e is None:
            return None
        pb = a["lp_per_byte"] - e["lp_per_byte"]
        return pb * a["n_bytes"] if summed else pb

    def A(self, cid: str, thought_ids: list[str], exclude: str | None, summed: bool) -> float | None:
        """Matched estimator: own (exclude = own thought id) -> LME over the others;
        external (exclude None) -> mean_j LME over ids != j."""
        ids = [x for x in thought_ids if x != exclude]
        if exclude is not None:
            b = [self.lift(cid, x, summed) for x in ids]
            return a_leg(b) if all(v is not None for v in b) and len(b) >= 1 else None
        vals = []
        for j in thought_ids:
            b = [self.lift(cid, x, summed) for x in thought_ids if x != j]
            if not b or any(v is None for v in b):
                return None
            vals.append(a_leg(b))
        return st.mean(vals)

    def R(self, xid: str, blind_ids: list[str], exclude: str | None) -> float | None:
        """Centred tempered LME of a_i = lpC(y_C^i|z) − lpC(y_C^i|∅) (per byte) over
        the blind refs' actions, 2-ref LOO-matched (own blind thought excluded /
        external thought averaged over the 2-ref subsets)."""
        a = {}
        for i, bid in enumerate(blind_ids):
            v = self.lift(f"bo{i}", xid, summed=False)
            if v is None:
                return None
            a[i] = v
        own = blind_ids.index(exclude) if exclude is not None else None
        return loo_R(a, own)


def cmd_report(args: argparse.Namespace) -> None:
    turns = read_jsonl(D7_TURNS)
    notes = {n["turn_id"]: n for n in read_jsonl(NOTES)}
    samples = [s for s in read_jsonl(SAMPLES) if "error" not in s]
    sample_errors = [s for s in read_jsonl(SAMPLES) if "error" in s]
    echoes = read_echoes()
    echo_errors = sum(1 for p in RESULTS.glob("echoes*.jsonl") for r in read_jsonl(p) if "error" in r)
    judge = [j for j in read_jsonl(JUDGE) if j.get("label")]
    d7_samples = [s for s in read_jsonl(D7_SAMPLES) if "error" not in s and s.get("variant") == "p_none"]
    d7_judge = read_jsonl(D7_JUDGE)
    d7_rep = json.loads((D7_RESULTS / "report.json").read_text())
    tidx = trace_index()
    by_turn = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in samples:
        by_turn[s["turn_id"]][s["variant"]].append(s)
    ctrl_by_turn = collections.defaultdict(list)
    for s in d7_samples:
        if s.get("parsed"):
            ctrl_by_turn[s["turn_id"]].append(s)

    rows = []
    for t in turns:
        tid, kind = t["turn_id"], t["dialect"]
        blind = t["refs"]
        blind_ys = [r["y"] for r in blind]
        sets = ref_sets(tid, samples)
        tools = load_envelope(*tidx[t["rollout_id"]])["trace"].get("tools") or []
        cands = candidates(t, sets, tools)
        ctxs = contexts(t, sets)
        E = TurnEchoes(tid, cands, ctxs, echoes)
        n = notes.get(tid) or {}
        row = {"turn_id": tid, "dialect": kind, "outcome": t["down"]["outcome"], "harness": t["harness"],
               "has_gen": "gen" in cands, "has_rep": "rep" in cands,
               "p_hind_model": n.get("p_hind_model"), "p_err_kind": n.get("p_err_kind"),
               "note_words": {"p_hind": len((n.get("p_hind") or "").split()) if n.get("p_hind") else None,
                              "p_err": len((n.get("p_err") or "").split()) if n.get("p_err") else None},
               "variants": {}, "legs": {}}
        B_ids = ["b0", "b1", "b2"]
        # ---- stored (vLLM) blind legs, D7-style
        a_bo = {}
        for j in range(K):
            a = [blind[i]["lp_cross"][j] - blind[i]["lp_empty"] for i in range(K) if i != j]
            a_bo[j] = clme(a, TAU)
        row["R_blind_blindown_stored"] = st.mean(a_bo.values())
        a_king = {}
        for p in t["king"]["pairs"]:
            i = next((i for i in range(K) if abs(blind[i]["lp_own"] - p["lpC_yc_zc"]) < 1e-12), None)
            if i is not None:
                a_king[i] = p["lpC_yc_za"] - p["lpC_yc_e"]
        row["R_blind_king_stored"] = loo_R(a_king, None) if len(a_king) == K else None
        # stored A_blind (summed), king + blind-own LOO
        bk = [(p["lpC_ya_zc"] - p["lpC_ya_e"]) * p["n_bytes_ya"] for p in t["king"]["pairs"]]
        row["A_blind_king_stored"] = st.mean(a_leg([b for m, b in enumerate(bk) if m != j]) for j in range(K)) if len(bk) == K else None
        abo = []
        for j in range(K):
            b = [(blind[j]["lp_cross"][i] - blind[j]["lp_empty"]) * blind[j]["n_bytes_y"] for i in range(K) if i != j]
            abo.append(a_leg(b))
        row["A_blind_blindown_stored"] = st.mean(abo)
        row["A_blind_blindown_stored_vals"] = abo
        # ---- Engy legs: blind thought set
        for summed, tag in ((True, "sum"), (False, "pb")):
            L = {}
            L["A_blind_blindown_vals"] = [E.A(f"bo{j}", B_ids, f"b{j}", summed) for j in range(K)]
            L["A_blind_blindown"] = _mean(L["A_blind_blindown_vals"]) if all(v is not None for v in L["A_blind_blindown_vals"]) else None
            L["A_blind_king"] = E.A("king", B_ids, None, summed)
            L["A_blind_gen"] = E.A("gen", B_ids, None, summed) if "gen" in cands else None
            L["A_blind_rep"] = E.A("rep", B_ids, None, summed) if "rep" in cands else None
            row["legs"][tag] = L
        row["R_blind_blindown_vals"] = [E.R(f"b{j}", B_ids, f"b{j}") for j in range(K)]
        row["R_blind_blindown"] = _mean(row["R_blind_blindown_vals"]) if all(v is not None for v in row["R_blind_blindown_vals"]) else None
        row["R_blind_king"] = E.R("kz", B_ids, None)
        row["blind_self_exact"] = st.mean(1.0 if norm_action(blind_ys[a], kind) == norm_action(blind_ys[b], kind) else 0.0
                                          for a in range(K) for b in range(a + 1, K))
        ctrl = [s["y"] for s in sorted(ctrl_by_turn.get(tid, []), key=lambda s: s["i"])]
        row["ctrl"] = {"exact_in_blind": _mean([1.0 if exact(y, blind_ys, kind) else 0.0 for y in ctrl]),
                       "jac_to_blind": _mean([agree(y, blind_ys) for y in ctrl]),
                       "change": _mean([1.0 if (not exact(y, blind_ys, kind) and agree(y, blind_ys) < 0.5) else 0.0 for y in ctrl])}

        for v, tag in (("p_hind", "h"), ("p_err", "r")):
            ss = sorted(by_turn[tid].get(v, []), key=lambda s: s["i"])
            if not ss:
                continue
            refs = [s for s in ss if s["parsed"]]
            p = ss[0].get("note") or ""
            ys = [s["y"] for s in refs]
            vr = {"n_sampled": len(ss), "n_parsed": len(refs),
                  "cap_hits": sum(1 for s in ss if s["finish"] == "length"),
                  "text_fallback": sum(1 for s in ss if s.get("kind_used") == "text"),
                  "len_z": [len(s["z"]) for s in refs], "len_y": [len(s["y"]) for s in refs],
                  "completion_tokens": sum((s.get("usage") or {}).get("completion_tokens") or 0 for s in ss),
                  "prompt_tokens": sum((s.get("usage") or {}).get("prompt_tokens") or 0 for s in ss),
                  "note_chars": len(p)}
            vr["exact_in_blind"] = _mean([1.0 if exact(y, blind_ys, kind) else 0.0 for y in ys])
            vr["jac_to_blind"] = _mean([agree(y, blind_ys) for y in ys])
            vr["change"] = _mean([1.0 if (not exact(y, blind_ys, kind) and agree(y, blind_ys) < 0.5) else 0.0 for y in ys])
            if ctrl:
                vr["exact_in_ctrl"] = _mean([1.0 if exact(y, ctrl, kind) else 0.0 for y in ys])
                vr["jac_to_ctrl"] = _mean([agree(y, ctrl) for y in ys])
            if len(ys) >= 2:
                vr["self_exact"] = st.mean(1.0 if norm_action(ys[a], kind) == norm_action(ys[b], kind) else 0.0
                                           for a in range(len(ys)) for b in range(a + 1, len(ys)))
                vr["self_jac"] = st.mean(jaccard(ys[a], ys[b]) for a in range(len(ys)) for b in range(a + 1, len(ys)))
            if t.get("y_rec"):
                vr["exact_rec"] = _mean([1.0 if norm_action(y, kind) == norm_action(t["y_rec"], kind) else 0.0 for y in ys])
            vr["leak_z"] = _mean([1.0 if leaks(p, s["z"]) else 0.0 for s in refs])
            vr["leak_y"] = _mean([1.0 if leaks(p, s["y"]) else 0.0 for s in refs])
            H_ids = [f"{tag}{s['i']}" for s in refs]
            vr["k_informed"] = len(H_ids)
            if len(H_ids) >= 2:
                # B licence of the informed refs (own thought -> own action, per byte)
                B = [E.lift(f"{tag}o{s['i']}", f"{tag}{s['i']}", summed=False) for s in refs]
                vr["B"] = B
                vr["B_pass"] = _mean([1.0 if (b is not None and b >= 0.02) else 0.0 for b in B]) if all(b is not None for b in B) else None
                for summed, ltag in ((True, "sum"), (False, "pb")):
                    L = {}
                    own = [E.A(f"{tag}o{s['i']}", H_ids, f"{tag}{s['i']}", summed) for s in refs]
                    L["A_priv_own_vals"] = own
                    L["A_priv_own"] = _mean(own) if all(v is not None for v in own) else None
                    bown = [E.A(f"bo{j}", H_ids, None, summed) for j in range(K)]
                    L["A_priv_blindown_vals"] = bown
                    L["A_priv_blindown"] = _mean(bown) if all(v is not None for v in bown) else None
                    L["A_priv_king"] = E.A("king", H_ids, None, summed)
                    L["A_priv_gen"] = E.A("gen", H_ids, None, summed) if "gen" in cands else None
                    L["A_priv_rep"] = E.A("rep", H_ids, None, summed) if "rep" in cands else None
                    # informed-own actions under the BLIND thoughts (does the blind teacher like them?)
                    vb = [E.A(f"{tag}o{s['i']}", B_ids, None, summed) for s in refs]
                    L["A_blind_informedown"] = _mean(vb) if all(v is not None for v in vb) else None
                    vr[ltag] = L
                # informed THOUGHTS under the blind refs' actions: R_blind(informed-own)
                rb = [E.R(x, B_ids, None) for x in H_ids]
                vr["R_blind_informedown"] = _mean(rb) if all(v is not None for v in rb) else None
            row["variants"][v] = vr
        rows.append(row)
    write_jsonl(RESULTS / "turn_metrics.jsonl", rows)

    # ---------------- σ per dialect (blind-own within-turn spread, Engy)
    def pooled_sd(vals_by_turn: list[list[float]]) -> float | None:
        v = [st.variance(x) for x in vals_by_turn if x and all(a is not None for a in x) and len(x) >= 2]
        return math.sqrt(st.mean(v)) if v else None

    sigma = {}
    for kind in ("bash", "tool_call", "terminus_json"):
        rs = [r for r in rows if r["dialect"] == kind]
        sigma[kind] = {
            "A_sum": pooled_sd([r["legs"]["sum"]["A_blind_blindown_vals"] for r in rs]),
            "A_pb": pooled_sd([r["legs"]["pb"]["A_blind_blindown_vals"] for r in rs]),
            "R": pooled_sd([r["R_blind_blindown_vals"] for r in rs]),
            "A_sum_stored": pooled_sd([r["A_blind_blindown_stored_vals"] for r in rs]),
        }
        for v in VARIANTS:
            for ltag in ("sum", "pb"):
                sigma[kind][f"A_priv_{ltag}_{v}"] = pooled_sd([r["variants"][v][ltag]["A_priv_blindown_vals"]
                                                               for r in rs if v in r["variants"] and ltag in r["variants"][v]])
    # ---------------- per-turn z's
    for r in rows:
        sg = sigma[r["dialect"]]
        r["z"] = {}
        for ltag, sk in (("sum", "A_sum"), ("pb", "A_pb")):
            L = r["legs"][ltag]
            mu = L["A_blind_blindown"]
            s_ = sg[sk]
            zz = {}
            if mu is not None and s_:
                for c in ("king", "gen", "rep"):
                    v = L.get(f"A_blind_{c}")
                    zz[f"A_blind_{c}"] = (v - mu) / s_ if v is not None else None
                zz["A_blind_blindown"] = 0.0
            r["z"][ltag] = zz
        muR, sR = r["R_blind_blindown"], sg["R"]
        r["z"]["R_king"] = (r["R_blind_king"] - muR) / sR if (muR is not None and sR and r["R_blind_king"] is not None) else None
        r["z"]["R_blindown"] = 0.0 if muR is not None else None
        for v in VARIANTS:
            vr = r["variants"].get(v)
            if not vr or "sum" not in vr:
                continue
            vr["z"] = {}
            for ltag, sk in (("sum", "A_sum"), ("pb", "A_pb")):
                L = vr[ltag]
                mu = L["A_priv_blindown"]
                # leg-consistent σ (as the live meter would anchor it): the blind-own actions'
                # within-turn spread UNDER THE INFORMED thoughts, pooled per dialect
                s_ = sg.get(f"A_priv_{ltag}_{v}") or sg[sk]
                zz = {}
                if mu is not None and s_:
                    for c in ("own", "king", "gen", "rep"):
                        val = L.get(f"A_priv_{c}")
                        zz[f"A_priv_{c}"] = (val - mu) / s_ if val is not None else None
                    zz["A_priv_blindown"] = 0.0
                    # informed-own under blind thoughts, in blind units
                    mub = r["legs"][ltag]["A_blind_blindown"]
                    zz["A_blind_informedown"] = (L["A_blind_informedown"] - mub) / sg[sk] if (L.get("A_blind_informedown") is not None and mub is not None) else None
                vr["z"][ltag] = zz
            vr["z"]["R_informedown"] = (vr["R_blind_informedown"] - muR) / sR if (vr.get("R_blind_informedown") is not None and muR is not None and sR) else None
            # V4 / V5 (summed form)
            zA = vr["z"]["sum"]
            zRk, zRi = r["z"]["R_king"], vr["z"]["R_informedown"]
            V4 = {}
            if zA:
                V4["king"] = min(zRk, zA["A_priv_king"]) if (zRk is not None and zA.get("A_priv_king") is not None) else None
                V4["blindown"] = min(0.0, 0.0)
                V4["informedown"] = min(zRi, zA["A_priv_own"]) if (zRi is not None and zA.get("A_priv_own") is not None) else None
                V4["gen"] = min(0.0, zA["A_priv_gen"]) if zA.get("A_priv_gen") is not None else None
                V4["rep"] = min(0.0, zA["A_priv_rep"]) if zA.get("A_priv_rep") is not None else None
            vr["V4"] = V4
            L, Lb = vr["sum"], r["legs"]["sum"]
            V5 = {}
            for c, kb in (("king", "A_blind_king"), ("blindown", "A_blind_blindown"), ("gen", "A_blind_gen"), ("rep", "A_blind_rep")):
                a, b = L.get(f"A_priv_{c}"), Lb.get(kb)
                V5[c] = (a - b) if (a is not None and b is not None) else None
            a, b = L.get("A_priv_own"), L.get("A_blind_informedown")
            V5["informedown"] = (a - b) if (a is not None and b is not None) else None
            vr["V5"] = V5
    write_jsonl(RESULTS / "turn_metrics.jsonl", rows)

    # ---------------- aggregate
    def agg(rs: list[dict], label: str) -> dict:
        out = {"label": label, "n_turns": len(rs), "variants": {}, "blind": {}}
        b = out["blind"]
        b["self_exact"] = _mean([r["blind_self_exact"] for r in rs])
        b["ctrl_exact_in_blind"] = _mean([r["ctrl"]["exact_in_blind"] for r in rs])
        b["ctrl_jac_to_blind"] = _mean([r["ctrl"]["jac_to_blind"] for r in rs])
        b["ctrl_change"] = _mean([r["ctrl"]["change"] for r in rs])
        for ltag in ("sum", "pb"):
            L = [r["legs"][ltag] for r in rs]
            b[f"A_blind_blindown_{ltag}"] = _mean([x["A_blind_blindown"] for x in L])
            b[f"A_blind_king_{ltag}"] = _mean([x["A_blind_king"] for x in L])
            b[f"A_blind_gen_{ltag}"] = _mean([x["A_blind_gen"] for x in L])
            b[f"A_blind_rep_{ltag}"] = _mean([x["A_blind_rep"] for x in L])
            b[f"hz_blindown_minus_king_{ltag}"] = _paired([x["A_blind_blindown"] - x["A_blind_king"] for x in L
                                                            if x["A_blind_blindown"] is not None and x["A_blind_king"] is not None])
            b[f"hz_king_minus_gen_{ltag}"] = _paired([x["A_blind_king"] - x["A_blind_gen"] for x in L
                                                       if x["A_blind_king"] is not None and x["A_blind_gen"] is not None])
            b[f"hz_king_minus_rep_{ltag}"] = _paired([x["A_blind_king"] - x["A_blind_rep"] for x in L
                                                       if x["A_blind_king"] is not None and x["A_blind_rep"] is not None])
            b[f"z_king_{ltag}"] = _mean([r["z"][ltag].get("A_blind_king") for r in rs])
            b[f"z_gen_{ltag}"] = _mean([r["z"][ltag].get("A_blind_gen") for r in rs])
            b[f"z_rep_{ltag}"] = _mean([r["z"][ltag].get("A_blind_rep") for r in rs])
        b["A_blind_blindown_stored"] = _mean([r["A_blind_blindown_stored"] for r in rs])
        b["A_blind_king_stored"] = _mean([r["A_blind_king_stored"] for r in rs])
        b["hz_blindown_minus_king_stored"] = _paired([r["A_blind_blindown_stored"] - r["A_blind_king_stored"] for r in rs
                                                      if r["A_blind_king_stored"] is not None])
        b["A_parity_engy_minus_stored_blindown"] = _mean([r["legs"]["sum"]["A_blind_blindown"] - r["A_blind_blindown_stored"] for r in rs
                                                          if r["legs"]["sum"]["A_blind_blindown"] is not None])
        b["A_parity_engy_minus_stored_king"] = _mean([r["legs"]["sum"]["A_blind_king"] - r["A_blind_king_stored"] for r in rs
                                                      if r["legs"]["sum"]["A_blind_king"] is not None and r["A_blind_king_stored"] is not None])
        b["R_blind_blindown"] = _mean([r["R_blind_blindown"] for r in rs])
        b["R_blind_blindown_stored"] = _mean([r["R_blind_blindown_stored"] for r in rs])
        b["R_blind_king"] = _mean([r["R_blind_king"] for r in rs])
        b["z_R_king"] = _mean([r["z"]["R_king"] for r in rs])
        b["hz_R_blindown_minus_king"] = _paired([r["R_blind_blindown"] - r["R_blind_king"] for r in rs
                                                 if r["R_blind_blindown"] is not None and r["R_blind_king"] is not None])
        # D7 p_none judge SAME on these turns
        tids = {r["turn_id"] for r in rs}
        jj = [j for j in d7_judge if j["turn_id"] in tids and j.get("same") is not None and j["variant"] == "p_none"]
        b["ctrl_judge_same"] = st.mean(1.0 if j["same"] else 0.0 for j in jj) if jj else None
        b["ctrl_judge_n"] = len(jj)
        jb = [j for j in d7_judge if j["turn_id"] in tids and j.get("same") is not None and j["variant"] == "blind_self"]
        b["blind_self_judge_same"] = st.mean(1.0 if j["same"] else 0.0 for j in jb) if jb else None
        for v in VARIANTS:
            vs = [(r, r["variants"][v]) for r in rs if v in r["variants"]]
            if not vs:
                continue
            g: dict = {"n_turns": len(vs)}
            ns = max(1, sum(x["n_sampled"] for _, x in vs))
            g["parse_rate"] = sum(x["n_parsed"] for _, x in vs) / ns
            g["cap_hit_rate"] = sum(x["cap_hits"] for _, x in vs) / ns
            g["text_fallback_rate"] = sum(x["text_fallback"] for _, x in vs) / ns
            g["yield3"] = _mean([1.0 if x["n_parsed"] >= 3 else 0.0 for _, x in vs])
            g["yield2"] = _mean([1.0 if x["n_parsed"] >= 2 else 0.0 for _, x in vs])
            lz = [l for _, x in vs for l in x["len_z"]]
            g["len_z_p50"] = st.median(lz) if lz else None
            ly = [l for _, x in vs for l in x["len_y"]]
            g["len_y_p50"] = st.median(ly) if ly else None
            g["completion_tokens_per_ref"] = sum(x["completion_tokens"] for _, x in vs) / ns
            g["prompt_tokens_per_ref"] = sum(x["prompt_tokens"] for _, x in vs) / ns
            g["note_chars_mean"] = _mean([x["note_chars"] for _, x in vs])
            for key in ("exact_in_blind", "jac_to_blind", "change", "exact_in_ctrl", "jac_to_ctrl", "self_exact",
                        "self_jac", "exact_rec", "leak_z", "leak_y", "B_pass"):
                g[key] = _mean([x.get(key) for _, x in vs])
            g["jac_drop_vs_ctrl"] = _paired([r["ctrl"]["jac_to_blind"] - x["jac_to_blind"] for r, x in vs
                                             if r["ctrl"]["jac_to_blind"] is not None and x.get("jac_to_blind") is not None])
            g["exact_drop_vs_ctrl"] = _paired([r["ctrl"]["exact_in_blind"] - x["exact_in_blind"] for r, x in vs
                                               if r["ctrl"]["exact_in_blind"] is not None and x.get("exact_in_blind") is not None])
            g["change_up_vs_ctrl"] = _paired([x["change"] - r["ctrl"]["change"] for r, x in vs
                                              if r["ctrl"]["change"] is not None and x.get("change") is not None])
            jj = [j for j in judge if j["turn_id"] in tids and j["variant"] == v]
            g["judge_n"] = len(jj)
            g["judge_same"] = st.mean(1.0 if j["label"] == "SAME" else 0.0 for j in jj) if jj else None
            g["judge_taxonomy"] = dict(collections.Counter(j["label"] for j in jj))
            # headroom + separation per form
            for ltag in ("sum", "pb"):
                Ls = [(r, x[ltag]) for r, x in vs if ltag in x]
                gg: dict = {"n": len(Ls)}
                for c in ("own", "blindown", "king", "gen", "rep"):
                    gg[f"A_priv_{c}"] = _mean([L.get(f"A_priv_{c}") for _, L in Ls])
                gg["A_blind_informedown"] = _mean([L.get("A_blind_informedown") for _, L in Ls])
                gg["A_blind_blindown"] = _mean([r["legs"][ltag]["A_blind_blindown"] for r, _ in Ls])
                gg["A_blind_king"] = _mean([r["legs"][ltag]["A_blind_king"] for r, _ in Ls])

                def pz(f):
                    return _paired([f(r, L) for r, L in Ls])
                gg["hz_own_minus_blindown"] = pz(lambda r, L: (L["A_priv_own"] - L["A_priv_blindown"]) if L.get("A_priv_own") is not None and L.get("A_priv_blindown") is not None else None)
                gg["hz_blindown_minus_king"] = pz(lambda r, L: (L["A_priv_blindown"] - L["A_priv_king"]) if L.get("A_priv_blindown") is not None and L.get("A_priv_king") is not None else None)
                gg["hz_own_minus_king"] = pz(lambda r, L: (L["A_priv_own"] - L["A_priv_king"]) if L.get("A_priv_own") is not None and L.get("A_priv_king") is not None else None)
                gg["hz_king_minus_gen"] = pz(lambda r, L: (L["A_priv_king"] - L["A_priv_gen"]) if L.get("A_priv_king") is not None and L.get("A_priv_gen") is not None else None)
                gg["hz_king_minus_rep"] = pz(lambda r, L: (L["A_priv_king"] - L["A_priv_rep"]) if L.get("A_priv_king") is not None and L.get("A_priv_rep") is not None else None)
                gg["hz_blindown_minus_gen"] = pz(lambda r, L: (L["A_priv_blindown"] - L["A_priv_gen"]) if L.get("A_priv_blindown") is not None and L.get("A_priv_gen") is not None else None)
                # same turns, blind thought set (separation kept?)
                gg["hz_blind_blindown_minus_king"] = pz(lambda r, L: (r["legs"][ltag]["A_blind_blindown"] - r["legs"][ltag]["A_blind_king"])
                                                        if r["legs"][ltag]["A_blind_blindown"] is not None and r["legs"][ltag]["A_blind_king"] is not None else None)
                gg["hz_blind_informedown_minus_blindown"] = pz(lambda r, L: (L["A_blind_informedown"] - r["legs"][ltag]["A_blind_blindown"])
                                                               if L.get("A_blind_informedown") is not None and r["legs"][ltag]["A_blind_blindown"] is not None else None)
                gg["hz_blind_informedown_minus_king"] = pz(lambda r, L: (L["A_blind_informedown"] - r["legs"][ltag]["A_blind_king"])
                                                           if L.get("A_blind_informedown") is not None and r["legs"][ltag]["A_blind_king"] is not None else None)
                # sd units
                zs = [(r, x["z"][ltag]) for r, x in vs if x.get("z") and ltag in x["z"]]
                for c in ("own", "king", "gen", "rep"):
                    gg[f"z_priv_{c}"] = _mean([z.get(f"A_priv_{c}") for _, z in zs])
                gg["z_blind_informedown"] = _mean([z.get("A_blind_informedown") for _, z in zs])
                gg["z_priv_own_minus_king"] = _paired([z["A_priv_own"] - z["A_priv_king"] for _, z in zs
                                                       if z.get("A_priv_own") is not None and z.get("A_priv_king") is not None])
                gg["z_priv_blindown_minus_king"] = _paired([0.0 - z["A_priv_king"] for _, z in zs if z.get("A_priv_king") is not None])
                gg["z_blind_blindown_minus_king"] = _paired([0.0 - r["z"][ltag]["A_blind_king"] for r, _ in zs if r["z"][ltag].get("A_blind_king") is not None])
                g[ltag] = gg
            g["R_blind_informedown"] = _mean([x.get("R_blind_informedown") for _, x in vs])
            g["z_R_informedown"] = _mean([(x.get("z") or {}).get("R_informedown") for _, x in vs])
            g["hz_R_informedown_minus_blindown"] = _paired([x["R_blind_informedown"] - r["R_blind_blindown"] for r, x in vs
                                                            if x.get("R_blind_informedown") is not None and r["R_blind_blindown"] is not None])
            g["hz_R_informedown_minus_king"] = _paired([x["R_blind_informedown"] - r["R_blind_king"] for r, x in vs
                                                        if x.get("R_blind_informedown") is not None and r["R_blind_king"] is not None])
            for c in ("king", "blindown", "informedown", "gen", "rep"):
                g[f"V4_{c}"] = _mean([(x.get("V4") or {}).get(c) for _, x in vs])
                g[f"V5_{c}"] = _mean([(x.get("V5") or {}).get(c) for _, x in vs])
                g[f"V4_{c}_minus_king"] = _paired([(x["V4"][c] - x["V4"]["king"]) for _, x in vs
                                                   if x.get("V4") and x["V4"].get(c) is not None and x["V4"].get("king") is not None])
                g[f"V5_{c}_z"] = _paired([(x.get("V5") or {}).get(c) for _, x in vs])
            out["variants"][v] = g
        return out

    rep = {"n_turns": len(rows), "dialects": dict(collections.Counter(r["dialect"] for r in rows)),
           "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
           "sample_errors": len(sample_errors), "echo_errors": echo_errors, "n_echoes": len(echoes),
           "sigma": sigma, "cost_usd_total": total_cost(),
           "d7_pooled": {"p_none": d7_rep["pooled"]["variants"].get("p_none"), "p_fr": d7_rep["pooled"]["variants"].get("p_fr"),
                         "judge": d7_rep["pooled"]["judge"]}}
    nl = list(notes.values())
    att = [a for n in nl for a in n["attempts"]]
    rep["notes"] = {
        "n": len(nl), "p_hind_ok": sum(1 for n in nl if n.get("p_hind")),
        "p_hind_first_try_ok": sum(1 for n in nl if n["attempts"] and n["attempts"][0].get("reject") is None and n["attempts"][0].get("note")),
        "attempts": len(att),
        "rejections": dict(collections.Counter(a.get("reject") for a in att if a.get("reject"))),
        "errors": sum(1 for a in att if a.get("error")),
        "models": dict(collections.Counter(n.get("p_hind_model") for n in nl if n.get("p_hind"))),
        "p_hind_words_p50": st.median([len(n["p_hind"].split()) for n in nl if n.get("p_hind")]) if any(n.get("p_hind") for n in nl) else None,
        "p_hind_words_mean": _mean([len(n["p_hind"].split()) for n in nl if n.get("p_hind")]),
        "p_hind_chars_mean": _mean([len(n["p_hind"]) for n in nl if n.get("p_hind")]),
        "p_err_kinds": dict(collections.Counter(n.get("p_err_kind") for n in nl)),
        "p_err_chars_mean": _mean([len(n["p_err"]) for n in nl if n.get("p_err")]),
        "fwd_msgs_p50": st.median([n["n_fwd"] for n in nl]) if nl else None,
        "fwd_chars_p50": st.median([n["fwd_chars"] for n in nl]) if nl else None,
        "note_prompt_tokens_mean": _mean([(a.get("usage") or {}).get("prompt_tokens") for a in att if a.get("usage")]),
        "note_completion_tokens_mean": _mean([(a.get("usage") or {}).get("completion_tokens") for a in att if a.get("usage")]),
        "note_usd_per_turn": sum((a.get("cost_usd") or 0) for a in att) / max(1, len(nl)),
    }
    rep["pooled"] = agg(rows, "pooled")
    rep["by_dialect"] = {k: agg([r for r in rows if r["dialect"] == k], k) for k in ("bash", "tool_call", "terminus_json")}
    rep["by_outcome"] = {k: agg([r for r in rows if r["outcome"] == k], k) for k in ("solved", "failed")}
    # p_hind change by outcome + taxonomy
    tax = {}
    for oc in ("solved", "failed"):
        tids = {r["turn_id"] for r in rows if r["outcome"] == oc}
        jj = [j for j in judge if j["turn_id"] in tids and j["variant"] == "p_hind"]
        tax[oc] = {"judge_n": len(jj), "judge_same": st.mean(1.0 if j["label"] == "SAME" else 0.0 for j in jj) if jj else None,
                   "taxonomy": dict(collections.Counter(j["label"] for j in jj))}
    rep["p_hind_by_outcome"] = tax
    (RESULTS / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print_report(rep)


def print_report(rep: dict) -> None:
    lines: list[str] = []
    P = lines.append
    P("Hindsight-privileged references read by the ACTION leg — probe W1 report")
    P(f"turns {rep['n_turns']} {rep['dialects']} outcomes {rep['outcomes']}; sample errors {rep['sample_errors']}, "
      f"echo errors {rep['echo_errors']}, echoes {rep['n_echoes']}; $ spent (all stages, all runs) {rep['cost_usd_total']:.2f}")
    n = rep["notes"]
    P("")
    P("== NOTES ==")
    P(f"p_hind ok {n['p_hind_ok']}/{n['n']} (first try {n['p_hind_first_try_ok']}); attempts {n['attempts']}, rejections {n['rejections']}, "
      f"errors {n['errors']}; models {n['models']}; words p50 {n['p_hind_words_p50']} mean {_f(n['p_hind_words_mean'],5,1)} chars {_f(n['p_hind_chars_mean'],5,0)}; "
      f"note prompt tok {_f(n['note_prompt_tokens_mean'],7,0)} compl tok {_f(n['note_completion_tokens_mean'],6,0)} ${n['note_usd_per_turn']:.4f}/turn")
    P(f"p_err kinds {n['p_err_kinds']} chars mean {_f(n['p_err_chars_mean'],5,0)}; forward part p50 {n['fwd_msgs_p50']} msgs / {n['fwd_chars_p50']} chars")
    P("")
    P("Terms: exactB/jacB = share of refs whose normalised action equals one of / best token-Jaccard to the 3 stored blind refs; change = norm-different from all 3 AND Jaccard < .5;")
    P("       exactC/jacC = same vs D7's p_none fresh-blind refs; dropJz / chg z = paired (p_none − variant) Jaccard / (variant − p_none) change over turns; self = within-family exact;")
    P("       judge = glm-5.3-flash taxonomy (ref i vs blind ref i, i = 0,1; first 60 turns): SAME rate + EXPLORE / ACT_FINISH / DIFFERENT_FIX counts; leak = >=30-char note substring in z / y;")
    P("       A_S(y) = tau·log mean_i exp(b_i/tau), b_i = [lpC(y|x,z_i) − lpC(y|x,∅)]·bytes(y) (summed, a_norm 1; 'pb' = per byte); S = blind thoughts (A_blind) or informed thoughts (A_priv);")
    P("       own = the informed refs' own actions LOO; blind-own = stored blind actions (LOO under blind thoughts, matched 2-thought mean under informed); king = stored king action;")
    P("       gen = `ls -la` in the dialect; rep = the previous assistant action re-issued; hz = paired z over turns; z(·) = (A − A_blind-own(turn)) / σ_A(dialect), σ_A = pooled within-turn sd of the blind-own LOO values;")
    P("       R_blind(z) = centred tempered LME of a_i = lpC(y_C^i|x,z) − lpC(y_C^i|x,∅) over the blind refs (2-ref LOO-matched); V4 = min(z_R_blind, z_A_priv); V5 = A_priv − A_blind (nats).")
    d7n, d7f, d7j = rep["d7_pooled"]["p_none"], rep["d7_pooled"]["p_fr"], rep["d7_pooled"]["judge"]
    for label, blk in [("POOLED", rep["pooled"])] + [(f"DIALECT {k}", v) for k, v in rep["by_dialect"].items()] + \
            [(f"OUTCOME {k}", v) for k, v in rep["by_outcome"].items()]:
        if not blk["n_turns"]:
            continue
        b = blk["blind"]
        P("")
        P(f"== {label} (n turns {blk['n_turns']}) ==")
        P("-- (a) action change (p_none = D7's fresh blind refs on the same turns; D7 p_fr pooled for comparison)")
        P(f"{'variant':<8} {'n':>4} {'exactB':>6} {'jacB':>6} {'change':>6} {'exactC':>6} {'jacC':>6} {'dropJz':>6} {'chg z':>6} {'self':>6} {'=rec':>6} {'judge':>9} {'EXPL':>5} {'ACTF':>5} {'DIFF':>5} {'leakZ':>6} {'leakY':>6} {'parse':>6} {'yld3':>6} {'cap':>6} {'lenZ':>5} {'lenY':>5}")
        P(f"{'p_none':<8} {blk['n_turns']:>4} {_f(b['ctrl_exact_in_blind'])} {_f(b['ctrl_jac_to_blind'])} {_f(b['ctrl_change'])} {'':>6} {'':>6} {'':>6} {'':>6} {_f(d7n['self_exact']) if label=='POOLED' else '':>6} {'':>6} "
          f"{_f(b['ctrl_judge_same'])}/{b['ctrl_judge_n']:<3}")
        if label == "POOLED" and d7f:
            P(f"{'p_fr(D7)':<8} {150:>4} {_f(d7f['exact_in_blind'])} {_f(d7f['jac_to_blind'])} {'':>6} {_f(d7f['exact_in_ctrl'])} {_f(d7f['jac_to_ctrl'])} {_f(d7f['jac_drop_vs_ctrl'])} {'':>6} {_f(d7f['self_exact'])} {_f(d7f['exact_rec'])} "
              f"{_f(d7j['p_fr']['same_rate'])}/{d7j['p_fr']['n']:<3} {'':>5} {'':>5} {'':>5} {_f(d7f['leak_z'])} {_f(d7f['leak_y'])} {_f(d7f['parse_rate'])} {'':>6} {_f(d7f['cap_hit_rate'])}")
        for v, g in blk["variants"].items():
            tx = g["judge_taxonomy"]
            P(f"{v:<8} {g['n_turns']:>4} {_f(g['exact_in_blind'])} {_f(g['jac_to_blind'])} {_f(g['change'])} {_f(g.get('exact_in_ctrl'))} {_f(g.get('jac_to_ctrl'))} {_f(g['jac_drop_vs_ctrl'])} {_f(g['change_up_vs_ctrl'])} {_f(g['self_exact'])} {_f(g['exact_rec'])} "
              f"{_f(g['judge_same'])}/{g['judge_n']:<3} {tx.get('EXPLORE',0):>5} {tx.get('ACT_FINISH',0):>5} {tx.get('DIFFERENT_FIX',0):>5} {_f(g['leak_z'])} {_f(g['leak_y'])} {_f(g['parse_rate'])} {_f(g['yield3'])} {_f(g['cap_hit_rate'])} {_f(g['len_z_p50'],5,0)} {_f(g['len_y_p50'],5,0)}")
        if b.get("blind_self_judge_same") is not None:
            P(f"   D7 judge blind-vs-blind SAME {b['blind_self_judge_same']:.3f}")
        P("-- (b) A leg, blind thought set (Engy re-echo; stored = the verdict's vLLM echoes): nats summed | z = sd units of the blind-own spread")
        P(f"{'form':<5} {'A_bown':>8} {'A_king':>8} {'A_gen':>8} {'A_rep':>8} {'hz b-k':>6} {'hz k-g':>6} {'hz k-r':>6} {'z_king':>6} {'z_gen':>6} {'z_rep':>6} | {'stored bown':>11} {'stored king':>11} {'hz b-k':>6} {'parity b':>8} {'parity k':>8}")
        for ltag in ("sum", "pb"):
            P(f"{ltag:<5} {_f(b[f'A_blind_blindown_{ltag}'],8,4)} {_f(b[f'A_blind_king_{ltag}'],8,4)} {_f(b[f'A_blind_gen_{ltag}'],8,4)} {_f(b[f'A_blind_rep_{ltag}'],8,4)} "
              f"{_f(b[f'hz_blindown_minus_king_{ltag}'])} {_f(b[f'hz_king_minus_gen_{ltag}'])} {_f(b[f'hz_king_minus_rep_{ltag}'])} {_f(b[f'z_king_{ltag}'],6,2)} {_f(b[f'z_gen_{ltag}'],6,2)} {_f(b[f'z_rep_{ltag}'],6,2)} | "
              + (f"{_f(b['A_blind_blindown_stored'],11,4)} {_f(b['A_blind_king_stored'],11,4)} {_f(b['hz_blindown_minus_king_stored'])} {_f(b['A_parity_engy_minus_stored_blindown'],8,4)} {_f(b['A_parity_engy_minus_stored_king'],8,4)}" if ltag == "sum" else ""))
        P(f"R_blind: blind-own {_f(b['R_blind_blindown'],7,4)} (stored {_f(b['R_blind_blindown_stored'],7,4)}) king {_f(b['R_blind_king'],7,4)} z_R(king) {_f(b['z_R_king'],5,2)} hz(bown−king) {_f(b['hz_R_blindown_minus_king'])}")
        P("-- (c) A_priv headroom: informed thought set as echo context; matched 2-thought estimator; n = turns with >=2 parsed informed refs")
        P(f"{'variant':<8} {'form':<4} {'n':>4} {'A_own':>8} {'A_bown':>8} {'A_king':>8} {'A_gen':>8} {'A_rep':>8} | {'hz o-b':>6} {'hz b-k':>6} {'hz o-k':>6} {'hz k-g':>6} {'hz k-r':>6} {'hz b-g':>6} | {'z_own':>6} {'z_king':>6} {'z_gen':>6} {'z_rep':>6} {'zo-k':>6} | {'blind: hz b-k':>13} {'zb-k':>6} | {'A_bl(own)':>9} {'hz io-b':>7} {'hz io-k':>7} {'z io':>6}")
        for v, g in blk["variants"].items():
            for ltag in ("sum", "pb"):
                gg = g[ltag]
                P(f"{v:<8} {ltag:<4} {gg['hz_own_minus_blindown']['n']:>4} {_f(gg['A_priv_own'],8,4)} {_f(gg['A_priv_blindown'],8,4)} {_f(gg['A_priv_king'],8,4)} {_f(gg['A_priv_gen'],8,4)} {_f(gg['A_priv_rep'],8,4)} | "
                  f"{_f(gg['hz_own_minus_blindown'])} {_f(gg['hz_blindown_minus_king'])} {_f(gg['hz_own_minus_king'])} {_f(gg['hz_king_minus_gen'])} {_f(gg['hz_king_minus_rep'])} {_f(gg['hz_blindown_minus_gen'])} | "
                  f"{_f(gg['z_priv_own'],6,2)} {_f(gg['z_priv_king'],6,2)} {_f(gg['z_priv_gen'],6,2)} {_f(gg['z_priv_rep'],6,2)} {_f(gg['z_priv_own_minus_king'])} | {_f(gg['hz_blind_blindown_minus_king'],13)} {_f(gg['z_blind_blindown_minus_king'])} | "
                  f"{_f(gg['A_blind_informedown'],9,4)} {_f(gg['hz_blind_informedown_minus_blindown'],7)} {_f(gg['hz_blind_informedown_minus_king'],7)} {_f(gg['z_blind_informedown'],6,2)}")
        P("-- (d) thought side + V4 / V5 (summed form): R_blind of the informed THOUGHTS; V4 = min(z_R_blind, z_A_priv) (gen/rep take z_R = 0 = a FLAT thought); V5 = A_priv − A_blind nats (paired z)")
        P(f"{'variant':<8} {'R_io':>7} {'z_R io':>6} {'hz io-b':>7} {'hz io-k':>7} | {'V4 king':>7} {'V4 bown':>7} {'V4 io':>7} {'V4 gen':>7} {'V4 rep':>7} {'io-k z':>6} {'b-k z':>6} {'gen-k z':>7} | {'V5 king':>7} {'z':>5} {'V5 bown':>7} {'z':>5} {'V5 io':>7} {'z':>5} {'V5 gen':>7} {'z':>5} {'V5 rep':>7} {'z':>5}")
        for v, g in blk["variants"].items():
            P(f"{v:<8} {_f(g['R_blind_informedown'],7,4)} {_f(g['z_R_informedown'],6,2)} {_f(g['hz_R_informedown_minus_blindown'],7)} {_f(g['hz_R_informedown_minus_king'],7)} | "
              f"{_f(g['V4_king'],7,2)} {_f(g['V4_blindown'],7,2)} {_f(g['V4_informedown'],7,2)} {_f(g['V4_gen'],7,2)} {_f(g['V4_rep'],7,2)} {_f(g['V4_informedown_minus_king'])} {_f(g['V4_blindown_minus_king'])} {_f(g['V4_gen_minus_king'],7)} | "
              f"{_f(g['V5_king'],7,3)} {_f(g['V5_king_z'],5)} {_f(g['V5_blindown'],7,3)} {_f(g['V5_blindown_z'],5)} {_f(g['V5_informedown'],7,3)} {_f(g['V5_informedown_z'],5)} {_f(g['V5_gen'],7,3)} {_f(g['V5_gen_z'],5)} {_f(g['V5_rep'],7,3)} {_f(g['V5_rep_z'],5)}")
    P("")
    P("== p_hind change by source-trajectory outcome (judge, first 60 turns) ==")
    for oc, x in rep["p_hind_by_outcome"].items():
        P(f"  {oc:<7} judge n {x['judge_n']:>3} SAME {_f(x['judge_same'])} taxonomy {x['taxonomy']}")
    P("σ per dialect = pooled within-turn sd of the 3 blind-own values (A_blind: under the blind thoughts; A_priv_<v>: the same blind-own actions under the informed thoughts):")
    for k, v in rep["sigma"].items():
        P(f"  {k:<14} A_sum {_f(v['A_sum'],6,3)} A_pb {_f(v['A_pb'],6,4)} R {_f(v['R'],6,4)} A_sum_stored {_f(v['A_sum_stored'],6,3)} | "
          + " ".join(f"A_priv_{ltag}_{vv} {_f(v.get(f'A_priv_{ltag}_{vv}'),6,3 if ltag=='sum' else 4)}" for vv in VARIANTS for ltag in ("sum", "pb")))
    txt = "\n".join(lines) + "\n"
    (RESULTS / "report.txt").write_text(txt)
    print(txt)
    print(f"-> {RESULTS / 'report.txt'} / report.json / turn_metrics.jsonl")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("notes")
    a.set_defaults(fn=cmd_notes)
    b = sub.add_parser("sample")
    b.add_argument("--concurrency", type=int, default=24)
    b.set_defaults(fn=cmd_sample)
    c = sub.add_parser("judge")
    c.add_argument("--n", type=int, default=60)
    c.set_defaults(fn=cmd_judge)
    d = sub.add_parser("echo")
    d.add_argument("--turns", type=int, default=0)
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--concurrency", type=int, default=24)
    d.set_defaults(fn=cmd_echo)
    e = sub.add_parser("report")
    e.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
