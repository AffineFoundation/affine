"""Privileged teacher references (design P1/P2/P4) — does a per-turn
privileged context p, shown to the teacher and never to the miner, change
the teacher's action beyond sampling noise while its thoughts stay inside
the BLIND typicality band under the live wvk-22 content-masked meter?

    python privileged_refs.py turns   [--n 150]        # select + materialize
    python privileged_refs.py notes                    # p_fr via glm-5.3 (T=0)
    python privileged_refs.py sample  [--turns N]      # k=3 teacher refs / variant
    python privileged_refs.py echo    [--turns N]      # teacher-forced echoes (blind x)
    python privileged_refs.py judge   [--n 60]         # glm-5.3-flash same-decision judge
    python privileged_refs.py report

Everything API-only through Engy (teacher qwen3.8-27b for sampling + echo,
glm-5.3 for the structural critique, glm-5.3-flash as judge). Work files
under research/results/frontier_arbiter/privileged_refs/ (jsonl, resumable).

Terms (one line each):
  x / z / y        turn prefix / thought / action (dialect action span).
  blind refs       the k=3 stored teacher refs of the wvk-22 verdict, ~T(.|x).
  p_none           fresh blind refs sampled now — the sampling-noise floor.
  p_obs            hindsight: first <=800 chars of the observation that followed
                   the recorded action + the trajectory's final outcome.
  p_out            outcome only (SOLVED/FAILED) + "consider what would change that".
  p_fr             glm-5.3 structural critique (<=60 words, no command/path/code).
  p_tpl            template reminder (repeated command / done-state), where a family applies.
  R_F(z)           centered tempered LME (tau 0.03) of a_i = lpC(y_i^F|x,z) - lpC(y_i^F|x,0)
                   over family F's refs; own thoughts leave-one-out (2 refs), external
                   thoughts averaged over the same 2-ref subsets (matched estimator).
  m_c              content-masked mean token logprob of z under x (live content_stats, theta=1).
  in-band          |m_c(z) - mean_3 m_c(blind)| <= 2 * sigma_Mc(dialect)  (live typ_c >= 0).
  B                lpC(y|x,z) - lpC(y|x,0) per byte; licence at >= 0.02.
  leak             a >=30-char substring of p appears in the ref's z or y.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import math
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    DATA, REPO, TEACHER_ENGY, Engy, TeacherEcho, agree, clme, exact, jaccard,
    load_verdict, materialize, norm_action, read_jsonl, append_jsonl, write_jsonl,
    reply_to_rollout, corpus_for, render_tool_call, repair_tool_call,
)
from affine import dialects  # noqa: E402
from affine.corpus.trace import sampled_paths, message_text  # noqa: E402
from affine.corpus.view import main_final_index, main_root_indices, rollout_outcome  # noqa: E402
from evalsrv import chat as live_chat  # noqa: E402
from evalsrv.sdmeter import content_stats  # noqa: E402

live_chat.set_thought_rendering("as_generated")

RESULTS = REPO / "research" / "results" / "frontier_arbiter" / "privileged_refs"
TURNS = RESULTS / "turns.jsonl"
NOTES = RESULTS / "notes.jsonl"
SAMPLES = RESULTS / "samples.jsonl"
ECHOES = RESULTS / "echoes.jsonl"
JUDGE = RESULTS / "judge.jsonl"
COST = RESULTS / "cost.jsonl"

VERDICTS = ("chal-00614", "chal-00621", "chal-00623")
DIALECTS = ("bash", "tool_call", "terminus_json")
MIN_DEPTH = 4
MAX_PREFIX_CHARS = 90_000
VARIANTS = ("p_none", "p_obs", "p_out", "p_fr", "p_tpl")
K = 3
TAU = 0.03
REF_TEMPERATURE = 0.8
REF_MAX_TOKENS = 4096
OBS_CHARS = 800
NOTE_HEAD = "Privileged note for this step (the assistant cannot see this): "
FRONTIER = "glm-5.3"
JUDGE_MODEL = "glm-5.3-flash"
TRACE_MIRROR = Path("/tmp/fa/chunks")
TRACE_INDEX = Path("/tmp/vr/rollouts.jsonl")
DATA_BASE = "https://data.affine.io/"
UA = {"User-Agent": "curl/8.5"}
LIVE_SPLIT = {"require_think_close": True, "text_fallback_at_tool_turns": True}


# ------------------------------------------------------------------ cost ledger
def log_cost(stage: str, engy: Engy, note: str = "") -> None:
    rec = {"at": time.time(), "stage": stage, "cost_usd": engy.cost_usd,
           "usage": engy.usage, "note": note}
    append_jsonl(COST, rec)
    print(f"  [$] {stage}: this run ${engy.cost_usd:.3f} | total so far ${total_cost():.2f} {note}",
          flush=True)


def total_cost() -> float:
    """Sum of the final ledger line of every (stage, run). Each run appends a
    cumulative `cost_usd` per checkpoint with note "done/total"; runs are told
    apart by (stage, total) — parallel echo workers carry distinct totals — and,
    within a group, by a cumulative value dropping (a fresh process)."""
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


# ------------------------------------------------------------------ traces
def trace_index() -> dict[str, tuple[str, int]]:
    out = {}
    with open(TRACE_INDEX) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            out[r["rollout_id"]] = (r["chunk"], r["line"])
    return out


_chunk_cache: dict[str, list[str]] = {}


def load_envelope(chunk: str, line: int) -> dict:
    if chunk not in _chunk_cache:
        if len(_chunk_cache) > 6:
            _chunk_cache.pop(next(iter(_chunk_cache)))
        path = TRACE_MIRROR / chunk
        if not path.exists():
            with httpx.Client(headers=UA, timeout=180) as cli:
                r = cli.get(DATA_BASE + "traces/chunks/" + chunk)
                r.raise_for_status()
                path.write_bytes(r.content)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _chunk_cache[chunk] = f.read().split("\n")
    return json.loads(_chunk_cache[chunk][line])


def newer_chunks(since_created: str) -> list[dict]:
    """Trace chunks published after the local mirror's manifest."""
    with httpx.Client(headers=UA, timeout=180) as cli:
        man = cli.get(DATA_BASE + "traces/manifest.json").json()
    return [c for c in man["chunks"] if c["created_at"] > since_created]


def index_new_chunks(chunks: list[dict], want: set[str]) -> dict[str, tuple[str, int]]:
    """Download newer chunks and index the rollout ids we still need."""
    found: dict[str, tuple[str, int]] = {}
    with httpx.Client(headers=UA, timeout=300) as cli:
        for c in chunks:
            name = c["key"].split("/")[-1]
            path = TRACE_MIRROR / name
            if not path.exists():
                r = cli.get(DATA_BASE + c["key"])
                r.raise_for_status()
                path.write_bytes(r.content)
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    # cheap scan before a full parse
                    m = re.search(r'"rollout_id":\s*"([0-9a-f]{32})"', line[:4000])
                    rid = m.group(1) if m else json.loads(line).get("rollout_id")
                    if rid in want:
                        found[rid] = (name, i)
    return found


def _msg_key(m: dict) -> tuple:
    return (m.get("role"), m.get("content"))


def downstream(trace: dict, turn_idx: int) -> dict:
    """What followed the recorded reply `turn_idx`: observation text (the
    tool / user messages before the next sampled reply on the same branch),
    whether the reply was the last one on its branch / the main root, the
    number of replies, and the trajectory outcome."""
    paths = sampled_paths(trace)
    if turn_idx >= len(paths):
        raise ValueError(f"turn_idx {turn_idx} beyond {len(paths)} sampled replies")
    me = paths[turn_idx]
    n = len(me)
    obs_msgs = None
    for j in range(turn_idx + 1, len(paths)):
        pj = paths[j]
        if len(pj) > n and all(_msg_key(a) == _msg_key(b) for a, b in zip(pj[:n], me)):
            obs_msgs = pj[n:-1]
            break
    obs = None
    if obs_msgs is not None:
        obs = "\n".join(message_text(m.get("content")) for m in obs_msgs).strip()
    main = main_root_indices(trace)
    final_idx = main_final_index(trace)
    next_main = None
    if turn_idx in main:
        pos = main.index(turn_idx)
        next_main = main[pos + 1] if pos + 1 < len(main) else None
    return {
        "obs": obs,
        "has_next": obs_msgs is not None,
        "is_last_main": bool(main) and main[-1] == turn_idx,
        "next_is_final": next_main is not None and next_main == final_idx,
        "n_replies": len(paths),
        "n_main": len(main),
        "outcome": rollout_outcome(trace),
        "stop": trace.get("stop_condition"),
        "reply_role_ok": me[-1].get("role") == "assistant",
    }


# ------------------------------------------------------------------ template family
def prefix_actions(prefix: list[dict], kind: str) -> list[tuple[int, str, str]]:
    """(assistant index, action, following observation) for earlier replies in x."""
    out = []
    a_idx = -1
    for i, m in enumerate(prefix):
        if m["role"] != "assistant":
            continue
        a_idx += 1
        _, y = dialects.split_action(m["content"], kind)
        if not y:
            continue
        nxt = prefix[i + 1]["content"] if i + 1 < len(prefix) and prefix[i + 1]["role"] == "user" else ""
        out.append((a_idx, y, nxt))
    return out


def template_note(prefix: list[dict], kind: str, y_rec: str, down: dict) -> dict | None:
    if y_rec:
        ny = norm_action(y_rec, kind)
        for a_idx, y_prev, obs_prev in prefix_actions(prefix, kind):
            if norm_action(y_prev, kind) == ny:
                snippet = re.sub(r"\s+", " ", obs_prev)[:300]
                return {"family": "repeat",
                        "note": (f"Reminder: this exact command was already run at assistant turn "
                                 f"{a_idx + 1} with this result: {snippet!r}. Do not repeat it; "
                                 "use what it showed.")}
    if down.get("next_is_final") and down.get("outcome") == "solved":
        return {"family": "done",
                "note": "The last observation already answers the task; finish now rather "
                        "than gathering more information."}
    return None


# ------------------------------------------------------------------ stage: turns
def eligible_turns(chal: str) -> list[dict]:
    d = load_verdict(chal)
    sl = d["verdict"]["slice"]
    corpus = corpus_for(sl["manifest_sha256"], sl.get("corpus_base_url"))
    rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
    sigma = ((d["verdict"].get("shadow") or {}).get("sd_meter") or {}).get("sigma_by_dialect") or {}
    king = {r["turn_id"]: r for r in d["king_rows"]}
    out = []
    for tid in d["turn_ids"]:
        row = rows.get(tid)
        if not row or row["action_kind"] not in DIALECTS:
            continue
        if int(row["turn_idx"]) < MIN_DEPTH or int(row["n_prefix_chars"]) > MAX_PREFIX_CHARS:
            continue
        refs = d["teacher_refs"].get(tid)
        if not refs or len(refs) != K:
            continue
        if any(r.get(k) is None for r in refs for k in
               ("z", "y", "lp_own", "lp_empty", "lp_thought", "lp_thought_e", "mc_thought", "lp_cross")):
            continue
        if any(len(r["lp_cross"]) != K or sum(v is None for v in r["lp_cross"]) != 1 for r in refs):
            continue
        kr = king.get(tid)
        if not kr or not kr.get("valid") or len(kr.get("pairs") or []) != K:
            continue
        if any(p.get("mc_za") is None or p.get("lpC_za_e") is None for p in kr["pairs"]):
            continue
        out.append({"turn_id": tid, "chal": chal, "dialect": row["action_kind"], "source": row["source"],
                    "turn_idx": int(row["turn_idx"]), "rollout_id": row["rollout_id"],
                    "n_prefix_chars": int(row["n_prefix_chars"]), "stratum": row["stratum"],
                    "refs": refs, "king": kr, "sigma_mc": (sigma.get(row["action_kind"]) or {}).get("Mc"),
                    "sigma_r": (sigma.get(row["action_kind"]) or {}).get("R")})
    return out


def cmd_turns(args: argparse.Namespace) -> None:
    rng = random.Random(20260921)
    cands: list[dict] = []
    for chal in VERDICTS:
        e = eligible_turns(chal)
        print(f"{chal}: {len(e)} eligible", collections.Counter(t["dialect"] for t in e))
        cands.extend(e)
    # one turn per rollout, dialect-stratified sample with a preference order
    by_roll: dict[str, list[dict]] = collections.defaultdict(list)
    for t in cands:
        by_roll[t["rollout_id"]].append(t)
    per_roll = [rng.choice(v) for v in by_roll.values()]
    rng.shuffle(per_roll)
    quota = {"bash": args.n * 0.4, "tool_call": args.n * 0.4, "terminus_json": args.n * 0.2}
    tidx = trace_index()
    missing = {t["rollout_id"] for t in per_roll if t["rollout_id"] not in tidx}
    if missing:
        print(f"{len(missing)} rollouts not in the local trace mirror; scanning newer chunks")
        newer = newer_chunks("2026-09-20T13:57")
        print(f"  {len(newer)} newer chunks")
        tidx.update(index_new_chunks(newer, missing))
    picked: list[dict] = []
    taken = collections.Counter()
    pool = collections.defaultdict(list)
    for t in per_roll:
        pool[t["dialect"]].append(t)
    # round-robin over dialects so quotas fill evenly; overflow to bash/tool_call
    order = sorted(pool, key=lambda k: -quota[k])
    exhausted = set()
    while len(picked) < args.n and len(exhausted) < len(pool):
        progressed = False
        for k in order:
            if k in exhausted:
                continue
            if taken[k] >= quota[k] and any(taken[o] < quota[o] and o not in exhausted for o in pool):
                continue
            while pool[k]:
                t = pool[k].pop()
                if t["rollout_id"] not in tidx:
                    continue
                try:
                    env = load_envelope(*tidx[t["rollout_id"]])
                    down = downstream(env["trace"], t["turn_idx"])
                except Exception as ex:  # noqa: BLE001 — drop the turn, keep going
                    print(f"  drop {t['turn_id']}: {type(ex).__name__}: {str(ex)[:120]}")
                    continue
                if not down["reply_role_ok"] or not down["obs"] or down["outcome"] not in ("solved", "failed"):
                    continue
                t["down"] = {k2: v for k2, v in down.items() if k2 != "obs"}
                t["obs"] = down["obs"][:OBS_CHARS]
                t["policy_id"] = (env.get("policy") or {}).get("id")
                t["harness"] = (env.get("policy") or {}).get("harness")
                picked.append(t)
                taken[k] += 1
                progressed = True
                break
            else:
                exhausted.add(k)
            if len(picked) >= args.n:
                break
        if not progressed and len(exhausted) >= len(pool):
            break
    print(f"picked {len(picked)}", dict(taken))
    # materialize prefixes per verdict, attach y_rec + template family
    out = []
    for chal in VERDICTS:
        sub = [t for t in picked if t["chal"] == chal]
        if not sub:
            continue
        mat = materialize(chal, [t["turn_id"] for t in sub])
        for t in sub:
            m = mat.get(t["turn_id"])
            if not m:
                print(f"  materialize miss {t['turn_id']}")
                continue
            t["prefix"] = m["prefix"]
            t["reference_turn"] = m["reference_turn"]
            _, y_rec = dialects.split_action(m["reference_turn"], t["dialect"])
            t["y_rec"] = y_rec
            t["tpl"] = template_note(t["prefix"], t["dialect"], y_rec, t["down"])
            out.append(t)
    write_jsonl(TURNS, out)
    n_tpl = collections.Counter((t["tpl"] or {}).get("family") for t in out)
    print(f"{len(out)} turns -> {TURNS}; template families {dict(n_tpl)}; "
          f"outcomes {collections.Counter(t['down']['outcome'] for t in out)}; "
          f"prefix chars p50 {st.median(t['n_prefix_chars'] for t in out):.0f} "
          f"max {max(t['n_prefix_chars'] for t in out)}")


# ------------------------------------------------------------------ stage: notes (p_fr)
FR_SYSTEM = (
    "You are a senior engineer reviewing an AI coding/terminal agent's transcript at one step. "
    "Write a STRUCTURAL note (at most 60 words) about what the agent should check or avoid at its "
    "NEXT step: reasoning traps, unverified assumptions, missing evidence, when to stop. "
    "Hard rules: do NOT give any command, file path, filename, function/variable name, code, "
    "or backticks. No quotes of the transcript. Plain prose, one paragraph, no lists."
)
PATH_RE = re.compile(r"(?<![\w])(?:\.{0,2}/)?[\w.-]+(?:/[\w.-]+)+|\b[\w-]+\.(?:py|js|ts|tsx|go|rs|java|json|toml|yaml|yml|md|txt|sh|c|h|cpp|rb|php|cs|sql|cfg|ini|lock)\b")


def render_transcript(prefix: list[dict], cap: int = 160_000) -> str:
    parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in prefix]
    s = "\n\n".join(parts)
    if len(s) > cap:
        s = s[: cap // 2] + "\n\n[... middle of transcript elided ...]\n\n" + s[-cap // 2:]
    return s


def validate_note(note: str) -> str | None:
    if not note or not note.strip():
        return "empty"
    if "`" in note:
        return "backtick"
    if PATH_RE.search(note):
        return "path"
    if len(note.split()) > 75:
        return "too_long"
    if re.search(r"[{}]|\w\(\)|\w\(\w|==|=>|->|\$\(|\$\w|<<|>>|\b\w+\.\w+\(", note):
        return "code_like"
    return None


async def gen_notes(turns: list[dict], engy: Engy) -> None:
    done = {r["turn_id"] for r in read_jsonl(NOTES) if r.get("note")}
    todo = [t for t in turns if t["turn_id"] not in done]
    print(f"p_fr notes: {len(todo)} to generate")

    async def one(t: dict) -> None:
        tr = render_transcript(t["prefix"])
        user = (f"Transcript so far (the agent must now produce its next step; dialect: {t['dialect']}):\n\n"
                f"{tr}\n\n--- END OF TRANSCRIPT ---\nWrite the note now (<= 60 words, no commands/paths/code/backticks).")
        msgs = [{"role": "system", "content": FR_SYSTEM}, {"role": "user", "content": user}]
        rec = {"turn_id": t["turn_id"], "attempts": []}
        for attempt in range(2):
            try:
                # glm-5.3 thinks before it answers; a small cap returns an
                # empty visible note (39/56 "empty" in the first run at 400).
                r = await engy.chat(FRONTIER, msgs, temperature=0.0, max_tokens=4000)
            except Exception as ex:  # noqa: BLE001
                rec["attempts"].append({"error": repr(ex)[:200]})
                break
            note = (r["content"] or "").strip()
            why = validate_note(note)
            rec["attempts"].append({"note": note, "reject": why, "usage": r["usage"], "cost_usd": r["cost_usd"],
                                    "finish": r["finish"], "reasoning_chars": len(r["reasoning"] or "")})
            if why is None:
                rec["note"] = note
                break
            msgs = msgs + [{"role": "assistant", "content": note},
                           {"role": "user", "content": f"Rejected ({why}). Rewrite: plain prose, <= 60 words, "
                                                       "absolutely no commands, paths, filenames, identifiers, code or backticks."}]
        append_jsonl(NOTES, rec)

    for i in range(0, len(todo), 16):
        await asyncio.gather(*[one(t) for t in todo[i:i + 16]])
        log_cost("notes", engy, f"{min(i + 16, len(todo))}/{len(todo)}")


def cmd_notes(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    engy = Engy(concurrency=8)
    asyncio.run(gen_notes(turns, engy))
    notes = read_jsonl(NOTES)
    ok = sum(1 for n in notes if n.get("note"))
    rej = collections.Counter(a.get("reject") for n in notes for a in n["attempts"] if a.get("reject"))
    print(f"notes ok {ok}/{len(notes)}; rejections {dict(rej)}")


# ------------------------------------------------------------------ stage: sample
def note_text(t: dict, variant: str, notes: dict[str, str]) -> str | None:
    outcome = t["down"]["outcome"].upper()
    if variant == "p_none":
        return ""
    if variant == "p_obs":
        return (f"Hindsight: after the step the assistant actually took here, the environment returned:\n"
                f"<<<\n{t['obs']}\n>>>\nThe trajectory eventually {outcome} the task.")
    if variant == "p_out":
        return f"This trajectory eventually {outcome} the task; consider what would change that at this step."
    if variant == "p_fr":
        n = notes.get(t["turn_id"])
        return n if n else None
    if variant == "p_tpl":
        return t["tpl"]["note"] if t.get("tpl") else None
    raise ValueError(variant)


def with_note(prefix: list[dict], p: str) -> list[dict]:
    if not p:
        return prefix
    msgs = [dict(m) for m in prefix]
    msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + NOTE_HEAD + p
    return msgs


def parse_reply(reply: dict, kind: str) -> dict:
    """Live-flag parse (require_think_close, text fallback at tool turns)."""
    r = reply_to_rollout(reply, kind)
    content = reply.get("content") or ""
    if reply.get("tool_calls"):
        content = content.rstrip() + "\n" + "\n".join(render_tool_call(c) for c in reply["tool_calls"])
    if kind == "tool_call":
        content, _ = repair_tool_call(content)
    text = (reply.get("reasoning") or "") + "\n" + live_chat.THINK_CLOSE + "\n" + content
    z, y = live_chat.split_rollout(text, kind, **LIVE_SPLIT)
    used = kind
    if y and not dialects.split_action(content, kind)[1]:
        used = "text"
    r.update({"z": z, "y": y, "parsed": bool(y), "kind_used": used if y else None,
              "think_closed": bool(reply.get("reasoning")) or live_chat.THINK_CLOSE in content})
    return r


async def sample_all(turns: list[dict], engy: Engy, notes: dict[str, str]) -> None:
    done = {(r["turn_id"], r["variant"], r["i"]) for r in read_jsonl(SAMPLES)}
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
                               "reasoning": r["reasoning"], "content": r["content"],
                               "tool_calls": r["tool_calls"], "finish": r["finish"], "usage": r["usage"],
                               "cost_usd": r["cost_usd"], **{k: parsed[k] for k in
                                                              ("z", "y", "parsed", "kind_used", "think_closed", "repaired")}})

    step = 48
    for s in range(0, len(jobs), step):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + step]])
        log_cost("sample", engy, f"{min(s + step, len(jobs))}/{len(jobs)}")


def cmd_sample(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)[: args.turns] if args.turns else read_jsonl(TURNS)
    notes = {n["turn_id"]: n["note"] for n in read_jsonl(NOTES) if n.get("note")}
    engy = Engy(concurrency=args.concurrency)
    asyncio.run(sample_all(turns, engy, notes))


# ------------------------------------------------------------------ stage: echo
def ref_sets(turn: dict, samples: list[dict]) -> dict[str, list[dict]]:
    """variant -> parsed refs [{i, z, y}] for one turn (blind = stored)."""
    out = {"blind": [{"i": i, "z": r["z"], "y": r["y"]} for i, r in enumerate(turn["refs"])]}
    for s in samples:
        if s["turn_id"] != turn["turn_id"] or not s.get("parsed"):
            continue
        out.setdefault(s["variant"], []).append({"i": s["i"], "z": s["z"], "y": s["y"]})
    return out


def read_echoes() -> list[dict]:
    """All echo records: the main file plus any parallel-worker part files."""
    out: list[dict] = []
    for p in sorted(RESULTS.glob("echoes*.jsonl")):
        out.extend(read_jsonl(p))
    return out


def echo_jobs(turn: dict, sets: dict[str, list[dict]], done: set[str]) -> list[tuple]:
    """(key, kind, prefix, z, y). key = f"{tid}|{what}|..." — everything is
    echoed under the BLIND prefix x."""
    tid = turn["turn_id"]
    prefix = turn["prefix"]
    jobs: list[tuple] = []
    blind = sets["blind"]
    king_z = turn["king"]["pairs"][0]["z_a"]

    def add(key, kind, z, y=None):
        if key not in done:
            jobs.append((key, kind, prefix, z, y))

    # per-turn once: blind thoughts + king thought re-echoed on this stack (m_c parity)
    for r in blind:
        add(f"{tid}|th|blind|{r['i']}", "thought", r["z"])
        add(f"{tid}|un|blind|{r['i']}", "uncond", r["z"])
    add(f"{tid}|th|king|0", "thought", king_z)
    add(f"{tid}|un|king|0", "uncond", king_z)
    for v, refs in sets.items():
        if v == "blind" or len(refs) < 2:
            continue
        for r in refs:
            add(f"{tid}|emp|{v}|{r['i']}", "action", "", r["y"])              # lpC(y^v_i | x, 0)
            add(f"{tid}|own|{v}|{r['i']}", "action", r["z"], r["y"])          # B numerator
            add(f"{tid}|th|{v}|{r['i']}", "thought", r["z"])                  # lpC(z^v_i | x) tokens
            add(f"{tid}|un|{v}|{r['i']}", "uncond", r["z"])                   # lpC(z^v_i | 0) tokens
            for r2 in refs:                                                   # within-family cross
                if r2["i"] != r["i"]:
                    add(f"{tid}|x|{v}|{r['i']}|{v}|{r2['i']}", "action", r2["z"], r["y"])
            for b in blind:                                                   # blind thoughts -> v actions
                add(f"{tid}|x|{v}|{r['i']}|blind|{b['i']}", "action", b["z"], r["y"])
                add(f"{tid}|x|blind|{b['i']}|{v}|{r['i']}", "action", r["z"], b["y"])   # v thoughts -> blind actions
            add(f"{tid}|x|{v}|{r['i']}|king|0", "action", king_z, r["y"])     # king thought -> v actions
    return jobs


async def echo_all(turns: list[dict], engy: Engy, start: int = 0) -> None:
    te = TeacherEcho(engy)
    samples = read_jsonl(SAMPLES)
    done = {r["key"] for r in read_echoes()}
    # parallel workers (--start > 0) append to their own part file so lines never interleave
    out_path = ECHOES if start == 0 else RESULTS / f"echoes.part{start}.jsonl"
    jobs: list[tuple] = []
    for t in turns:
        jobs.extend(echo_jobs(t, ref_sets(t, samples), done))
    print(f"echo: {len(jobs)} echoes to run")

    async def one(key, kind, prefix, z, y):
        try:
            if kind == "action":
                r = await te.lp_action(prefix, z, y)
            elif kind == "thought":
                r = await te.lp_thought(prefix, z, tokens=True)
            else:
                r = await te.lp_thought_uncond(z, tokens=True)
            append_jsonl(out_path, {"key": key, **r})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(out_path, {"key": key, "error": repr(ex)[:300]})

    step = 96
    # order: per turn, so the prefix stays hot in whatever cache Engy has
    for s in range(0, len(jobs), step):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + step]])
        log_cost("echo", engy, f"{min(s + step, len(jobs))}/{len(jobs)}")


def cmd_echo(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)[args.start:]
    turns = turns[: args.turns] if args.turns else turns
    engy = Engy(concurrency=args.concurrency)
    asyncio.run(echo_all(turns, engy, args.start))


# ------------------------------------------------------------------ stage: judge
JUDGE_SYSTEM = (
    "You compare two candidate next actions of a coding/terminal agent at the same point of a "
    "transcript. Answer whether they represent the SAME DECISION (same intent / same next step, "
    "e.g. both inspect the same thing, both apply the same fix, both finish) even if the exact "
    "text differs. Reply with exactly one word: SAME or DIFFERENT."
)


async def judge_all(turns: list[dict], engy: Engy, n_turns: int) -> None:
    samples = read_jsonl(SAMPLES)
    # only a decided verdict counts as done: the thinking judge returns empty content when
    # its reasoning eats the token budget, and those rows are retried
    done = {(r["turn_id"], r["variant"], r["pair"]) for r in read_jsonl(JUDGE) if r.get("same") is not None}
    by_turn = collections.defaultdict(list)
    for s in samples:
        if s.get("parsed"):
            by_turn[s["turn_id"]].append(s)
    jobs = []
    for t in turns[:n_turns]:
        blind = turn_refs = t["refs"]
        tail = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in t["prefix"][-3:])
        if len(tail) > 12_000:
            tail = tail[-12_000:]
        sets = collections.defaultdict(list)
        for s in by_turn[t["turn_id"]]:
            sets[s["variant"]].append(s)
        for v, refs in sets.items():
            refs = sorted(refs, key=lambda s: s["i"])
            # pair 0: variant ref 0 vs blind ref 0; pair 1: variant ref 1 vs blind ref 1
            for pi in range(min(2, len(refs), len(blind))):
                if (t["turn_id"], v, pi) in done:
                    continue
                jobs.append((t["turn_id"], v, pi, tail, blind[pi]["y"], refs[pi]["y"]))
        # blind self-agreement floor: blind ref 1 vs blind ref 2
        if (t["turn_id"], "blind_self", 0) not in done and len(blind) >= 3:
            jobs.append((t["turn_id"], "blind_self", 0, tail, blind[1]["y"], blind[2]["y"]))
    print(f"judge: {len(jobs)} comparisons")

    async def one(tid, v, pi, tail, ya, yb):
        user = (f"Recent transcript context:\n{tail}\n\n--- Candidate action A ---\n{ya[:3000]}\n\n"
                f"--- Candidate action B ---\n{yb[:3000]}\n\nSAME or DIFFERENT?")
        try:
            r = await engy.chat(JUDGE_MODEL, [{"role": "system", "content": JUDGE_SYSTEM},
                                              {"role": "user", "content": user}], temperature=0.0, max_tokens=2000)
            ans = (r["content"] or "").strip().upper()
            if not ans.startswith(("SAME", "DIFF")):
                # thinking model: fall back to the last verdict word in the reasoning
                words = re.findall(r"\b(SAME|DIFFERENT)\b", (r["content"] or "") + " " + (r["reasoning"] or ""), re.I)
                ans = words[-1].upper() if words else ans
            same = ans.startswith("SAME") if ans.startswith(("SAME", "DIFF")) else None
            append_jsonl(JUDGE, {"turn_id": tid, "variant": v, "pair": pi, "same": same, "raw": ans[:40],
                                 "cost_usd": r["cost_usd"]})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(JUDGE, {"turn_id": tid, "variant": v, "pair": pi, "same": None, "error": repr(ex)[:200]})

    for s in range(0, len(jobs), 32):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + 32]])
        log_cost("judge", engy, f"{min(s + 32, len(jobs))}/{len(jobs)}")


def cmd_judge(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    engy = Engy(concurrency=16)
    asyncio.run(judge_all(turns, engy, args.n))


# ------------------------------------------------------------------ stage: report
def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return st.mean(v) if v else None


def _paired(diffs: list[float]) -> dict:
    d = [x for x in diffs if x is not None and math.isfinite(x)]
    if len(d) < 3:
        return {"n": len(d), "mean": _mean(d), "se": None, "z": None}
    se = st.stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "mean": st.mean(d), "se": se, "z": (st.mean(d) / se) if se > 0 else None}


def loo_R(a_by_ref: dict[int, float], own: int | None, tau: float = TAU) -> float | None:
    """Matched 2-ref estimator. own=j: clme over i != j. own=None (external
    thought): mean over j of clme over i != j. Needs 3 refs."""
    ids = sorted(a_by_ref)
    if own is not None:
        vals = [a_by_ref[i] for i in ids if i != own]
        return clme(vals, tau) if len(vals) >= 2 else None
    if len(ids) < 3:
        return clme([a_by_ref[i] for i in ids], tau) if len(ids) >= 2 else None
    return st.mean(clme([a_by_ref[i] for i in ids if i != j], tau) for j in ids)


def full_R(a_by_ref: dict[int, float], tau: float = TAU) -> float | None:
    vals = list(a_by_ref.values())
    return clme(vals, tau) if len(vals) >= 2 else None


def leaks(p: str, text: str, n: int = 30) -> bool:
    if not p or not text:
        return False
    pn = re.sub(r"\s+", " ", p)
    tn = re.sub(r"\s+", " ", text)
    for s in range(0, max(1, len(pn) - n + 1), 3):
        w = pn[s:s + n]
        if len(w) == n and w in tn:
            return True
    return False


def cmd_report(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    samples = [s for s in read_jsonl(SAMPLES) if "error" not in s]
    sample_errors = [s for s in read_jsonl(SAMPLES) if "error" in s]
    echoes = {r["key"]: r for r in read_echoes() if "error" not in r}
    echo_errors = [r for r in read_echoes() if "error" in r]
    judge = read_jsonl(JUDGE)
    notes = {n["turn_id"]: n for n in read_jsonl(NOTES)}
    by_turn = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in samples:
        by_turn[s["turn_id"]][s["variant"]].append(s)
    variants = [v for v in VARIANTS]
    rep: dict = {"n_turns": len(turns), "dialects": dict(collections.Counter(t["dialect"] for t in turns)),
                 "outcomes": dict(collections.Counter(t["down"]["outcome"] for t in turns)),
                 "sample_errors": len(sample_errors), "echo_errors": len(echo_errors)}
    per_turn_rows: list[dict] = []

    for t in turns:
        tid = t["turn_id"]
        kind = t["dialect"]
        blind = t["refs"]
        blind_ys = [r["y"] for r in blind]
        king = t["king"]
        king_z = king["pairs"][0]["z_a"]
        sig = t.get("sigma_mc")
        # blind band (Engy stack m_c) + stored
        mc_blind_engy = []
        for i in range(K):
            ex_ = echoes.get(f"{tid}|th|blind|{i}")
            eu = echoes.get(f"{tid}|un|blind|{i}")
            if ex_ and eu:
                cs = content_stats([tuple(x) for x in ex_["tokens"]], [tuple(x) for x in eu["tokens"]], 1.0)
                mc_blind_engy.append(cs["mc"])
            else:
                mc_blind_engy.append(None)
        mc_blind_stored = [r["mc_thought"] for r in blind]
        row = {"turn_id": tid, "dialect": kind, "chal": t["chal"], "outcome": t["down"]["outcome"],
               "n_prefix_chars": t["n_prefix_chars"], "mc_blind_engy": mc_blind_engy,
               "mc_blind_stored": mc_blind_stored, "sigma_mc": sig, "sigma_r": t.get("sigma_r"),
               "variants": {}}
        # blind R (stored, LOO) for blind own thoughts and king
        a_blind_own = {}
        for j in range(K):
            a = {i: blind[i]["lp_cross"][j] - blind[i]["lp_empty"] for i in range(K) if i != j}
            a_blind_own[j] = clme(list(a.values()), TAU)
        row["R_blind_blindown"] = st.mean(a_blind_own.values())
        # king pairs align with refs by lpC_yc_zc == lp_own
        a_king = {}
        for p in king["pairs"]:
            i = next((i for i in range(K) if abs(blind[i]["lp_own"] - p["lpC_yc_zc"]) < 1e-12), None)
            if i is not None:
                a_king[i] = p["lpC_yc_za"] - p["lpC_yc_e"]
        row["R_blind_king"] = loo_R(a_king, None) if len(a_king) == K else None
        row["R_blind_king_full"] = full_R(a_king) if len(a_king) >= 2 else None
        ek = echoes.get(f"{tid}|th|king|0")
        ku = echoes.get(f"{tid}|un|king|0")
        row["mc_king_engy"] = content_stats([tuple(x) for x in ek["tokens"]], [tuple(x) for x in ku["tokens"]], 1.0)["mc"] if ek and ku else None
        row["mc_king_stored"] = king["pairs"][0]["mc_za"]
        row["blind_B"] = [r["lp_own"] - r["lp_empty"] for r in blind]
        row["blind_self_exact"] = st.mean(1.0 if norm_action(blind_ys[a], kind) == norm_action(blind_ys[b], kind) else 0.0
                                          for a in range(K) for b in range(a + 1, K))
        row["blind_self_jac"] = st.mean(jaccard(blind_ys[a], blind_ys[b]) for a in range(K) for b in range(a + 1, K))
        row["blind_len_z"] = [len(r["z"]) for r in blind]

        for v in variants:
            ss = sorted(by_turn[tid].get(v, []), key=lambda s: s["i"])
            if not ss:
                continue
            vr = {"n_sampled": len(ss), "n_parsed": sum(1 for s in ss if s["parsed"]),
                  "cap_hits": sum(1 for s in ss if s["finish"] == "length"),
                  "think_closed": sum(1 for s in ss if s["think_closed"]),
                  "text_fallback": sum(1 for s in ss if s.get("kind_used") == "text"),
                  "len_z": [len(s["z"]) for s in ss if s["parsed"]],
                  "len_y": [len(s["y"]) for s in ss if s["parsed"]],
                  "completion_tokens": sum((s.get("usage") or {}).get("completion_tokens") or 0 for s in ss),
                  "prompt_tokens": sum((s.get("usage") or {}).get("prompt_tokens") or 0 for s in ss),
                  "note_chars": len(ss[0].get("note") or "")}
            refs = [s for s in ss if s["parsed"]]
            p = ss[0].get("note") or ""
            ys = [s["y"] for s in refs]
            # (a) action agreement vs stored blind; self-agreement
            vr["exact_in_blind"] = _mean([1.0 if exact(y, blind_ys, kind) else 0.0 for y in ys])
            vr["jac_to_blind"] = _mean([agree(y, blind_ys) for y in ys])
            ctrl = [s["y"] for s in sorted(by_turn[tid].get("p_none", []), key=lambda s: s["i"]) if s["parsed"]]
            if v != "p_none" and ctrl:
                vr["exact_in_ctrl"] = _mean([1.0 if exact(y, ctrl, kind) else 0.0 for y in ys])
                vr["jac_to_ctrl"] = _mean([agree(y, ctrl) for y in ys])
            if len(ys) >= 2:
                vr["self_exact"] = st.mean(1.0 if norm_action(ys[a], kind) == norm_action(ys[b], kind) else 0.0
                                           for a in range(len(ys)) for b in range(a + 1, len(ys)))
                vr["self_jac"] = st.mean(jaccard(ys[a], ys[b]) for a in range(len(ys)) for b in range(a + 1, len(ys)))
                vr["identical_all"] = 1.0 if len({norm_action(y, kind) for y in ys}) == 1 else 0.0
            # matches recorded action?
            if t.get("y_rec"):
                vr["exact_rec"] = _mean([1.0 if norm_action(y, kind) == norm_action(t["y_rec"], kind) else 0.0 for y in ys])
            # (d) leakage
            if v != "p_none":
                vr["leak_z"] = _mean([1.0 if leaks(p, s["z"]) else 0.0 for s in refs])
                vr["leak_y"] = _mean([1.0 if leaks(p, s["y"]) else 0.0 for s in refs])
            # echoes
            if len(refs) >= 2:
                emp = {s["i"]: echoes.get(f"{tid}|emp|{v}|{s['i']}") for s in refs}
                own = {s["i"]: echoes.get(f"{tid}|own|{v}|{s['i']}") for s in refs}
                if all(emp.values()) and all(own.values()):
                    B = [own[i]["lp_per_byte"] - emp[i]["lp_per_byte"] for i in emp]
                    vr["B"] = B
                    vr["B_pass"] = st.mean(1.0 if b >= 0.02 else 0.0 for b in B)
                    ids = [s["i"] for s in refs]

                    def a_of(thought_key: str) -> dict[int, float] | None:
                        out = {}
                        for i in ids:
                            e = echoes.get(f"{tid}|x|{v}|{i}|{thought_key}")
                            if e is None:
                                return None
                            out[i] = e["lp_per_byte"] - emp[i]["lp_per_byte"]
                        return out
                    # p-own (LOO): a_i for i != j from cross echoes + own echo is not used
                    own_vals = []
                    for j in ids:
                        a = {}
                        ok = True
                        for i in ids:
                            if i == j:
                                continue
                            e = echoes.get(f"{tid}|x|{v}|{i}|{v}|{j}")
                            if e is None:
                                ok = False
                                break
                            a[i] = e["lp_per_byte"] - emp[i]["lp_per_byte"]
                        if ok and len(a) >= 2:
                            own_vals.append(clme(list(a.values()), TAU))
                    if len(own_vals) == len(ids) and len(ids) == K:
                        vr["R_priv_own"] = st.mean(own_vals)
                    bl = []
                    bl_full = []
                    for b in range(K):
                        a = a_of(f"blind|{b}")
                        if a is not None and len(a) == K:
                            bl.append(loo_R(a, None))
                            bl_full.append(full_R(a))
                    if len(bl) == K:
                        vr["R_priv_blindown"] = st.mean(bl)
                        vr["R_priv_blindown_full"] = st.mean(bl_full)
                    a = a_of("king|0")
                    if a is not None and len(a) == K:
                        vr["R_priv_king"] = loo_R(a, None)
                        vr["R_priv_king_full"] = full_R(a)
                    # v thoughts under BLIND refs
                    rb = []
                    for s in refs:
                        a = {}
                        for b in range(K):
                            e = echoes.get(f"{tid}|x|blind|{b}|{v}|{s['i']}")
                            if e is None:
                                break
                            a[b] = e["lp_per_byte"] - blind[b]["lp_empty"]
                        if len(a) == K:
                            rb.append(loo_R(a, None))
                    if len(rb) == len(refs):
                        vr["R_blind_vown"] = st.mean(rb)
                # (c) typicality
                mcs = []
                n_content = []
                for s in refs:
                    ex_ = echoes.get(f"{tid}|th|{v}|{s['i']}")
                    eu = echoes.get(f"{tid}|un|{v}|{s['i']}")
                    if ex_ and eu:
                        cs = content_stats([tuple(x) for x in ex_["tokens"]], [tuple(x) for x in eu["tokens"]], 1.0)
                        mcs.append(cs["mc"])
                        n_content.append(cs["n_content"])
                vr["mc"] = mcs
                vr["n_content"] = n_content
                mb = [m for m in mc_blind_engy if m is not None]
                if len(mb) == K and mcs:
                    mu, sd = st.mean(mb), st.stdev(mb)
                    zs_turn = [((m - mu) / sd) if (m is not None and sd > 0) else None for m in mcs]
                    zs_live = [((m - mu) / sig) if (m is not None and sig) else None for m in mcs]
                    vr["z_turn"] = zs_turn
                    vr["z_live"] = zs_live
                    vr["in_band_live"] = _mean([1.0 if (z is not None and abs(z) <= 2.0 and nc >= 10) else 0.0
                                                for z, nc in zip(zs_live, n_content)])
                    vr["in_band_turn"] = _mean([1.0 if (z is not None and abs(z) <= 2.0) else 0.0 for z in zs_turn])
                    vr["content_floor"] = _mean([1.0 if nc < 10 else 0.0 for nc in n_content])
            row["variants"][v] = vr
        per_turn_rows.append(row)
    write_jsonl(RESULTS / "turn_metrics.jsonl", per_turn_rows)

    # ---------------- aggregate
    def agg(rows: list[dict], label: str) -> dict:
        out: dict = {"label": label, "n_turns": len(rows), "variants": {}}
        # blind reference numbers
        out["blind"] = {
            "self_exact": _mean([r["blind_self_exact"] for r in rows]),
            "self_jac": _mean([r["blind_self_jac"] for r in rows]),
            "B_pass": _mean([st.mean(1.0 if b >= 0.02 else 0.0 for b in r["blind_B"]) for r in rows]),
            "R_blind_blindown": _mean([r["R_blind_blindown"] for r in rows]),
            "R_blind_king": _mean([r["R_blind_king"] for r in rows]),
            "len_z_p50": st.median([x for r in rows for x in r["blind_len_z"]]) if rows else None,
            "mc_parity_engy_minus_stored": _mean([e - s_ for r in rows for e, s_ in zip(r["mc_blind_engy"], r["mc_blind_stored"]) if e is not None]),
            "mc_parity_abs": _mean([abs(e - s_) for r in rows for e, s_ in zip(r["mc_blind_engy"], r["mc_blind_stored"]) if e is not None]),
            "king_mc_parity": _mean([r["mc_king_engy"] - r["mc_king_stored"] for r in rows if r["mc_king_engy"] is not None]),
        }
        # king in-band vs the blind band (live sigma) — a reference point
        kz = []
        for r in rows:
            mb = [m for m in r["mc_blind_engy"] if m is not None]
            if len(mb) == K and r["mc_king_engy"] is not None and r["sigma_mc"]:
                kz.append(abs(r["mc_king_engy"] - st.mean(mb)) / r["sigma_mc"])
        out["blind"]["king_in_band_live"] = _mean([1.0 if z <= 2 else 0.0 for z in kz])
        out["blind"]["blind_R_blindown_minus_king"] = _paired([r["R_blind_blindown"] - r["R_blind_king"] for r in rows if r["R_blind_king"] is not None])
        # blind LOO in-band: each blind ref vs the other two (turn sd undefined with 2 → live sigma only)
        bz = []
        for r in rows:
            mb = r["mc_blind_engy"]
            if all(m is not None for m in mb) and r["sigma_mc"]:
                for j in range(K):
                    others = [mb[i] for i in range(K) if i != j]
                    bz.append(abs(mb[j] - st.mean(others)) / r["sigma_mc"])
        out["blind"]["blind_loo_in_band_live"] = _mean([1.0 if z <= 2 else 0.0 for z in bz])
        for v in variants:
            vs = [(r, r["variants"][v]) for r in rows if v in r["variants"]]
            if not vs:
                continue
            g: dict = {"n_turns": len(vs)}
            g["parse_rate"] = sum(x["n_parsed"] for _, x in vs) / max(1, sum(x["n_sampled"] for _, x in vs))
            g["cap_hit_rate"] = sum(x["cap_hits"] for _, x in vs) / max(1, sum(x["n_sampled"] for _, x in vs))
            g["think_close_rate"] = sum(x["think_closed"] for _, x in vs) / max(1, sum(x["n_sampled"] for _, x in vs))
            g["text_fallback_rate"] = sum(x["text_fallback"] for _, x in vs) / max(1, sum(x["n_sampled"] for _, x in vs))
            g["yield_ge2"] = _mean([1.0 if x["n_parsed"] >= 2 else 0.0 for _, x in vs])
            lz = [l for _, x in vs for l in x["len_z"]]
            g["len_z_p50"] = st.median(lz) if lz else None
            ly = [l for _, x in vs for l in x["len_y"]]
            g["len_y_p50"] = st.median(ly) if ly else None
            g["completion_tokens_per_ref"] = sum(x["completion_tokens"] for _, x in vs) / max(1, sum(x["n_sampled"] for _, x in vs))
            g["prompt_tokens_per_ref"] = sum(x["prompt_tokens"] for _, x in vs) / max(1, sum(x["n_sampled"] for _, x in vs))
            g["note_chars_mean"] = _mean([x["note_chars"] for _, x in vs])
            for key in ("exact_in_blind", "jac_to_blind", "exact_in_ctrl", "jac_to_ctrl", "self_exact", "self_jac",
                        "identical_all", "exact_rec", "leak_z", "leak_y", "B_pass", "in_band_live", "in_band_turn",
                        "content_floor"):
                g[key] = _mean([x.get(key) for _, x in vs])
            # action change beyond noise: paired (ctrl agreement − v agreement) per turn
            if v != "p_none":
                d = [(r["variants"]["p_none"].get("jac_to_blind") - x.get("jac_to_blind"))
                     for r, x in vs if "p_none" in r["variants"] and x.get("jac_to_blind") is not None
                     and r["variants"]["p_none"].get("jac_to_blind") is not None]
                g["jac_drop_vs_ctrl"] = _paired(d)
                d = [(r["variants"]["p_none"].get("exact_in_blind") - x.get("exact_in_blind"))
                     for r, x in vs if "p_none" in r["variants"] and x.get("exact_in_blind") is not None
                     and r["variants"]["p_none"].get("exact_in_blind") is not None]
                g["exact_drop_vs_ctrl"] = _paired(d)
            # headroom
            g["R_priv_own"] = _mean([x.get("R_priv_own") for _, x in vs])
            g["R_priv_blindown"] = _mean([x.get("R_priv_blindown") for _, x in vs])
            g["R_priv_king"] = _mean([x.get("R_priv_king") for _, x in vs])
            g["R_blind_vown"] = _mean([x.get("R_blind_vown") for _, x in vs])
            g["hz_own_minus_blindown"] = _paired([x["R_priv_own"] - x["R_priv_blindown"] for _, x in vs
                                                  if x.get("R_priv_own") is not None and x.get("R_priv_blindown") is not None])
            g["hz_blindown_minus_king"] = _paired([x["R_priv_blindown"] - x["R_priv_king"] for _, x in vs
                                                   if x.get("R_priv_blindown") is not None and x.get("R_priv_king") is not None])
            g["hz_own_minus_king"] = _paired([x["R_priv_own"] - x["R_priv_king"] for _, x in vs
                                              if x.get("R_priv_own") is not None and x.get("R_priv_king") is not None])
            g["hz_blind_vown_minus_blindown"] = _paired([x["R_blind_vown"] - r["R_blind_blindown"] for r, x in vs
                                                         if x.get("R_blind_vown") is not None])
            g["hz_blind_vown_minus_king"] = _paired([x["R_blind_vown"] - r["R_blind_king"] for r, x in vs
                                                     if x.get("R_blind_vown") is not None and r["R_blind_king"] is not None])
            # effect sizes in the live meter's sd units: mean over turns of diff / σ_R(dialect)
            g["eff_sd_own_minus_blindown"] = _mean([(x["R_priv_own"] - x["R_priv_blindown"]) / r["sigma_r"] for r, x in vs
                                                    if x.get("R_priv_own") is not None and x.get("R_priv_blindown") is not None and r.get("sigma_r")])
            g["eff_sd_blindown_minus_king"] = _mean([(x["R_priv_blindown"] - x["R_priv_king"]) / r["sigma_r"] for r, x in vs
                                                     if x.get("R_priv_blindown") is not None and x.get("R_priv_king") is not None and r.get("sigma_r")])
            g["eff_sd_own_minus_king"] = _mean([(x["R_priv_own"] - x["R_priv_king"]) / r["sigma_r"] for r, x in vs
                                                if x.get("R_priv_own") is not None and x.get("R_priv_king") is not None and r.get("sigma_r")])
            # cost per turn (tokens): duel-time = k refs sampled with the note in the prompt (+ the
            # echo work, identical to today's); fold-time = the note's own generation cost
            g["sample_prompt_tokens_per_turn"] = g["prompt_tokens_per_ref"] * K
            g["sample_completion_tokens_per_turn"] = g["completion_tokens_per_ref"] * K
            if v == "p_fr":
                nt = [notes[r["turn_id"]] for r, _ in vs if r["turn_id"] in notes]
                att = [a for n in nt for a in n["attempts"] if a.get("usage")]
                g["note_calls_per_turn"] = len(att) / max(1, len(nt))
                g["note_prompt_tokens_per_turn"] = sum((a["usage"].get("prompt_tokens") or 0) for a in att) / max(1, len(nt))
                g["note_completion_tokens_per_turn"] = sum((a["usage"].get("completion_tokens") or 0) for a in att) / max(1, len(nt))
                g["note_usd_per_turn"] = sum((a.get("cost_usd") or 0) for a in att) / max(1, len(nt))
            # in-band in sd units of the live sigma: mean |z|
            zl = [abs(z) for _, x in vs for z in (x.get("z_live") or []) if z is not None]
            g["abs_z_live_mean"] = _mean(zl)
            g["abs_z_live_p50"] = st.median(zl) if zl else None
            zs = [z for _, x in vs for z in (x.get("z_live") or []) if z is not None]
            g["z_live_mean"] = _mean(zs)
            out["variants"][v] = g
        # judge
        jt = {r["turn_id"] for r in rows}
        jj = [j for j in judge if j["turn_id"] in jt and j.get("same") is not None]
        out["judge"] = {}
        for v in variants + ["blind_self"]:
            sub = [j["same"] for j in jj if j["variant"] == v]
            if sub:
                out["judge"][v] = {"n": len(sub), "same_rate": st.mean(1.0 if s_ else 0.0 for s_ in sub)}
        return out

    rep["pooled"] = agg(per_turn_rows, "pooled")
    rep["by_dialect"] = {k: agg([r for r in per_turn_rows if r["dialect"] == k], k) for k in DIALECTS}
    rep["by_outcome"] = {k: agg([r for r in per_turn_rows if r["outcome"] == k], k) for k in ("solved", "failed")}
    rep["notes"] = {"n": len(notes), "ok": sum(1 for n in notes.values() if n.get("note")),
                    "rejections": dict(collections.Counter(a.get("reject") for n in notes.values() for a in n["attempts"] if a.get("reject")))}
    rep["tpl_families"] = dict(collections.Counter((t.get("tpl") or {}).get("family") for t in turns))
    rep["cost_usd_total"] = total_cost()
    cost_by_stage = collections.defaultdict(float)
    for c in read_jsonl(COST):
        cost_by_stage[c["stage"]] = max(cost_by_stage[c["stage"]], c["cost_usd"])
    rep["cost_by_stage_last_run"] = dict(cost_by_stage)
    (RESULTS / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print_report(rep)


def _f(x, w=6, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    if isinstance(x, dict):
        return _f(x.get("z"), w, 2)
    return f"{x:{w}.{p}f}"


def print_report(rep: dict) -> None:
    lines = []
    P = lines.append
    P("Privileged teacher references — probe report")
    P(f"turns {rep['n_turns']} {rep['dialects']} outcomes {rep['outcomes']}; sample errors {rep['sample_errors']}, echo errors {rep['echo_errors']}")
    P(f"p_fr notes ok {rep['notes']['ok']}/{rep['notes']['n']} (rejections {rep['notes']['rejections']}); template families {rep['tpl_families']}")
    P(f"$ spent (all stages, all runs): {rep['cost_usd_total']:.2f}")
    P("")
    P("Terms: exact/jac to blind = share of p-refs whose normalised action equals one of / best token-Jaccard to the 3 stored blind refs;")
    P("       p_none = fresh blind refs = the sampling-noise floor; drop z = paired (p_none − variant) over turns, mean/SE;")
    P("       self = within-family pairwise exact / Jaccard; ident = all parsed refs share one action; judge = glm-5.3-flash SAME rate (ref0 vs blind ref0, ref1 vs blind1);")
    P("       R_priv(z) = centered tempered LME (tau .03) of a_i = lpC(y_i^p|x,z) − lpC(y_i^p|x,∅) over the variant's refs, 2-ref LOO-matched;")
    P("       own = the variant's own thoughts, blind-own = stored blind thoughts, king = the stored king thought; hz = paired z of the difference;")
    P("       R_blind = same under the stored blind refs (blind-own from lp_cross, king from the verdict pairs, v-own from new echoes);")
    P("       in-band(live) = |m_c(z) − mean_3 m_c(blind)| ≤ 2·σ_Mc(dialect) AND ≥10 content tokens (live typ_c ≥ 0); in-band(turn) uses the per-turn sd of the 3 blind m_c;")
    P("       leak = ≥30-char substring of the note in z / y; B pass = lpC(y|x,z) − lpC(y|x,∅) ≥ 0.02; cap = finish_reason length at 4096.")
    for label, blk in [("POOLED", rep["pooled"])] + [(f"DIALECT {k}", v) for k, v in rep["by_dialect"].items()] + \
            [(f"OUTCOME {k}", v) for k, v in rep["by_outcome"].items()]:
        if not blk["n_turns"]:
            continue
        P("")
        P(f"== {label} (n turns {blk['n_turns']}) ==")
        b = blk["blind"]
        P(f"blind stored refs: self exact {_f(b['self_exact'])} self jac {_f(b['self_jac'])} B pass {_f(b['B_pass'])} "
          f"R_blind(blind-own) {_f(b['R_blind_blindown'],7,4)} R_blind(king) {_f(b['R_blind_king'],7,4)} "
          f"z(blind-own − king) {_f(b['blind_R_blindown_minus_king'])} | m_c parity Engy−live {_f(b['mc_parity_engy_minus_stored'],6,3)} (abs {_f(b['mc_parity_abs'],5,3)}) "
          f"| blind LOO in-band(live) {_f(b['blind_loo_in_band_live'])} king in-band(live) {_f(b['king_in_band_live'])} | len z p50 {b['len_z_p50']}")
        P("-- (a) action change")
        P(f"{'variant':<8} {'n':>4} {'exactB':>6} {'jacB':>6} {'exactC':>6} {'jacC':>6} {'dropJz':>6} {'dropEz':>6} {'self':>6} {'selfJ':>6} {'ident':>6} {'=rec':>6} {'judge':>6}")
        for v, g in blk["variants"].items():
            j = blk["judge"].get(v, {})
            P(f"{v:<8} {g['n_turns']:>4} {_f(g['exact_in_blind'])} {_f(g['jac_to_blind'])} {_f(g.get('exact_in_ctrl'))} {_f(g.get('jac_to_ctrl'))} "
              f"{_f(g.get('jac_drop_vs_ctrl'))} {_f(g.get('exact_drop_vs_ctrl'))} {_f(g['self_exact'])} {_f(g['self_jac'])} {_f(g['identical_all'])} {_f(g['exact_rec'])} "
              f"{_f(j.get('same_rate'))}{('/' + str(j['n'])) if j else ''}")
        if blk["judge"].get("blind_self"):
            P(f"   judge blind-vs-blind (ref1 vs ref2) SAME rate {blk['judge']['blind_self']['same_rate']:.3f} / n {blk['judge']['blind_self']['n']}")
        P("-- (b) headroom  (R per byte; hz = paired z over turns; d/σ = mean diff in live σ_R(dialect) units; n = turns with 3/3 parsed refs)")
        P(f"{'variant':<8} {'n':>4} {'Rp_own':>8} {'Rp_bown':>8} {'Rp_king':>8} {'hz o-b':>6} {'d/σ':>5} {'hz b-k':>6} {'d/σ':>5} {'hz o-k':>6} {'d/σ':>5} | {'Rb_vown':>8} {'Rb_bown':>8} {'Rb_king':>8} {'hz v-b':>6} {'hz v-k':>6}")
        for v, g in blk["variants"].items():
            P(f"{v:<8} {g['hz_own_minus_blindown']['n']:>4} {_f(g['R_priv_own'],8,4)} {_f(g['R_priv_blindown'],8,4)} {_f(g['R_priv_king'],8,4)} {_f(g['hz_own_minus_blindown'])} {_f(g['eff_sd_own_minus_blindown'],5,2)} "
              f"{_f(g['hz_blindown_minus_king'])} {_f(g['eff_sd_blindown_minus_king'],5,2)} {_f(g['hz_own_minus_king'])} {_f(g['eff_sd_own_minus_king'],5,2)} | {_f(g['R_blind_vown'],8,4)} {_f(b['R_blind_blindown'],8,4)} {_f(b['R_blind_king'],8,4)} "
              f"{_f(g['hz_blind_vown_minus_blindown'])} {_f(g['hz_blind_vown_minus_king'])}")
        P("-- (c) typicality under the live content mask; (d) leakage + B; (e) parse / cap / length")
        P(f"{'variant':<8} {'inband':>6} {'inbT':>6} {'|z|mn':>6} {'|z|p50':>6} {'zmean':>6} {'<10ct':>6} | {'leakZ':>6} {'leakY':>6} {'Bpass':>6} | {'parse':>6} {'yld2':>6} {'cap':>6} {'think':>6} {'txtfb':>6} {'lenZ':>6} {'lenY':>6} {'ctok':>6} {'ptok':>7} {'note':>5}")
        for v, g in blk["variants"].items():
            P(f"{v:<8} {_f(g['in_band_live'])} {_f(g['in_band_turn'])} {_f(g['abs_z_live_mean'],6,2)} {_f(g['abs_z_live_p50'],6,2)} {_f(g['z_live_mean'],6,2)} {_f(g['content_floor'])} | "
              f"{_f(g['leak_z'])} {_f(g['leak_y'])} {_f(g['B_pass'])} | {_f(g['parse_rate'])} {_f(g['yield_ge2'])} {_f(g['cap_hit_rate'])} {_f(g['think_close_rate'])} {_f(g['text_fallback_rate'])} "
              f"{_f(g['len_z_p50'],6,0)} {_f(g['len_y_p50'],6,0)} {_f(g['completion_tokens_per_ref'],6,0)} {_f(g['prompt_tokens_per_ref'],7,0)} {_f(g['note_chars_mean'],5,0)}")
    P("")
    P("== DECISION TABLE (pooled) — change = action-change beyond the fresh-blind noise floor (paired Jaccard-drop z ≥ 2 AND judge SAME below p_none);")
    P("   headroom = R_priv(p-own) − R_priv(blind-own) paired z ≥ 2; band = p-thoughts inside the live blind band ≥ 0.80; leak = note quoted in z or y ≤ 0.05 ==")
    P(f"{'variant':<8} {'dropJz':>6} {'judge':>6} {'change':>6} | {'hz o-b':>6} {'headrm':>6} | {'inband':>6} {'band':>6} | {'leakZ':>6} {'leakY':>6} {'leak':>6} | {'verdict':>8}")
    pj = (rep["pooled"]["judge"].get("p_none") or {}).get("same_rate")
    for v, g in rep["pooled"]["variants"].items():
        if v == "p_none":
            continue
        dz = (g.get("jac_drop_vs_ctrl") or {}).get("z")
        js = (rep["pooled"]["judge"].get(v) or {}).get("same_rate")
        change = dz is not None and dz >= 2 and (js is None or pj is None or js < pj)
        hz = (g.get("hz_own_minus_blindown") or {}).get("z")
        headroom = hz is not None and hz >= 2
        band = g.get("in_band_live") is not None and g["in_band_live"] >= 0.80
        lk = max(g.get("leak_z") or 0.0, g.get("leak_y") or 0.0)
        leak_ok = lk <= 0.05
        verdict = "PASS" if (change and headroom and band and leak_ok) else "FAIL"
        P(f"{v:<8} {_f(dz)} {_f(js)} {'yes' if change else 'no':>6} | {_f(hz)} {'yes' if headroom else 'no':>6} | "
          f"{_f(g.get('in_band_live'))} {'yes' if band else 'no':>6} | {_f(g.get('leak_z'))} {_f(g.get('leak_y'))} {'yes' if leak_ok else 'no':>6} | {verdict:>8}")
    P("")
    P("== COST per turn (tokens; k=3 refs) — duel-time = refs sampled with the note in the prompt (echo work unchanged vs today);")
    P("   fold-time = producing the note once per D turn: p_obs/p_out read the stored trace (free), p_fr = one glm-5.3 call per turn ==")
    P(f"{'variant':<8} {'ref prompt tok/turn':>20} {'ref compl tok/turn':>19} {'note chars':>10} {'note calls':>10} {'note prompt tok':>15} {'note compl tok':>14} {'note $/turn':>11}")
    for v, g in rep["pooled"]["variants"].items():
        P(f"{v:<8} {_f(g['sample_prompt_tokens_per_turn'],20,0)} {_f(g['sample_completion_tokens_per_turn'],19,0)} {_f(g['note_chars_mean'],10,0)} "
          f"{_f(g.get('note_calls_per_turn'),10,2)} {_f(g.get('note_prompt_tokens_per_turn'),15,0)} {_f(g.get('note_completion_tokens_per_turn'),14,0)} {_f(g.get('note_usd_per_turn'),11,4)}")
    P(f"Engy $ by stage (last run): {rep['cost_by_stage_last_run']}")
    txt = "\n".join(lines) + "\n"
    (RESULTS / "report.txt").write_text(txt)
    print(txt)
    print(f"-> {RESULTS / 'report.txt'} / report.json / turn_metrics.jsonl")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("turns")
    a.add_argument("--n", type=int, default=150)
    a.set_defaults(fn=cmd_turns)
    b = sub.add_parser("notes")
    b.set_defaults(fn=cmd_notes)
    c = sub.add_parser("sample")
    c.add_argument("--turns", type=int, default=0)
    c.add_argument("--concurrency", type=int, default=24)
    c.set_defaults(fn=cmd_sample)
    d = sub.add_parser("echo")
    d.add_argument("--turns", type=int, default=0)
    d.add_argument("--start", type=int, default=0, help="skip the first N turns (parallel workers)")
    d.add_argument("--concurrency", type=int, default=24)
    d.set_defaults(fn=cmd_echo)
    e = sub.add_parser("judge")
    e.add_argument("--n", type=int, default=60)
    e.set_defaults(fn=cmd_judge)
    f = sub.add_parser("report")
    f.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
