#!/usr/bin/env python
"""First-divergence experiment (docs/divergence-analysis.md §8), reign 13.

For 300 reign-13 king rollouts on tasks the teacher solved: at every king turn
(cap 40) sample 3 teacher references at the king's own prefix (Engy
qwen3.8-27b, T 0.8, 4,096 tokens), stamp the turn stop-eligible (a reference
stops / answers in prose / issues the harness completion), test whether the
king's action is one of the reference actions (a_match), and find the first
turn where it is not. Echo the king thought and the reference thoughts at the
first-divergence turn and at one control turn (per-byte lpC(z|x) -> G band).
"""
from __future__ import annotations
import argparse, collections, concurrent.futures as cf, gzip, hashlib, json, os, random, re, sys, time
from pathlib import Path
REPO = Path.home() / "subnet120"
sys.path.insert(0, str(REPO / "affine")); sys.path.insert(0, str(REPO / "ops" / "king-review")); sys.path.insert(0, str(REPO / "rollouts"))
import httpx
from affine import dialects
from affine.corpus.completion import completion_kind
from affine.corpus.trace import sampled_paths
import krlib
from krlib import parse_rollout, norm_action, load_env_groups

TRACES = REPO / "affine/state/king_review/traces"
OUT = Path(os.environ.get("DIV_OUT", str(REPO / "affine/state/king_divergence"))); OUT.mkdir(parents=True, exist_ok=True)
KING12 = os.environ.get("DIV_KING", "6d0ee567e33e")
KING = f"king-{KING12}"
CLASSES = {
    "agent": {"multiswe", "swesmith", "scaleswe", "r2e_gym", "swelego", "swerebench_v2", "terminal_bench_2", "terminal_lego", "nl2repobench", "affine_tmax", "affine_nl2lib"},
    "tool": {"affine_agent", "affine_sql", "affine_eog", "affine_autobench", "affine_wikispeedia", "affine_uuidctf", "affine_prolog", "affine_wiki"},
    "notool": {"affine_when2call", "affine_notool"},
}
PRICE_IN, PRICE_OUT, PRICE_CACHE = 0.045e-6, 0.32e-6, 0.015e-6
FOREIGN_FENCE_RE = re.compile(r"```(?:mswea_bash_command|sh|shell|console|zsh)[ \t]*\n")
ANY_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
MARKUP_RE = re.compile(r"```|<tool_call>|\"commands\"\s*:|\\boxed\{")


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


def wire(m: dict) -> dict:
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m["role"] == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = [{"id": tc.get("id") or f"call_{i}", "type": "function",
                              "function": {"name": tc.get("name") or (tc.get("function") or {}).get("name", ""),
                                           "arguments": tc.get("arguments") if tc.get("arguments") is not None else (tc.get("function") or {}).get("arguments", "{}")}}
                             for i, tc in enumerate(m["tool_calls"])]
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id") or ""
        if m.get("name"): out["name"] = m["name"]
    return out


def reply_action(content: str, tool_calls, kind: str) -> str:
    if tool_calls:
        return json.dumps([{"name": (tc.get("function") or tc).get("name"), "arguments": (tc.get("function") or tc).get("arguments")} for tc in tool_calls], sort_keys=True)
    text = FOREIGN_FENCE_RE.sub("```bash\n", content or "")
    try:
        acts = dialects.get(kind).actions(text)
    except Exception:
        acts = []
    if acts:
        return acts[-1]
    try:
        acts = dialects.get("tool_call").actions(text)
    except Exception:
        acts = []
    if acts:
        return acts[-1]
    fences = ANY_FENCE_RE.findall(text)
    return fences[-1].strip() if fences else ""


def is_stop(content: str, tool_calls, kind: str, action: str) -> bool:
    """Stop-eligible reply: harness completion (submit / finish tool / task_complete),
    or visible prose with no action in the dialect."""
    if tool_calls and any("finish" in str((tc.get("function") or tc).get("name", "")).lower() or "complete" in str((tc.get("function") or tc).get("name", "")).lower() for tc in tool_calls):
        return True
    try:
        ck = completion_kind(content or "", kind)
    except Exception:
        ck = None
    if ck:
        return True
    return (not action) and bool((content or "").strip()) and not MARKUP_RE.search(content or "")


class Engy:
    def __init__(self, key):
        self.h = httpx.Client(base_url="https://api.engy.ai/v1", headers={"Authorization": f"Bearer {key}"}, timeout=600)
        self.usage = collections.Counter()

    def chat(self, messages, n=1):
        for attempt in range(5):
            try:
                r = self.h.post("/chat/completions", json={"model": "qwen3.8-27b", "messages": messages, "temperature": 0.8, "max_tokens": 4096})
                r.raise_for_status(); d = r.json()
                u = d.get("usage") or {}
                self.usage["in"] += u.get("prompt_tokens", 0); self.usage["out"] += u.get("completion_tokens", 0)
                self.usage["cache"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                ch = d["choices"][0]; m = ch["message"]
                return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "tool_calls": m.get("tool_calls"), "finish": ch.get("finish_reason")}
            except Exception as e:
                if attempt == 4: return {"content": "", "reasoning": "", "tool_calls": None, "finish": f"error:{type(e).__name__}"}
                time.sleep(3 * (attempt + 1))

    def echo_span(self, ids: list[int], start: int):
        for attempt in range(4):
            try:
                r = self.h.post("/completions", json={"model": "qwen3.8-27b", "prompt": ids, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0, "logprob_start_len": start})
                r.raise_for_status(); d = r.json()
                u = d.get("usage") or {}; self.usage["in"] += u.get("prompt_tokens", 0)
                return d["choices"][0]["logprobs"]["token_logprobs"]
            except Exception:
                if attempt == 3: return None
                time.sleep(3 * (attempt + 1))

    def cost(self):
        return self.usage["in"] * PRICE_IN + self.usage["out"] * PRICE_OUT


def per_byte_lp(engy, tok, prefix_msgs, thought):
    """Approximate lpC(z|x) per byte: score the thought span of
    template(prefix)+<think>thought</think> with Engy's span echo (token ids +
    logprob_start_len at the prefix/thought boundary)."""
    if not thought or not thought.strip():
        return None
    pre = tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
    if not pre.endswith("<think>\n"):
        pre = pre + "<think>\n"
    body = thought.strip() + "\n</think>"
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    ids = tok(pre + body, add_special_tokens=False).input_ids
    L = 0
    while L < min(len(pre_ids), len(ids)) and pre_ids[L] == ids[L]:
        L += 1
    start = max(1, L - 1)
    lps = engy.echo_span(ids, start)
    if not lps: return None
    vals = [x for x in lps[1:] if x is not None]
    if not vals: return None
    return sum(vals) / max(1, len(thought.strip().encode()))


def select_failed(index_rows, cap=300, seed=13):
    """Every FAILED rollout of the king on a task the teacher solved (multi-turn first), capped."""
    rnd = random.Random(seed)
    teacher_solved = {r["sid"] for r in index_rows if not r.get("king") and r.get("outcome") == "solved"}
    pool = [r for r in index_rows if r.get("king") == KING and r.get("outcome") == "failed" and r["sid"] in teacher_solved]
    def cls_of(src):
        for c, srcs in CLASSES.items():
            if src in srcs: return c
        return "other"
    multi = [r for r in pool if r.get("n_replies", 0) >= 2]; single = [r for r in pool if r.get("n_replies", 0) < 2]
    rnd.shuffle(multi); rnd.shuffle(single)
    take = multi[:cap] + single[: max(0, cap - len(multi))]
    log(f"{KING}: failed pool {len(pool)} (multi {len(multi)}, single {len(single)}) -> {len(take)}")
    return [dict(r, cls=cls_of(r["source"])) for r in take]


def select(index_rows, seed=13):
    rnd = random.Random(seed)
    teacher_solved = {r["sid"] for r in index_rows if not r.get("king") and r.get("outcome") == "solved"}
    king_rows = [r for r in index_rows if r.get("king") == KING and r.get("outcome") in ("solved", "failed") and r["sid"] in teacher_solved]
    picked = []
    for cls, srcs in CLASSES.items():
        pool = [r for r in king_rows if r["source"] in srcs and (cls == "notool" or r.get("n_replies", 0) >= 3)]
        solved = [r for r in pool if r["outcome"] == "solved"]; failed = [r for r in pool if r["outcome"] == "failed"]
        rnd.shuffle(solved); rnd.shuffle(failed)
        take = []
        # balance outcomes, spread over sources
        for lst in (failed, solved):
            per_src = collections.Counter()
            for r in lst:
                if per_src[r["source"]] >= (50 if cls == "notool" else 14): continue
                take.append(r); per_src[r["source"]] += 1
                if len(take) >= (50 if lst is failed else 100): break
        picked += [dict(r, cls=cls) for r in take[:100]]
        log(f"class {cls}: pool {len(pool)} (solved {len(solved)} / failed {len(failed)}) -> picked {len(take[:100])}")
    return picked


def load_envelope(row):
    with gzip.open(TRACES / "chunks" / row["chunk"], "rt") as f:
        for i, line in enumerate(f):
            if i == row["line"]:
                return json.loads(line)
    return None


def probe_rollout(engy, tok, row, env_groups, max_turns, echo_control):
    env = load_envelope(row)
    if not env: return None
    ro = parse_rollout(env, env_groups)
    trace = env["trace"]; kind = ro.action_kind
    paths = sampled_paths(trace)
    turns = []
    n = min(len(paths), len(ro.turns), max_turns)
    first_div = None
    for k in range(n):
        path = paths[k]
        if not path or path[-1]["role"] != "assistant": continue
        prefix = [wire(m) for m in path[:-1]]
        kmsg = path[-1]
        k_action = reply_action(kmsg.get("content") or "", kmsg.get("tool_calls"), kind)
        k_stop = is_stop(kmsg.get("content") or "", kmsg.get("tool_calls"), kind, k_action)
        refs = list(cf.ThreadPoolExecutor(3).map(lambda _: engy.chat(prefix), range(3)))
        ref_rows = []
        for r in refs:
            a = reply_action(r["content"], r["tool_calls"], kind)
            ref_rows.append({"action": a, "stop": is_stop(r["content"], r["tool_calls"], kind, a), "finish": r["finish"],
                             "thought": r["reasoning"][:6000], "visible": r["content"][:3000], "valid": bool(a) or r["finish"] not in ("length", None) and bool(r["content"].strip())})
        ref_acts = [norm_action(r["action"]) for r in ref_rows if r["action"]]
        ref_stop = sum(r["stop"] for r in ref_rows)
        a_match = bool(k_action) and norm_action(k_action) in ref_acts
        agree = (a_match) or (k_stop and ref_stop >= 1)
        unanimous = len(ref_acts) == 3 and len(set(ref_acts)) == 1
        row_t = {"turn_idx": k, "node_id": ro.turns[k].node_id, "kind": kind, "prefix_chars": sum(len(m["content"]) for m in prefix),
                 "king_action": k_action[:2000], "king_stop": k_stop, "king_thought_len": len(ro.turns[k].reasoning or ""),
                 "king_finish": ro.turns[k].finish, "king_loop": ro.turns[k].loop, "obs_kind": ro.turns[k].obs_kind,
                 "ref_n_valid": sum(bool(r["action"]) or r["stop"] for r in ref_rows), "ref_stop": ref_stop, "stop_eligible": ref_stop >= 1,
                 "ref_unanimous": unanimous, "a_match": a_match, "agree": agree, "refs": ref_rows}
        if first_div is None and not agree and (k_action or k_stop is False):
            first_div = k; row_t["first_divergence"] = True
        turns.append(row_t)
    # echoes: at first divergence and at one control (agreeing) turn
    def do_echo(t):
        path = paths[t["turn_idx"]]; prefix = [wire(m) for m in path[:-1]]
        kth = ro.turns[t["turn_idx"]].reasoning or ""
        m = per_byte_lp(engy, tok, prefix, kth)
        ts = [per_byte_lp(engy, tok, prefix, r["thought"]) for r in t["refs"] if r["thought"]]
        ts = [x for x in ts if x is not None]
        t["echo"] = {"m": m, "t": ts}
        if m is not None and len(ts) >= 2:
            import statistics as st
            mu = st.mean(ts); sd = st.pstdev(ts) if len(ts) > 1 else 0.0
            for c in (2.0, 4.0):
                w = max(c * sd, 0.002); t["echo"][f"G_c{int(c)}"] = min(m - (mu - w), (mu + w) - m)
    if first_div is not None:
        do_echo(turns[first_div])
    if echo_control:
        ctrl = [t for t in turns if t["agree"] and t["turn_idx"] != first_div]
        if ctrl: do_echo(random.choice(ctrl))
    return {"rollout_id": ro.rollout_id, "traj_id": ro.traj_id, "source": ro.source, "cls": row["cls"], "harness": ro.harness, "kind": kind,
            "outcome": ro.outcome, "stop_condition": ro.stop_condition, "n_turns_total": len(ro.turns), "n_probed": len(turns),
            "first_divergence": first_div, "_sys": " ".join((m.get("content") or "")[:20000] for m in (paths[0][:1] if paths else []) if m.get("role") in ("system","user")), "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--budget-usd", type=float, default=90.0)
    ap.add_argument("--no-echo", action="store_true"); ap.add_argument("--cls", default=""); ap.add_argument("--suffix", default=""); ap.add_argument("--failed-only", action="store_true"); ap.add_argument("--cap", type=int, default=300)
    a = ap.parse_args()
    key = os.environ.get("ENGY_EVAL") or os.environ.get("ENGY_2")
    if not key:
        for line in (Path.home() / ".affine-validator.env").read_text().splitlines():
            if line.startswith("ENGY_EVAL="): key = line.split("=", 1)[1].strip().strip('"\'')
    engy = Engy(key)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    index_rows = [json.loads(l) for l in open(TRACES / "rollout_index.jsonl")]
    picked = (select_failed(index_rows, a.cap) if a.failed_only else select(index_rows))[: a.n]
    if a.cls: picked = [r for r in picked if r["cls"] in a.cls.split(",")]
    env_groups = load_env_groups()
    out_path = OUT / f"{KING12}.turns{a.suffix}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["rollout_id"] for l in open(out_path)}
    todo = [r for r in picked if r["rollout_id"] not in done]
    log(f"{len(picked)} rollouts selected, {len(todo)} to probe; budget ${a.budget_usd}")
    with open(out_path, "a") as f, cf.ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(probe_rollout, engy, tok, r, env_groups, a.max_turns, not a.no_echo): r for r in todo}
        n = 0
        for fut in cf.as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                log(f"rollout failed: {e!r}"); continue
            if res:
                f.write(json.dumps(res) + "\n"); f.flush(); n += 1
            if n % 10 == 0:
                log(f"{n}/{len(todo)} rollouts, cost ${engy.cost():.2f}, tokens in {engy.usage['in']/1e6:.1f}M out {engy.usage['out']/1e6:.2f}M")
            if engy.cost() > a.budget_usd:
                log("budget reached; stopping submissions"); 
                for ff in futs: ff.cancel()
                break
    log(f"done: cost ${engy.cost():.2f}")


if __name__ == "__main__":
    main()
