"""IPRM probe (N1) — stage 1: build the evaluation rows and the SFT corpus.

Implicit process reward from the success-conditioned teacher:
    C  = frozen teacher Qwen/Qwen3.8-27B
    C+ = C fine-tuned (LoRA) on the assistant turns of SOLVED teacher trajectories
    term(y | x) = lpC+(y|x) - lpC(y|x)   summed over the ACTION BODY tokens

Evaluation ground truth = every state with continuation-graded first actions
(research/results/frontier_arbiter/{split_states,outcome}/continuations.jsonl):
  arm T            teacher finished the episode; first action = its first reply
  arm F1/F1b/F1c   a frontier proposal was FORCED as the first action, teacher finished
  arm F            the frontier (glm-5.3) finished the episode itself; first action = its reply
Label of a row = the continuation's outcome. Extra unlabelled candidates per state:
the stored original teacher action, the stored king action, the deliberated
frontier samples, and two attack rows (generic `ls -la`, repeat-last).

Training corpus = solved teacher_* rollouts of the agentic sources (mini-swe
textbased, verifiers bash, terminus_2, pi, claude_code, ...), every task that
appears in an evaluation state HELD OUT, capped per (source, harness), rendered
exactly as the duel renders a prefix (affine.corpus.view.build_view_record ->
node chain -> teacher chat template; prior assistant turns carry an empty think
block, the same bytes gen_prompt shows the model at duel time). Loss spans =
assistant content + <|im_end|>. Long trajectories are tiled into windows of
<= MAX_TOKENS: head (system + task) + a contiguous block of turns.

Outputs (/tmp/iprm): eval_states.jsonl, eval_rows.jsonl, train.jsonl (token ids
+ label mask), train_meta.json.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

sys.path.insert(0, str(common.REPO / "affine"))          # datagen.slicer (stale tree ok)
from affine.corpus.view import build_view_record  # noqa: E402
from affine.corpus.materialize import node_path  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402

RES = common.REPO / "research" / "results" / "frontier_arbiter"
STATE_DIR = Path("/tmp/fa_outcome/states/states")
CHUNKS = Path("/tmp/fa/chunks")
INDEX = Path("/tmp/vr/rollouts.jsonl")
OUT = Path("/tmp/iprm")

MSWEA_FENCE = "```mswea_bash_command\n"
AGENTIC_SOURCES = ("multiswe", "r2e_gym", "scaleswe", "swelego", "swerebench_v2", "swesmith",
                   "terminal_bench_2", "terminal_lego", "affine_tmax", "affine_sql", "affine_oolong",
                   "affine_prolog", "affine_uuidctf", "affine_numina", "affine_deshuffle")
# claude_code is out: its system block alone exceeds the window (no turns fit)
HARNESSES = ("mini_swe_textbased", "bash", "terminus_2", "pi", "hermes_agent", "kimi_code")
ASSISTANT_HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
IM_END = "<|im_end|>"


def base_id(sid: str) -> str:
    return ":".join(sid.split(":")[:2])


def fix_fence(content: str) -> str:
    return content.replace(MSWEA_FENCE, "```bash\n")


def parse_action(content: str, kind: str, tool_calls=None) -> str | None:
    rep = {"reasoning": "", "content": fix_fence(content or ""), "tool_calls": tool_calls or []}
    r = common.reply_to_rollout(rep, kind)
    return r["y"] if r["parsed"] else None


# ------------------------------------------------------------------ action body
def action_body_span(y: str, kind: str) -> tuple[int, int]:
    """(start, end) char offsets INTO y of the action body: the bytes a miner
    actually chooses — command text without the fence, the tool-call JSON
    without the <tool_call> envelope, the `commands` value of a Terminus reply."""
    if kind == "bash":
        m = re.match(r"^\s*```[a-zA-Z_]*\n(.*?)\n?```\s*$", y, re.S)
        if m:
            return m.start(1), m.end(1)
        return 0, len(y)
    if kind == "tool_call":
        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", y, re.S)
        if m:
            return m.start(1), m.end(1)
        return 0, len(y)
    if kind == "terminus_json":
        m = re.search(r'"commands"\s*:\s*(\[.*\])', y, re.S)
        if m:
            # trim to the matching bracket
            depth, i = 0, m.start(1)
            for j in range(m.start(1), len(y)):
                if y[j] == "[":
                    depth += 1
                elif y[j] == "]":
                    depth -= 1
                    if depth == 0:
                        return i, j + 1
            return m.start(1), m.end(1)
        return 0, len(y)
    return 0, len(y)


def generic_action(kind: str, prefix: list[dict]) -> str:
    if kind == "bash":
        return "```bash\nls -la\n```"
    if kind == "terminus_json":
        return json.dumps({"analysis": "Let me look at the current directory.",
                           "plan": "List the files to see what is here.",
                           "commands": [{"keystrokes": "ls -la\n", "duration": 1.0}]}, indent=2)
    if kind == "tool_call":
        # tool name the harness exposes: the verifiers `bash` harness uses `bash`;
        # pi / claude_code expose `bash`/`Bash`. Read the system prompt for a hint.
        sysm = next((m["content"] for m in prefix if m["role"] == "system"), "")
        name = "bash"
        for cand in ("bash", "Bash", "execute_bash", "shell", "run_command", "terminal"):
            if f'"name": "{cand}"' in sysm or f"<name>{cand}</name>" in sysm:
                name = cand
                break
        return "<tool_call>\n" + json.dumps({"name": name, "arguments": {"command": "ls -la"}}) + "\n</tool_call>"
    return "ls -la"


def repeat_last_action(kind: str, prefix: list[dict]) -> str | None:
    for m in reversed(prefix):
        if m["role"] == "assistant":
            content = m["content"]
            if content.endswith("\n</think>"):
                content = content[: -len("\n</think>")]
            y = parse_action(content, kind)
            if y:
                return y
    return None


# ------------------------------------------------------------------ eval set
def load_states(baker: ToolBaker | None = None) -> dict[str, dict]:
    """State files keep the harness's NATIVE messages (tool_calls / role=tool
    for the verifiers `bash` harness); the duel scores the BAKED plain form
    (affine.toolbake.ToolBaker.bake, parity-checked against the template)."""
    states = {}
    for f in sorted(STATE_DIR.iterdir()):
        d = json.load(open(f))
        msgs = d["messages"]
        if baker is not None and (d.get("tools") or any(m.get("tool_calls") or m["role"] == "tool" for m in msgs)):
            assert baker.parity_ok(msgs, d.get("tools") or [], baker.bake(msgs, d.get("tools") or []))
            d["messages"] = baker.bake(msgs, d.get("tools") or [])
            d["baked"] = True
        states[base_id(d["state_id"])] = d
    return states


def build_eval(out: Path, baker: ToolBaker) -> tuple[list[dict], list[dict]]:
    states = load_states(baker)
    conts = common.read_jsonl(RES / "split_states" / "continuations.jsonl") + \
        common.read_jsonl(RES / "outcome" / "continuations.jsonl")
    props = {p["state_id"]: p for p in common.read_jsonl(RES / "split_states" / "proposals.jsonl")}
    kept = {k["state_id"]: k for k in common.read_jsonl(RES / "outcome" / "kept.jsonl")}
    delib = common.read_jsonl(RES / "deliberated" / "samples.jsonl")
    selected = {base_id(s["state_id"]): s for s in common.read_jsonl(RES / "split_states" / "selected.jsonl")}

    rows: list[dict] = []
    used_states: set[str] = set()
    n_unparsed = 0
    for c in conts:
        if c.get("outcome") not in ("solved", "failed"):
            continue
        b = base_id(c["state_id"])
        arm = c["arm"]
        st = states[b]
        kind = st["action_kind"]
        fid = b + ":frontier"
        y = None
        origin = None
        if arm == "T":
            fr = c.get("first_reply") or {}
            y = parse_action(fr.get("content") or "", kind, fr.get("tool_calls"))
            origin = "teacher_cont"
        elif arm == "F":
            fr = c.get("first_reply") or {}
            y = parse_action(fr.get("content") or "", kind, fr.get("tool_calls"))
            origin = "frontier_cont"
        elif arm in ("F1", "F1b", "F1c"):
            p = props.get(fid)
            if p:
                for q in p["proposals"]:
                    if q["arm"] == arm:
                        y = q["y"]
            if y is None and arm == "F1" and fid in kept and kept[fid].get("frontier_greedy"):
                y = kept[fid]["frontier_greedy"]["y"]
            if y is not None:
                y = fix_fence(y)
                y = parse_action(y, kind) or y
            origin = "frontier_forced"
        if not y:
            n_unparsed += 1
            continue
        rows.append({"state": b, "kind": kind, "y": y, "label": 1 if c["outcome"] == "solved" else 0,
                     "arm": arm, "origin": origin, "continuation": c.get("continuation"),
                     "cont_state_id": c["state_id"]})
        used_states.add(b)

    # unlabelled candidates
    for b in sorted(used_states):
        st = states[b]
        kind = st["action_kind"]
        prefix = st["messages"]
        for key, origin in (("teacher_action", "teacher_orig"), ("king_action", "king_stored")):
            if st.get(key):
                y = parse_action(fix_fence(st[key]), kind) or fix_fence(st[key])
                rows.append({"state": b, "kind": kind, "y": y, "label": None, "arm": None, "origin": origin})
        rows.append({"state": b, "kind": kind, "y": generic_action(kind, prefix), "label": None,
                     "arm": None, "origin": "attack_generic"})
        rl = repeat_last_action(kind, prefix)
        if rl:
            rows.append({"state": b, "kind": kind, "y": rl, "label": None, "arm": None, "origin": "attack_repeat"})
        fid = b + ":frontier"
        if fid in props:
            for a in props[fid].get("teacher_first_actions") or []:
                rows.append({"state": b, "kind": kind, "y": parse_action(fix_fence(a), kind) or fix_fence(a),
                             "label": None, "arm": None, "origin": "teacher_first_unlabelled"})
        if fid in kept:
            for t in kept[fid].get("teacher") or []:
                if t.get("y"):
                    rows.append({"state": b, "kind": kind, "y": t["y"], "label": None, "arm": None,
                                 "origin": "teacher_sample_unlabelled"})
    for d in delib:
        b = base_id(d["state_id"])
        if b in used_states and d.get("parsed") and d.get("y"):
            rows.append({"state": b, "kind": states[b]["action_kind"], "y": d["y"], "label": None, "arm": None,
                         "origin": f"deliberated_{d['variant']}", "model": d["model"]})

    # dedupe identical (state, y) keeping labels: aggregate labels per distinct action
    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["state"], r["y"])
        a = agg.setdefault(k, {"state": r["state"], "kind": r["kind"], "y": r["y"], "origins": [],
                               "labels": [], "arms": []})
        a["origins"].append(r["origin"])
        if r["label"] is not None:
            a["labels"].append(r["label"])
            a["arms"].append(r["arm"])
    out_rows = []
    for i, ((b, y), a) in enumerate(sorted(agg.items())):
        s, e = action_body_span(y, a["kind"])
        a.update({"row_id": hashlib.sha1(f"{b}|{y}".encode()).hexdigest()[:12], "body_start": s, "body_end": e,
                  "body": y[s:e], "n_labels": len(a["labels"]),
                  "p_solved": (sum(a["labels"]) / len(a["labels"])) if a["labels"] else None})
        out_rows.append(a)

    st_rows = []
    for b in sorted(used_states):
        st = states[b]
        sel = selected.get(b)
        st_rows.append({"state": b, "kind": st["action_kind"], "source": st["source"], "harness": st["harness"],
                        "depth": st.get("depth"), "task_sid": st["task"]["sid"], "orig_outcome": st.get("orig_outcome"),
                        "prefix_chars": st.get("prefix_chars"), "primary": sel is not None,
                        "tier": sel.get("tier") if sel else None, "prefix": st["messages"]})
    common.write_jsonl(out / "eval_states.jsonl", st_rows)
    common.write_jsonl(out / "eval_rows.jsonl", out_rows)
    lab = [r for r in out_rows if r["n_labels"]]
    both = collections.Counter()
    for r in lab:
        both[r["state"]] += 1
    print(f"eval: {len(st_rows)} states, {len(out_rows)} distinct (state, action) rows, {len(lab)} labelled "
          f"(unparsed continuations {n_unparsed}); labelled rows per state: {sorted(collections.Counter(both.values()).items())}")
    print("origins:", collections.Counter(o for r in out_rows for o in r["origins"]))
    return st_rows, out_rows


# ------------------------------------------------------------------ training set
MAJOR_SOURCES = ("multiswe", "r2e_gym", "scaleswe", "swelego", "swerebench_v2", "swesmith",
                 "terminal_bench_2", "terminal_lego", "affine_tmax")


def pick_training(holdout_sids: set[str], cap: int, seed: int, cap_minor: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(INDEX)]
    teacher = [r for r in rows if (r["policy_id"] or "").startswith("teacher_") and r["outcome"] in ("solved", "failed")
               and r["source"] in AGENTIC_SOURCES and r["harness"] in HARNESSES]
    per_task = collections.defaultdict(collections.Counter)
    for r in teacher:
        per_task[(r["source"], r["sid"])][r["outcome"]] += 1
    solved = [r for r in teacher if r["outcome"] == "solved" and r["sid"] not in holdout_sids]
    rng = random.Random(seed)
    rng.shuffle(solved)
    # mixed tasks first (their solved trajectories are the difficulty-controlled signal)
    solved.sort(key=lambda r: 0 if per_task[(r["source"], r["sid"])]["failed"] else 1)
    picked, n_cell = [], collections.Counter()
    for r in solved:
        cell = (r["source"], r["harness"])
        if n_cell[cell] >= (cap if r["source"] in MAJOR_SOURCES or cap_minor is None else cap_minor):
            continue
        n_cell[cell] += 1
        r["mixed_task"] = bool(per_task[(r["source"], r["sid"])]["failed"])
        picked.append(r)
    print(f"train pool: {len(teacher)} teacher agentic rollouts, {len(solved)} solved & not held out, picked {len(picked)}")
    for cell, n in sorted(n_cell.items()):
        print(f"   {cell[0]:18s} {cell[1]:18s} {n}")
    return picked


def conversation_of(envelope: dict, baker: ToolBaker) -> list[dict] | None:
    rec = build_view_record(envelope, baker=baker)
    if rec is None or not rec["turns"]:
        return None
    last = max(rec["turns"], key=lambda t: t["turn_idx"])
    # node_path returns root -> node, the reply node itself last
    return [{"role": m["role"], "content": m["content"]} for m in node_path(rec["nodes"], last["node_id"])]


def render_window(tok, msgs: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids + label mask (1 = loss) for one window rendered through the
    teacher's chat template (no generation prompt). Loss on every assistant
    content + its <|im_end|>."""
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    spans = []
    pos = 0
    for m in msgs:
        if m["role"] != "assistant":
            continue
        # the template trims the assistant content (leading "\n\n" of the
        # harness replies is dropped); try the trimmed forms in order
        i, body = -1, m["content"]
        for cand in (m["content"], m["content"].strip(), m["content"].lstrip(), m["content"].rstrip()):
            i = text.find(ASSISTANT_HEAD + cand + IM_END, pos)
            if i >= 0:
                body = cand
                break
        if i < 0:
            raise ValueError("assistant turn not found in rendering")
        s = i + len(ASSISTANT_HEAD)
        e = s + len(body) + len(IM_END)
        spans.append((s, e))
        pos = e
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    labels = [0] * len(ids)
    for k, (a, b) in enumerate(enc["offset_mapping"]):
        if any(s <= a < e for s, e in spans):
            labels[k] = 1
    return ids, labels


def windows_of(tok, msgs: list[dict], max_tokens: int, max_windows: int, rng: random.Random) -> list[tuple[list[int], list[int]]]:
    """Tile the assistant turns into blocks; each window = head (system + first
    user) + contiguous messages ending at a block end, filled backwards up to
    max_tokens. The deepest block always kept; earlier blocks sampled."""
    head_n = 2 if len(msgs) >= 2 and msgs[0]["role"] == "system" else 1
    head = msgs[:head_n]
    body = msgs[head_n:]
    if not any(m["role"] == "assistant" for m in body):
        return []
    lens = [len(tok(m["content"], add_special_tokens=False)["input_ids"]) + 8 for m in msgs]
    head_len = sum(lens[:head_n]) + 16
    if head_len > max_tokens * 0.6:
        # giant scaffold prompt (Claude Code's system block): no room for turns
        return []
    # blocks: walk from the end
    ends = []
    j = len(body)
    while j > 0:
        tot, i = head_len, j
        while i > 0 and tot + lens[head_n + i - 1] <= max_tokens:
            tot += lens[head_n + i - 1]
            i -= 1
        if i == j:      # single message larger than the budget: skip it
            j -= 1
            continue
        ends.append((i, j))
        j = i
    if not ends:
        return []
    chosen = [ends[0]] + rng.sample(ends[1:], min(max_windows - 1, len(ends) - 1))
    out = []
    for i, j in chosen:
        block = body[i:j]
        # a window must start on a user message and contain an assistant turn
        while block and block[0]["role"] != "user":
            block = block[1:]
        if not any(m["role"] == "assistant" for m in block):
            continue
        while block and block[-1]["role"] != "assistant":
            block = block[:-1]
        if not block:
            continue
        try:
            ids, labels = render_window(tok, head + block)
        except ValueError:
            continue
        if len(ids) > max_tokens + 64 or sum(labels) == 0:
            continue
        out.append((ids, labels))
    return out


def build_train(out: Path, holdout_sids: set[str], cap: int, max_tokens: int, max_windows: int,
                token_budget: int, seed: int, cap_minor: int | None = None) -> None:
    tok = common.teacher_tokenizer()
    baker = ToolBaker(tok)
    picked = pick_training(holdout_sids, cap, seed, cap_minor)
    by_chunk = collections.defaultdict(list)
    for r in picked:
        by_chunk[r["chunk"]].append(r)
    rng = random.Random(seed)
    n_tok = n_loss = n_win = n_traj = n_fail = 0
    meta_cells = collections.Counter()
    with open(out / "train.jsonl", "w") as f:
        chunk_items = sorted(by_chunk.items())
        rng.shuffle(chunk_items)          # the token budget must not cut whole sources
        for ci, (chunk, rs) in enumerate(chunk_items):
            want = {r["line"]: r for r in rs}
            with gzip.open(CHUNKS / chunk, "rt") as g:
                for ln, line in enumerate(g):
                    if ln not in want:
                        continue
                    r = want[ln]
                    env = json.loads(line)
                    try:
                        conv = conversation_of(env, baker)
                    except Exception as e:  # ToolParityError etc.
                        n_fail += 1
                        continue
                    if not conv:
                        n_fail += 1
                        continue
                    wins = windows_of(tok, conv, max_tokens, max_windows, rng)
                    if not wins:
                        n_fail += 1
                        continue
                    n_traj += 1
                    for ids, labels in wins:
                        f.write(json.dumps({"rollout_id": r["rollout_id"], "source": r["source"], "harness": r["harness"],
                                            "mixed": r["mixed_task"], "ids": ids, "labels": labels}) + "\n")
                        n_tok += len(ids)
                        n_loss += sum(labels)
                        n_win += 1
                        meta_cells[(r["source"], r["harness"])] += 1
            if ci % 50 == 0:
                print(f"  chunk {ci}/{len(by_chunk)}: traj {n_traj} windows {n_win} tokens {n_tok/1e6:.1f}M loss {n_loss/1e6:.1f}M fail {n_fail}", flush=True)
            if n_tok >= token_budget:
                print("token budget reached")
                break
    meta = {"n_traj": n_traj, "n_windows": n_win, "n_tokens": n_tok, "n_loss_tokens": n_loss, "n_failed": n_fail,
            "max_tokens": max_tokens, "max_windows": max_windows, "cap_per_cell": cap, "seed": seed,
            "cells": {f"{s}|{h}": n for (s, h), n in sorted(meta_cells.items())},
            "holdout_sids": len(holdout_sids)}
    json.dump(meta, open(out / "train_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cap", type=int, default=70, help="solved trajectories per (source, harness), major sources")
    ap.add_argument("--cap-minor", type=int, default=30, help="same for the small affine_* sources")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-windows", type=int, default=2)
    ap.add_argument("--token-budget", type=int, default=16_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    baker = ToolBaker(common.teacher_tokenizer())
    st_rows, _ = build_eval(a.out, baker)
    if a.eval_only:
        return
    holdout = {s["task_sid"] for s in st_rows}
    # hold out every task of every stored state (246), not only the graded ones
    for f in STATE_DIR.iterdir():
        holdout.add(json.load(open(f))["task"]["sid"])
    build_train(a.out, holdout, a.cap, a.max_tokens, a.max_windows, a.token_budget, a.seed, a.cap_minor)


if __name__ == "__main__":
    main()
