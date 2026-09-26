"""IPRM probe — stage 3a (local): render every evaluation (state, action) row
to teacher token ids exactly as the duel forces an action with NO thought:

    text = gen_prompt(prefix) + "\\n</think>" + "\\n\\n" + y       (common.force_text(prefix, "", y))

and mark the tokens whose start offset lies inside the ACTION BODY span
(fence / envelope bytes excluded). Also emits the prefix-swap control rows:
the same action body scored under (i) the same trajectory cut two assistant
turns earlier and (ii) a random other eval state's prefix of the same
dialect. Output: /tmp/iprm/score_in.jsonl (ids, body_tok, meta).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

OUT = Path("/tmp/iprm")


def render(tok, prefix: list[dict], y: str, body: tuple[int, int]) -> dict:
    full = common.force_text(prefix, "", y)
    y_start = len(full) - len(y)
    bs, be = y_start + body[0], y_start + body[1]
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    body_tok = [i for i, (a, _) in enumerate(offs) if bs <= a < be]
    if not body_tok:
        # a tiny body (terminus `"commands": []`) can be swallowed by a token that starts
        # on the preceding space; fall back to the tokens overlapping the span
        body_tok = [i for i, (a, b) in enumerate(offs) if a < be and b > bs] or \
                   [i for i, (a, b) in enumerate(offs) if a <= bs < b]
    act_tok = [i for i, (a, _) in enumerate(offs) if y_start <= a < len(full)]
    return {"ids": ids, "body_tok": body_tok, "act_tok": act_tok, "n_body_bytes": len(full[bs:be].encode()),
            "n_act_bytes": len(y.encode()), "body_fallback": not any(bs <= offs[i][0] < be for i in body_tok)}


def earlier_prefix(prefix: list[dict], back: int = 2) -> list[dict] | None:
    """The same trajectory cut `back` assistant turns earlier (ends on user)."""
    idx = [i for i, m in enumerate(prefix) if m["role"] == "assistant"]
    if len(idx) <= back:
        return None
    cut = idx[-back]                  # drop from this assistant turn on
    p = prefix[:cut]
    while p and p[-1]["role"] != "user":
        p = p[:-1]
    return p if len(p) >= 2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "score_in.jsonl")
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    tok = common.teacher_tokenizer()
    states = {s["state"]: s for s in common.read_jsonl(OUT / "eval_states.jsonl")}
    rows = common.read_jsonl(OUT / "eval_rows.jsonl")
    rng = random.Random(a.seed)
    out = []
    n_skip = 0
    by_kind: dict[str, list[str]] = {}
    for s in states.values():
        by_kind.setdefault(s["kind"], []).append(s["state"])
    for r in rows:
        st = states[r["state"]]
        body = (r["body_start"], r["body_end"])
        rec = render(tok, st["prefix"], r["y"], body)
        if len(rec["ids"]) > a.max_tokens:
            n_skip += 1
            continue
        out.append({"row_id": r["row_id"], "state": r["state"], "ctx": "own", "ctx_state": r["state"], **rec})
        # prefix-swap controls only for labelled + attack + king/teacher rows
        ctl = bool(r["n_labels"]) or any(o.startswith("attack") or o in ("king_stored", "teacher_orig") for o in r["origins"])
        if not ctl:
            continue
        ep = earlier_prefix(st["prefix"])
        if ep:
            rec2 = render(tok, ep, r["y"], body)
            out.append({"row_id": r["row_id"], "state": r["state"], "ctx": "earlier", "ctx_state": r["state"], **rec2})
        others = [b for b in by_kind[r["kind"]] if b != r["state"]]
        if others:
            o = rng.choice(others)
            rec3 = render(tok, states[o]["prefix"], r["y"], body)
            if len(rec3["ids"]) <= a.max_tokens:
                out.append({"row_id": r["row_id"], "state": r["state"], "ctx": "other_task", "ctx_state": o, **rec3})
    common.write_jsonl(a.out, out)
    n_tok = sum(len(o["ids"]) for o in out)
    print(f"{len(out)} scoring rows ({n_skip} skipped over {a.max_tokens} tokens), {n_tok/1e6:.1f}M tokens; "
          f"ctx: own {sum(1 for o in out if o['ctx']=='own')} earlier {sum(1 for o in out if o['ctx']=='earlier')} "
          f"other {sum(1 for o in out if o['ctx']=='other_task')}")


if __name__ == "__main__":
    main()
