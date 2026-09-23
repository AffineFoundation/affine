"""IPRM probe — stage 3b (runs ON THE POD): teacher-forced logprobs of every
rendered evaluation row under C (adapter disabled) and C+ (adapter enabled),
on the same weights, same tokens.

Input  score_in.jsonl  {row_id, state, ctx, ctx_state, ids, body_tok, act_tok, ...}
Output score_out.jsonl {row_id, ctx, ctx_state, model: base|plus, lp_body, n_body,
                        lp_act, n_act, lp_tokens_body: [...]}

Rows sharing (ctx_state) share the rendered prefix; the longest common token
prefix of a group is run once with a KV/recurrent-state cache and each row's
tail is scored from a copy of that cache (falls back to a full forward when
the cache cannot be copied or the body starts inside the common prefix).
A parity check (--check N) re-scores N rows with a full forward.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3.8-27B"


def common_prefix_len(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    L = 0
    while L < n and all(s[L] == seqs[0][L] for s in seqs):
        L += 1
    return L


@torch.no_grad()
def full_logprobs(model, ids: list[int]) -> torch.Tensor:
    """lp[t] = log p(ids[t] | ids[:t]) for t >= 1 (lp[0] = 0)."""
    x = torch.tensor(ids, device="cuda").unsqueeze(0)
    out = model(input_ids=x, use_cache=False)
    logits = out.logits[0].float()
    lp = F.log_softmax(logits[:-1], dim=-1).gather(1, x[0, 1:].unsqueeze(1)).squeeze(1)
    return torch.cat([torch.zeros(1, device="cuda"), lp])


@torch.no_grad()
def cached_group(model, rows: list[dict]) -> list[torch.Tensor]:
    seqs = [r["ids"] for r in rows]
    L = common_prefix_len(seqs)
    need = min(min(r["body_tok"] or r["act_tok"] or [1]) for r in rows)
    if L < 8 or need <= L - 1 or any(len(s) <= L for s in seqs):
        return [full_logprobs(model, s) for s in seqs]
    # leave one token before the first scored position so the suffix always has >= 1 token
    L = min(L, need) - 0
    x = torch.tensor(seqs[0][:L], device="cuda").unsqueeze(0)
    out = model(input_ids=x, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[0, -1].float()
    results = []
    for s in seqs:
        c = copy.deepcopy(cache)
        tail = torch.tensor(s[L:], device="cuda").unsqueeze(0)
        o = model(input_ids=tail, past_key_values=c, use_cache=True)
        logits = torch.cat([last_logits.unsqueeze(0), o.logits[0, :-1].float()], dim=0)   # predict s[L:]
        lp_tail = F.log_softmax(logits, dim=-1).gather(1, tail[0].unsqueeze(1)).squeeze(1)
        lp = torch.zeros(len(s), device="cuda")
        lp[L:] = lp_tail
        results.append(lp)
        del c, o
    del cache
    return results


def summarize(row: dict, lp: torch.Tensor) -> dict:
    lp = lp.tolist()
    body = [lp[i] for i in row["body_tok"]]
    act = [lp[i] for i in row["act_tok"]]
    return {"lp_body": sum(body), "n_body": len(body), "lp_act": sum(act), "n_act": len(act),
            "lp_tokens_body": [round(v, 4) for v in body]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp)]
    if a.limit:
        rows = rows[: a.limit]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["ctx_state"] + "|" + r["ctx"]].append(r)
    print(f"{len(rows)} rows in {len(groups)} prefix groups", flush=True)

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0},
                                                attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(a.adapter))
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    done = set()
    if a.out.exists():
        for l in open(a.out):
            d = json.loads(l)
            done.add((d["row_id"], d["ctx"], d["ctx_state"], d["model"]))
    fout = open(a.out, "a")
    n_done = 0
    t0 = time.time()
    checks = []
    for gi, (gk, grows) in enumerate(sorted(groups.items())):
        for mode in ("base", "plus"):
            todo = [r for r in grows if (r["row_id"], r["ctx"], r["ctx_state"], mode) not in done]
            if not todo:
                continue
            ctx = model.disable_adapter() if mode == "base" else torch.no_grad()
            with ctx:
                if a.no_cache:
                    lps = [full_logprobs(model, r["ids"]) for r in todo]
                else:
                    try:
                        lps = cached_group(model, todo)
                    except Exception as e:  # cache copy unsupported -> full forwards
                        print(f"cache path failed ({type(e).__name__}: {str(e)[:120]}), full forward", flush=True)
                        lps = [full_logprobs(model, r["ids"]) for r in todo]
                if a.check and len(checks) < a.check and not a.no_cache:
                    ref = full_logprobs(model, todo[0]["ids"])
                    s1, s2 = summarize(todo[0], lps[0]), summarize(todo[0], ref)
                    checks.append({"row_id": todo[0]["row_id"], "model": mode, "cached": s1["lp_body"], "full": s2["lp_body"]})
                    print("parity", checks[-1], flush=True)
            for r, lp in zip(todo, lps):
                rec = {"row_id": r["row_id"], "state": r["state"], "ctx": r["ctx"], "ctx_state": r["ctx_state"],
                       "model": mode, **summarize(r, lp)}
                fout.write(json.dumps(rec) + "\n")
                n_done += 1
            fout.flush()
        if gi % 10 == 0:
            el = time.time() - t0
            print(f"group {gi}/{len(groups)}  rows {n_done}  {el/60:.1f} min  mem {torch.cuda.max_memory_allocated()/1e9:.0f} GB", flush=True)
    json.dump(checks, open(a.out.with_suffix(".checks.json"), "w"), indent=1)
    print("DONE", n_done, f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
