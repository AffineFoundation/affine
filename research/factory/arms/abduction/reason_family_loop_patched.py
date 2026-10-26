#!/usr/bin/env python3
"""Isomorphism factory: RAFT loop for the reason-family scoring rules.

One loop, four arms (--score):
  reason       lift = [lp_T(y | h, z) - lp_T(y | h)] / n_tok(y)
  contrastive  lift = reason_lift(y_true) - max over distractors y' of same
  grounded     lift = [lp_T(o | h, a, z) - lp_T(o | h, a)] / n_tok(o)
  abduction    lift = [lp_T(e | a, z) - lp_T(e | a)] / n_tok(e)

Round structure (miner improves the thing the score rewards):
  1. sample k thoughts per turn from the miner server (current adapter)
  2. score each thought with the arm's lift under the frozen teacher
  3. keep the best positive thought per turn
  4. train: CE backbone on teacher rollouts mixed 1:1 with (prompt -> z_best)
  5. checkpoint adapter, hot-reload it onto the miner server
  6. every round, score 1 miner thought on held-out turns -> the S curve

Teacher-side baselines (the no-thought term) never change; they are cached in
<out>/baselines.jsonl across rounds and restarts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gad_common import load_rollout_cache, load_turns, bash_body  # noqa: E402

ELICIT = {
    "reason": "Briefly explain which action should be taken next and why.",
    "contrastive": "Briefly explain which action should be taken next and why.",
    "grounded": ("Briefly explain what the environment will output in "
                 "response to this action."),
    "abduction": ("Briefly state the key facts of the current environment "
                  "state that determine what happens next."),
}


def plain_prompt(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def lp_of(url, model, tok, prompt, completion, timeout=600):
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    body = {"model": model, "prompt": prompt + completion, "max_tokens": 1,
            "temperature": 0.0, "echo": True, "logprobs": 0}
    r = requests.post(f"{url}/v1/completions", json=body, timeout=timeout)
    r.raise_for_status()
    lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    tail = [x for x in lps[n_prompt:len(lps) - 1] if x is not None]
    return sum(tail), max(len(tail), 1)


class Scorer:
    """Arm-specific lift. Baselines cached: key -> (lp, n)."""

    def __init__(self, args, tok, turns, cache, obs):
        self.a, self.tok, self.turns, self.cache, self.obs = args, tok, turns, cache, obs
        self.base_path = os.path.join(args.out, "baselines.jsonl")
        self.baselines = {}
        if os.path.exists(self.base_path):
            for line in open(self.base_path):
                r = json.loads(line)
                self.baselines[r["key"]] = (r["lp"], r["n"])
        self.lock = threading.Lock()
        self.bfh = open(self.base_path, "a")

    def _base(self, key, prompt, completion):
        if key in self.baselines:
            return self.baselines[key]
        lp, n = lp_of(self.a.teacher_url, self.a.teacher, self.tok,
                      prompt, completion)
        with self.lock:
            self.baselines[key] = (lp, n)
            self.bfh.write(json.dumps({"key": key, "lp": lp, "n": n}) + "\n")
            self.bfh.flush()
        return lp, n

    def targets(self, tid):
        """(scoring_prompt_base, completion_target, baseline_key) or None."""
        a = self.a
        if a.score in ("reason", "contrastive"):
            y = self.cache[tid][0]["raw"]
            p = plain_prompt(self.tok, self.turns[tid])
            return p, y, f"y:{tid}"
        if a.score == "grounded":
            ent = self.obs.get(tid + "#obs")
            if ent is None:
                return None
            p = plain_prompt(self.tok, ent["prefix"])
            return p, ent["obs"], f"o:{tid}"
        if a.score == "abduction":
            msgs = self.turns[tid]
            if not msgs or msgs[-1].get("role") != "user":
                return None
            e = (msgs[-1].get("content") or "").strip()[:8000]
            y = self.cache[tid][0]["raw"]
            p = f"Action taken:\n{y}\n\n"
            return p, "The environment state that led to this action:\n" + e, f"e:{tid}"
        raise ValueError(a.score)

    def lift(self, tid, z):
        t = self.targets(tid)
        if t is None:
            return None
        p, target, key = t
        lp0, n0 = self._base(key, p, target)
        if self.a.score == "abduction":
            pz = p + "Explanation:\n" + z.strip() + "\n\n"
        else:
            pz = p + z.strip() + "\n\n"
        lp1, n1 = lp_of(self.a.teacher_url, self.a.teacher, self.tok, pz, target)
        val = (lp1 - lp0) / max(min(n0, n1), 1)
        if self.a.score == "contrastive":
            rolls = self.cache[tid]
            dis = None
            if len(rolls) > 1 and bash_body(rolls[1]["raw"]) != bash_body(rolls[0]["raw"]):
                dis = rolls[1]["raw"]
            if dis is not None:
                dk = f"d:{tid}"
                dlp0, dn0 = self._base(dk, p, dis)
                dlp1, dn1 = lp_of(self.a.teacher_url, self.a.teacher,
                                  self.tok, pz, dis)
                val -= max((dlp1 - dlp0) / max(min(dn0, dn1), 1), 0.0)
        return val


def gen_thoughts(args, tok, prompt, k, model_tag):
    body = {"model": model_tag, "prompt": prompt, "n": k,
            "temperature": 0.9, "max_tokens": args.max_z_tokens}
    r = requests.post(f"{args.miner_url}/v1/completions", json=body, timeout=600)
    r.raise_for_status()
    return [c.get("text") or "" for c in r.json()["choices"]]


def elicit_prompt(args, tok, turns, obs, tid):
    """Prompt the miner writes its thought from (also the aux SFT prompt)."""
    if args.score == "grounded":
        ent = obs.get(tid + "#obs")
        if ent is None:
            return None
        msgs = ent["prefix"] + [{"role": "user", "content": ELICIT[args.score]}]
    else:
        msgs = turns[tid] + [{"role": "user", "content": ELICIT[args.score]}]
    return plain_prompt(tok, msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True,
                    choices=["reason", "contrastive", "grounded", "abduction"])
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--teacher", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--teacher-url", default="http://127.0.0.1:8010")
    ap.add_argument("--miner-url", default="http://127.0.0.1:8021")
    ap.add_argument("--rollouts", default="/root/work/teacher_train.jsonl")
    ap.add_argument("--heldout", default="/root/work/teacher_heldout.jsonl")
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--witness-turns", default="/root/work/witness_turns.jsonl.gz")
    ap.add_argument("--witness-obs", default="/root/work/witness_obs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--steps-per-round", type=int, default=50)
    ap.add_argument("--turns-per-round", type=int, default=384)
    ap.add_argument("--heldout-n", type=int, default=96)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-z-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--score-workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ttok = AutoTokenizer.from_pretrained(args.teacher)

    turns = load_turns(args.turns)
    cache = load_rollout_cache(args.rollouts)
    hcache = load_rollout_cache(args.heldout)

    obs = {}
    if args.score == "grounded" and os.path.exists(args.witness_obs):
        wturns = load_turns(args.witness_turns)
        for line in open(args.witness_obs):
            r = json.loads(line)
            wid = r["turn_id"]
            if wid in wturns:
                obs[wid] = {"prefix": wturns[wid],
                            "obs": r["rollouts"][0]["raw"]}

    train_tids = [t for t in cache if t in turns]
    held_tids = [t for t in hcache if t in turns][:args.heldout_n]
    if args.score == "grounded":
        train_tids = [t for t in train_tids if t + "#obs" in obs]
        held_tids = [t for t in held_tids if t + "#obs" in obs]
    print(f"arm={args.score} train_turns={len(train_tids)} "
          f"held={len(held_tids)}", flush=True)

    # backbone CE examples (identical to CTRL)
    backbone = []
    for tid in train_tids:
        p_ids = tok(plain_prompt(tok, turns[tid]),
                    add_special_tokens=False)["input_ids"]
        for roll in cache[tid]:
            c_ids = tok(roll["raw"], add_special_tokens=False)["input_ids"]
            c_ids = c_ids + [tok.eos_token_id]
            if len(p_ids) + len(c_ids) <= args.max_len:
                backbone.append((p_ids, c_ids))
    print(f"backbone examples={len(backbone)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.to("cuda")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = args.rounds * args.steps_per_round
    warmup = max(10, total_steps // 20)

    def lr_at(s):
        if s < warmup:
            return args.lr * s / warmup
        t = (s - warmup) / max(total_steps - warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    # PATCH (abduction pod): Scorer must see held-out rollouts too, else
    # targets() KeyErrors on held tids and S_held is nan every round.
    scorer = Scorer(args, ttok, turns, {**hcache, **cache}, obs)
    rng = random.Random(args.seed)
    slog = open(os.path.join(args.out, "s_curve.jsonl"), "a")
    gstep = 0
    cur_tag = args.base  # miner server model tag for generation

    def reload_adapter(tag, path):
        nonlocal cur_tag
        try:
            requests.post(f"{args.miner_url}/v1/load_lora_adapter",
                          json={"lora_name": tag, "lora_path": path},
                          timeout=300)
            served = requests.get(f"{args.miner_url}/v1/models",
                                  timeout=30).json()
            names = [m["id"] for m in served.get("data", [])]
            if tag in names:
                cur_tag = tag
                return True
        except Exception as e:
            print(f"WARN adapter reload failed: {e}", flush=True)
        return False

    errlog = {"gen": 0, "lift": 0}

    def collect(tids, k):
        """Generate + score thoughts; returns {tid: (best_z, best_lift)}."""
        res = {}
        lk = threading.Lock()

        def work(tid):
            ep = elicit_prompt(args, tok, turns, obs, tid)
            if ep is None or len(ep) > 60000:
                return
            try:
                zs = gen_thoughts(args, tok, ep, k, cur_tag)
            except Exception as ex:
                with lk:
                    if errlog["gen"] < 3:
                        errlog["gen"] += 1
                        print(f"WARN gen fail {tid}: {ex!r}", flush=True)
                return
            best = None
            for z in zs:
                z = z.strip()
                if not z:
                    continue
                try:
                    v = scorer.lift(tid, z)
                except Exception as ex:
                    with lk:
                        if errlog["lift"] < 3:
                            errlog["lift"] += 1
                            print(f"WARN lift fail {tid}: {ex!r}", flush=True)
                    continue
                if v is not None and (best is None or v > best[1]):
                    best = (z, v)
            if best is not None:
                with lk:
                    res[tid] = best

        with ThreadPoolExecutor(max_workers=args.score_workers) as ex:
            list(ex.map(work, tids))
        return res

    for rnd in range(1, args.rounds + 1):
        # -- held-out S (the curve the verdict reads) --------------------
        hres = collect(held_tids, 1)
        hvals = [v for _, v in hres.values()]
        smean = sum(hvals) / len(hvals) if hvals else float("nan")
        rec = {"round": rnd, "gstep": gstep, "S_held": round(smean, 5),
               "n": len(hvals)}
        slog.write(json.dumps(rec) + "\n")
        slog.flush()
        print(f"ROUND {rnd} S_held={smean:.5f} n={len(hvals)}", flush=True)

        # -- collect train thoughts --------------------------------------
        batch_tids = rng.sample(train_tids,
                                min(args.turns_per_round, len(train_tids)))
        tres = collect(batch_tids, args.k)
        kept = {t: (z, v) for t, (z, v) in tres.items() if v > 0}
        vals = sorted(v for _, v in tres.values())
        print(f"  thoughts kept={len(kept)}/{len(tres)} "
              f"median_lift={vals[len(vals)//2] if vals else float('nan'):.4f}",
              flush=True)

        aux = []
        for tid, (z, _) in kept.items():
            ep = elicit_prompt(args, tok, turns, obs, tid)
            p_ids = tok(ep, add_special_tokens=False)["input_ids"]
            c_ids = tok(z, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(p_ids) + len(c_ids) <= args.max_len:
                aux.append((p_ids, c_ids))

        # -- train steps_per_round on 1:1 mix -----------------------------
        pool_b = backbone[:]
        rng.shuffle(pool_b)
        pool_a = aux[:] if aux else []
        rng.shuffle(pool_a)
        bi = ai = mb = 0
        for _ in range(args.steps_per_round * args.accum):
            use_aux = pool_a and (mb % 2 == 1)
            src, idx = (pool_a, ai) if use_aux else (pool_b, bi)
            batch = []
            for _ in range(args.batch):
                if not src:
                    break
                batch.append(src[idx % len(src)])
                idx += 1
            if use_aux:
                ai = idx
            else:
                bi = idx
            if not batch:
                continue
            n = max(len(p) + len(c) for p, c in batch)
            ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
            lbl = torch.full((len(batch), n), -100, dtype=torch.long)
            msk = torch.zeros((len(batch), n), dtype=torch.long)
            for k2, (p, c) in enumerate(batch):
                seq = p + c
                ids[k2, :len(seq)] = torch.tensor(seq)
                msk[k2, :len(seq)] = 1
                lbl[k2, len(p):len(seq)] = torch.tensor(c)
            out = model(input_ids=ids.cuda(), attention_mask=msk.cuda())
            lg = out.logits[:, :-1].float()
            tgt = lbl[:, 1:].cuda()
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                   tgt.reshape(-1), ignore_index=-100) / args.accum
            loss.backward()
            mb += 1
            if mb % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 25 == 0:
                    print(f"  step {gstep}/{total_steps}", flush=True)

        tag = f"step{gstep}"
        d = os.path.join(args.out, tag)
        model.save_pretrained(d)
        print(f"CKPT_SAVED {d}", flush=True)
        reload_adapter(f"{args.score}_{tag}", d)

    # final held-out S after the last round of training
    hres = collect(held_tids, 1)
    hvals = [v for _, v in hres.values()]
    smean = sum(hvals) / len(hvals) if hvals else float("nan")
    slog.write(json.dumps({"round": args.rounds + 1, "gstep": gstep,
                           "S_held": round(smean, 5), "n": len(hvals)}) + "\n")
    slog.close()
    print(f"REASON_LOOP_DONE steps={gstep}", flush=True)


if __name__ == "__main__":
    main()
