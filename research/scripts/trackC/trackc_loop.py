#!/usr/bin/env python3
"""Track C phase 2: generator rounds against a FROZEN discriminator.

Each round:
  1. sample n candidates per prefix from the current G (vLLM, LoRA hot-swap)
  2. reward = frozen-D probability the candidate is the teacher, paired with a
     cached Qwen3-32B rollout for the same turn, averaged over both orderings;
     no valid closed bash block -> reward 0
  3. best-of-n rejection SFT on the top candidate per prefix (LoRA, capped)
  4. eval on held-out prefixes: fool rate, valid-action rate, action agreement
     with teacher, thought stats; one status line per round

The discriminator adapter is loaded once at startup and NEVER updated: the
whole point of this track is watching what G does to a static judge.

Guards: reward 0 for invalid actions, capped SFT steps, rollback-and-halve on
degeneration (valid rate < 50% or junk thoughts), all rollbacks logged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

from trackc_common import (action_inner, build_gen_prompt, cand_record,
                           candidate_text, jaccard, load_rollouts, load_turns,
                           parse_prefix, sample_completions, word_set)
from disc_ab_train import ABData


# --------------------------------------------------------------------------
# frozen-D scoring via vLLM logprobs
# --------------------------------------------------------------------------
class DScorer:
    """Scores (candidate vs teacher) pairs with the served frozen judge.

    Prompt token ids are built by the SAME ABData code the judge was trained
    with, then sent to vLLM as token-id prompts, so training and scoring see
    byte-identical inputs. P(teacher slot) is read off the A/B token logprobs
    and averaged over both orderings.
    """

    N_TOK = 8  # parallel prompt builders; fast tokenizers are not thread-safe,
    # so each builder thread gets its own instance ("Already borrowed" guard)

    def __init__(self, urls, model_name, d_base, tok, turns, max_len,
                 workers=32):
        self.urls, self.model, self.tok = urls, model_name, tok
        self.turns, self.max_len, self.workers = turns, max_len, workers
        self.a_tok, self.b_tok = "A", "B"
        from transformers import AutoTokenizer
        self.toks = [tok] + [AutoTokenizer.from_pretrained(d_base)
                             for _ in range(self.N_TOK - 1)]

    def _query(self, idx, ids):
        body = {"model": self.model, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0, "logprobs": 20}
        for attempt in range(3):
            url = self.urls[(idx + attempt) % len(self.urls)]
            try:
                r = requests.post(f"{url}/v1/completions", json=body,
                                  timeout=300)
                r.raise_for_status()
                lp = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
                pa = math.exp(lp[self.a_tok]) if self.a_tok in lp else 1e-9
                pb = math.exp(lp[self.b_tok]) if self.b_tok in lp else 1e-9
                return pa / (pa + pb)
            except Exception:
                time.sleep(3 * (attempt + 1))
        return None

    def score(self, items):
        """items: list of (key, turn_id, cand_text, ref_text).
        Returns key -> P(candidate judged teacher), order-averaged, or None."""
        n = len(items)
        if not n:
            return {}
        chunk_sz = max(1, (n + self.N_TOK - 1) // self.N_TOK)
        chunks = [(ci, items[ci * chunk_sz:(ci + 1) * chunk_sz])
                  for ci in range(self.N_TOK) if items[ci * chunk_sz:]]
        jobs = []  # (item_index, ids, label)
        jlock = threading.Lock()

        def build(arg):
            ci, chunk = arg
            tok = self.toks[ci % len(self.toks)]
            pairs = [{"turn_id": t, "repo": t.split(":")[0],
                      "mine": c, "ref": rf} for _, t, c, rf in chunk]
            local = []
            for sw in (False, True):
                ds = ABData(pairs, self.turns, tok, self.max_len, swap=sw)
                for j in range(len(chunk)):
                    item = ds[j]
                    local.append((ci * chunk_sz + j, item["ids"],
                                  item["label"]))
            with jlock:
                jobs.extend(local)

        with ThreadPoolExecutor(max_workers=self.N_TOK) as ex:
            list(ex.map(build, chunks))

        acc = {}
        alock = threading.Lock()

        def work(jidx):
            i, ids, label = jobs[jidx]
            # label 0 -> teacher sits in slot A
            p_a = self._query(jidx, ids)
            if p_a is None:
                return
            p_teacher = p_a if label == 0 else 1.0 - p_a
            with alock:
                acc.setdefault(i, []).append(1.0 - p_teacher)

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(work, range(len(jobs))))
        out = {}
        for i, (key, *_rest) in enumerate(items):
            ps = acc.get(i)
            out[key] = sum(ps) / len(ps) if ps else None
        return out


# --------------------------------------------------------------------------
# generator LoRA trainer (rejection SFT)
# --------------------------------------------------------------------------
class GTrainer:
    def __init__(self, base, lora_r=16, max_seq=8192):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        self.tok = AutoTokenizer.from_pretrained(base)
        self.max_seq = max_seq
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        self.model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])).cuda()

    def make_example(self, prompt: str, target: str):
        pi = self.tok(prompt, add_special_tokens=False)["input_ids"]
        ti = self.tok(target, add_special_tokens=False)["input_ids"]
        ti = ti + [self.tok.eos_token_id]
        if len(pi) + len(ti) > self.max_seq:
            return None
        return {"ids": pi + ti, "n_prompt": len(pi)}

    def train_round(self, examples, lr, batch, max_steps, seed):
        rng = random.Random(seed)
        exs = list(examples)
        rng.shuffle(exs)
        exs = exs[: batch * max_steps]
        exs.sort(key=lambda e: len(e["ids"]))  # bucket to limit padding
        batches = [exs[i:i + batch] for i in range(0, len(exs), batch)]
        rng.shuffle(batches)

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        self.model.train()
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        losses = []
        for bt in batches:
            n = max(len(e["ids"]) for e in bt)
            ids = torch.full((len(bt), n), pad, dtype=torch.long)
            labels = torch.full((len(bt), n), -100, dtype=torch.long)
            mask = torch.zeros((len(bt), n), dtype=torch.long)
            for i, e in enumerate(bt):
                L = len(e["ids"])
                ids[i, :L] = torch.tensor(e["ids"])
                mask[i, :L] = 1
                labels[i, e["n_prompt"]:L] = ids[i, e["n_prompt"]:L]
            out = self.model(input_ids=ids.cuda(), attention_mask=mask.cuda(),
                             labels=labels.cuda())
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
        self.model.eval()
        return (sum(losses) / len(losses)) if losses else float("nan"), len(batches)

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        state = load_file(os.path.join(path, "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model, state)


# --------------------------------------------------------------------------
# server LoRA hot-swap
# --------------------------------------------------------------------------
def load_lora(urls, name, path):
    for url in urls:
        last = None
        for attempt in range(5):
            try:
                r = requests.post(f"{url}/v1/load_lora_adapter",
                                  json={"lora_name": name, "lora_path": path},
                                  timeout=300)
                if r.status_code == 200 or "already" in (r.text or "").lower():
                    last = None
                    break
                last = RuntimeError(f"{r.status_code}: {r.text[:200]}")
            except Exception as e:
                last = e
            time.sleep(10 * (attempt + 1))
        if last:
            raise last


def unload_lora(urls, name):
    for url in urls:
        try:
            requests.post(f"{url}/v1/unload_lora_adapter",
                          json={"lora_name": name}, timeout=60)
        except Exception:
            pass


# --------------------------------------------------------------------------
# held-out evaluation
# --------------------------------------------------------------------------
def eval_heldout(g_urls, model_name, tids, prompts, teacher_ref, teacher_all,
                 scorer, temp, max_tokens):
    jobs = [(t, prompts[t]) for t in tids]
    res = sample_completions(g_urls, model_name, jobs, 1, temp, max_tokens,
                             workers=64)
    n_choices = n_valid = 0
    per = {}
    score_items = []
    for tid in tids:
        ch = res.get(tid)
        if not ch:
            continue
        c = cand_record(ch[0]["text"], ch[0]["finish"])
        n_choices += 1
        if not c["valid"]:
            continue
        n_valid += 1
        per[tid] = c
        score_items.append((tid, tid, candidate_text(c["z"], c["y"]),
                            teacher_ref[tid]))
    rewards = scorer.score(score_items) if score_items else {}

    fooled = []
    probs = []
    ag_exact = []
    ag_tok = []
    th_len = []
    th_jac = []
    for tid, c in per.items():
        p = rewards.get(tid)
        if p is not None:
            probs.append(p)
            fooled.append(1.0 if p > 0.5 else 0.0)
        mine_act = action_inner(c["y"])
        t_actions = [action_inner(t["y"]) for t in teacher_all[tid]]
        ag_exact.append(1.0 if any(mine_act == a for a in t_actions) else 0.0)
        ag_tok.append(max(jaccard(word_set(mine_act), word_set(a))
                          for a in t_actions))
        th_len.append(len(c["z"]))
        th_jac.append(max(jaccard(word_set(c["z"]), word_set(t["z"]))
                          for t in teacher_all[tid]))

    def mean(v):
        return sum(v) / len(v) if v else float("nan")

    return {
        "eval_n": n_choices,
        "valid_eval": n_valid / max(n_choices, 1),
        "fool_rate": mean(fooled),
        "fool_p": mean(probs),
        "agree_exact": mean(ag_exact),
        "agree_tok": mean(ag_tok),
        "think_len": mean(th_len),
        "think_jac": mean(th_jac),
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/work/research/data/disc_pairs")
    ap.add_argument("--workdir", default="/root/work/trackC")
    ap.add_argument("--g-base", default="Qwen/Qwen3-4B")
    ap.add_argument("--d-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--d-name", default="discD")
    ap.add_argument("--g-url", action="append", required=True)
    ap.add_argument("--d-url", action="append", required=True)
    ap.add_argument("--teacher-train", required=True)
    ap.add_argument("--teacher-heldout", required=True)
    ap.add_argument("--per-round", type=int, default=256)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=4096)
    ap.add_argument("--sft-lr", type=float, default=5e-5)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--sft-max-steps", type=int, default=40)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-heldout-n", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--start-adapter", default=None)
    args = ap.parse_args()

    wd = args.workdir
    os.makedirs(os.path.join(wd, "G_ckpt"), exist_ok=True)
    status_path = os.path.join(wd, "status.log")
    metrics_path = os.path.join(wd, "metrics.jsonl")

    def status(line):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(status_path, "a") as fh:
            fh.write(f"[{stamp}] {line}\n")
        print(f"STATUS {line}", flush=True)

    from transformers import AutoTokenizer
    tok_d = AutoTokenizer.from_pretrained(args.d_base)
    turns_full = load_turns(os.path.join(args.data, "turns.jsonl.gz"))
    turns_msgs = {k: (parse_prefix(v.get("prefix")) or [])
                  for k, v in turns_full.items()}

    teacher_train = load_rollouts(args.teacher_train)
    teacher_held = load_rollouts(args.teacher_heldout)
    t_ref_train = {t: candidate_text(cs[0]["z"], cs[0]["y"])
                   for t, cs in teacher_train.items()}
    t_ref_held = {t: candidate_text(cs[0]["z"], cs[0]["y"])
                  for t, cs in teacher_held.items()}

    pool = [t for t in json.load(open(os.path.join(wd, "train_pool.json")))
            if t in t_ref_train]
    heldout = [t for t in json.load(open(os.path.join(wd, "heldout.json")))
               if t in t_ref_held][: args.eval_heldout_n]

    trainer = GTrainer(args.g_base, lora_r=args.lora_r)
    tok_g = trainer.tok
    prompts = {}
    for tid in pool + heldout:
        msgs = turns_msgs.get(tid)
        if msgs:
            prompts[tid] = build_gen_prompt(tok_g, msgs)
    pool = [t for t in pool if t in prompts]
    heldout = [t for t in heldout if t in prompts]

    scorer = DScorer(args.d_url, args.d_name, args.d_base, tok_d, turns_msgs,
                     args.d_max_len)
    status(f"LOOP START pool={len(pool)} heldout={len(heldout)} "
           f"per_round={args.per_round} n={args.n} sft_lr={args.sft_lr} "
           f"sft_max_steps={args.sft_max_steps} (frozen D: {args.d_name})")

    state = {"cur_name": args.g_base, "cur_lr": args.sft_lr,
             "prev_good": None, "wd": wd}
    if args.start_adapter:
        trainer.load(args.start_adapter)
        state["prev_good"] = args.start_adapter
        name = f"G-r{args.start_round - 1}"
        try:
            load_lora(args.g_url, name, args.start_adapter)
            state["cur_name"] = name
        except Exception as e:
            status(f"BLOCKER resume adapter load failed: {e}; using base")

    # round 0 = untrained baseline
    if args.start_round <= 1:
        ev = eval_heldout(args.g_url, state["cur_name"], heldout, prompts,
                          t_ref_held,
                          teacher_held, scorer, args.temp, args.max_tokens)
        status(f"round=0 fool={ev['fool_rate']:.3f} fool_p={ev['fool_p']:.3f} "
               f"valid_eval={ev['valid_eval']:.3f} "
               f"agree_exact={ev['agree_exact']:.3f} "
               f"agree_tok={ev['agree_tok']:.3f} "
               f"think_len={ev['think_len']:.0f} "
               f"think_jac={ev['think_jac']:.3f} baseline")
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps({"round": 0, **ev}) + "\n")

    fails = 0
    for r in range(args.start_round, args.rounds + 1):
        try:
            run_round(r, args, pool, heldout, prompts, t_ref_train,
                      t_ref_held, teacher_held, trainer, scorer,
                      status, metrics_path, state)
            fails = 0
        except Exception as e:
            fails += 1
            status(f"round={r} BLOCKER {type(e).__name__}: {str(e)[:300]} "
                   f"(consecutive={fails}); sleeping {60 * fails}s and continuing")
            time.sleep(min(60 * fails, 600))


def run_round(r, args, pool, heldout, prompts, t_ref_train, t_ref_held,
              teacher_held, trainer, scorer, status, metrics_path, state):
    if True:  # keep original indentation below
        cur_name = state["cur_name"]
        cur_lr = state["cur_lr"]
        prev_good = state["prev_good"]
        wd = state["wd"]
        t0 = time.time()
        lo = ((r - 1) * args.per_round) % len(pool)
        tids = [pool[(lo + i) % len(pool)] for i in range(args.per_round)]

        # 1. sample candidates
        jobs = [(t, prompts[t]) for t in tids]
        res = sample_completions(args.g_url, cur_name, jobs, args.n,
                                 args.temp, args.max_tokens, workers=64)
        cands = {}
        n_total = n_valid = n_fail = 0
        for tid in tids:
            ch = res.get(tid)
            if ch is None:
                n_fail += 1
                continue
            cs = [cand_record(c["text"], c["finish"]) for c in ch]
            n_total += len(cs)
            n_valid += sum(c["valid"] for c in cs)
            cands[tid] = cs
        valid_train = n_valid / max(n_total, 1)
        t_sample = time.time() - t0

        think_lens = [len(c["z"]) for cs in cands.values()
                      for c in cs if c["valid"]]
        mean_think = sum(think_lens) / max(len(think_lens), 1)

        # guards BEFORE updating (they reflect the current adapter)
        if r > args.start_round and (valid_train < 0.5 or mean_think < 30):
            if prev_good:
                trainer.load(prev_good)
                name = f"G-r{r}"
                load_lora(args.g_url, name, prev_good)
                unload_lora(args.g_url, cur_name)
                state["cur_name"] = name
            state["cur_lr"] = max(cur_lr / 2, 1e-6)
            status(f"round={r} ROLLBACK valid_train={valid_train:.3f} "
                   f"think_len={mean_think:.0f} -> restored {prev_good}, "
                   f"lr={state['cur_lr']:.2e}")
            return

        # 2. reward with the frozen D
        items = []
        for tid, cs in cands.items():
            for i, c in enumerate(cs):
                if c["valid"]:
                    items.append(((tid, i), tid,
                                  candidate_text(c["z"], c["y"]),
                                  t_ref_train[tid]))
        rewards = scorer.score(items)
        n_scored = sum(1 for v in rewards.values() if v is not None)
        t_score = time.time() - t0 - t_sample

        # 3. best-of-n -> SFT examples
        examples = []
        top_rewards = []
        for tid, cs in cands.items():
            best_i, best_p = None, -1.0
            for i, c in enumerate(cs):
                p = rewards.get((tid, i))
                if c["valid"] and p is not None and p > best_p:
                    best_i, best_p = i, p
            if best_i is None:
                continue
            top_rewards.append(best_p)
            c = cs[best_i]
            if c["finish"] != "stop":
                continue
            ex = trainer.make_example(prompts[tid], c["raw"])
            if ex:
                examples.append(ex)

        loss, steps = trainer.train_round(examples, cur_lr, args.sft_batch,
                                          args.sft_max_steps, seed=r)
        t_train = time.time() - t0 - t_sample - t_score

        # 4. checkpoint + hot-swap
        ckpt = os.path.join(wd, "G_ckpt", f"round_{r}")
        trainer.save(ckpt)
        name = f"G-r{r}"
        load_lora(args.g_url, name, ckpt)
        old = cur_name
        cur_name = name
        state["cur_name"] = name
        if old != args.g_base:
            unload_lora(args.g_url, old)
        state["prev_good"] = ckpt

        # 5. held-out eval
        ev = eval_heldout(args.g_url, cur_name, heldout, prompts, t_ref_held,
                          teacher_held, scorer, args.temp, args.max_tokens)
        t_all = time.time() - t0
        mean_top = sum(top_rewards) / max(len(top_rewards), 1)
        status(f"round={r} fool={ev['fool_rate']:.3f} "
               f"fool_p={ev['fool_p']:.3f} valid_train={valid_train:.3f} "
               f"valid_eval={ev['valid_eval']:.3f} "
               f"agree_exact={ev['agree_exact']:.3f} "
               f"agree_tok={ev['agree_tok']:.3f} "
               f"think_len={ev['think_len']:.0f} "
               f"think_jac={ev['think_jac']:.3f} "
               f"train_reward_top={mean_top:.3f} n_sft={len(examples)} "
               f"steps={steps} loss={loss:.3f} lr={cur_lr:.1e} "
               f"t={t_all:.0f}s (s{t_sample:.0f}/j{t_score:.0f}/t{t_train:.0f})")
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps({
                "round": r, "valid_train": valid_train,
                "train_reward_top": mean_top, "n_sft": len(examples),
                "steps": steps, "loss": loss, "lr": cur_lr,
                "n_scored": n_scored, "wall_s": t_all, **ev}) + "\n")


if __name__ == "__main__":
    main()
