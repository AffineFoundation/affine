#!/usr/bin/env python3
"""Track B driver: alternating GAN-style distillation rounds.

Topology (one process, GPU-pinned by CUDA_VISIBLE_DEVICES at launch):
  this process (GPU a): holds discriminator D (Qwen3-8B + LoRA) resident;
                        scores pairs, trains D, evaluates held-out metrics.
  vLLM server (GPU b):  serves G = Qwen3-4B base + hot-swapped LoRA.
  g_sft.py (GPU c):     spawned per round for the best-of-n G update; runs
                        concurrently with the D update (different GPUs).

Round:  sample G candidates -> score vs cached teacher rollouts with D ->
        best-of-n SFT for G  ||  fresh+replay training for D ->
        hot-swap new G into vLLM -> held-out metrics -> checkpoint + log.

Reward for a candidate = P_D(candidate is judged the teacher), order-averaged.
Invalid candidates (no closed bash block) never enter scoring => reward 0.
Collapse guards: rollback + halve G steps when valid rate < 0.5 or candidate
self-similarity explodes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import requests
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gad_common import (bash_body, both_text, build_prompt, load_rollout_cache,  # noqa: E402
                        load_turn_meta, load_turns, split_rollout, token_jaccard)

QUESTION = (
    "Above is the transcript of a software-engineering task in progress.\n"
    "Two assistants each proposed the next step. One of them is the reference "
    "assistant.\n\nWhich one is the reference assistant? Reply with a single "
    "letter, A or B."
)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def render_prefix(messages, max_chars):
    parts = [f"<{m.get('role','?')}>\n{m.get('content','')}" for m in messages]
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


def ab_token_ids(tok):
    out = []
    for letter in ("A", "B"):
        for form in (letter, " " + letter):
            enc = tok(form, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                out.append(enc[0])
                break
        else:
            raise SystemExit(f"cannot map {letter!r} to a single token")
    return out


class Disc:
    """Resident discriminator: A/B judging via the LM head's A-vs-B logits."""

    def __init__(self, model_name, lora_r, max_len):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        m.config.use_cache = False
        m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
        # fresh LoRA is an exact identity, so pre-training D is the true
        # zero-shot pretrained judge
        self.model = get_peft_model(m, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        self.model.to("cuda")
        self.ab = ab_token_ids(self.tok)
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=2e-5, weight_decay=0.0)

    def _fit(self, prefix, a_text, b_text):
        chars = min(len(prefix), self.max_len * 6)
        ids = None
        for _ in range(5):
            body = (f"{prefix[-chars:] if chars else ''}\n\n"
                    f"=== Candidate A ===\n{a_text}\n\n"
                    f"=== Candidate B ===\n{b_text}\n\n{QUESTION}")
            msgs = [{"role": "user", "content": body}]
            try:
                text = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                text = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= self.max_len:
                return ids
            over = len(ids) - self.max_len
            nxt = chars - int(over * 4.0) - 256
            if nxt <= 0:
                break
            chars = nxt
        return ids[-self.max_len:]

    def _batch(self, examples):
        n = max(len(e) for e in examples)
        ids = torch.full((len(examples), n), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(examples), n), dtype=torch.long)
        for i, e in enumerate(examples):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = 1
        return ids.cuda(), mask.cuda()

    def _logits(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask).logits
        idx = mask.sum(1) - 1
        last = out[torch.arange(out.size(0), device=out.device), idx]
        return last[:, self.ab].float()

    @torch.no_grad()
    def score(self, pairs, batch=8):
        """pairs: [{prefix_text, mine, ref}] -> p_teacher per pair,
        averaged over both slot orderings (cancels position bias)."""
        self.model.eval()
        p_sum = [0.0] * len(pairs)
        for swap in (False, True):
            exs = []
            for p in pairs:
                a, b = (p["ref"], p["mine"]) if not swap else (p["mine"], p["ref"])
                exs.append(self._fit(p["prefix_text"], a, b))
            lbl = 0 if not swap else 1  # slot holding the teacher
            for i0 in range(0, len(exs), batch):
                ids, mask = self._batch(exs[i0:i0 + batch])
                pr = torch.softmax(self._logits(ids, mask), -1)[:, lbl]
                for j, v in enumerate(pr.tolist()):
                    p_sum[i0 + j] += v / 2.0
        self.model.train()
        return p_sum

    def train_pairs(self, pairs, epochs=1.5, batch=4, accum=4, max_steps=150,
                    seed=0):
        """CE on the A/B letter, teacher slot randomised per example."""
        self.model.train()
        rng = random.Random(seed)
        step = mb = 0
        losses = []
        n_ep = 0
        while n_ep < epochs - 1e-9 and step < max_steps:
            order = list(range(len(pairs)))
            rng.shuffle(order)
            n_ep += 1
            for i0 in range(0, len(order), batch):
                chunk = [pairs[j] for j in order[i0:i0 + batch]]
                exs, lbls = [], []
                for p in chunk:
                    t_in_a = rng.random() < 0.5
                    a, b = (p["ref"], p["mine"]) if t_in_a else (p["mine"], p["ref"])
                    exs.append(self._fit(p["prefix_text"], a, b))
                    lbls.append(0 if t_in_a else 1)
                ids, mask = self._batch(exs)
                lg = self._logits(ids, mask)
                loss = F.cross_entropy(
                    lg, torch.tensor(lbls, device=lg.device)) / accum
                loss.backward()
                losses.append(loss.item() * accum)
                mb += 1
                if mb % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    step += 1
                    if step >= max_steps:
                        break
        self.opt.zero_grad(set_to_none=True)
        return step, sum(losses) / max(len(losses), 1)

    def save(self, path):
        self.model.save_pretrained(path)


def sample_g(urls, model_name, tok, turns, tids, n, temp, max_tokens,
             max_prompt_chars=60000, timeout=2400):
    """Sample n candidates per turn, round-robin across G replicas. Returns
    turn_id -> list[{z, y, raw}] and the valid-rollout rate."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    if isinstance(urls, str):
        urls = urls.split(",")
    out = {}
    lock = threading.Lock()
    stats = {"total": 0, "valid": 0, "err": 0}
    rr = {"i": 0}

    def work(tid):
        prompt = build_prompt(tok, turns[tid])
        if len(prompt) > max_prompt_chars:
            return
        with lock:
            rr["i"] += 1
            url = urls[rr["i"] % len(urls)]
        body = {"model": model_name, "prompt": prompt, "n": n,
                "temperature": temp, "max_tokens": max_tokens}
        try:
            r = requests.post(f"{url}/v1/completions", json=body, timeout=timeout)
            r.raise_for_status()
            choices = r.json()["choices"]
        except Exception:
            with lock:
                stats["err"] += 1
            return
        cands = []
        for c in choices:
            raw = c.get("text") or ""
            z, y = split_rollout(raw)
            with lock:
                stats["total"] += 1
            if z and y:
                cands.append({"z": z, "y": y, "raw": raw})
                with lock:
                    stats["valid"] += 1
        if cands:
            with lock:
                out[tid] = cands

    with ThreadPoolExecutor(max_workers=48) as ex:
        list(ex.map(work, tids))
    vr = stats["valid"] / max(stats["total"], 1)
    return out, vr, stats


def load_adapter(urls, name, path, drop=()):
    """Load adapter `name` on every G replica; best-effort unload of stale
    names to stay under each server's --max-loras budget. All replicas must
    accept the load, otherwise raise: serving a round on mixed adapters
    would silently corrupt the metrics."""
    if isinstance(urls, str):
        urls = urls.split(",")
    for url in urls:
        for stale in set(drop) | {name}:
            try:
                requests.post(f"{url}/v1/unload_lora_adapter",
                              json={"lora_name": stale}, timeout=60)
            except Exception:
                pass
        r = requests.post(f"{url}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path},
                          timeout=300)
        r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--meta", default="/root/work/turn_meta.jsonl.gz")
    ap.add_argument("--teacher-train", default="/dshare/gad/teacher_train.jsonl")
    ap.add_argument("--teacher-heldout", default="/dshare/gad/teacher_heldout.jsonl")
    ap.add_argument("--g-url", default="http://127.0.0.1:8001")
    ap.add_argument("--g-base", default="Qwen/Qwen3-4B")
    ap.add_argument("--d-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sft-gpu", default="2")
    ap.add_argument("--sft-python", default="/root/trainenv/bin/python")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--batch-turns", type=int, default=300)
    ap.add_argument("--n-cands", type=int, default=6)
    ap.add_argument("--heldout-n", type=int, default=200)
    ap.add_argument("--heldout-cands", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=3584)
    ap.add_argument("--d-steps", type=int, default=150)
    ap.add_argument("--g-steps", type=int, default=80)
    ap.add_argument("--replay-frac", type=float, default=0.30)
    ap.add_argument("--work", default="/dshare/gad")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    status_path = os.path.join(args.work, "status.log")
    metrics_path = os.path.join(args.work, "metrics.jsonl")
    replay_path = os.path.join(args.work, "replay.jsonl")
    state_path = os.path.join(args.work, "state.json")

    def status(line):
        with open(status_path, "a") as fh:
            fh.write(f"{now()} {line}\n")
        print(f"STATUS {line}", flush=True)

    from transformers import AutoTokenizer
    g_tok = AutoTokenizer.from_pretrained(args.g_base)

    turns = load_turns(args.turns)
    meta = load_turn_meta(args.meta)
    t_train = load_rollout_cache(args.teacher_train)
    t_held = load_rollout_cache(args.teacher_heldout)
    train_tids = sorted(t for t in t_train if t in turns)
    held_tids = sorted(t for t in t_held if t in turns)[: args.heldout_n]
    print(f"train turns with teacher rollouts: {len(train_tids)}  "
          f"held-out: {len(held_tids)}", flush=True)

    # prefix text cache for D prompts
    pfx = {}
    for t in set(train_tids) | set(held_tids):
        pfx[t] = render_prefix(turns[t], args.d_max_len * 6)

    disc = Disc(args.d_model, lora_r=16, max_len=args.d_max_len)
    # resume D from its latest per-round checkpoint (a restarted driver used
    # to silently reset D to the zero-shot judge)
    d_dir = os.path.join(args.work, "d_lora")
    if os.path.isdir(d_dir):
        cks = [d for d in os.listdir(d_dir) if d[0] == "r" and d[1:].isdigit()]
        if cks:
            latest = max(cks, key=lambda s: int(s[1:]))
            import safetensors.torch
            from peft import set_peft_model_state_dict
            sd = safetensors.torch.load_file(
                os.path.join(d_dir, latest, "adapter_model.safetensors"))
            set_peft_model_state_dict(disc.model, sd)
            print(f"D resumed from d_lora/{latest}", flush=True)
    status(f"driver up: D={args.d_model} G={args.g_base} "
           f"train_turns={len(train_tids)} held={len(held_tids)}")

    state = {"round": 0, "g_lora": "", "g_serve": args.g_base,
             "g_steps": args.g_steps, "cursor": 0}
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"resuming state: {state}", flush=True)
        # make sure every replica serves the resumed adapter (fresh replicas
        # start with the bare base model)
        if state["g_serve"] != args.g_base and state["g_lora"]:
            try:
                load_adapter(args.g_url, state["g_serve"], state["g_lora"])
            except Exception as e:
                print(f"resume adapter push failed ({e}); serving base",
                      flush=True)
                state["g_serve"] = args.g_base

    rng = random.Random(120)
    shuffled = train_tids[:]
    rng.shuffle(shuffled)

    def teacher_text(tid, cache, k=None, seed=0):
        rolls = cache[tid]
        r = rolls[0] if k == 0 else rolls[random.Random(seed ^ hash(tid) & 0xffff).randrange(len(rolls))]
        return both_text(r["z"], r["y"]), r

    def heldout_eval(round_no, serve_name):
        """Sample held-out candidates from current G, score with current D."""
        cands, vr, st = sample_g(args.g_url, serve_name, g_tok, turns,
                                 held_tids, args.heldout_cands, args.temp,
                                 args.max_tokens)
        pairs, first_idx = [], []
        agree_ex, agree_tok, th_len, th_jac, self_jac = [], [], [], [], []
        for tid in held_tids:
            if tid not in cands:
                continue
            t_first = t_held[tid][0]
            ref_txt = both_text(t_first["z"], t_first["y"])
            cs = cands[tid]
            for ci, c in enumerate(cs):
                if ci == 0:
                    first_idx.append(len(pairs))
                pairs.append({"prefix_text": pfx[tid], "ref": ref_txt,
                              "mine": both_text(c["z"], c["y"])})
            c0 = cs[0]
            gb, tb = bash_body(c0["y"]), bash_body(t_first["y"])
            agree_ex.append(1.0 if gb == tb else 0.0)
            agree_tok.append(token_jaccard(gb, tb))
            th_len.append(len(c0["z"]))
            th_jac.append(token_jaccard(c0["z"], t_first["z"]))
            if len(cs) >= 2:
                self_jac.append(token_jaccard(cs[0]["z"], cs[1]["z"]))
        p_teacher = disc.score(pairs) if pairs else []
        fool = sum(1 - p for p in p_teacher) / max(len(p_teacher), 1)
        d_acc = (sum(1 for i in first_idx if p_teacher[i] > 0.5)
                 / max(len(first_idx), 1))
        # length-bar control on the first-candidate pairs
        lc = lt = 0
        for i in first_idx:
            dl = len(pairs[i]["ref"]) - len(pairs[i]["mine"])
            lc += dl > 0
            lt += dl == 0
        lacc = (lc + 0.5 * lt) / max(len(first_idx), 1)
        return {
            "valid_rate": round(vr, 4),
            "fool": round(fool, 4),
            "d_acc": round(d_acc, 4),
            "len_bar": round(max(lacc, 1 - lacc), 4),
            "agree_exact": round(sum(agree_ex) / max(len(agree_ex), 1), 4),
            "agree_tok": round(sum(agree_tok) / max(len(agree_tok), 1), 4),
            "think_len": int(sum(th_len) / max(len(th_len), 1)),
            "think_jacc": round(sum(th_jac) / max(len(th_jac), 1), 4),
            "self_jacc": round(sum(self_jac) / max(len(self_jac), 1), 4),
            "n_pairs": len(pairs),
            "sample_errs": st["err"],
        }

    # ---------------- round 0: baseline (zero-shot D, raw G0) ---------------
    if state["round"] == 0:
        m = heldout_eval(0, state["g_serve"])
        m["round"] = 0
        m["note"] = "baseline_G0_zeroshotD"
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        status(f"round=0 fool={m['fool']} d_acc={m['d_acc']} "
               f"len_bar={m['len_bar']} valid={m['valid_rate']} "
               f"agree_exact={m['agree_exact']} agree_tok={m['agree_tok']} "
               f"think_len={m['think_len']} think_jacc={m['think_jacc']} "
               f"note=baseline")
        state["round"] = 1
        json.dump(state, open(state_path, "w"))

    # ---------------- alternating rounds ------------------------------------
    while state["round"] <= args.rounds:
        r = state["round"]
        t0 = time.time()
        note = "ok"

        # 1. fresh train batch (cycle through the shuffled pool)
        bt = []
        while len(bt) < args.batch_turns:
            if state["cursor"] >= len(shuffled):
                state["cursor"] = 0
                rng.shuffle(shuffled)
            bt.append(shuffled[state["cursor"]])
            state["cursor"] += 1

        # 2. sample candidates from current G
        cands, vr_train, st = sample_g(args.g_url, state["g_serve"], g_tok,
                                       turns, bt, args.n_cands, args.temp,
                                       args.max_tokens)
        print(f"[r{r}] sampled {len(cands)}/{len(bt)} turns "
              f"valid_rate={vr_train:.3f} errs={st['err']}", flush=True)

        # 3. score all candidates against a cached teacher rollout
        flat, owners = [], []
        for tid, cs in cands.items():
            ttxt, _ = teacher_text(tid, t_train, seed=r)
            for c in cs:
                flat.append({"prefix_text": pfx[tid], "ref": ttxt,
                             "mine": both_text(c["z"], c["y"])})
                owners.append((tid, c))
        p_teacher = disc.score(flat)
        rewards = [1 - p for p in p_teacher]

        # 4a. winners -> G SFT (subprocess on its own GPU)
        best = {}
        for (tid, c), rw in zip(owners, rewards):
            if tid not in best or rw > best[tid][1]:
                best[tid] = (c, rw)
        win_path = os.path.join(args.work, f"winners_r{r}.jsonl")
        with open(win_path, "w") as fh:
            for tid, (c, rw) in best.items():
                fh.write(json.dumps({
                    "turn_id": tid, "reward": rw,
                    "prompt": build_prompt(g_tok, turns[tid]),
                    "completion": c["raw"]}) + "\n")
        mean_win_rw = sum(rw for _, rw in best.values()) / max(len(best), 1)
        new_lora = os.path.join(args.work, "g_lora", f"r{r}")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.sft_gpu
        sft_log = open(os.path.join(args.work, f"g_sft_r{r}.log"), "w")
        sft = subprocess.Popen(
            [args.sft_python, os.path.join(os.path.dirname(os.path.abspath(__file__)), "g_sft.py"),
             "--base", args.g_base, "--winners", win_path,
             "--lora-in", state["g_lora"], "--lora-out", new_lora,
             "--max-steps", str(state["g_steps"])],
            env=env, stdout=sft_log, stderr=subprocess.STDOUT)

        # 4b. D update in parallel: fresh pairs (top + one random candidate
        #     per turn) mixed with ~30% replay of earlier rounds
        fresh = []
        by_tid = {}
        for (tid, c), pt in zip(owners, p_teacher):
            by_tid.setdefault(tid, []).append((c, pt))
        for tid, lst in by_tid.items():
            ttxt, _ = teacher_text(tid, t_train, seed=r)
            lst.sort(key=lambda x: x[1])  # lowest p_teacher = most fooling
            take = [lst[0]]
            if len(lst) > 1:
                take.append(random.Random(r * 7919 + hash(tid) % 1000).choice(lst[1:]))
            for c, _ in take:
                fresh.append({"turn_id": tid, "prefix_text": pfx[tid],
                              "ref": ttxt, "mine": both_text(c["z"], c["y"])})
        replay = []
        if os.path.exists(replay_path):
            pool = [json.loads(x) for x in open(replay_path)]
            k = int(len(fresh) * args.replay_frac / max(1 - args.replay_frac, 1e-9))
            if pool and k:
                replay = random.Random(r).sample(pool, min(k, len(pool)))
                for p in replay:
                    p["prefix_text"] = pfx.get(p["turn_id"], "")
        d_steps, d_loss = disc.train_pairs(fresh + replay, seed=r,
                                           max_steps=args.d_steps)
        print(f"[r{r}] D trained: steps={d_steps} loss={d_loss:.4f} "
              f"fresh={len(fresh)} replay={len(replay)}", flush=True)
        with open(replay_path, "a") as fh:
            for p in fresh:
                fh.write(json.dumps({"turn_id": p["turn_id"], "ref": p["ref"],
                                     "mine": p["mine"]}) + "\n")

        # 4c. wait for G SFT, hot-swap the new adapter into vLLM
        sft_rc = sft.wait()
        sft_log.close()
        if sft_rc != 0:
            note = "g_sft_failed"
            status(f"round={r} ERROR g_sft exit={sft_rc}; keeping previous G")
        else:
            serve_name = f"g_r{r}"
            try:
                prev = [state["g_serve"]] if state["g_serve"] != args.g_base else []
                load_adapter(args.g_url, serve_name, new_lora, drop=prev)
            except Exception as e:
                note = f"lora_load_failed:{type(e).__name__}"
                serve_name = state["g_serve"]
        # 5. held-out metrics with (possibly) new G and new D
        cand_serve = serve_name if sft_rc == 0 and "failed" not in note else state["g_serve"]
        m = heldout_eval(r, cand_serve)

        # 6. collapse guards
        rolled_back = False
        if m["valid_rate"] < 0.5 or (m["self_jacc"] > 0.92 and m["n_pairs"] > 20):
            rolled_back = True
            state["g_steps"] = max(8, state["g_steps"] // 2)
            note = f"ROLLBACK(valid={m['valid_rate']},self_jacc={m['self_jacc']})->g_steps={state['g_steps']}"
            try:  # revert serving to the previous good adapter
                if state["g_lora"]:
                    load_adapter(args.g_url, f"g_r{r}_reverted", state["g_lora"])
                    state["g_serve"] = f"g_r{r}_reverted"
                else:
                    state["g_serve"] = args.g_base
            except Exception:
                state["g_serve"] = args.g_base
        elif sft_rc == 0 and "failed" not in note:
            state["g_lora"] = new_lora
            state["g_serve"] = cand_serve

        disc.save(os.path.join(args.work, "d_lora", f"r{r}"))
        m.update({"round": r, "note": note, "train_valid_rate": round(vr_train, 4),
                  "mean_winner_reward": round(mean_win_rw, 4),
                  "d_train_steps": d_steps, "d_train_loss": round(d_loss, 4),
                  "g_steps": state["g_steps"],
                  "wall_s": round(time.time() - t0, 1)})
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        status(f"round={r} fool={m['fool']} d_acc={m['d_acc']} "
               f"len_bar={m['len_bar']} valid={m['valid_rate']} "
               f"agree_exact={m['agree_exact']} agree_tok={m['agree_tok']} "
               f"think_len={m['think_len']} think_jacc={m['think_jacc']} "
               f"self_jacc={m['self_jacc']} win_rw={m['mean_winner_reward']} "
               f"wall={m['wall_s']}s note={note}")
        state["round"] = r + 1
        json.dump(state, open(state_path, "w"))
        if rolled_back and os.path.isdir(new_lora):
            os.rename(new_lora, new_lora + "_rolledback")

    status("driver finished all rounds")


if __name__ == "__main__":
    main()
