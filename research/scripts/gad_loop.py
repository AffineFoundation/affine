#!/usr/bin/env python3
"""Continuous adversarial distillation: generator vs discriminator, on the SWE corpus.

The two players, both training continuously:

  Discriminator D  sees (prefix, candidate A, candidate B) where one candidate is
                   the teacher's rollout and the other is the generator's, and
                   answers which is the teacher by comparing the logits of the
                   tokens "A" and "B". A candidate block is the thought AND the
                   bash action together, so D judges the whole response.

  Generator G      samples several candidate rollouts per turn and is trained
                   toward the ones D mistakes for the teacher.

G is updated by reinforcing its own best samples (rejection fine-tuning) rather
than by a policy-gradient estimator. Both maximise the same objective, but
reinforcing selected samples is far more stable to run unattended, which matters
more here than the last few percent of sample efficiency.

Guards, all mandatory, because "fool the discriminator" is trivially gamed by
producing text that reads teacher-like while the action degenerates:

  * a candidate whose action is not a valid closed bash block scores zero and can
    never be selected for training;
  * G takes a capped number of optimiser steps per round;
  * collapse (valid-action rate or reward collapsing, or D saturating) triggers a
    rollback to the previous adapter and halves the learning rate.

LoRA is confined to the attention projections. The generator is a
mixture-of-experts model, and adapters on expert weights are not reliably
loadable by the inference server, so keeping to attention keeps the trained
adapter and the served adapter identical.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import re
import shutil
import sys
import time

import requests
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from disc_text import normalize  # noqa: E402

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BASH_RE = re.compile(r"```bash\n.*?\n```", re.DOTALL)
THOUGHT_LABEL_RE = re.compile(r"^\s*THOUGHT:\s*")

QUESTION = ("One of the two candidates above was written by the reference model "
            "and the other by a different model. Which candidate was written by "
            "the reference model? Answer with a single letter, A or B.")


# --------------------------------------------------------------------------- #
# rollout handling (validator contract)
# --------------------------------------------------------------------------- #

def split_rollout(text: str) -> tuple[str, str]:
    if THINK_CLOSE in text:
        latent, _, rest = text.partition(THINK_CLOSE)
    else:
        latent, rest = "", text
    matches = list(BASH_RE.finditer(rest))
    if not matches:
        return "", ""
    m = matches[-1]
    y = m.group(0)
    visible = THOUGHT_LABEL_RE.sub("", rest[: m.start()].strip())
    z = "\n".join(s for s in (latent.strip(), visible.strip()) if s)
    return z, y


def both_channel(z: str, y: str) -> str:
    """The discriminator's candidate block: thought, blank line, action."""
    return (normalize(z) + "\n\n" + normalize(y)).strip()


def valid_action(y: str) -> bool:
    return bool(y) and bool(BASH_RE.fullmatch(y.strip()))


def parse_prefix(raw):
    import ast
    if isinstance(raw, list):
        return raw
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(raw)
            if isinstance(v, list):
                return v
        except Exception:
            continue
    return None


def load_turns(path, limit=0):
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("prefix"):
                out[r["turn_id"]] = r
            if limit and len(out) >= limit:
                break
    return out


def load_teacher(path):
    """turn_id -> teacher rollout (z, y), first usable candidate."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for c in r.get("candidates") or []:
                if c.get("z") and valid_action(c.get("y") or ""):
                    out[r["turn_id"]] = (c["z"], c["y"])
                    break
    return out


# --------------------------------------------------------------------------- #
# discriminator
# --------------------------------------------------------------------------- #

class Judge:
    """Qwen3 causal LM used as an A/B chooser via the logits of "A" and "B"."""

    def __init__(self, model_name, device, lora_r=16, max_len=3072, lr=5e-5):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.max_len = device, max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16, attn_implementation="sdpa")
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        m.config.use_cache = False
        m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
        self.model = get_peft_model(m, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )).to(device)
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)
        ids = [self.tok.encode(t, add_special_tokens=False) for t in ("A", "B")]
        self.ab = [i[0] for i in ids]

    def _prompt(self, prefix_text, a_text, b_text):
        body = (f"{prefix_text}\n\n=== Candidate A ===\n{a_text}\n\n"
                f"=== Candidate B ===\n{b_text}\n\n{QUESTION}")
        msgs = [{"role": "user", "content": body}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def _encode(self, prefix_text, a_text, b_text):
        """Trim the prefix until the whole prompt fits; candidates are never cut."""
        chars = min(len(prefix_text), self.max_len * 6)
        for _ in range(6):
            text = self._prompt(prefix_text[-chars:] if chars else "", a_text, b_text)
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= self.max_len or chars == 0:
                return ids[-self.max_len:]
            chars = int(chars * 0.5)
        return ids[-self.max_len:]

    def _batch(self, items):
        """items: (prefix, a_text, b_text). Returns padded ids/mask."""
        seqs = [self._encode(*it) for it in items]
        n = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s)   # left pad
            att[i, n - len(s):] = 1
        return ids.to(self.device), att.to(self.device)

    def _logits_ab(self, ids, att):
        out = self.model(input_ids=ids, attention_mask=att).logits[:, -1, :]
        return out[:, self.ab]        # [batch, 2] -> (A, B)

    @torch.no_grad()
    def p_teacher(self, prefix, cand, ref, batch=4):
        """P(D says `cand` is the teacher), averaged over both slot orders.

        Averaging removes the judge's positional bias, which is large enough to
        dominate a single-order reading.
        """
        self.model.eval()
        items = [(prefix, cand, ref), (prefix, ref, cand)]
        ids, att = self._batch(items)
        lg = self._logits_ab(ids, att).float()
        p = F.softmax(lg, dim=-1)
        # order 1: cand in slot A -> P(cand)=P(A); order 2: cand in B -> P(B)
        return 0.5 * (p[0, 0] + p[1, 1]).item()

    def train_step(self, samples, accum=1):
        """samples: (prefix, teacher_text, gen_text). D learns to find the teacher."""
        self.model.train()
        rnd = random.Random(0)
        losses, correct = [], 0
        self.opt.zero_grad(set_to_none=True)
        for i, (prefix, tref, gen) in enumerate(samples):
            teacher_in_a = rnd.random() < 0.5     # balance slots to kill bias
            a, b = (tref, gen) if teacher_in_a else (gen, tref)
            ids, att = self._batch([(prefix, a, b)])
            lg = self._logits_ab(ids, att).float()
            target = torch.tensor([0 if teacher_in_a else 1], device=self.device)
            loss = F.cross_entropy(lg, target)
            (loss / max(1, accum)).backward()
            losses.append(loss.item())
            correct += int(lg.argmax(-1).item() == target.item())
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
        return (sum(losses) / max(1, len(losses)),
                correct / max(1, len(samples)))


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #

class Generator:
    """Sampled from over HTTP (fast server), trained locally with LoRA."""

    def __init__(self, model_name, device, url, lora_r=16, max_len=8192,
                 lr=1e-5, served_name=None):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.url = model_name, url.rstrip("/")
        self.served = served_name or model_name
        self.device, self.max_len = device, max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16, attn_implementation="sdpa")
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        m.config.use_cache = False
        m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
        # attention-only: expert adapters do not load reliably in the server
        self.model = get_peft_model(m, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )).to(device)
        self.lr = lr
        self._loaded = None
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def set_lr(self, lr):
        self.lr = lr
        for g in self.opt.param_groups:
            g["lr"] = lr

    def render(self, prefix_messages):
        p = self.tok.apply_chat_template(prefix_messages, tokenize=False,
                                        add_generation_prompt=True)
        if not p.rstrip().endswith(THINK_OPEN):
            p += THINK_OPEN
        return p

    def sample(self, prompt, k, temp, max_tokens, adapter=None, timeout=1800):
        body = {"model": adapter or self.served, "prompt": prompt, "n": k,
                "temperature": temp, "max_tokens": max_tokens}
        r = requests.post(f"{self.url}/v1/completions", json=body, timeout=timeout)
        r.raise_for_status()
        return [c.get("text") or "" for c in r.json()["choices"]]

    def sft_step(self, pairs, accum=4):
        """pairs: (prompt_text, completion_text). Loss on completion tokens only."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        losses = []
        for i, (prompt, completion) in enumerate(pairs):
            p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = self.tok(completion, add_special_tokens=False)["input_ids"]
            if not c_ids:
                continue
            keep = self.max_len - len(c_ids)
            if keep < 64:
                c_ids = c_ids[: self.max_len - 64]
                keep = 64
            p_ids = p_ids[-keep:]
            ids = torch.tensor([p_ids + c_ids], device=self.device)
            labels = torch.tensor([[-100] * len(p_ids) + c_ids], device=self.device)
            out = self.model(input_ids=ids, labels=labels)
            (out.loss / max(1, accum)).backward()
            losses.append(out.loss.item())
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
        if len(losses) % accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return sum(losses) / max(1, len(losses))

    def save_adapter(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        return path

    def load_adapter_into_server(self, name, path):
        """Hot-swap the adapter in the inference server; False if unsupported.

        The previously loaded adapter is dropped first: the server keeps every
        registered adapter, and each round uses a new name, so without this the
        registrations pile up until the slot limit is hit.
        """
        try:
            r = requests.post(f"{self.url}/v1/load_lora_adapter",
                              json={"lora_name": name, "lora_path": path},
                              timeout=600)
            ok = r.status_code < 300
        except Exception:
            return False
        if ok:
            for stale in (self._loaded, "probe"):
                if stale and stale != name:
                    try:
                        requests.post(f"{self.url}/v1/unload_lora_adapter",
                                      json={"lora_name": stale}, timeout=120)
                    except Exception:
                        pass
            self._loaded = name
        return ok


# --------------------------------------------------------------------------- #
# loop
# --------------------------------------------------------------------------- #

def status(logfile, msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | {msg}"
    print(line, flush=True)
    if logfile:
        try:
            with open(logfile, "a") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--teacher-rollouts", required=True)
    ap.add_argument("--gen-model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--gen-url", default="http://127.0.0.1:8004")
    ap.add_argument("--disc-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--gen-device", default="cuda:4")
    ap.add_argument("--disc-device", default="cuda:6")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--turns-per-round", type=int, default=16)
    ap.add_argument("--k", type=int, default=6, help="candidates per turn")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--g-steps-cap", type=int, default=8,
                    help="hard cap on generator optimiser steps per round")
    ap.add_argument("--d-samples", type=int, default=24,
                    help="discriminator training samples per round")
    ap.add_argument("--replay", type=int, default=400,
                    help="replay buffer size, so D does not forget old G behaviour")
    ap.add_argument("--gen-lr", type=float, default=1e-5)
    ap.add_argument("--disc-lr", type=float, default=5e-5)
    ap.add_argument("--reward-floor", type=float, default=0.0,
                    help="absolute floor on a winner's reward; keep at 0 with a "
                         "trained discriminator, which scores all samples low")
    ap.add_argument("--min-valid-rate", type=float, default=0.5,
                    help="collapse trigger: valid-action rate below this")
    ap.add_argument("--ckpt-dir", default="/opt/ckpt/gadA")
    ap.add_argument("--status-log", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in --ckpt-dir")
    ap.add_argument("--disc-init", default=None,
                    help="pre-trained discriminator adapter to start D from")
    ap.add_argument("--gen-max-len", type=int, default=8192)
    ap.add_argument("--disc-max-len", type=int, default=3072)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    status(args.status_log, f"LOOP init | gen={args.gen_model} disc={args.disc_model} "
                            f"k={args.k} turns/round={args.turns_per_round}")

    turns = load_turns(args.turns)
    teacher = load_teacher(args.teacher_rollouts)
    usable = [t for t in teacher if t in turns]
    random.Random(0).shuffle(usable)
    status(args.status_log,
           f"LOOP data | turns={len(turns)} teacher_rollouts={len(teacher)} "
           f"usable={len(usable)}")
    if not usable:
        status(args.status_log, "LOOP abort | no turns with a teacher rollout")
        return

    gen = Generator(args.gen_model, args.gen_device, args.gen_url,
                    max_len=args.gen_max_len, lr=args.gen_lr)
    judge = Judge(args.disc_model, args.disc_device,
                  max_len=args.disc_max_len, lr=args.disc_lr)
    status(args.status_log, "LOOP models_loaded")

    def load_into(model, path, what):
        """Load a saved LoRA state into a live peft model."""
        f = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(f):
            return False
        try:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(f))
            status(args.status_log, f"LOOP loaded {what} | {path}")
            return True
        except Exception as e:
            status(args.status_log,
                   f"LOOP load {what} failed | {type(e).__name__}")
            return False

    if args.disc_init:
        load_into(judge.model, args.disc_init, "disc_init")

    replay = []
    cursor = 0
    prev_ckpt = None
    adapter_name = None
    lr = args.gen_lr

    # Resuming matters for an unattended run: a crash should cost one round, not
    # the whole night's training.
    start = 0
    if args.resume:
        cks = sorted(glob.glob(os.path.join(args.ckpt_dir, "round*")))
        for ck in reversed(cks):
            if not load_into(gen.model, ck, "gen_resume"):
                continue
            start = int(os.path.basename(ck)[5:])
            prev_ckpt = ck
            cursor = (start * args.turns_per_round) % len(usable)
            load_into(judge.model, os.path.join(ck, "disc"), "disc_resume")
            status(args.status_log,
                   f"LOOP resumed | from {os.path.basename(ck)}")
            break

    rnd = start
    while rnd < args.rounds:
        t0 = time.time()
        batch = [usable[(cursor + i) % len(usable)]
                 for i in range(args.turns_per_round)]

        # ---- sample candidates from G -----------------------------------
        groups, n_raw, n_valid = [], 0, 0
        errs = {}
        for tid in batch:
            prefix_msgs = parse_prefix(turns[tid].get("prefix"))
            if not prefix_msgs:
                continue
            prompt = gen.render(prefix_msgs)
            try:
                texts = gen.sample(prompt, args.k, args.temp, args.max_tokens,
                                   adapter=adapter_name)
            except Exception as e:
                errs[type(e).__name__] = errs.get(type(e).__name__, 0) + 1
                continue
            cands = []
            for tx in texts:
                n_raw += 1
                z, y = split_rollout(tx)
                if not z or not valid_action(y):
                    continue          # guard: invalid action is never trainable
                n_valid += 1
                cands.append({"z": z, "y": y, "text": tx})
            if cands:
                groups.append({"tid": tid, "prompt": prompt, "cands": cands})

        valid_rate = n_valid / max(1, n_raw)
        if not groups:
            # Usually the sampler is restarting. Wait for it instead of burning
            # through the round budget, and do not consume a round number.
            status(args.status_log,
                   f"ROUND {rnd + 1} stalled | no usable samples "
                   f"(valid_rate={valid_rate:.3f} errors={errs}) - waiting 90s")
            time.sleep(90)
            continue

        rnd += 1
        cursor = (cursor + args.turns_per_round) % len(usable)
        if errs:
            status(args.status_log, f"ROUND {rnd} partial_sample_errors {errs}")

        # ---- reward each candidate with D -------------------------------
        prefix_text_cache = {}
        for g in groups:
            tid = g["tid"]
            if tid not in prefix_text_cache:
                msgs = parse_prefix(turns[tid].get("prefix")) or []
                prefix_text_cache[tid] = "\n".join(
                    str(m.get("content", "")) for m in msgs)
            ptext = prefix_text_cache[tid]
            tz, ty = teacher[tid]
            ref = both_channel(tz, ty)
            for c in g["cands"]:
                c["reward"] = judge.p_teacher(ptext, both_channel(c["z"], c["y"]), ref)
            g["cands"].sort(key=lambda c: -c["reward"])

        rewards = [c["reward"] for g in groups for c in g["cands"]]
        mean_r = sum(rewards) / max(1, len(rewards))
        best_r = [g["cands"][0]["reward"] for g in groups]
        mean_best = sum(best_r) / max(1, len(best_r))

        # ---- generator update: reinforce its own best samples -----------
        # Selection is relative, not absolute. A well-trained discriminator puts
        # every generator sample near zero, so an absolute floor would reject the
        # whole batch and stop training; what carries information is which of
        # the k samples the judge rated highest.
        winners = []
        for g in groups:
            cs = g["cands"]
            if cs[0]["reward"] < args.reward_floor:
                continue
            if len(cs) > 1 and cs[0]["reward"] <= cs[-1]["reward"]:
                continue      # judge saw no difference; nothing to learn here
            winners.append((g["prompt"], cs[0]["text"]))
        winners = winners[: args.g_steps_cap * 4]
        g_loss = float("nan")
        if winners:
            g_loss = gen.sft_step(winners, accum=4)

        # ---- discriminator update: fresh pairs + replay -----------------
        fresh = []
        for g in groups:
            tid = g["tid"]
            tz, ty = teacher[tid]
            fresh.append((prefix_text_cache[tid], both_channel(tz, ty),
                          both_channel(g["cands"][0]["z"], g["cands"][0]["y"])))
        replay.extend(fresh)
        replay = replay[-args.replay:]
        d_batch = fresh + random.sample(replay, min(len(replay),
                                                   max(0, args.d_samples - len(fresh))))
        d_loss, d_acc = judge.train_step(d_batch, accum=4)

        # ---- checkpoint every round ------------------------------------
        ck = os.path.join(args.ckpt_dir, f"round{rnd:04d}")
        gen.save_adapter(ck)
        # D is saved alongside G so a restart does not reset the reward scale.
        try:
            judge.model.save_pretrained(os.path.join(ck, "disc"))
        except Exception:
            pass
        ok = gen.load_adapter_into_server(f"gadA_r{rnd}", ck)
        if ok:
            adapter_name = f"gadA_r{rnd}"
        elif rnd == 1:
            status(args.status_log,
                   "LOOP warn | server rejected adapter hot-load; G keeps sampling "
                   "from the base policy, so training will not compound")

        status(args.status_log,
               f"ROUND {rnd} | valid_rate={valid_rate:.3f} mean_r={mean_r:.3f} "
               f"best_r={mean_best:.3f} g_loss={g_loss:.4f} d_loss={d_loss:.4f} "
               f"d_acc={d_acc:.3f} winners={len(winners)} lr={lr:.2e} "
               f"secs={time.time()-t0:.0f} ckpt={os.path.basename(ck)} "
               f"served={'yes' if ok else 'no'}")

        # ---- collapse detection: roll back and halve --------------------
        collapsed = valid_rate < args.min_valid_rate
        if collapsed and prev_ckpt:
            lr = max(lr * 0.5, 1e-7)
            gen.set_lr(lr)
            try:
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                sd = load_file(os.path.join(prev_ckpt, "adapter_model.safetensors"))
                set_peft_model_state_dict(gen.model, sd)
                status(args.status_log,
                       f"ROUND {rnd} | COLLAPSE valid_rate={valid_rate:.3f} -> "
                       f"rolled back to {os.path.basename(prev_ckpt)}, lr={lr:.2e}")
            except Exception as e:
                status(args.status_log,
                       f"ROUND {rnd} | COLLAPSE rollback failed {type(e).__name__}")
        else:
            prev_ckpt = ck

        # keep the checkpoint directory from growing without bound
        if rnd > 6:
            old = os.path.join(args.ckpt_dir, f"round{rnd-6:04d}")
            if old != prev_ckpt and os.path.isdir(old) and (rnd - 6) % 5:
                shutil.rmtree(old, ignore_errors=True)


if __name__ == "__main__":
    main()
