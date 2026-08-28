#!/usr/bin/env python3
"""Track P driver: PRODUCTION-SPEC adversarial distillation loop.

  G (miner)    Qwen/Qwen3.6-35B-A3B, served by vLLM (GPU 0) with runtime
               LoRA hot-swap; trained per round by g_sft.py (GPUs 6,7),
               attention-only LoRA -- Track A's exact g-step method.
  T (teacher)  Qwen/Qwen3.8-27B; rollouts CACHED (copied from Track A pod,
               same teacher); live server on GPU 1 only for ad-hoc use.
  D (judge)    the SAME Qwen3.8-27B base + LoRA -- the novel element.
               Served by vLLM (GPUs 2,3, TP2) with runtime LoRA updating;
               scored via the A-vs-B first-token logprobs; trained every
               d_interval-th round by d_train.py (GPUs 4,5).

BALANCED REGIME FROM ROUND ZERO (the other arms collapsed when D trained
every round): D trains only every --d-interval(=3)rd round on pairs
accumulated since its last update; if held-out d_acc > 0.85 for 3
consecutive D-train rounds, D is frozen for twice the interval.

Round 0 logs the ZERO-SHOT judge: the raw 27B teacher prompted as an A/B
judge (no adapter), with position-bias / matched-pair diagnostics.
All candidate text is normalised (reasoning markers stripped) before D
sees it -- this closed a format leak in the earlier discriminator work.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gad_common import (bash_body, both_text, build_prompt, load_rollout_cache,  # noqa: E402
                        load_turns, split_rollout, token_jaccard)
from judge_common import fit_ids  # noqa: E402


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def render_prefix(messages, max_chars):
    parts = [f"<{m.get('role','?')}>\n{m.get('content','')}" for m in messages]
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


class DClient:
    """A/B judge over a vLLM server: first-token logprobs of 'A' vs 'B'."""

    def __init__(self, url, base_name, tok, max_len):
        self.url = url.rstrip("/")
        self.base = base_name
        self.serve = base_name          # adapter name once one is loaded
        self.tok = tok
        self.max_len = max_len
        self.miss = 0

    def _one(self, ids):
        body = {"model": self.serve, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0, "logprobs": 20}
        r = requests.post(f"{self.url}/v1/completions", json=body, timeout=300)
        r.raise_for_status()
        lp = r.json()["choices"][0].get("logprobs") or {}
        tops = (lp.get("top_logprobs") or [{}])[0] or {}
        import math
        pa = pb = 0.0
        for t, v in tops.items():
            s = t.strip()
            if s == "A":
                pa += math.exp(v)
            elif s == "B":
                pb += math.exp(v)
        if pa + pb <= 0:
            self.miss += 1
            return 0.5
        return pa / (pa + pb)

    def score_pairs(self, pairs, workers=24):
        """pairs: [{prefix_text, mine, ref}] -> list of
        {p_teacher, p_ta, p_tb} with p_ta = P(A) when teacher sits in slot A
        and p_tb = P(B) when teacher sits in slot B (both-order eval)."""
        jobs = []
        for i, p in enumerate(pairs):
            ids_ta = fit_ids(self.tok, p["prefix_text"], p["ref"], p["mine"],
                             self.max_len)
            ids_tb = fit_ids(self.tok, p["prefix_text"], p["mine"], p["ref"],
                             self.max_len)
            jobs.append((i, "ta", ids_ta))
            jobs.append((i, "tb", ids_tb))
        out = [{} for _ in pairs]

        def work(job):
            i, order, ids = job
            try:
                p_a = self._one(ids)
            except Exception:
                p_a = None
            return i, order, p_a

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, order, p_a in ex.map(work, jobs):
                out[i][order] = p_a
        res = []
        for o in out:
            p_ta = o.get("ta")          # prob mass on slot A, teacher in A
            p_tb = o.get("tb")          # prob mass on slot A, teacher in B
            if p_ta is None or p_tb is None:
                res.append(None)
                continue
            res.append({"p_teacher": 0.5 * (p_ta + (1.0 - p_tb)),
                        "p_ta": p_ta, "p_tb": 1.0 - p_tb,
                        "pick_a": 0.5 * (p_ta + p_tb)})
        return res

    def load_adapter(self, name, path):
        stale = self.serve if self.serve != self.base else None
        r = requests.post(f"{self.url}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path},
                          timeout=600)
        r.raise_for_status()
        if stale:
            try:
                requests.post(f"{self.url}/v1/unload_lora_adapter",
                              json={"lora_name": stale}, timeout=120)
            except Exception:
                pass
        self.serve = name


def sample_g(url, model_name, tok, turns, tids, n, temp, max_tokens,
             max_prompt_chars=60000, timeout=2400, workers=24):
    """n candidates per turn from the G server. Validity gate: rollout must
    split into (thought, action) with a closed bash block; invalid never
    enters scoring (reward 0 by construction)."""
    import threading
    out = {}
    lock = threading.Lock()
    stats = {"total": 0, "valid": 0, "err": 0}

    def work(tid):
        prompt = build_prompt(tok, turns[tid])
        if len(prompt) > max_prompt_chars:
            return
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

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, tids))
    vr = stats["valid"] / max(stats["total"], 1)
    return out, vr, stats


def load_g_adapter(url, name, path, drop=()):
    for stale in set(drop) | {name}:
        try:
            requests.post(f"{url}/v1/unload_lora_adapter",
                          json={"lora_name": stale}, timeout=60)
        except Exception:
            pass
    r = requests.post(f"{url}/v1/load_lora_adapter",
                      json={"lora_name": name, "lora_path": path}, timeout=600)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--teacher-train", default="/dshare/gad/teacher_train.jsonl")
    ap.add_argument("--teacher-heldout", default="/dshare/gad/teacher_heldout.jsonl")
    ap.add_argument("--g-url", default="http://127.0.0.1:8001")
    ap.add_argument("--g-base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--d-url", default="http://127.0.0.1:8002")
    ap.add_argument("--d-base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--sft-gpus", default="6,7")
    ap.add_argument("--dtrain-gpus", default="4,5")
    ap.add_argument("--train-python", default="/root/trainenv/bin/python")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--batch-turns", type=int, default=32)
    ap.add_argument("--n-cands", type=int, default=5)
    ap.add_argument("--heldout-n", type=int, default=200)
    ap.add_argument("--heldout-cands", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=3584)
    ap.add_argument("--d-interval", type=int, default=3)
    ap.add_argument("--d-steps", type=int, default=150)
    ap.add_argument("--d-steps-first", type=int, default=250)
    ap.add_argument("--d-lr", type=float, default=1e-5)
    ap.add_argument("--d-freeze-acc", type=float, default=0.85)
    ap.add_argument("--g-steps", type=int, default=40)
    ap.add_argument("--replay-frac", type=float, default=0.30)
    ap.add_argument("--work", default="/dshare/gad")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    status_path = os.path.join(args.work, "status.log")
    metrics_path = os.path.join(args.work, "metrics.jsonl")
    replay_path = os.path.join(args.work, "replay.jsonl")
    pending_path = os.path.join(args.work, "d_pending.jsonl")
    state_path = os.path.join(args.work, "state.json")

    def status(line):
        with open(status_path, "a") as fh:
            fh.write(f"{now()} {line}\n")
        print(f"STATUS {line}", flush=True)

    from transformers import AutoTokenizer
    g_tok = AutoTokenizer.from_pretrained(args.g_base)
    d_tok = AutoTokenizer.from_pretrained(args.d_base)

    turns = load_turns(args.turns)
    t_train = load_rollout_cache(args.teacher_train)
    t_held = load_rollout_cache(args.teacher_heldout)
    train_tids = sorted(t for t in t_train if t in turns)
    held_tids = sorted(t for t in t_held if t in turns)[: args.heldout_n]
    print(f"train turns with teacher rollouts: {len(train_tids)}  "
          f"held-out: {len(held_tids)}", flush=True)

    pfx = {}
    for t in set(train_tids) | set(held_tids):
        pfx[t] = render_prefix(turns[t], args.d_max_len * 6)

    disc = DClient(args.d_url, args.d_base, d_tok, args.d_max_len)

    state = {"round": 0, "g_lora": "", "g_serve": args.g_base,
             "d_lora": "", "d_serve": args.d_base,
             "g_steps": args.g_steps, "cursor": 0,
             "d_trains": 0, "d_hot_streak": 0, "d_frozen_until": 0,
             "d_acc_history": []}
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))
        print(f"resuming state: {state}", flush=True)
        if state["g_serve"] != args.g_base and state["g_lora"]:
            try:
                load_g_adapter(args.g_url, state["g_serve"], state["g_lora"])
            except Exception as e:
                print(f"G resume adapter push failed ({e}); serving base", flush=True)
                state["g_serve"] = args.g_base
        if state["d_serve"] != args.d_base and state["d_lora"]:
            try:
                disc.load_adapter(state["d_serve"], state["d_lora"])
            except Exception as e:
                print(f"D resume adapter push failed ({e}); serving base", flush=True)
                state["d_serve"] = args.d_base
        disc.serve = state["d_serve"]

    status(f"driver up: T=D_base={args.d_base} G={args.g_base} "
           f"D=27B+LoRA(vLLM {args.d_url}) train_turns={len(train_tids)} "
           f"held={len(held_tids)} d_interval={args.d_interval} "
           f"gpu_map[G_serve=0 T_serve=1 D_serve=2,3 D_train={args.dtrain_gpus} "
           f"G_sft={args.sft_gpus}]")

    rng = random.Random(120)
    shuffled = train_tids[:]
    rng.shuffle(shuffled)

    def teacher_text(tid, cache):
        r = cache[tid][0]
        return both_text(r["z"], r["y"])

    def heldout_eval(round_no, serve_name):
        """Sample held-out candidates from current G, judge with current D.
        Returns headline metrics + D diagnostics (pos bias, matched acc)."""
        cands, vr, st = sample_g(args.g_url, serve_name, g_tok, turns,
                                 held_tids, args.heldout_cands, args.temp,
                                 args.max_tokens)
        pairs, first_idx = [], []
        agree_ex, agree_tok, th_len, th_jac, self_jac = [], [], [], [], []
        for tid in held_tids:
            if tid not in cands:
                continue
            ref_txt = teacher_text(tid, t_held)
            cs = cands[tid]
            for ci, c in enumerate(cs):
                if ci == 0:
                    first_idx.append(len(pairs))
                pairs.append({"prefix_text": pfx[tid], "ref": ref_txt,
                              "mine": both_text(c["z"], c["y"])})
            c0 = cs[0]
            t0r = t_held[tid][0]
            gb, tb = bash_body(c0["y"]), bash_body(t0r["y"])
            agree_ex.append(1.0 if gb == tb else 0.0)
            agree_tok.append(token_jaccard(gb, tb))
            th_len.append(len(c0["z"]))
            th_jac.append(token_jaccard(c0["z"], t0r["z"]))
            if len(cs) >= 2:
                self_jac.append(token_jaccard(cs[0]["z"], cs[1]["z"]))
        scores = disc.score_pairs(pairs) if pairs else []
        good = [(i, s) for i, s in enumerate(scores) if s]
        fool = sum(1 - s["p_teacher"] for _, s in good) / max(len(good), 1)
        fi = [i for i in first_idx if scores[i]]
        d_acc = sum(1 for i in fi if scores[i]["p_teacher"] > 0.5) / max(len(fi), 1)
        pos_bias = sum(scores[i]["pick_a"] for i in fi) / max(len(fi), 1)
        matched = sum(1 for i in fi
                      if scores[i]["p_ta"] > 0.5 and scores[i]["p_tb"] > 0.5)
        matched_acc = matched / max(len(fi), 1)
        lc = lt = 0
        for i in fi:
            dl = len(pairs[i]["ref"]) - len(pairs[i]["mine"])
            lc += dl > 0
            lt += dl == 0
        lacc = (lc + 0.5 * lt) / max(len(fi), 1)
        return {
            "valid_rate": round(vr, 4),
            "fool": round(fool, 4),
            "d_acc": round(d_acc, 4),
            "d_pos_bias": round(pos_bias, 4),
            "d_matched_acc": round(matched_acc, 4),
            "len_bar": round(max(lacc, 1 - lacc), 4),
            "agree_exact": round(sum(agree_ex) / max(len(agree_ex), 1), 4),
            "agree_tok": round(sum(agree_tok) / max(len(agree_tok), 1), 4),
            "think_len": int(sum(th_len) / max(len(th_len), 1)),
            "think_jacc": round(sum(th_jac) / max(len(th_jac), 1), 4),
            "self_jacc": round(sum(self_jac) / max(len(self_jac), 1), 4),
            "n_pairs": len(pairs),
            "judge_miss": disc.miss,
            "sample_errs": st["err"],
        }

    # ------------- round 0: zero-shot judge + raw-G0 baseline ---------------
    if state["round"] == 0:
        m = heldout_eval(0, state["g_serve"])
        m["round"] = 0
        m["note"] = "baseline_G0_zeroshot27Bjudge"
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        status(f"round=0 ZEROSHOT-JUDGE fool={m['fool']} d_acc={m['d_acc']} "
               f"pos_bias={m['d_pos_bias']} matched_acc={m['d_matched_acc']} "
               f"len_bar={m['len_bar']} valid={m['valid_rate']} "
               f"agree_exact={m['agree_exact']} agree_tok={m['agree_tok']} "
               f"think_len={m['think_len']} note=baseline (27B base as judge, "
               f"no LoRA; cf trackB separate-8B zero-shot 0.64)")
        state["round"] = 1
        json.dump(state, open(state_path, "w"))

    # ------------- rounds ----------------------------------------------------
    while state["round"] <= args.rounds:
        r = state["round"]
        t0 = time.time()
        note = "ok"

        bt = []
        while len(bt) < args.batch_turns:
            if state["cursor"] >= len(shuffled):
                state["cursor"] = 0
                rng.shuffle(shuffled)
            bt.append(shuffled[state["cursor"]])
            state["cursor"] += 1

        cands, vr_train, st = sample_g(args.g_url, state["g_serve"], g_tok,
                                       turns, bt, args.n_cands, args.temp,
                                       args.max_tokens)
        print(f"[r{r}] sampled {len(cands)}/{len(bt)} turns "
              f"valid_rate={vr_train:.3f} errs={st['err']}", flush=True)
        if not cands:
            status(f"round={r} stalled: no usable samples "
                   f"(errs={st['err']}); waiting 120s")
            time.sleep(120)
            continue

        # score candidates against the cached teacher rollout
        flat, owners = [], []
        for tid, cs in cands.items():
            ttxt = teacher_text(tid, t_train)
            for c in cs:
                flat.append({"prefix_text": pfx[tid], "ref": ttxt,
                             "mine": both_text(c["z"], c["y"])})
                owners.append((tid, c))
        scores = disc.score_pairs(flat)
        rewards = [(1 - s["p_teacher"]) if s else 0.0 for s in scores]

        # winners -> G SFT subprocess (Track A method: attention-only LoRA)
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
        new_g_lora = os.path.join(args.work, "g_lora", f"r{r}")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.sft_gpus
        sft_log = open(os.path.join(args.work, f"g_sft_r{r}.log"), "w")
        here = os.path.dirname(os.path.abspath(__file__))
        sft = subprocess.Popen(
            [args.train_python, os.path.join(here, "g_sft.py"),
             "--base", args.g_base, "--winners", win_path,
             "--lora-in", state["g_lora"], "--lora-out", new_g_lora,
             "--max-steps", str(state["g_steps"])],
            env=env, stdout=sft_log, stderr=subprocess.STDOUT)

        # accumulate D pairs (most-fooling + one random candidate per turn)
        by_tid = {}
        for (tid, c), s in zip(owners, scores):
            if s:
                by_tid.setdefault(tid, []).append((c, s["p_teacher"]))
        with open(pending_path, "a") as fh:
            for tid, lst in by_tid.items():
                ttxt = teacher_text(tid, t_train)
                lst.sort(key=lambda x: x[1])
                take = [lst[0]]
                if len(lst) > 1:
                    take.append(random.Random(r * 7919 + hash(tid) % 1000)
                                .choice(lst[1:]))
                for c, _ in take:
                    fh.write(json.dumps({"turn_id": tid, "prefix_text": pfx[tid],
                                         "ref": ttxt,
                                         "mine": both_text(c["z"], c["y"])})
                             + "\n")

        # D-step: only every d_interval-th round, and not while frozen
        d_due = (r % args.d_interval == 0) and (r >= state["d_frozen_until"])
        d_steps_done, d_note = 0, ""
        dtrain = None
        if d_due and os.path.exists(pending_path):
            fresh = [json.loads(x) for x in open(pending_path)]
            replay = []
            if os.path.exists(replay_path):
                pool = [json.loads(x) for x in open(replay_path)]
                k = int(len(fresh) * args.replay_frac /
                        max(1 - args.replay_frac, 1e-9))
                if pool and k:
                    replay = random.Random(r).sample(pool, min(k, len(pool)))
            dp = os.path.join(args.work, f"d_pairs_r{r}.jsonl")
            with open(dp, "w") as fh:
                for p in fresh + replay:
                    fh.write(json.dumps(p) + "\n")
            new_d_lora = os.path.join(args.work, "d_lora", f"r{r}")
            steps = args.d_steps_first if state["d_trains"] == 0 else args.d_steps
            denv = dict(os.environ)
            denv["CUDA_VISIBLE_DEVICES"] = args.dtrain_gpus
            d_log = open(os.path.join(args.work, f"d_train_r{r}.log"), "w")
            dtrain = subprocess.Popen(
                [args.train_python, os.path.join(here, "d_train.py"),
                 "--base", args.d_base, "--pairs", dp,
                 "--lora-in", state["d_lora"], "--lora-out", new_d_lora,
                 "--max-steps", str(steps), "--lr", str(args.d_lr),
                 "--max-len", str(args.d_max_len), "--seed", str(r)],
                env=denv, stdout=d_log, stderr=subprocess.STDOUT)
            status(f"round={r} D-train launched: fresh={len(fresh)} "
                   f"replay={len(replay)} steps<={steps} "
                   f"(train #{state['d_trains'] + 1})")

        # wait G SFT; hot-swap G adapter
        sft_rc = sft.wait()
        sft_log.close()
        serve_name = state["g_serve"]
        if sft_rc != 0:
            note = "g_sft_failed"
            status(f"round={r} ERROR g_sft exit={sft_rc}; keeping previous G")
        else:
            serve_name = f"g_r{r}"
            try:
                prev = [state["g_serve"]] if state["g_serve"] != args.g_base else []
                load_g_adapter(args.g_url, serve_name, new_g_lora, drop=prev)
                state["g_lora"] = new_g_lora
                state["g_serve"] = serve_name
            except Exception as e:
                note = f"g_lora_load_failed:{type(e).__name__}"
                serve_name = state["g_serve"]

        # wait D train (if launched); hot-swap D adapter; clear pending
        if dtrain is not None:
            d_rc = dtrain.wait()
            d_log.close()
            meta = {}
            mp = os.path.join(args.work, "d_lora", f"r{r}", "train_meta.json")
            if os.path.exists(mp):
                meta = json.load(open(mp))
            d_steps_done = meta.get("steps", 0)
            if d_rc == 0 and d_steps_done:
                try:
                    disc.load_adapter(f"d_r{r}", os.path.join(args.work, "d_lora", f"r{r}"))
                    state["d_lora"] = os.path.join(args.work, "d_lora", f"r{r}")
                    state["d_serve"] = f"d_r{r}"
                    state["d_trains"] += 1
                    d_note = (f"d_trained steps={d_steps_done} "
                              f"examples={meta.get('examples_seen')} "
                              f"loss={round(meta.get('loss', 0), 4)}")
                    # fold pending into replay, reset pending
                    with open(replay_path, "a") as rf:
                        for line in open(pending_path):
                            rf.write(line)
                    os.remove(pending_path)
                except Exception as e:
                    d_note = f"d_lora_load_failed:{type(e).__name__}"
            else:
                d_note = f"d_train_failed rc={d_rc}"
            status(f"round={r} D-step: {d_note}")

        # held-out metrics with current G and current D
        m = heldout_eval(r, state["g_serve"])

        # freeze logic: 3 consecutive hot D-train evals -> longer freeze
        if dtrain is not None and d_steps_done:
            state["d_acc_history"].append(m["d_acc"])
            if m["d_acc"] > args.d_freeze_acc:
                state["d_hot_streak"] += 1
            else:
                state["d_hot_streak"] = 0
            if state["d_hot_streak"] >= 3:
                state["d_frozen_until"] = r + 2 * args.d_interval
                state["d_hot_streak"] = 0
                note = f"D_FROZEN_until_r{state['d_frozen_until']}"
                status(f"round={r} d_acc>{args.d_freeze_acc} 3x in a row -> "
                       f"freezing D until round {state['d_frozen_until']}")

        m.update({"round": r, "note": note, "d_note": d_note,
                  "train_valid_rate": round(vr_train, 4),
                  "mean_winner_reward": round(mean_win_rw, 4),
                  "d_train_steps": d_steps_done,
                  "d_serve": state["d_serve"], "g_serve": state["g_serve"],
                  "g_steps": state["g_steps"],
                  "wall_s": round(time.time() - t0, 1)})
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        status(f"round={r} fool={m['fool']} d_acc={m['d_acc']} "
               f"pos_bias={m['d_pos_bias']} matched={m['d_matched_acc']} "
               f"valid={m['valid_rate']} agree_exact={m['agree_exact']} "
               f"agree_tok={m['agree_tok']} think_len={m['think_len']} "
               f"self_jacc={m['self_jacc']} win_rw={m['mean_winner_reward']} "
               f"d_steps={d_steps_done} wall={m['wall_s']}s note={note}")

        # collapse guard (Track A style: rollback G, halve budget)
        if m["valid_rate"] < 0.5 or (m["self_jacc"] > 0.92 and m["n_pairs"] > 20):
            state["g_steps"] = max(8, state["g_steps"] // 2)
            status(f"round={r} COLLAPSE-GUARD valid={m['valid_rate']} "
                   f"self_jacc={m['self_jacc']} -> g_steps={state['g_steps']}, "
                   f"reverting to previous adapter")
            try:
                prev_lora = os.path.join(args.work, "g_lora", f"r{r - 1}")
                if os.path.isdir(prev_lora):
                    load_g_adapter(args.g_url, f"g_r{r}_reverted", prev_lora)
                    state["g_lora"] = prev_lora
                    state["g_serve"] = f"g_r{r}_reverted"
                else:
                    state["g_serve"] = args.g_base
                    state["g_lora"] = ""
            except Exception:
                state["g_serve"] = args.g_base
                state["g_lora"] = ""

        state["round"] = r + 1
        json.dump(state, open(state_path, "w"))

        # prune old G adapters (keep every 5th)
        old = os.path.join(args.work, "g_lora", f"r{r - 8}")
        if (r - 8) > 0 and (r - 8) % 5 and os.path.isdir(old) \
                and old != state["g_lora"]:
            import shutil
            shutil.rmtree(old, ignore_errors=True)

    status("driver finished all rounds")


if __name__ == "__main__":
    main()
