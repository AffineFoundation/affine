#!/usr/bin/env python3
"""Track S: self-play control -- is "perfect distillation" a stable
equilibrium of the GAN-style distillation game?

One box, one base model (Qwen/Qwen3.8-27B) in three roles:
  Teacher T  the base model, served plain.
  Generator  G = base + LoRA initialised EMPTY (lora_B = 0, so round 0 is an
             exact copy of the teacher). Trained per round with Track M's
             miner recipe: best-of-k rejection SFT on the candidates D rates
             most teacher-like (g_sft.py, attention-only LoRA, lr 1e-5).
  Discrim.   D = base + separate LoRA A/B judge (judge_common prompt, both
             slot orders averaged), ONLINE: warm-start update every
             --d-every rounds on recent (G, teacher) pairs, lr 1e-5.

FAIRNESS INVARIANT (the experiment is void without it): teacher references
are sampled FRESH every round through the SAME vLLM server, with the SAME
request body (temperature / max_tokens / chat template) as G's rollouts --
only the model name differs (base vs adapter). No cached teacher text is
ever scored. At round 0 every scored pair is therefore two draws from the
same distribution: any sustained held-out D accuracy away from 0.5 is a
pipeline artifact, and is treated as a first-class result (LEAK).

Leak instrumentation, evaluated at every D update on held-out turns that no
one ever trains on:
  held_live  D accuracy on (fresh teacher, fresh current-G) pairs
  held_ctrl  D accuracy on (fresh teacher, fresh G-round-0) pairs; G-r0 is
             the frozen zero adapter, so truth stays 0.5 FOREVER -- any
             sustained excess is a serving-path artifact (adapter route,
             cache, template), independent of G's drift
  held_tt    D accuracy on (fresh teacher, fresh teacher) pairs -- pure
             sampling/order artifact floor, truth 0.5 by construction

GPU map (8xB200):
  0,1  vLLM 27B TP2 :8000  sampler -- teacher (base) + G adapters (g_r*)
  2,3  vLLM 27B TP2 :8001  judge   -- base + D adapters (d_v*)
  4,5  trackS_d_train.py (DDP, one replica per GPU), every --d-every rounds
  6,7  g_sft.py per round; SWE proxy panel hits :8000 (loop pauses)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

# The HF cache MUST be the /dshare copy: /root is a slow encrypted FUSE
# mount, and a watchdog relaunch without this env once produced a second,
# CORRUPT 53GB cache there (missing shard -> D update failures, 2026-08-22).
os.environ["HF_HOME"] = "/dshare/hf"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gad_common import (bash_body, both_text, build_prompt, load_turns,  # noqa: E402
                        token_jaccard)
from trackM_common import (Judge, gen_effect, judge_effect, load_adapter,  # noqa: E402
                           now, render_prefix, sample_model)

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--train-tids", default="/root/work/train_tids.txt")
    ap.add_argument("--held-tids", default="/root/work/held_tids.txt")
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--sample-url", default="http://127.0.0.1:8000")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8001")
    ap.add_argument("--work", default="/dshare/trackS")
    ap.add_argument("--batch-turns", type=int, default=28)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=3584)
    ap.add_argument("--g-steps", type=int, default=30)
    ap.add_argument("--d-every", type=int, default=3)
    ap.add_argument("--d-steps", type=int, default=200)
    ap.add_argument("--d-v0-steps", type=int, default=100)
    ap.add_argument("--d-window", type=int, default=300)
    ap.add_argument("--d-epochs", type=float, default=4.0)
    ap.add_argument("--r0-turns", type=int, default=96)
    ap.add_argument("--held-n", type=int, default=64)
    ap.add_argument("--typicality-n", type=int, default=12)
    ap.add_argument("--bench-every", type=int, default=15)
    ap.add_argument("--sft-gpus", default="6,7")
    ap.add_argument("--dtrain-gpus", default="4,5")
    ap.add_argument("--train-python", default="/root/trainenv/bin/python")
    ap.add_argument("--bench-python", default="/root/benchenv/bin/python")
    ap.add_argument("--rounds", type=int, default=10000)
    args = ap.parse_args()

    W = args.work
    os.makedirs(f"{W}/g_lora", exist_ok=True)
    os.makedirs(f"{W}/d_versions", exist_ok=True)
    state_path = f"{W}/state.json"

    def status(line):
        with open(f"{W}/status.log", "a") as fh:
            fh.write(f"{now()} [loop] {line}\n")
        print(f"STATUS {line}", flush=True)

    import requests
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    turns = load_turns(args.turns)
    train_tids = [t for t in open(args.train_tids).read().split()
                  if t in turns]
    held_tids = [t for t in open(args.held_tids).read().split()
                 if t in turns]
    held_fixed = sorted(held_tids)[: args.held_n]

    pfx_cache = {}

    def prefix(tid):
        if tid not in pfx_cache:
            pfx_cache[tid] = render_prefix(turns[tid], args.d_max_len * 6)
        return pfx_cache[tid]

    judge = Judge(args.judge_url, args.base, tok, args.d_max_len)
    state = {"round": 0, "g_lora": "", "g_serve": "", "dver": "",
             "d_lora": "", "r0_done": False}
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))

    def save():
        json.dump(state, open(state_path, "w"))

    # ---------------------------------------------------------------- helpers
    def g_sft_subprocess(winners_path, lora_in, lora_out, max_steps,
                         epochs=None, log_path=None):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.sft_gpus
        cmd = [args.train_python, f"{HERE}/g_sft.py", "--base", args.base,
               "--winners", winners_path, "--lora-in", lora_in,
               "--lora-out", lora_out, "--max-steps", str(max_steps)]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        out = open(log_path or f"{W}/g_sft_last.log", "w")
        return subprocess.run(cmd, env=env, stdout=out,
                              stderr=subprocess.STDOUT).returncode

    def d_train_subprocess(pairs_path, lora_in, lora_out, max_steps,
                           log_path):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.dtrain_gpus
        ngpu = len(args.dtrain_gpus.split(","))
        cmd = [args.train_python, "-m", "torch.distributed.run",
               f"--nproc-per-node={ngpu}", f"{HERE}/trackS_d_train.py",
               "--base", args.base, "--pairs", pairs_path,
               "--lora-in", lora_in, "--lora-out", lora_out,
               "--max-steps", str(max_steps), "--lr", "1e-5",
               "--batch", "2", "--accum", "2",
               "--epochs", str(args.d_epochs),
               "--max-len", str(args.d_max_len)]
        out = open(log_path, "w")
        return subprocess.run(cmd, env=env, stdout=out,
                              stderr=subprocess.STDOUT).returncode

    def mean_logprob_under_base(prompt, completion):
        """Typicality: mean per-token logprob of a G completion under the
        plain teacher. Token-id prompt so the completion region is exact."""
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if not c_ids:
            return None
        # prompt logprobs materialise a full-vocab fp32 log_softmax over the
        # WHOLE prompt (~600KB/token): a 12k-token request = ~7GB and killed
        # the engine once (2026-08-22). Cap the window hard.
        keep_p = max(256, 4096 - len(c_ids))
        ids = p_ids[-keep_p:] + c_ids
        n_c = min(len(c_ids), len(ids) - 1)
        body = {"model": args.base, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0, "echo": True, "logprobs": 0}
        try:
            r = requests.post(f"{args.sample_url}/v1/completions", json=body,
                              timeout=300)
            r.raise_for_status()
            lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
            comp = [v for v in lps[len(ids) - n_c: len(ids)]
                    if v is not None]
            return sum(comp) / len(comp) if comp else None
        except Exception:
            return None

    def score_ref_vs(cands_by_tid, refs_by_tid, use_idx=0):
        """Build judge pairs (ref = teacher rollout 0, mine = candidate)."""
        pairs, owners = [], []
        for tid, cs in cands_by_tid.items():
            refs = refs_by_tid.get(tid) or []
            if not refs or not refs[0]["valid"]:
                continue
            ref = both_text(refs[0]["z"], refs[0]["y"])
            c = cs[use_idx] if len(cs) > use_idx else None
            if c is None or not c["valid"]:
                continue
            pairs.append({"prefix_text": prefix(tid), "ref": ref,
                          "mine": both_text(c["z"], c["y"])})
            owners.append(tid)
        return pairs, owners

    def held_eval():
        """Fresh rollouts on the fixed held set; D acc for live/ctrl/tt."""
        t_ref, _, _ = sample_model(args.sample_url, args.base, tok, turns,
                                   held_fixed, 2, args.temp, args.max_tokens)
        g_cur, _, _ = sample_model(args.sample_url, state["g_serve"], tok,
                                   turns, held_fixed, 1, args.temp,
                                   args.max_tokens)
        g_zero, _, _ = sample_model(args.sample_url, "g_r0", tok, turns,
                                    held_fixed, 1, args.temp,
                                    args.max_tokens)
        out = {}
        for name, mine_by_tid, idx in (("live", g_cur, 0),
                                       ("ctrl", g_zero, 0),
                                       ("tt", t_ref, 1)):
            pairs, _ = score_ref_vs(mine_by_tid, t_ref, use_idx=idx)
            scores = judge.score_pairs(pairs)
            ok = [s for s in scores if s]
            if ok:
                out[name] = round(
                    sum(1 for s in ok if s["p_teacher"] > 0.5) / len(ok), 4)
                out[f"{name}_p"] = round(mean([s["p_teacher"] for s in ok]), 4)
                out[f"{name}_n"] = len(ok)
            else:
                out[name] = float("nan")
                out[f"{name}_n"] = 0
        return out

    def load_g(name, path, drop_prev=True):
        drop = []
        if drop_prev and state["g_serve"] not in ("", "g_r0", name):
            drop = [state["g_serve"]]
        load_adapter(args.sample_url, name, path, drop=drop)

    def probe_pairs():
        p = f"{W}/probe_pairs.jsonl"
        return [json.loads(x) for x in open(p)] if os.path.exists(p) else []

    # ------------------------------------------------------------- round 0
    rng = random.Random(120)
    shuffled = train_tids[:]
    rng.shuffle(shuffled)

    if not state["r0_done"]:
        status(f"driver up: base={args.base} G=base+zero-LoRA (exact copy) "
               f"D=base+LoRA online(every {args.d_every}r, {args.d_steps} "
               f"steps, lr 1e-5) k={args.k} turns/round={args.batch_turns} "
               f"temp={args.temp} max_tok={args.max_tokens} train_turns="
               f"{len(train_tids)} held={len(held_tids)} (fixed "
               f"{len(held_fixed)}) | gpu_map[0,1=sampler :8000 (T=base + "
               f"g_r*) 2,3=judge :8001 (base + d_v*) 4,5=d_train 6,7=g_sft]")

        # G round-0 adapter: fresh attention LoRA (lora_B=0 -> exact teacher),
        # built by the production g_sft code path with zero optimiser steps so
        # keys/remap are byte-identical to every later round.
        g0 = f"{W}/g_lora/r0"
        if not os.path.exists(f"{g0}/adapter_model.safetensors"):
            status("building zero-init G adapter (g_sft --epochs 0)")
            dummy = f"{W}/g0_dummy_winners.jsonl"
            tid0 = shuffled[0]
            with open(dummy, "w") as fh:
                fh.write(json.dumps({"turn_id": tid0,
                                     "prompt": build_prompt(tok, turns[tid0]),
                                     "completion": "placeholder"}) + "\n")
            rc = g_sft_subprocess(dummy, "", g0, 0, epochs=0,
                                  log_path=f"{W}/g0_init.log")
            if rc != 0:
                status(f"FATAL g0 init failed rc={rc}")
                return
        load_g("g_r0", g0, drop_prev=False)
        state["g_lora"] = g0
        state["g_serve"] = "g_r0"
        save()

        # identity check: the zero adapter must NOT change the model.
        probes = [build_prompt(tok, turns[t]) for t in shuffled[:3]]
        noise = max(gen_effect(args.sample_url, args.base, args.base, p)
                    for p in probes)
        eff0 = max(gen_effect(args.sample_url, args.base, "g_r0", p)
                   for p in probes)
        verdict = "PASS" if eff0 <= max(1e-3, 10 * noise) else "FAIL"
        status(f"IDENTITY CHECK g_r0 vs base: eff={eff0:.6f} "
               f"noise_floor={noise:.6f} -> {verdict}"
               + ("" if verdict == "PASS" else
                  " -- LoRA serving path is NOT identity; LEAK-ARTIFACT, "
                  "investigate before trusting any D readout"))

        # round-0 pairs: two fresh draws from the same distribution.
        r0_tids = shuffled[: args.r0_turns]
        t0 = time.time()
        t_ref, t_vr, _ = sample_model(args.sample_url, args.base, tok, turns,
                                      r0_tids, 2, args.temp, args.max_tokens)
        g_ref, g_vr, _ = sample_model(args.sample_url, "g_r0", tok, turns,
                                      r0_tids, 2, args.temp, args.max_tokens)
        v0_path = f"{W}/d_v0_pairs.jsonl"
        n_pairs = 0
        with open(v0_path, "w") as fh:
            for tid in r0_tids:
                ts, gs = t_ref.get(tid) or [], g_ref.get(tid) or []
                for i in range(2):
                    if (len(ts) > i and ts[i]["valid"]
                            and len(gs) > i and gs[i]["valid"]):
                        fh.write(json.dumps({
                            "turn_id": tid, "prefix_text": prefix(tid),
                            "ref": both_text(ts[i]["z"], ts[i]["y"]),
                            "mine": both_text(gs[i]["z"], gs[i]["y"]),
                        }) + "\n")
                        n_pairs += 1
        status(f"round-0 pairs sampled: n={n_pairs} teacher_valid={t_vr:.3f} "
               f"g0_valid={g_vr:.3f} wall={time.time()-t0:.0f}s "
               f"(both distributions IDENTICAL by construction)")

        # fixed probe pairs for stale-cache / adapter-effect checks
        with open(f"{W}/probe_pairs.jsonl", "w") as fh:
            for row in [json.loads(x) for x in open(v0_path)][:6]:
                fh.write(json.dumps(row) + "\n")
        json.dump(probes, open(f"{W}/probe_prompts.json", "w"))

        # D v0: brief, fresh round-0 pairs ONLY (no cached anything)
        status(f"training D v0 (fresh LoRA, {args.d_v0_steps} steps, "
               f"round-0 pairs only)")
        t0 = time.time()
        rc = d_train_subprocess(v0_path, "", f"{W}/d_versions/v0",
                                args.d_v0_steps, f"{W}/d_train_v0.log")
        if rc != 0:
            status(f"FATAL D v0 training failed rc={rc}")
            return
        judge.load_adapter("d_v0", f"{W}/d_versions/v0")
        eff = judge_effect(args.judge_url, args.base, "d_v0", tok,
                           probe_pairs()[:4])
        status(f"D v0 loaded; adapter-effect vs base = {eff:.5f} "
               f"({'OK' if eff >= 0.005 else 'NO-OP -- FATAL'}) "
               f"wall={time.time()-t0:.0f}s")
        if eff < 0.005:
            return
        state["dver"] = "v0"
        state["d_lora"] = f"{W}/d_versions/v0"
        save()

        held = held_eval()
        leak = ("LEAK-FREE so far" if held["ctrl"] <= 0.58 and
                held["tt"] <= 0.58 else "LEAK SUSPECTED -- investigate")
        status(f"D UPDATE dver=v0 held_live={held['live']} "
               f"held_ctrl={held['ctrl']} held_tt={held['tt']} "
               f"p(live/ctrl/tt)={held.get('live_p')}/{held.get('ctrl_p')}/"
               f"{held.get('tt_p')} n={held.get('live_n')} | truth=0.5 for "
               f"all three at round 0 -> {leak}")
        state["r0_done"] = True
        save()
    else:
        # resume: make servers consistent with state
        if state["g_lora"]:
            try:
                load_adapter(args.sample_url, "g_r0", f"{W}/g_lora/r0")
                if state["g_serve"] not in ("", "g_r0"):
                    load_adapter(args.sample_url, state["g_serve"],
                                 state["g_lora"])
            except Exception as e:
                status(f"resume G adapter reload failed {type(e).__name__}")
        if state["dver"]:
            try:
                judge.load_adapter(f"d_{state['dver']}", state["d_lora"])
            except Exception as e:
                status(f"resume D adapter reload failed {type(e).__name__}")
        judge.serve = f"d_{state['dver']}"
        status(f"resumed at round={state['round']} g={state['g_serve']} "
               f"d={state['dver']}")

    # ------------------------------------------------------------ main loop
    cursor = state["round"] * args.batch_turns % len(shuffled)
    while state["round"] < args.rounds:
        r = state["round"] + 1
        t0 = time.time()
        bt = []
        while len(bt) < args.batch_turns:
            if cursor >= len(shuffled):
                cursor = 0
                rng.shuffle(shuffled)
            bt.append(shuffled[cursor])
            cursor += 1

        # ---- fresh samples: G (k) and teacher (2), same server/params ----
        g_cands, g_vr, g_st = sample_model(args.sample_url, state["g_serve"],
                                           tok, turns, bt, args.k, args.temp,
                                           args.max_tokens)
        t_refs, t_vr, t_st = sample_model(args.sample_url, args.base, tok,
                                          turns, bt, 2, args.temp,
                                          args.max_tokens)
        if not g_cands or not t_refs:
            status(f"round={r} stalled (g={len(g_cands)} t={len(t_refs)} "
                   f"errs={g_st['err']}/{t_st['err']}); waiting 90s "
                   f"+ best-effort adapter reload (server may have restarted)")
            time.sleep(90)
            try:
                load_adapter(args.sample_url, "g_r0", f"{W}/g_lora/r0")
                if state["g_serve"] not in ("", "g_r0"):
                    load_adapter(args.sample_url, state["g_serve"],
                                 state["g_lora"])
            except Exception:
                pass
            try:
                if state["dver"]:
                    judge.load_adapter(f"d_{state['dver']}", state["d_lora"])
            except Exception:
                pass
            continue

        # ---- score all valid G candidates against fresh ref1 -------------
        pairs, owners = [], []
        for tid, cs in g_cands.items():
            refs = t_refs.get(tid) or []
            if not refs or not refs[0]["valid"]:
                continue
            ref = both_text(refs[0]["z"], refs[0]["y"])
            for c in cs:
                if c["valid"]:
                    pairs.append({"prefix_text": prefix(tid), "ref": ref,
                                  "mine": both_text(c["z"], c["y"])})
                    owners.append((tid, c))
        scores = judge.score_pairs(pairs)
        rewards = [(1 - s["p_teacher"]) if s else 0.0 for s in scores]

        best = {}
        for (tid, c), rw in zip(owners, rewards):
            c["reward"] = rw
            if tid not in best or rw > best[tid]["reward"]:
                best[tid] = c
        fool = mean(rewards)
        fool_best = mean([c["reward"] for c in best.values()])
        fool_win = (sum(1 for x in rewards if x > 0.5) / len(rewards)
                    if rewards else float("nan"))

        # ---- agreement / drift metrics -----------------------------------
        ag_ex, ag_tok, tt_ex, tt_tok, gg_tok, zlens = [], [], [], [], [], []
        for tid, c in best.items():
            refs = t_refs[tid]
            rb = bash_body(refs[0]["y"])
            cb = bash_body(c["y"])
            ag_ex.append(1.0 if cb == rb else 0.0)
            ag_tok.append(token_jaccard(cb, rb))
            zlens.append(len(c["z"]))
            if len(refs) > 1 and refs[1]["valid"]:
                r2 = bash_body(refs[1]["y"])
                tt_ex.append(1.0 if r2 == rb else 0.0)
                tt_tok.append(token_jaccard(r2, rb))
            valid_cs = [x for x in g_cands[tid] if x["valid"]]
            if len(valid_cs) > 1:
                o = sorted(valid_cs, key=lambda x: -x.get("reward", 0.0))
                gg_tok.append(token_jaccard(bash_body(o[0]["y"]),
                                            bash_body(o[1]["y"])))

        typ_items = list(best.items())[: args.typicality_n]
        typs = [mean_logprob_under_base(build_prompt(tok, turns[tid]),
                                        c["raw"]) for tid, c in typ_items]
        typ = mean(typs)

        # ---- G update: best-of-k rejection SFT ---------------------------
        win_path = f"{W}/winners_r{r}.jsonl"
        with open(win_path, "w") as fh:
            for tid, c in best.items():
                fh.write(json.dumps({"turn_id": tid, "reward": c["reward"],
                                     "prompt": build_prompt(tok, turns[tid]),
                                     "completion": c["raw"]}) + "\n")
        lora_in = f"{state['g_lora']}/hf" if state["g_lora"] else ""
        new_lora = f"{W}/g_lora/r{r}"
        rc = g_sft_subprocess(win_path, lora_in, new_lora, args.g_steps,
                              log_path=f"{W}/g_sft_r{r}.log")
        g_loaded = "kept"
        if rc == 0:
            try:
                load_g(f"g_r{r}", new_lora)
                eff = gen_effect(args.sample_url, args.base, f"g_r{r}",
                                 build_prompt(tok, turns[bt[0]]))
                if eff < 1e-4:
                    raise RuntimeError(f"no-op adapter (eff={eff})")
                state["g_lora"] = new_lora
                state["g_serve"] = f"g_r{r}"
                g_loaded = f"g_r{r}(eff={eff:.4f})"
            except Exception as e:
                status(f"round={r} G adapter hot-load REJECTED "
                       f"{type(e).__name__} {str(e)[:100]}; keeping "
                       f"{state['g_serve']}")
        else:
            status(f"round={r} g_sft FAILED rc={rc}; keeping "
                   f"{state['g_serve']}")

        # ---- accumulate D pairs (fresh ref vs this round's best G) -------
        with open(f"{W}/d_pending.jsonl", "a") as fh:
            for tid, c in best.items():
                fh.write(json.dumps({
                    "turn_id": tid, "round": r, "prefix_text": prefix(tid),
                    "ref": both_text(t_refs[tid][0]["z"],
                                     t_refs[tid][0]["y"]),
                    "mine": both_text(c["z"], c["y"])}) + "\n")

        status(f"round={r} fool={fool:.4f} fool_best={fool_best:.4f} "
               f"fool_win={fool_win:.3f} valid={g_vr:.3f} t_valid={t_vr:.3f} "
               f"agree_exact={mean(ag_ex):.3f} agree_tok={mean(ag_tok):.3f} "
               f"tt_exact={mean(tt_ex):.3f} tt_tok={mean(tt_tok):.3f} "
               f"g_self_tok={mean(gg_tok):.3f} typ_lp={typ:.4f} "
               f"think_len={mean(zlens):.0f} winners={len(best)} "
               f"g={g_loaded} dver={state['dver']} "
               f"errs={g_st['err']}+{t_st['err']} wall={time.time()-t0:.0f}s")
        state["round"] = r
        save()

        # ---- D online update every d-every rounds -------------------------
        if r % args.d_every == 0:
            t1 = time.time()
            rows = [json.loads(x) for x in open(f"{W}/d_pending.jsonl")]
            recent = rows[-args.d_window:]
            dp = f"{W}/d_train_r{r}.jsonl"
            with open(dp, "w") as fh:
                for row in recent:
                    fh.write(json.dumps(row) + "\n")
            new_ver = f"v{r}"
            out_dir = f"{W}/d_versions/{new_ver}"
            # [fix] FROM-SCRATCH per update (Track M's crown recipe): the
            # warm-start + 400-pair window at lr 1e-5 was too weak -- v12
            # missed a 3x think-length gap a trivial length probe classifies
            # at 0.69 (audited 2026-08-22, see remeasure/diag logs).
            rc = d_train_subprocess(dp, "", out_dir, args.d_steps,
                                    f"{W}/d_train_{new_ver}.log")
            if rc != 0:
                status(f"round={r} D train FAILED rc={rc}; keeping "
                       f"{state['dver']}")
            else:
                pr = probe_pairs()[:6]
                before = judge.score_pairs(pr)
                old_ver, old_lora = state["dver"], state["d_lora"]
                swapped = False
                delta = eff = float("nan")
                try:
                    judge.load_adapter(f"d_{new_ver}", out_dir,
                                       drop=[f"d_{old_ver}"])
                    after = judge.score_pairs(pr)
                    delta = mean([abs(a["p_teacher"] - b["p_teacher"])
                                  for a, b in zip(after, before) if a and b])
                    eff = judge_effect(args.judge_url, args.base,
                                       f"d_{new_ver}", tok, pr[:4])
                    if eff < 0.005:
                        raise RuntimeError(f"no-op vs base (eff={eff:.5f})")
                    swapped = True
                except Exception as e:
                    status(f"round={r} D adapter swap REJECTED "
                           f"{type(e).__name__} {str(e)[:120]}; reverting "
                           f"to {old_ver}")
                    try:
                        judge.load_adapter(f"d_{old_ver}", old_lora)
                    except Exception:
                        pass
                if swapped:
                    if delta < 1e-4:
                        status(f"round={r} D swap WARNING probe_delta="
                               f"{delta:.6f} -- stale cache or null update")
                    state["dver"] = new_ver
                    state["d_lora"] = out_dir
                    save()
                    meta = {}
                    try:
                        meta = json.load(open(f"{out_dir}/train_meta.json"))
                    except Exception:
                        pass
                    held = held_eval()
                    status(f"D UPDATE dver={new_ver} held_live={held['live']} "
                           f"held_ctrl={held['ctrl']} held_tt={held['tt']} "
                           f"p(l/c/t)={held.get('live_p')}/"
                           f"{held.get('ctrl_p')}/{held.get('tt_p')} "
                           f"n={held.get('live_n')} probe_delta={delta:.4f} "
                           f"eff_vs_base={eff:.4f} train_acc="
                           f"{meta.get('train_acc_tail')} pairs={len(recent)} "
                           f"wall={time.time()-t1:.0f}s")

        # ---- SWE proxy panel every bench-every rounds ---------------------
        if r % args.bench_every == 0:
            status(f"BENCH launching SWE proxy panel on {state['g_serve']} "
                   f"(loop paused)")
            try:
                subprocess.run(
                    [args.bench_python, f"{HERE}/run_panel.py",
                     "--pin", f"{HERE}/panels/panel_proxy.json",
                     "--model", state["g_serve"], "--port", "8000",
                     "--tag", f"proxy_r{r}", "--workers", "8",
                     "--bench-dir", "/dshare/bench", "--work", W],
                    stdout=open(f"{W}/bench_proxy_r{r}.log", "w"),
                    stderr=subprocess.STDOUT, timeout=7200)
            except Exception as e:
                status(f"BENCH proxy_r{r} crashed {type(e).__name__}")

        # ---- prune old G adapters (keep r0, last 3, every 10th) -----------
        old = f"{W}/g_lora/r{r - 4}"
        if (r - 4) > 0 and (r - 4) % 10 and os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
        # prune old winners/d_train logs of similar age
        for f in (f"{W}/winners_r{r - 8}.jsonl", f"{W}/g_sft_r{r - 8}.log"):
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
