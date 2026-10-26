#!/usr/bin/env python3
"""Track T: pure-GAN training of a RAW student -- no SFT warm start anywhere.

Question: can a raw student (strongest Affine 35B-A3B checkpoint, served
as-is) gain real SWE ability from the adversarial signal ALONE, i.e. from
being rewarded for fooling an online judge into thinking its rollouts are
the 27B teacher's?

Hard constraints (the experiment is void without them):
  * NO imitation SFT. The student is never trained on teacher text. The
    only student training data is its OWN rollouts, weighted by the judge
    (best-of-k rejection SFT, Track M's miner g-step).
  * The cached teacher rollout corpus is used ONLY as (a) judge-training
    references and (b) scoring references. It never enters winners.jsonl.
  * Loop rewards are scored against FRESH teacher rollouts sampled per
    round from the served 27B (temperature/template identical to cache
    conventions).

Roles:
  Teacher  Qwen/Qwen3.8-27B, served plain (black box).
  Student  dendriteholdings/albedo-qwen3.6-35b-king-genesis (35B-A3B MoE
           miner arch) + LoRA trained by the loop. Round 0 = raw checkpoint.
  Judge    the 27B base + LoRA A/B chooser (judge_common prompt, both slot
           orders averaged, disc_text normalisation via gad_common). ONLINE:
           pretrained to the 0.6-0.85 held-out band BEFORE the loop, then
           warm-start retrained every --d-every rounds; retrain SKIPPED
           while held-out acc > 0.85 (balanced-regime throttle).

GPU map (8xB300, pins are CUDA_VISIBLE_DEVICES, invisible in ps):
  0,1  vLLM 27B TP2 :8000  teacher (base) + judge (d_v* adapter/request)
  2    vLLM student :8001  student (base + m_r* adapters)
  3,4  g_sft.py per round (student LoRA; device_map=auto shards the 35B)
  5    trackT_d_train.py online judge retrains (DDPx1, grad-ckpt);
       judge PRETRAIN borrows 3,4 too (DDPx3) since g_sft is idle then
  6,7  vLLM student TP2 :8006  bench server (panels run here, loop unpaused)
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

# /root is a slow encrypted FUSE mount; a stray HF_HOME there once produced
# a second, corrupt 53GB cache on Track S. Pin the real one unconditionally.
os.environ["HF_HOME"] = "/dshare/hf"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gad_common import (bash_body, both_text, build_prompt,  # noqa: E402
                        load_rollout_cache, load_turns, token_jaccard)
from trackM_common import (Judge, eval_held, gen_effect, judge_effect,  # noqa: E402
                           load_adapter, now, render_prefix, sample_model)

HERE = os.path.dirname(os.path.abspath(__file__))
STUDENT = "dendriteholdings/albedo-qwen3.6-35b-king-genesis"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--teacher-train", default="/root/work/teacher_train.jsonl")
    ap.add_argument("--teacher-heldout",
                    default="/root/work/teacher_heldout.jsonl")
    ap.add_argument("--teacher-base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--student-base", default=STUDENT)
    ap.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    ap.add_argument("--student-url", default="http://127.0.0.1:8001")
    ap.add_argument("--bench-url", default="http://127.0.0.1:8006")
    ap.add_argument("--work", default="/dshare/trackT")
    ap.add_argument("--batch-turns", type=int, default=28)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=3584)
    ap.add_argument("--g-steps", type=int, default=30)
    ap.add_argument("--d-every", type=int, default=3)
    ap.add_argument("--d-steps", type=int, default=150)
    ap.add_argument("--d-window", type=int, default=400)
    ap.add_argument("--d-replay", type=int, default=100)
    ap.add_argument("--d-throttle", type=float, default=0.85)
    ap.add_argument("--pretrain-turns", type=int, default=800)
    ap.add_argument("--pretrain-k", type=int, default=2)
    ap.add_argument("--pretrain-steps", type=int, default=120)
    ap.add_argument("--held-n", type=int, default=64)
    ap.add_argument("--typicality-n", type=int, default=12)
    ap.add_argument("--bench-every", type=int, default=15)
    ap.add_argument("--full-every-s", type=int, default=12600,
                    help="full-150 panel cadence, seconds of loop time (~3.5h)")
    ap.add_argument("--sft-gpus", default="3,4")
    ap.add_argument("--dtrain-gpus", default="5")
    ap.add_argument("--pretrain-gpus", default="3,4,5")
    ap.add_argument("--train-python", default="/root/trainenv/bin/python")
    ap.add_argument("--bench-python", default="/root/benchenv/bin/python")
    ap.add_argument("--rounds", type=int, default=10000)
    args = ap.parse_args()

    W = args.work
    os.makedirs(f"{W}/m_lora", exist_ok=True)
    os.makedirs(f"{W}/d_versions", exist_ok=True)
    state_path = f"{W}/state.json"

    def status(line, tag="loop"):
        with open(f"{W}/status.log", "a") as fh:
            fh.write(f"{now()} [{tag}] {line}\n")
        print(f"STATUS [{tag}] {line}", flush=True)

    def gpu_hours():
        try:
            t0 = int(open(f"{W}/pod_t0").read().strip())
        except Exception:
            t0 = int(time.time())
        return round((time.time() - t0) / 3600 * 8, 1)

    import requests
    from transformers import AutoTokenizer
    m_tok = AutoTokenizer.from_pretrained(args.student_base)
    d_tok = AutoTokenizer.from_pretrained(args.teacher_base)

    turns = load_turns(args.turns)
    t_train = load_rollout_cache(args.teacher_train)
    t_held = load_rollout_cache(args.teacher_heldout)
    train_tids = sorted(t for t in t_train if t in turns)
    held_tids = sorted(t for t in t_held if t in turns)
    held_fixed = held_tids[: args.held_n]
    pfx_cache = {}

    def prefix(tid):
        if tid not in pfx_cache:
            pfx_cache[tid] = render_prefix(turns[tid], args.d_max_len * 6)
        return pfx_cache[tid]

    judge = Judge(args.teacher_url, args.teacher_base, d_tok, args.d_max_len)
    state = {"round": 0, "m_lora": "", "m_serve": args.student_base,
             "dver": "", "d_lora": "", "pretrain_done": False,
             "held_acc": None, "last_full_t": 0.0, "bench_serve": ""}
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))

    def save():
        json.dump(state, open(state_path, "w"))

    # ---------------------------------------------------------------- helpers
    def g_sft_subprocess(winners_path, lora_in, lora_out, max_steps, log_path):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.sft_gpus
        cmd = [args.train_python, f"{HERE}/g_sft.py",
               "--base", args.student_base,
               "--winners", winners_path, "--lora-in", lora_in,
               "--lora-out", lora_out, "--max-steps", str(max_steps)]
        return subprocess.run(cmd, env=env, stdout=open(log_path, "w"),
                              stderr=subprocess.STDOUT).returncode

    def d_train_subprocess(pairs_path, lora_in, lora_out, max_steps,
                           log_path, gpus, seed=0):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        ngpu = len(gpus.split(","))
        cmd = [args.train_python, "-m", "torch.distributed.run",
               f"--nproc-per-node={ngpu}", f"{HERE}/trackT_d_train.py",
               "--base", args.teacher_base, "--pairs", pairs_path,
               "--lora-in", lora_in, "--lora-out", lora_out,
               "--max-steps", str(max_steps), "--lr", "1e-5",
               "--batch", "2", "--accum", "2",
               "--max-len", str(args.d_max_len), "--seed", str(seed)]
        return subprocess.run(cmd, env=env, stdout=open(log_path, "w"),
                              stderr=subprocess.STDOUT).returncode

    def mean_logprob_under_teacher(prompt, completion):
        """Typicality drift metric: mean per-token logprob of a student
        completion under the plain 27B teacher (27B tokenizer)."""
        p_ids = d_tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = d_tok(completion, add_special_tokens=False)["input_ids"]
        if not c_ids:
            return None
        # prompt logprobs materialise full-vocab fp32 over the whole prompt;
        # cap the window hard (a 12k-token request once killed the engine).
        keep_p = max(256, 4096 - len(c_ids))
        ids = p_ids[-keep_p:] + c_ids
        n_c = min(len(c_ids), len(ids) - 1)
        body = {"model": args.teacher_base, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0, "echo": True, "logprobs": 0}
        try:
            r = requests.post(f"{args.teacher_url}/v1/completions", json=body,
                              timeout=300)
            r.raise_for_status()
            lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
            comp = [v for v in lps[len(ids) - n_c: len(ids)] if v is not None]
            return sum(comp) / len(comp) if comp else None
        except Exception:
            return None

    def probe_pairs():
        p = f"{W}/probe_pairs.jsonl"
        return [json.loads(x) for x in open(p)] if os.path.exists(p) else []

    def gen_held_pairs():
        """Fresh student rollouts on the fixed held turns vs cached teacher
        held refs (cache use = scoring reference, allowed)."""
        cands, _, _ = sample_model(args.student_url, state["m_serve"], m_tok,
                                   turns, held_fixed, 1, args.temp,
                                   args.max_tokens)
        out = []
        for tid in held_fixed:
            cs = cands.get(tid)
            if not cs or not cs[0]["valid"]:
                continue
            ref = both_text(t_held[tid][0]["z"], t_held[tid][0]["y"])
            out.append({"turn_id": tid, "prefix_text": prefix(tid),
                        "ref": ref, "mine": both_text(cs[0]["z"],
                                                      cs[0]["y"])})
        return out

    def load_student(name, path):
        drop = ([state["m_serve"]]
                if state["m_serve"] not in ("", args.student_base, name)
                else [])
        load_adapter(args.student_url, name, path, drop=drop)

    def launch_bench(tag, model_name, adapter_path, round_no):
        """Panels run on the DEDICATED bench server (:8006, GPUs 6,7); the
        loop keeps going. bench.lock serialises proxy vs full runs."""
        lock = f"{W}/bench.lock"
        if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 4 * 3600:
            status(f"SKIP tag={tag} (bench busy: {open(lock).read().strip()})",
                   tag="bench")
            return
        with open(lock, "w") as fh:
            fh.write(tag)
        cmd = [args.bench_python, f"{HERE}/trackT_bench.py",
               "--tag", tag, "--model", model_name,
               "--adapter", adapter_path or "", "--round", str(round_no),
               "--work", W]
        subprocess.Popen(cmd, stdout=open(f"{W}/bench_{tag}.driver.log", "w"),
                         stderr=subprocess.STDOUT)
        status(f"LAUNCH tag={tag} ckpt={model_name} (GPUs 6,7, background)",
               tag="bench")

    # ----------------------------------------------------- phase 0: pretrain
    rng = random.Random(120)
    shuffled = train_tids[:]
    rng.shuffle(shuffled)

    if not state["pretrain_done"]:
        status(
            f"driver up: student={args.student_base} (RAW, no SFT warm "
            f"start) teacher={args.teacher_base} judge=27B+LoRA online "
            f"(every {args.d_every}r warm-start {args.d_steps} steps lr "
            f"1e-5, throttle held_acc>{args.d_throttle}) k={args.k} "
            f"turns/round={args.batch_turns} temp={args.temp} max_tok="
            f"{args.max_tokens} train_turns={len(train_tids)} held="
            f"{len(held_tids)} (fixed {len(held_fixed)}) | gpu_map[0,1="
            f"teacher+judge :8000 2=student :8001 3,4=g_sft 5=d_train "
            f"6,7=bench :8006]", tag="infra")

        # 1) student rollouts on train turns -> judge pretrain pairs.
        # Teacher side = CACHED rollouts (judge-training use, allowed).
        t0 = time.time()
        pre_tids = shuffled[: args.pretrain_turns]
        cands, vr, st = sample_model(args.student_url, args.student_base,
                                     m_tok, turns, pre_tids, args.pretrain_k,
                                     args.temp, args.max_tokens, workers=48)
        pre_path = f"{W}/d_pretrain_pairs.jsonl"
        n_pairs = 0
        with open(pre_path, "w") as fh:
            for tid, cs in cands.items():
                ref = both_text(t_train[tid][0]["z"], t_train[tid][0]["y"])
                for c in cs:
                    if c["valid"]:
                        fh.write(json.dumps({
                            "turn_id": tid, "prefix_text": prefix(tid),
                            "ref": ref,
                            "mine": both_text(c["z"], c["y"])}) + "\n")
                        n_pairs += 1
        status(f"student rollouts done: turns={len(cands)} pairs={n_pairs} "
               f"valid={vr:.3f} errs={st['err']} wall={time.time()-t0:.0f}s",
               tag="pretrain")
        if n_pairs < 200:
            status(f"FATAL too few pretrain pairs ({n_pairs})", tag="pretrain")
            return

        # fixed probe pairs for adapter-effect / stale-cache checks
        with open(f"{W}/probe_pairs.jsonl", "w") as fh:
            for row in [json.loads(x) for x in open(pre_path)][:6]:
                fh.write(json.dumps(row) + "\n")

        # 2) held gate pairs (student fresh vs cached held teacher refs)
        held_pairs = gen_held_pairs()
        with open(f"{W}/held_pairs_v0.jsonl", "w") as fh:
            for p in held_pairs:
                fh.write(json.dumps(p) + "\n")
        status(f"held gate pairs: n={len(held_pairs)}", tag="pretrain")

        # 3) judge v0: fresh LoRA, DDPx3 on the idle trainer GPUs, trained
        # INTO the 0.6-0.85 band (fewer steps on overshoot, more on under).
        steps = args.pretrain_steps
        acc = None
        for attempt in range(1, 4):
            t1 = time.time()
            out_dir = f"{W}/d_versions/v0"
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            status(f"training judge v0 attempt{attempt}: steps={steps} "
                   f"pairs={n_pairs} gpus={args.pretrain_gpus}",
                   tag="pretrain")
            rc = d_train_subprocess(pre_path, "", out_dir, steps,
                                    f"{W}/d_train_v0_a{attempt}.log",
                                    args.pretrain_gpus, seed=attempt)
            if rc != 0:
                status(f"FATAL judge v0 train failed rc={rc} "
                       f"(see d_train_v0_a{attempt}.log)", tag="pretrain")
                return
            judge.load_adapter("d_v0", out_dir)
            eff = judge_effect(args.teacher_url, args.teacher_base, "d_v0",
                               d_tok, probe_pairs()[:4])
            if eff < 0.005:
                status(f"FATAL judge v0 is a serving NO-OP (eff={eff:.5f}) "
                       f"-- check key remap", tag="pretrain")
                return
            gate = eval_held(judge, held_pairs)
            acc = gate["held_acc"]
            wall = time.time() - t1
            meta = json.load(open(f"{out_dir}/train_meta.json"))
            status(f"PRETRAIN dver=v0 attempt={attempt} held_acc={acc} "
                   f"pos_bias={gate['pos_bias']} matched="
                   f"{gate['matched_acc']} n={gate['n']} steps="
                   f"{meta['steps']} train_acc={meta.get('train_acc_tail')} "
                   f"eff={eff:.4f} pairs={n_pairs} wall={wall:.0f}s "
                   f"gpu_hours={gpu_hours()}", tag="judge")
            if 0.6 <= acc <= 0.85:
                break
            if acc > 0.85:
                steps = max(30, steps // 2)
                status(f"held_acc={acc}>0.85 (saturated: no graded reward) "
                       f"-- retraining with steps={steps}", tag="pretrain")
            else:
                steps = min(400, steps * 2)
                status(f"held_acc={acc}<0.60 (no signal) -- retraining with "
                       f"steps={steps}", tag="pretrain")
        if acc is None:
            return
        if not (0.6 <= acc <= 0.85):
            status(f"WARNING judge v0 outside band after retries "
                   f"(held_acc={acc}); proceeding -- throttle/updates will "
                   f"correct online", tag="pretrain")
        state.update({"dver": "v0", "d_lora": f"{W}/d_versions/v0",
                      "held_acc": acc, "pretrain_done": True,
                      "last_full_t": time.time()})
        save()
        status("pretrain complete; entering GAN loop", tag="pretrain")
    else:
        # resume: make servers consistent with state
        if state["m_serve"] != args.student_base and state["m_lora"]:
            try:
                load_adapter(args.student_url, state["m_serve"],
                             state["m_lora"])
            except Exception as e:
                status(f"resume student adapter reload failed "
                       f"{type(e).__name__}", tag="infra")
        if state["dver"]:
            try:
                judge.load_adapter(f"d_{state['dver']}", state["d_lora"])
            except Exception as e:
                status(f"resume judge adapter reload failed "
                       f"{type(e).__name__}", tag="infra")
        judge.serve = f"d_{state['dver']}"
        status(f"resumed at round={state['round']} m={state['m_serve']} "
               f"d={state['dver']}", tag="infra")

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

        # ---- fresh samples: student k, teacher 2 (scoring ref + tt) ------
        g_cands, g_vr, g_st = sample_model(args.student_url,
                                           state["m_serve"], m_tok, turns,
                                           bt, args.k, args.temp,
                                           args.max_tokens)
        t_refs, t_vr, t_st = sample_model(args.teacher_url,
                                          args.teacher_base, d_tok, turns,
                                          bt, 2, args.temp, args.max_tokens)
        if not g_cands or not t_refs:
            status(f"round={r} stalled (g={len(g_cands)} t={len(t_refs)} "
                   f"errs={g_st['err']}/{t_st['err']}); waiting 90s + "
                   f"best-effort adapter reload")
            time.sleep(90)
            try:
                if state["m_serve"] != args.student_base and state["m_lora"]:
                    load_adapter(args.student_url, state["m_serve"],
                                 state["m_lora"])
            except Exception:
                pass
            try:
                if state["dver"]:
                    judge.load_adapter(f"d_{state['dver']}", state["d_lora"])
            except Exception:
                pass
            continue

        # ---- validity gate + judge scores vs FRESH teacher ref -----------
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

        # ---- agreement / drift metrics ------------------------------------
        ag_ex, ag_tok, tt_tok, zlens = [], [], [], []
        for tid, c in best.items():
            refs = t_refs[tid]
            rb = bash_body(refs[0]["y"])
            cb = bash_body(c["y"])
            ag_ex.append(1.0 if cb == rb else 0.0)
            ag_tok.append(token_jaccard(cb, rb))
            zlens.append(len(c["z"]))
            if len(refs) > 1 and refs[1]["valid"]:
                tt_tok.append(token_jaccard(bash_body(refs[1]["y"]), rb))
        typ_items = list(best.items())[: args.typicality_n]
        typs = [mean_logprob_under_teacher(build_prompt(m_tok, turns[tid]),
                                           c["raw"]) for tid, c in typ_items]
        typ = mean(typs)

        # ---- G update: best-of-k rejection SFT on OWN rollouts only ------
        win_path = f"{W}/winners_r{r}.jsonl"
        with open(win_path, "w") as fh:
            for tid, c in best.items():
                fh.write(json.dumps({"turn_id": tid, "reward": c["reward"],
                                     "prompt": build_prompt(m_tok,
                                                            turns[tid]),
                                     "completion": c["raw"]}) + "\n")
        lora_in = f"{state['m_lora']}/hf" if state["m_lora"] else ""
        new_lora = f"{W}/m_lora/r{r}"
        rc = g_sft_subprocess(win_path, lora_in, new_lora, args.g_steps,
                              f"{W}/g_sft_r{r}.log")
        g_loaded = "kept"
        if rc == 0:
            try:
                load_student(f"m_r{r}", new_lora)
                eff = gen_effect(args.student_url, args.student_base,
                                 f"m_r{r}", build_prompt(m_tok, turns[bt[0]]))
                if eff < 1e-4:
                    raise RuntimeError(f"no-op adapter (eff={eff})")
                state["m_lora"] = new_lora
                state["m_serve"] = f"m_r{r}"
                g_loaded = f"m_r{r}(eff={eff:.4f})"
            except Exception as e:
                status(f"round={r} student adapter hot-load REJECTED "
                       f"{type(e).__name__} {str(e)[:100]}; keeping "
                       f"{state['m_serve']}")
        else:
            status(f"round={r} g_sft FAILED rc={rc}; keeping "
                   f"{state['m_serve']}")

        # ---- accumulate judge pairs (fresh teacher ref vs best student) --
        with open(f"{W}/d_pending.jsonl", "a") as fh:
            for tid, c in best.items():
                fh.write(json.dumps({
                    "turn_id": tid, "round": r, "prefix_text": prefix(tid),
                    "ref": both_text(t_refs[tid][0]["z"],
                                     t_refs[tid][0]["y"]),
                    "mine": both_text(c["z"], c["y"])}) + "\n")

        status(f"round={r} fool={fool:.4f} fool_best={fool_best:.4f} "
               f"valid={g_vr:.3f} agree_exact={mean(ag_ex):.3f} "
               f"agree_tok={mean(ag_tok):.3f} typ_lp={typ:.4f} "
               f"winners={len(best)} dver={state['dver']} "
               f"wall={time.time()-t0:.0f}s fool_win={fool_win:.3f} "
               f"t_valid={t_vr:.3f} tt_tok={mean(tt_tok):.3f} "
               f"think_len={mean(zlens):.0f} g={g_loaded} "
               f"errs={g_st['err']}+{t_st['err']}")
        state["round"] = r
        save()

        # ---- judge online update every d-every rounds ---------------------
        if r % args.d_every == 0:
            t1 = time.time()
            held_pairs = gen_held_pairs()
            gate0 = eval_held(judge, held_pairs)
            state["held_acc"] = gate0["held_acc"]
            save()
            if gate0["held_acc"] > args.d_throttle:
                status(f"SKIP round={r} held_acc={gate0['held_acc']} "
                       f"(> {args.d_throttle} balanced-regime throttle) "
                       f"pos_bias={gate0['pos_bias']} n={gate0['n']}",
                       tag="judge")
            else:
                rows = [json.loads(x) for x in open(f"{W}/d_pending.jsonl")]
                recent = rows[-args.d_window:]
                replay = [json.loads(x) for x in open(
                    f"{W}/d_pretrain_pairs.jsonl")]
                replay = random.Random(r).sample(
                    replay, min(args.d_replay, len(replay)))
                dp = f"{W}/d_train_r{r}.jsonl"
                with open(dp, "w") as fh:
                    for row in recent + replay:
                        fh.write(json.dumps(row) + "\n")
                new_ver = f"v{r}"
                out_dir = f"{W}/d_versions/{new_ver}"
                lin = f"{state['d_lora']}/hf" if state["d_lora"] else ""
                rc = d_train_subprocess(dp, lin, out_dir, args.d_steps,
                                        f"{W}/d_train_{new_ver}.log",
                                        args.dtrain_gpus, seed=r)
                if rc != 0:
                    status(f"round={r} judge train FAILED rc={rc}; keeping "
                           f"{state['dver']}", tag="judge")
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
                                      for a, b in zip(after, before)
                                      if a and b])
                        eff = judge_effect(args.teacher_url,
                                           args.teacher_base,
                                           f"d_{new_ver}", d_tok, pr[:4])
                        if eff < 0.005:
                            raise RuntimeError(
                                f"no-op vs base (eff={eff:.5f})")
                        # stale prefix-cache guard: scores must move
                        if delta < 1e-4:
                            status(f"round={r} WARNING probe_delta="
                                   f"{delta:.6f} -- stale cache or null "
                                   f"update", tag="judge")
                        swapped = True
                    except Exception as e:
                        status(f"round={r} judge adapter swap REJECTED "
                               f"{type(e).__name__} {str(e)[:120]}; "
                               f"reverting to {old_ver}", tag="judge")
                        try:
                            judge.load_adapter(f"d_{old_ver}", old_lora)
                        except Exception:
                            pass
                    if swapped:
                        state["dver"] = new_ver
                        state["d_lora"] = out_dir
                        save()
                        meta = {}
                        try:
                            meta = json.load(
                                open(f"{out_dir}/train_meta.json"))
                        except Exception:
                            pass
                        gate = eval_held(judge, held_pairs)
                        state["held_acc"] = gate["held_acc"]
                        save()
                        status(f"UPDATE dver={new_ver} held_acc="
                               f"{gate['held_acc']} pos_bias="
                               f"{gate['pos_bias']} matched="
                               f"{gate['matched_acc']} n={gate['n']} "
                               f"train_acc={meta.get('train_acc_tail')} "
                               f"pairs={len(recent)}+{len(replay)}replay "
                               f"probe_delta={delta:.4f} eff={eff:.4f} "
                               f"wall={time.time()-t1:.0f}s", tag="judge")

        # ---- benches on the dedicated server (loop unpaused) --------------
        if r % args.bench_every == 0 and state["m_lora"]:
            launch_bench(f"proxy_r{r}", state["m_serve"], state["m_lora"], r)
        if (time.time() - state["last_full_t"] > args.full_every_s
                and state["m_lora"]):
            state["last_full_t"] = time.time()
            save()
            launch_bench(f"full_r{r}", state["m_serve"], state["m_lora"], r)

        # ---- prune old adapters (keep last 3 + every 10th) ----------------
        old = f"{W}/m_lora/r{r - 4}"
        if (r - 4) > 0 and (r - 4) % 10 and os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
        for f in (f"{W}/winners_r{r - 8}.jsonl", f"{W}/g_sft_r{r - 8}.log"):
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
