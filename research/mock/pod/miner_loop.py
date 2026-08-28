#!/usr/bin/env python3
"""Track M MINER BOX loop: the one white-box miner iterating against the
published judge -- exactly what a real SN120 miner would run.

GPU map (B300, Docker vLLM only):
  0    vLLM Qwen3.8-27B :8002  local judge copy (base + published adapter)
  1    vLLM 35B-A3B    :8001  miner model (base + own training adapter)
  2,3  g_sft.py per round (attention-only LoRA, Track A's g-step config)

Round: pull/refresh published judge -> sample k rollouts per turn ->
validity gate -> reward = P(judged teacher) via local judge (both orders)
-> best-of-k SFT on winners -> hot-swap adapter -> log. Every
--submit-every rounds: ship the adapter to the eval box challenger slot
(submissions/ dir + READY marker).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from gad_common import both_text, build_prompt, load_rollout_cache, load_turns  # noqa: E402
from trackM_common import Judge, gen_effect, load_adapter, now, render_prefix, sample_model  # noqa: E402


class DualJudge:
    """Two identical judge servers (GPUs 0 and 4) halve reward-scoring time."""

    def __init__(self, j1, j2):
        self.js = [j1, j2]

    @property
    def serve(self):
        return self.js[0].serve

    def load_adapter(self, name, path, drop=()):
        for j in self.js:
            j.load_adapter(name, path, drop=drop)

    def score_pairs(self, pairs, workers=24):
        half = (len(pairs) + 1) // 2
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(self.js[0].score_pairs, pairs[:half], workers)
            f2 = ex.submit(self.js[1].score_pairs, pairs[half:], workers)
            return f1.result() + f2.result()


def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--teacher-train", default="/dshare/koth/teacher_train.jsonl")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8002")
    ap.add_argument("--judge2-url", default="http://127.0.0.1:8004")
    ap.add_argument("--miner-url", default="http://127.0.0.1:8001")
    ap.add_argument("--base-27b", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--base-35b", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--eval-ssh", default="-p 40301 root@86.38.182.105")
    ap.add_argument("--eval-koth", default="/dshare/koth")
    ap.add_argument("--batch-turns", type=int, default=16)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=3584)
    ap.add_argument("--g-steps", type=int, default=30)
    ap.add_argument("--sft-gpus", default="2,3")
    ap.add_argument("--train-python", default="/root/trainenv/bin/python")
    ap.add_argument("--submit-every", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10000)
    ap.add_argument("--work", default="/dshare/koth")
    args = ap.parse_args()

    W = args.work
    os.makedirs(f"{W}/judges", exist_ok=True)
    os.makedirs(f"{W}/m_lora", exist_ok=True)
    status_path = f"{W}/status.log"
    state_path = f"{W}/miner_state.json"
    here = os.path.dirname(os.path.abspath(__file__))
    SSH = f"ssh -o StrictHostKeyChecking=no {args.eval_ssh}"

    def status(line):
        with open(status_path, "a") as fh:
            fh.write(f"{now()} [miner] {line}\n")
        print(f"STATUS {line}", flush=True)

    from transformers import AutoTokenizer
    m_tok = AutoTokenizer.from_pretrained(args.base_35b)
    d_tok = AutoTokenizer.from_pretrained(args.base_27b)

    turns = load_turns(args.turns)
    t_train = load_rollout_cache(args.teacher_train)
    train_tids = sorted(t for t in t_train if t in turns)
    pfx = {}

    def prefix(tid):
        if tid not in pfx:
            pfx[tid] = render_prefix(turns[tid], args.d_max_len * 6)
        return pfx[tid]

    judge = DualJudge(Judge(args.judge_url, args.base_27b, d_tok, args.d_max_len),
                      Judge(args.judge2_url, args.base_27b, d_tok, args.d_max_len))
    state = {"round": 0, "m_lora": "", "m_serve": args.base_35b, "dver": ""}
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))
        if state["m_serve"] != args.base_35b and state["m_lora"]:
            try:
                load_adapter(args.miner_url, state["m_serve"], state["m_lora"])
            except Exception:
                state["m_serve"] = args.base_35b

    def save():
        json.dump(state, open(state_path, "w"))

    def pull_judge():
        """Poll the eval box VERSION file; pull + hot-load a new adapter."""
        r = sh(f"{SSH} 'cat {args.eval_koth}/VERSION 2>/dev/null'")
        ver = (r.stdout or "").strip()
        if not ver or ver == state["dver"]:
            return False
        dst = f"{W}/judges/{ver}"
        if not os.path.isdir(dst):
            os.makedirs(dst, exist_ok=True)
            r = sh(f"{SSH} 'cd {args.eval_koth}/d_versions && tar czf - {ver}' "
                   f"| tar xzf - -C {W}/judges/", timeout=1200)
            if r.returncode != 0:
                status(f"judge pull {ver} FAILED: {r.stderr[-200:]}")
                shutil.rmtree(dst, ignore_errors=True)
                return False
        try:
            judge.load_adapter(f"d_{ver}", dst,
                               drop=[f"d_{state['dver']}"] if state["dver"] else [])
            gp = f"{W}/seed_pairs.jsonl"
            if os.path.exists(gp):
                from trackM_common import judge_effect
                pairs = [json.loads(x) for x in open(gp)][:4]
                eff = judge_effect(args.judge_url, args.base_27b, f"d_{ver}",
                                   d_tok, pairs)
                if eff < 0.005:
                    raise RuntimeError(f"published judge {ver} is a serving "
                                       f"no-op here (eff={eff:.5f})")
        except Exception as e:
            status(f"judge adapter load {ver} FAILED {type(e).__name__} "
                   f"{str(e)[:120]}")
            return False
        old = state["dver"]
        state["dver"] = ver
        save()
        status(f"JUDGE UPDATE: {old or 'none'} -> {ver} (hot-swapped "
               f"mid-training; new reign)")
        return True

    def submit(r):
        if not state["m_lora"]:
            status("submit skipped: no trained adapter yet")
            return
        sub = f"sub_r{r:04d}_{int(time.time())}"
        rem = f"{args.eval_koth}/submissions/{sub}"
        meta = {"round": r, "local_fool": state.get("last_fool"),
                "dver": state["dver"], "ts": now()}
        cmd = (f"{SSH} 'mkdir -p {rem}/adapter' && "
               f"cd {state['m_lora']} && tar czf - . | "
               f"{SSH} 'tar xzf - -C {rem}/adapter' && "
               f"echo '{json.dumps(meta)}' | {SSH} 'cat > {rem}/meta.json' && "
               f"{SSH} 'touch {rem}/READY'")
        res = sh(cmd, timeout=1200)
        if res.returncode == 0:
            status(f"SUBMIT {sub}: round={r} local_fool={state.get('last_fool')} "
                   f"dver={state['dver']} -> eval box challenger slot")
        else:
            status(f"SUBMIT FAILED rc={res.returncode}: {res.stderr[-200:]}")

    status(f"miner up: base={args.base_35b} judge={args.base_27b}+published "
           f"train_turns={len(train_tids)} k={args.k} submit_every="
           f"{args.submit_every} | gpu_map[0=judge :8002 1=miner :8001 "
           f"2,3=g_sft]")
    while not pull_judge():
        status("waiting for published judge v0...")
        time.sleep(60)

    rng = random.Random(42)
    shuffled = train_tids[:]
    rng.shuffle(shuffled)
    cursor = state["round"] * args.batch_turns % len(shuffled)

    while state["round"] < args.rounds:
        r = state["round"] + 1
        t0 = time.time()
        pull_judge()

        bt = []
        while len(bt) < args.batch_turns:
            if cursor >= len(shuffled):
                cursor = 0
                rng.shuffle(shuffled)
            bt.append(shuffled[cursor])
            cursor += 1

        cands, vr, st = sample_model(args.miner_url, state["m_serve"], m_tok,
                                     turns, bt, args.k, args.temp,
                                     args.max_tokens)
        pairs, owners = [], []
        for tid, cs in cands.items():
            ref = both_text(t_train[tid][0]["z"], t_train[tid][0]["y"])
            for c in cs:
                if c["valid"]:
                    pairs.append({"prefix_text": prefix(tid), "ref": ref,
                                  "mine": both_text(c["z"], c["y"])})
                    owners.append((tid, c))
        scores = judge.score_pairs(pairs)
        rewards = [(1 - s["p_teacher"]) if s else 0.0 for s in scores]

        best = {}
        all_r = []
        for (tid, c), rw in zip(owners, rewards):
            all_r.append(rw)
            if tid not in best or rw > best[tid][1]:
                best[tid] = (c, rw)
        mean_fool = sum(all_r) / max(len(all_r), 1)
        state["last_fool"] = round(mean_fool, 4)
        frac_tiny = sum(1 for x in all_r if x < 0.01) / max(len(all_r), 1)

        win_path = f"{W}/winners_r{r}.jsonl"
        with open(win_path, "w") as fh:
            for tid, (c, rw) in best.items():
                fh.write(json.dumps({"turn_id": tid, "reward": rw,
                                     "prompt": build_prompt(m_tok, turns[tid]),
                                     "completion": c["raw"]}) + "\n")

        new_lora = f"{W}/m_lora/r{r}"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.sft_gpus
        lora_in = f"{state['m_lora']}/hf" if state["m_lora"] else ""
        rc = subprocess.run(
            [args.train_python, f"{here}/g_sft.py", "--base", args.base_35b,
             "--winners", win_path, "--lora-in", lora_in,
             "--lora-out", new_lora, "--max-steps", str(args.g_steps)],
            env=env, stdout=open(f"{W}/g_sft_r{r}.log", "w"),
            stderr=subprocess.STDOUT).returncode
        if rc == 0:
            try:
                prev = ([state["m_serve"]]
                        if state["m_serve"] != args.base_35b else [])
                load_adapter(args.miner_url, f"m_r{r}", new_lora, drop=prev)
                # adapter-effect guard: vLLM silently no-ops mis-keyed LoRAs
                eff = gen_effect(args.miner_url, args.base_35b, f"m_r{r}",
                                 build_prompt(m_tok, turns[bt[0]]))
                if eff < 1e-4:
                    raise RuntimeError(f"no-op adapter (eff={eff})")
                state["m_lora"] = new_lora
                state["m_serve"] = f"m_r{r}"
            except Exception as e:
                status(f"round={r} adapter hot-load REJECTED "
                       f"{type(e).__name__} {str(e)[:100]}; keeping "
                       f"{state['m_serve']}")
        else:
            status(f"round={r} g_sft FAILED rc={rc}; keeping {state['m_serve']}")

        best_mean = (sum(rw for _, rw in best.values()) / max(len(best), 1))
        status(f"round={r} fool_local={mean_fool:.4f} best_fool="
               f"{best_mean:.4f} valid={vr:.3f} winners={len(best)} "
               f"rewards<0.01={frac_tiny:.2f} dver={state['dver']} "
               f"errs={st['err']} wall={time.time()-t0:.0f}s")
        if frac_tiny > 0.8:
            status(f"round={r} WARNING >80% of rewards < 0.01 -- judge "
                   f"saturated (balance lesson); eval box should shorten "
                   f"next crown-train")
        state["round"] = r
        save()
        if r % args.submit_every == 0:
            submit(r)
        # prune old adapters (keep last 3 + every 10th)
        old = f"{W}/m_lora/r{r - 4}"
        if (r - 4) > 0 and (r - 4) % 10 and os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)


if __name__ == "__main__":
    main()
