#!/usr/bin/env python3
"""Track M EVAL BOX driver: continuous KOTH duel loop with crown-triggered,
from-scratch, frozen-per-reign discriminator retraining.

GPU map (pins are CUDA_VISIBLE_DEVICES, invisible in ps -- documented here
and in every status line):
  0,1  vLLM Qwen3.8-27B TP2 :8000  teacher + D scoring (adapter per request)
  2    vLLM 35B-A3B      :8001  KING   (base + king adapter)
  3    vLLM 35B-A3B      :8003  CHALLENGER (miner submissions)
  4,5  d_train.py at crown events (from scratch, full archive + replay)
  6,7  SWE proxy/full panel benches (bench server :8006)

Cycle: eval batches -> (challenger adopted from submissions/) -> paired duel
-> crown when ch_fool >= king_fool + eps over >= N pairs (reject at 2N) ->
PAUSE -> retrain D from scratch on full archive (+high-fool replay) ->
held-out gate -> publish (VERSION + d_versions/vK) -> new reign under new
judge. Everything D sees is normalised text (disc_text conventions).
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

from gad_common import bash_body, both_text, build_prompt, load_rollout_cache, load_turns, token_jaccard  # noqa: E402
from trackM_common import Judge, eval_held, gen_effect, judge_effect, load_adapter, now, render_prefix, sample_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--teacher-train", default="/dshare/gad/teacher_train.jsonl")
    ap.add_argument("--teacher-heldout", default="/dshare/gad/teacher_heldout.jsonl")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000")
    ap.add_argument("--king-url", default="http://127.0.0.1:8001")
    ap.add_argument("--chal-url", default="http://127.0.0.1:8003")
    ap.add_argument("--base-27b", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--base-35b", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--batch-turns", type=int, default=24)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=3584)
    ap.add_argument("--crown-eps", type=float, default=0.03,
                    help="economic floor: challenger fool >= king fool + eps")
    ap.add_argument("--crown-z", type=float, default=2.5,
                    help="paired z-score gate (p<0.01): mean(d)/(std(d)/sqrt(n)) "
                         "on per-turn d_i = fool_c_i - fool_k_i; at fool~0.18 "
                         "the +3pp floor alone is inside noise ~10% of the time")
    ap.add_argument("--crown-n", type=int, default=400,
                    help="first test checkpoint; tests only at n, 2n, 3n, 4n "
                         "to avoid sequential-peeking inflation")
    ap.add_argument("--duel-max", type=int, default=1600,
                    help="REJECT if conditions unmet by this many pairs")
    ap.add_argument("--duel-early-reject", type=int, default=800,
                    help="REJECT early if z<0 at this checkpoint")
    ap.add_argument("--d-steps", type=int, default=150)
    ap.add_argument("--d-lr", type=float, default=1e-5)
    ap.add_argument("--d-batch", type=int, default=2, help="per-GPU")
    ap.add_argument("--d-accum", type=int, default=1)
    ap.add_argument("--dtrain-gpus", default="2,3,4,5",
                    help="king/chal GPUs are borrowed during the pause (their "
                         "containers are stopped) + the dedicated 4,5")
    ap.add_argument("--train-python", default="/root/trainenv/bin/python")
    ap.add_argument("--ratchet-n", type=int, default=100,
                    help="pairs used for the reign-start king fool rate")
    ap.add_argument("--work", default="/dshare/koth")
    args = ap.parse_args()

    W = args.work
    os.makedirs(f"{W}/d_versions", exist_ok=True)
    os.makedirs(f"{W}/submissions", exist_ok=True)
    status_path = f"{W}/status.log"
    archive_path = f"{W}/archive.jsonl"
    state_path = f"{W}/state.json"
    here = os.path.dirname(os.path.abspath(__file__))

    def status(line):
        with open(status_path, "a") as fh:
            fh.write(f"{now()} [eval] {line}\n")
        print(f"STATUS {line}", flush=True)

    from transformers import AutoTokenizer
    m_tok = AutoTokenizer.from_pretrained(args.base_35b)
    d_tok = AutoTokenizer.from_pretrained(args.base_27b)

    turns = load_turns(args.turns)
    t_train = load_rollout_cache(args.teacher_train)
    t_held = load_rollout_cache(args.teacher_heldout)
    train_tids = sorted(t for t in t_train if t in turns)
    held_tids = sorted(t for t in t_held if t in turns)
    pfx = {}

    def prefix(tid):
        if tid not in pfx:
            pfx[tid] = render_prefix(turns[tid], args.d_max_len * 6)
        return pfx[tid]

    def teacher_ref(tid, cache):
        r = cache[tid][0]
        return both_text(r["z"], r["y"])

    dver0 = "v0"
    if os.path.exists(f"{W}/VERSION"):
        dver0 = open(f"{W}/VERSION").read().strip()
    FRESH_DUEL = {"n": 0, "kf": 0.0, "cf": 0.0, "sd": 0.0, "sd2": 0.0,
                  "ck": 0}
    state = {
        "reign": 0, "dver": dver0,
        "king_name": args.base_35b, "king_adapter": "",
        "chal": None,                      # {name, adapter, sub, local_fool}
        "duel": dict(FRESH_DUEL),
        "reign_turns": 0, "reign_t0": time.time(),
        "ratchet": {"n": 0, "f": 0.0, "logged": False},
        "agree": {"n": 0, "ex": 0.0, "tok": 0.0},
        "cursor": 0, "batch": 0,
    }
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))
        if "sd" not in state["duel"]:
            # duel begun under the old rule: restart it under the z-gate
            state["duel"] = dict(FRESH_DUEL)
            if state["chal"]:
                status(f"duel with {state['chal']['sub']} RESTARTED under new "
                       f"crown rule (paired z>={args.crown_z} added)")
        status(f"resume state reign={state['reign']} dver={state['dver']} "
               f"king={state['king_name']} chal={bool(state['chal'])}")

    judge = Judge(args.judge_url, args.base_27b, d_tok, args.d_max_len)
    # (re)load serving adapters after restart
    dv = state["dver"]
    if os.path.isdir(f"{W}/d_versions/{dv}"):
        judge.load_adapter(f"d_{dv}", f"{W}/d_versions/{dv}")
        status(f"judge adapter d_{dv} loaded")
    if state["king_adapter"]:
        load_adapter(args.king_url, state["king_name"], state["king_adapter"])
    if state["chal"]:
        load_adapter(args.chal_url, state["chal"]["name"],
                     state["chal"]["adapter"])

    rng = random.Random(777)
    shuffled = train_tids[:]
    rng.shuffle(shuffled)

    def save():
        json.dump(state, open(state_path, "w"))

    def duel_stats(d):
        """Paired duel statistics from running sums of d_i = fool_c - fool_k
        (continuous judge probabilities, per turn)."""
        import math
        n = max(d["n"], 1)
        mean = d["sd"] / n
        var = max(d["sd2"] / n - mean * mean, 0.0) * (n / max(n - 1, 1))
        se = math.sqrt(var / n) if n > 1 else float("inf")
        z = mean / se if se > 0 else 0.0
        return {"n": d["n"], "kf": d["kf"] / n, "cf": d["cf"] / n,
                "diff": mean, "se": se, "z": z}

    def archive_rows(rows):
        with open(archive_path, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    guard_pairs = [json.loads(x)
                   for x in open(f"{W}/seed_pairs.jsonl")][:8]
    guard_prompt = build_prompt(m_tok, turns[guard_pairs[0]["turn_id"]])

    def adopt_submission():
        subs = sorted(d for d in os.listdir(f"{W}/submissions")
                      if os.path.exists(f"{W}/submissions/{d}/READY"))
        if not subs:
            return
        sub = subs[-1]
        meta = {}
        mp = f"{W}/submissions/{sub}/meta.json"
        if os.path.exists(mp):
            meta = json.load(open(mp))
        name = f"chal_{sub}"
        try:
            load_adapter(args.chal_url, name, f"{W}/submissions/{sub}/adapter")
            eff = gen_effect(args.chal_url, args.base_35b, name, guard_prompt)
            if eff < 1e-4:
                raise RuntimeError(f"adapter-effect guard: no-op (eff={eff})")
        except Exception as e:
            status(f"submission {sub} adapter REJECTED at load: "
                   f"{type(e).__name__} {str(e)[:120]}")
            os.rename(f"{W}/submissions/{sub}/READY",
                      f"{W}/submissions/{sub}/FAILED")
            return
        state["chal"] = {"name": name, "adapter": f"{W}/submissions/{sub}/adapter",
                         "sub": sub, "local_fool": meta.get("local_fool")}
        state["duel"] = dict(FRESH_DUEL)
        os.rename(f"{W}/submissions/{sub}/READY",
                  f"{W}/submissions/{sub}/CONSUMED")
        status(f"DUEL start: challenger={sub} miner_round={meta.get('round')} "
               f"miner_local_fool={meta.get('local_fool')} vs king="
               f"{state['king_name']} (rule: margin>={args.crown_eps} AND "
               f"z>={args.crown_z}, tests at n={args.crown_n},x2,x3,x4, "
               f"reject at {args.duel_max})")
        save()

    def crown(st_):
        """Pause evals, retrain D from scratch, gate, publish, new reign."""
        t_crown = time.time()
        old_king = state["king_name"]
        ch = state["chal"]
        new_reign = state["reign"] + 1
        new_dver = f"v{new_reign}"
        status(f"CROWN reign={state['reign']}->{new_reign}: {ch['sub']} "
               f"dethrones {old_king} | n={st_['n']} "
               f"king_fool={st_['kf']:.4f} ch_fool={st_['cf']:.4f} "
               f"diff={st_['diff']:+.4f} paired_se={st_['se']:.4f} "
               f"z={st_['z']:.2f} "
               f"miner_local_fool={ch.get('local_fool')} (overfit gap = "
               f"local - measured = "
               f"{(ch.get('local_fool') or 0) - st_['cf']:+.4f}) | "
               f"reign_len_turns={state['reign_turns']} "
               f"reign_wall_s={time.time()-state['reign_t0']:.0f} | "
               f"EVALS PAUSED for D retrain")
        # 1. new king takes GPU-2 slot
        new_king_name = f"king_r{new_reign}"
        king_dir = f"{W}/kings/{new_king_name}"
        os.makedirs(f"{W}/kings", exist_ok=True)
        if os.path.isdir(king_dir):
            shutil.rmtree(king_dir)
        shutil.copytree(ch["adapter"], king_dir)
        load_adapter(args.king_url, new_king_name, king_dir,
                     drop=[old_king] if state["king_adapter"] else [])
        state.update({"king_name": new_king_name, "king_adapter": king_dir,
                      "chal": None, "duel": dict(FRESH_DUEL)})

        # 2. held-out gate data FIRST (needs the king/chal servers, which are
        # about to be stopped so their GPUs can join the DDP retrain)
        held_pairs = gen_held_pairs()

        # 3. D retrain from scratch: full archive + replay of high-fool rows
        pairs_path = f"{W}/d_pairs_{new_dver}.jsonl"
        n_all = n_replay = 0
        with open(pairs_path, "w") as fh:
            for line in open(archive_path):
                r = json.loads(line)
                if not r.get("valid") or r["turn_id"] not in turns:
                    continue
                row = json.dumps({"turn_id": r["turn_id"],
                                  "prefix_text": prefix(r["turn_id"]),
                                  "ref": r["ref"],
                                  "mine": both_text(r["z"], r["y"])})
                fh.write(row + "\n")
                n_all += 1
                sc = r.get("score")
                if sc is not None and (1 - sc) >= 0.5:   # fooled the judge
                    fh.write(row + "\n")
                    n_replay += 1
        # borrow the king/chal GPUs for DDP: their servers are paused anyway
        gpus = args.dtrain_gpus.split(",")
        subprocess.run("docker stop vllm_8001 vllm_8003", shell=True,
                       capture_output=True, timeout=300)
        d_steps = args.d_steps
        acc = None
        for attempt in (1, 2):
            t0 = time.time()
            out_dir = f"{W}/d_versions/{new_dver}"
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = args.dtrain_gpus
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log_p = f"{W}/d_train_{new_dver}.log"
            # --grad-ckpt is REQUIRED on this arch: qwen3_5's gated-delta
            # layers OOM a 141GB H200 at batch 2 without it (v1 crown, rc=1)
            rc = subprocess.run(
                [f"{os.path.dirname(args.train_python)}/torchrun",
                 f"--nproc-per-node={len(gpus)}", f"{here}/d_train.py",
                 "--base", args.base_27b, "--pairs", pairs_path,
                 "--lora-out", out_dir, "--grad-ckpt",
                 "--max-steps", str(d_steps), "--lr", str(args.d_lr),
                 "--batch", str(args.d_batch), "--accum", str(args.d_accum),
                 "--max-len", str(args.d_max_len), "--seed", str(new_reign)],
                env=env, stdout=open(log_p, "w"),
                stderr=subprocess.STDOUT).returncode
            wall = time.time() - t0
            if rc != 0:
                status(f"CROWN {new_dver} D-train FAILED rc={rc} (see "
                       f"{log_p}); keeping {dv_current()} judge")
                subprocess.run("docker start vllm_8001 vllm_8003", shell=True,
                               capture_output=True, timeout=300)
                wait_and_repin_king()
                save()
                return
            # 4. gates: adapter-effect guard (no-op class) + held-out band
            judge.load_adapter(f"d_{new_dver}_cand", out_dir)
            eff = judge_effect(args.judge_url, args.base_27b,
                               f"d_{new_dver}_cand", d_tok, guard_pairs)
            if eff < 0.01:
                status(f"CROWN {new_dver} PUBLISH REFUSED: adapter-effect "
                       f"guard eff={eff:.5f} (<0.01) -- adapter is a "
                       f"serving no-op; keeping {dv_current()}")
                subprocess.run("docker start vllm_8001 vllm_8003", shell=True,
                               capture_output=True, timeout=300)
                save()
                return
            gate = eval_held(judge, held_pairs)
            acc = gate["held_acc"]
            meta = json.load(open(f"{out_dir}/train_meta.json"))
            status(f"CROWN {new_dver} D-train attempt{attempt}: steps="
                   f"{meta['steps']} pairs={n_all}(+{n_replay} replay) "
                   f"wall={wall:.0f}s effect={eff:.4f} held_acc={acc} "
                   f"pos_bias={gate['pos_bias']} matched={gate['matched_acc']} "
                   f"ab_mass={gate['ab_mass']} n={gate['n']}")
            if acc <= 0.92 or attempt == 2:
                if acc > 0.92:
                    status(f"CROWN {new_dver} WARNING held_acc={acc}>0.92 "
                           f"after retry; publishing anyway (watch miner "
                           f"reward distribution)")
                if acc > 0.9:
                    status(f"CROWN NOTE: acc>{0.9} -- next crown will use "
                           f"fewer steps (balance lesson)")
                break
            d_steps = max(30, d_steps // 2)
            status(f"CROWN {new_dver} held_acc={acc}>0.92: retraining "
                   f"from scratch with steps={d_steps}")
        # 4b. SOFT answer-mass gate (replaces the A3 hard answer-first gate,
        # user decision 2026-08-22 ~21:00Z): with direct logit reading the
        # A/B ratio is defined regardless of what the judge WOULD generate,
        # so format drift is harmless for scoring. Only refuse when raw
        # mass(A)+mass(B) collapses -- renormalized readings get noisy when
        # the judge puts almost no probability on either letter.
        if gate["ab_mass"] < 0.5:
            status(f"CROWN {new_dver} PUBLISH REFUSED: mean answer mass "
                   f"{gate['ab_mass']:.4f} < 0.5 on held pairs -- judge "
                   f"barely considers A/B; keeping {dv_current()}")
            subprocess.run("docker start vllm_8001 vllm_8003", shell=True,
                           capture_output=True, timeout=300)
            wait_and_repin_king()
            save()
            return
        # 5. resume the sampling servers, re-pin the new king adapter
        subprocess.run("docker start vllm_8001 vllm_8003", shell=True,
                       capture_output=True, timeout=300)
        wait_and_repin_king()
        # 6. publish
        judge.load_adapter(f"d_{new_dver}", f"{W}/d_versions/{new_dver}",
                           drop=[f"d_{new_dver}_cand"])
        with open(f"{W}/VERSION", "w") as fh:
            fh.write(new_dver + "\n")
        state.update({"reign": new_reign, "dver": new_dver,
                      "reign_turns": 0, "reign_t0": time.time(),
                      "ratchet": {"n": 0, "f": 0.0, "logged": False},
                      "agree": {"n": 0, "ex": 0.0, "tok": 0.0}})
        save()
        status(f"PUBLISH {new_dver}: adapter={W}/d_versions/{new_dver} "
               f"VERSION bumped | crown_total_wall={time.time()-t_crown:.0f}s "
               f"| EVALS RESUME under {new_dver}, king={new_king_name}")
        # 5. benches in background (proxy per reign; full panel on the king)
        try:
            subprocess.Popen(
                ["bash", f"{here}/bench_king.sh", new_king_name,
                 king_dir, str(new_reign)],
                stdout=open(f"{W}/bench_r{new_reign}.log", "w"),
                stderr=subprocess.STDOUT)
            status(f"bench launched for {new_king_name} (GPUs 6,7, background)")
        except Exception as e:
            status(f"bench launch failed {type(e).__name__} (non-blocking)")

    def dv_current():
        return state["dver"]

    def wait_and_repin_king():
        """After a sampling-server restart, runtime adapters are gone; the
        king adapter must be re-loaded or every request 404s."""
        import requests as rq
        for port in (8001, 8003):
            for _ in range(80):
                try:
                    rq.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                    break
                except Exception:
                    time.sleep(10)
        if state["king_adapter"]:
            load_adapter(args.king_url, state["king_name"],
                         state["king_adapter"])

    def gen_held_pairs():
        cands, vr, _ = sample_model(args.king_url, state["king_name"], m_tok,
                                    turns, held_tids, 1, args.temp,
                                    args.max_tokens)
        out = []
        for tid in held_tids:
            cs = cands.get(tid)
            if not cs or not cs[0]["valid"]:
                continue
            out.append({"turn_id": tid, "prefix_text": prefix(tid),
                        "ref": teacher_ref(tid, t_held),
                        "mine": both_text(cs[0]["z"], cs[0]["y"])})
        return out

    def reject(st_):
        ch = state["chal"]
        status(f"REJECT: challenger {ch['sub']} failed to dethrone "
               f"{state['king_name']} | n={st_['n']} "
               f"king_fool={st_['kf']:.4f} ch_fool={st_['cf']:.4f} "
               f"diff={st_['diff']:+.4f} paired_se={st_['se']:.4f} "
               f"z={st_['z']:.2f} (needed diff>=+{args.crown_eps} AND "
               f"z>={args.crown_z}) | miner_local_fool="
               f"{ch.get('local_fool')} -- noise-floor calibration point")
        state["chal"] = None
        state["duel"] = dict(FRESH_DUEL)
        save()

    status(f"eval driver up: reign={state['reign']} dver={state['dver']} "
           f"king={state['king_name']} train_turns={len(train_tids)} "
           f"held={len(held_tids)} eps={args.crown_eps} N={args.crown_n} | "
           f"gpu_map[0,1=judge/teacher :8000 2=king :8001 3=chal :8003 "
           f"4,5=d_train 6,7=bench]")

    while True:
        if not state["chal"]:
            adopt_submission()
        bt = []
        while len(bt) < args.batch_turns:
            if state["cursor"] >= len(shuffled):
                state["cursor"] = 0
                rng.shuffle(shuffled)
            bt.append(shuffled[state["cursor"]])
            state["cursor"] += 1
        state["batch"] += 1
        b = state["batch"]
        t0 = time.time()

        # king and challenger sample in parallel (separate GPUs/servers)
        with ThreadPoolExecutor(max_workers=2) as pool:
            k_fut = pool.submit(sample_model, args.king_url,
                                state["king_name"], m_tok, turns, bt, 1,
                                args.temp, args.max_tokens)
            c_fut = None
            if state["chal"]:
                c_fut = pool.submit(sample_model, args.chal_url,
                                    state["chal"]["name"], m_tok, turns, bt,
                                    1, args.temp, args.max_tokens)
            k_cands, k_vr, k_st = k_fut.result()
            c_cands, c_vr = {}, None
            if c_fut:
                c_cands, c_vr, _ = c_fut.result()

        # judge all valid rollouts vs the cached teacher ref
        pairs, owners = [], []
        for who, cands in (("king", k_cands), ("challenger", c_cands)):
            for tid, cs in cands.items():
                c = cs[0]
                if c["valid"]:
                    pairs.append({"prefix_text": prefix(tid),
                                  "ref": teacher_ref(tid, t_train),
                                  "mine": both_text(c["z"], c["y"])})
                    owners.append((who, tid, c))
        scores = judge.score_pairs(pairs)

        fool = {"king": {}, "challenger": {}}
        rows = []
        ts = now()
        n_scr_err = 0
        for (who, tid, c), s in zip(owners, scores):
            # [fix] transport/scoring errors are EXCLUDED from every stat
            # (never scored 0.5 or 0.0); archived with score_err for audit.
            # Absent from fool[who], the turn also drops out of the paired
            # duel intersection below.
            if s is None:
                n_scr_err += 1
            else:
                fool[who][tid] = 1 - s["p_teacher"]
            rows.append({"ts": ts, "turn_id": tid, "model": who,
                         "ckpt": state["king_name"] if who == "king"
                         else state["chal"]["name"],
                         "dver": state["dver"], "z": c["z"], "y": c["y"],
                         "ref": teacher_ref(tid, t_train),
                         "score": (s["p_teacher"] if s else None),
                         "valid": True,
                         **({"score_err": True} if s is None else {})})
        # invalid rollouts: archived, fool 0
        for who, cands in (("king", k_cands), ("challenger", c_cands)):
            for tid, cs in cands.items():
                c = cs[0]
                if not c["valid"]:
                    fool[who][tid] = 0.0
                    rows.append({"ts": ts, "turn_id": tid, "model": who,
                                 "ckpt": state["king_name"] if who == "king"
                                 else state["chal"]["name"],
                                 "dver": state["dver"], "z": c["z"],
                                 "y": c["y"],
                                 "ref": teacher_ref(tid, t_train),
                                 "score": None, "valid": False})
        archive_rows(rows)

        if not k_cands:
            # king server unreachable/misloaded: back off instead of spinning
            status(f"batch={b} king sampling returned NOTHING "
                   f"(errs={k_st['err']}); repinning king and backing off 30s")
            try:
                wait_and_repin_king()
            except Exception as e:
                status(f"king repin failed {type(e).__name__}")
            time.sleep(30)
            continue

        kf_list = list(fool["king"].values())
        kf = sum(kf_list) / max(len(kf_list), 1)
        state["reign_turns"] += len(kf_list)

        # king-vs-teacher agreement (reign metric)
        for tid, cs in k_cands.items():
            c = cs[0]
            if c["valid"]:
                tb = bash_body(t_train[tid][0]["y"])
                gb = bash_body(c["y"])
                state["agree"]["n"] += 1
                state["agree"]["ex"] += 1.0 if gb == tb else 0.0
                state["agree"]["tok"] += token_jaccard(gb, tb)

        # ratchet metric: king fool over first pairs of the reign, fresh judge
        rat = state["ratchet"]
        if not rat["logged"]:
            rat["n"] += len(kf_list)
            rat["f"] += sum(kf_list)
            if rat["n"] >= args.ratchet_n:
                status(f"RATCHET reign={state['reign']} dver={state['dver']} "
                       f"king={state['king_name']} "
                       f"king_fool_fresh_judge={rat['f']/rat['n']:.4f} "
                       f"n={rat['n']}")
                rat["logged"] = True

        cf = None
        if state["chal"]:
            both = [t for t in fool["king"] if t in fool["challenger"]]
            d = state["duel"]
            for t in both:
                di = fool["challenger"][t] - fool["king"][t]
                d["n"] += 1
                d["kf"] += fool["king"][t]
                d["cf"] += fool["challenger"][t]
                d["sd"] += di
                d["sd2"] += di * di
            cf = (sum(fool["challenger"][t] for t in both) / max(len(both), 1))

        ag = state["agree"]
        b_mass = [s["mass"] for s in scores if s]
        b_mass = sum(b_mass) / len(b_mass) if b_mass else float("nan")
        status(f"batch={b} reign={state['reign']} dver={state['dver']} "
               f"king_fool={kf:.4f} "
               f"ch_fool={'%.4f' % cf if cf is not None else '-'} "
               f"duel_n={state['duel']['n'] if state['chal'] else 0} "
               f"valid_k={k_vr:.3f} "
               f"valid_c={'%.3f' % c_vr if c_vr is not None else '-'} "
               f"agree_exact={ag['ex']/max(ag['n'],1):.3f} "
               f"agree_tok={ag['tok']/max(ag['n'],1):.3f} "
               f"errs={k_st['err']} scr_errs={n_scr_err} "
               f"judge_errs={judge.err} judge_miss={judge.miss} "
               f"judge_mass={b_mass:.3f} "
               f"wall={time.time()-t0:.0f}s")
        if b_mass == b_mass and b_mass < 0.5:
            status(f"ALARM: judge answer mass {b_mass:.3f} < 0.5 this batch "
                   f"-- scoring is renormalizing near-zero probabilities; "
                   f"judge {state['dver']} format health degraded")
        save()

        # crown rule: margin floor AND paired z-gate, tested only at fixed
        # checkpoints (n, 2n, 3n, 4n) to avoid sequential-peeking inflation
        if state["chal"]:
            d = state["duel"]
            checkpoints = [args.crown_n * i for i in (1, 2, 3, 4)]
            due = [c for c in checkpoints[d["ck"]:] if d["n"] >= c]
            if due:
                d["ck"] += len(due)
                st_ = duel_stats(d)
                decision = "CONTINUE"
                if (st_["diff"] >= args.crown_eps
                        and st_["z"] >= args.crown_z):
                    decision = "CROWN"
                elif d["n"] >= args.duel_max:
                    decision = "REJECT"
                elif (d["n"] >= args.duel_early_reject and st_["z"] < 0):
                    decision = "REJECT (early: z<0)"
                status(f"DUEL checkpoint: n={st_['n']} "
                       f"king_fool={st_['kf']:.4f} ch_fool={st_['cf']:.4f} "
                       f"diff={st_['diff']:+.4f} paired_se={st_['se']:.4f} "
                       f"z={st_['z']:.2f} decision={decision}")
                if decision == "CROWN":
                    crown(st_)
                elif decision.startswith("REJECT"):
                    reject(st_)
                save()


if __name__ == "__main__":
    main()
