"""Judge-baseline hardening: run a real LLM judge on the SAME turns S sees.

Apples-to-apples: for each king, feed the judge the turn context and that king's
actual stored step (thought + bash action), get a 0-10 quality score, average per
king, correlate with swe-rebench. Compare to S (thought-KL) on the same kings.

Two judge models (production GLM-5.2 for fidelity + gpt-4.1-mini for family
robustness) × two prompt variants (holistic, reference-anchored) show the judge's
weak correlation isn't a strawman or a single-prompt artifact.

Cost control: results cached in results/judge_same_turns_calls.jsonl; reruns free.
Usage: source .venv/bin/activate && OPENROUTER_API_KEY=... python scripts/judge_same_turns.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

import httpx
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rt3_calibration import PAIR_FILES, flat, load_pairs, mix  # noqa: E402

RESULTS = ROOT / "results"
CACHE = RESULTS / "judge_same_turns_calls.jsonl"
CORPUS = ROOT / "data" / "turns_minicoder.jsonl"
REF = RESULTS / "ref_minicoder.jsonl"
SWE = {"genesis": 58.2, "XCIX": 39.8, "I": 38.4, "II": 37.2, "VIII": 36.2,
       "XLI": 34.2, "VII": 33.2, "C": 32.0, "XLVI": 13.2, "CI": 12.4, "LI": 11.6}
KINGS = list(SWE)
N_TURNS = 50
MODELS = ["z-ai/glm-5.2", "openai/gpt-4.1-mini"]
VARIANTS = ["holistic", "anchored"]
CONCURRENCY = 12


def stored_steps() -> dict[str, dict[str, dict]]:
    """by[king][turn_id] = {'z': thought, 'y': action} (first stored sample)."""
    by: dict[str, dict[str, dict]] = {k: {} for k in [f"king-{s}" for s in KINGS]}
    for pf in PAIR_FILES:
        if not pf.exists():
            continue
        for line in open(pf):
            r = json.loads(line)
            if r.get("miner") not in by or not (r.get("valid") and "pairs" in r):
                continue
            if r["turn_id"] in by[r["miner"]]:
                continue
            p = r["pairs"][0]
            by[r["miner"]][r["turn_id"]] = {"z": p.get("z_a", ""), "y": p.get("y_a", "")}
    return by


def load_prefix_and_ref() -> tuple[dict[str, str], dict[str, str]]:
    ref_action: dict[str, str] = {}
    if REF.exists():
        for line in open(REF):
            r = json.loads(line)
            rr = r.get("ref") or []
            if rr:
                ref_action[r["turn_id"]] = rr[0].get("y", "")
    prefix: dict[str, str] = {}
    wanted = set()
    steps = stored_steps()
    for turns in steps.values():
        wanted.update(turns)
    for line in open(CORPUS):
        r = json.loads(line)
        tid = f"{r['traj_id']}:{r['turn_idx']}"
        if tid in wanted:
            pfx = r.get("prefix", [])
            prefix[tid] = "\n".join(str(x) for x in pfx)[-4000:]
    return prefix, ref_action


def build_prompt(variant: str, ctx: str, step: dict, ref_y: str) -> str:
    base = (
        "You are grading a single step taken by a coding agent working through a "
        "software task. Below is the recent context, then the agent's next step "
        "(its private reasoning and the shell command it runs).\n\n"
        f"=== CONTEXT (truncated) ===\n{ctx}\n\n"
        f"=== AGENT STEP ===\nREASONING: {step['z'][:1500]}\nCOMMAND: {step['y'][:600]}\n\n"
    )
    if variant == "anchored" and ref_y:
        base += (f"=== REFERENCE (a strong model's command for this step) ===\n{ref_y[:600]}\n\n"
                 "Rate 0-10 how well the agent's step advances the task toward the same "
                 "goal as the reference (10 = as good or better; 0 = useless/harmful).")
    else:
        base += ("Rate 0-10 how good this step is at advancing the task "
                 "(10 = expert-level, correct and efficient; 0 = useless/harmful).")
    base += " Reply with ONLY the integer score."
    return base


def parse_score(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    if not m:
        return None
    v = float(m.group())
    return max(0.0, min(10.0, v))


def load_cache() -> dict[tuple, float]:
    c: dict[tuple, float] = {}
    if CACHE.exists():
        for line in open(CACHE):
            r = json.loads(line)
            if r.get("score") is not None:
                c[(r["model"], r["variant"], r["king"], r["turn_id"])] = r["score"]
    return c


async def call_judge(client, key, model, prompt) -> float | None:
    for attempt in range(4):
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 2000, "temperature": 0.0},
                timeout=120.0)
            if resp.status_code != 200:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            return parse_score(resp.json()["choices"][0]["message"]["content"])
        except Exception:
            await asyncio.sleep(2 * (attempt + 1))
    return None


async def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("no OPENROUTER_API_KEY"); return
    steps = stored_steps()
    prefix, ref_action = load_prefix_and_ref()
    cache = load_cache()
    turn_ids = {k: sorted(steps[f"king-{k}"])[:N_TURNS] for k in KINGS}

    jobs = []
    for model in MODELS:
        for variant in VARIANTS:
            for k in KINGS:
                for tid in turn_ids[k]:
                    if (model, variant, k, tid) in cache:
                        continue
                    if tid not in prefix:
                        continue
                    jobs.append((model, variant, k, tid))
    print(f"{len(jobs)} judge calls to make ({len(cache)} cached)")

    sem = asyncio.Semaphore(CONCURRENCY)
    out_f = open(CACHE, "a")
    lock = asyncio.Lock()
    done = [0]

    async with httpx.AsyncClient() as client:
        async def one(model, variant, k, tid):
            async with sem:
                step = steps[f"king-{k}"][tid]
                prompt = build_prompt(variant, prefix.get(tid, ""), step, ref_action.get(tid, ""))
                score = await call_judge(client, key, model, prompt)
            async with lock:
                rec = {"model": model, "variant": variant, "king": k,
                       "turn_id": tid, "score": score}
                out_f.write(json.dumps(rec) + "\n"); out_f.flush()
                if score is not None:
                    cache[(model, variant, k, tid)] = score
                done[0] += 1
                if done[0] % 50 == 0:
                    print(f"  {done[0]}/{len(jobs)}", flush=True)
        await asyncio.gather(*[one(*j) for j in jobs])
    out_f.close()

    by = load_pairs()
    S = {k: st.mean(mix(p, "l1lift") for p in flat(by[f"king-{k}"])) for k in KINGS}
    print("\n== judge-on-same-turns vs swe (per-king mean), compared to S ==")
    print(f"{'model':22s} {'variant':9s} {'n_kings':>7s} {'ρ(judge,swe)':>13s} {'coverage':>9s}")
    for model in MODELS:
        for variant in VARIANTS:
            jm, sw = [], []
            for k in KINGS:
                vals = [cache[(model, variant, k, tid)] for tid in turn_ids[k]
                        if (model, variant, k, tid) in cache]
                if len(vals) >= 15:
                    jm.append(st.mean(vals)); sw.append(SWE[k])
            if len(jm) >= 4:
                rho = stats.spearmanr(jm, sw)[0]
                cov = st.mean([len([1 for tid in turn_ids[k]
                              if (model, variant, k, tid) in cache]) for k in KINGS])
                print(f"{model:22s} {variant:9s} {len(jm):7d} {rho:+13.3f} {cov:8.0f}")
    kk = [k for k in KINGS]
    rho_s = stats.spearmanr([S[k] for k in kk], [SWE[k] for k in kk])[0]
    print(f"\nS (thought-KL) vs swe on these {len(kk)} kings: ρ={rho_s:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
