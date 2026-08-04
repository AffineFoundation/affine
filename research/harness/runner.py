"""E-KINGS runner: score a set of miners against the teacher over a turn file.

Usage (on the pod):
  python -m harness.runner --turns /root/data/turns_minicoder.jsonl \
      --miners king-genesis:8001 king-I:8002 ... --n-turns 100 \
      --out /root/results/ekings_run1.jsonl

Assumes the teacher and all listed miners are already serving (see serve.py).
Resumable: skips (turn_id, miner) pairs already present in the output file.
"""

import argparse
import asyncio
import json
import pathlib
import random

import httpx

from .chat import BASH_RE, action_kind, _extract_tool_json
from .client import VllmModel
from .config import RunCfg, TEACHER, ModelCfg, king_cfg
from .terms import miner_terms, teacher_reference


def gold_action(rec: dict) -> str | None:
    """Parse a force-target action from the trajectory's reference_turn."""
    ref = rec.get("reference_turn") or ""
    kind = rec.get("action_kind", "bash")
    if kind == "tool":
        return _extract_tool_json(ref)
    matches = list(BASH_RE.finditer(ref))
    return matches[-1].group(0) if matches else None


async def sample_force_rollouts(miner: VllmModel, prefix, n: int,
                                temperature: float, max_tokens: int,
                                y_gold: str) -> list[tuple[str, str]]:
    """Sample miner thoughts; pin action to the gold trajectory action."""
    raw = await asyncio.gather(*[
        miner.sample(prefix, temperature, max_tokens) for _ in range(n)
    ])
    out = []
    for z, _y in raw:
        out.append((z or "", y_gold))
    return out


def load_turns(path: str, n: int, seed: int = 7):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def turn_id(rec: dict) -> str:
    return f"{rec['traj_id']}:{rec['turn_idx']}"


async def run(args):
    rc = RunCfg()
    if getattr(args, "concurrency", None):
        rc.max_concurrency = args.concurrency
    sem = asyncio.Semaphore(rc.max_concurrency)
    async with httpx.AsyncClient() as http:
        teacher = VllmModel(TEACHER, http, sem)
        miners = []
        for spec in args.miners:
            name, port = spec.rsplit(":", 1)
            if "=" in name:
                mname, repo = name.split("=", 1)
                cfg = ModelCfg(name=mname, repo=repo, port=int(port),
                               family="generic", gpus="?")
            else:
                cfg = king_cfg(name.removeprefix("king-"), int(port), "?")
            miners.append(VllmModel(cfg, http, sem))

        turns = load_turns(args.turns, args.n_turns)
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if outp.exists():
            with open(outp) as f:
                for line in f:
                    r = json.loads(line)
                    done.add((r["turn_id"], r["miner"]))

        # Teacher rollouts are cached per turn so runs in separate waves (or
        # resumed runs) score every miner against identical references.
        ref_path = pathlib.Path(args.ref_cache)
        ref_cache = {}
        if ref_path.exists():
            with open(ref_path) as f:
                for line in f:
                    r = json.loads(line)
                    ref_cache[r["turn_id"]] = r["ref"]
        ref_f = open(ref_path, "a")

        out_f = open(outp, "a")
        for ti, rec in enumerate(turns):
            tid = turn_id(rec)
            prefix = rec["prefix"]
            kind = rec.get("action_kind", "bash")
            tok = action_kind.set(kind)
            try:
                if tid in ref_cache:
                    ref = ref_cache[tid]
                else:
                    try:
                        ref = await teacher_reference(
                            teacher, prefix, rc.n_teacher_samples, rc.temperature,
                            rc.max_thought_tokens, rc.max_action_tokens)
                    except Exception as e:
                        print(f"[{ti}] {tid} teacher failed: {e}")
                        continue
                    ref_f.write(json.dumps({"turn_id": tid, "ref": ref}) + "\n")
                    ref_f.flush()
                if not ref:
                    print(f"[{ti}] {tid} no teacher rollouts, skipping")
                    continue
                causality = sum(r["lp_own"] - r["lp_empty"] for r in ref) / len(ref)

                y_gold = gold_action(rec) if args.force_y_from_ref else None
                if args.force_y_from_ref and not y_gold:
                    print(f"[{ti}] {tid} no gold action in reference_turn, skip")
                    continue

                async def one_miner(m: VllmModel, kind: str = kind,
                                    y_gold: str | None = y_gold):
                    if (tid, m.cfg.name) in done:
                        return None
                    ttok = action_kind.set(kind)
                    try:
                        rollouts = None
                        if y_gold is not None:
                            rollouts = await sample_force_rollouts(
                                m, prefix, rc.n_miner_samples, rc.temperature,
                                rc.max_thought_tokens + rc.max_action_tokens,
                                y_gold)
                        t = await miner_terms(
                            teacher, m, prefix, ref, rc.n_miner_samples,
                            rc.temperature, rc.max_thought_tokens,
                            rc.max_action_tokens, score_bank=args.score_bank,
                            rollouts=rollouts)
                    except Exception as e:
                        return {"turn_id": tid, "miner": m.cfg.name,
                                "valid": False, "error": str(e)[:200]}
                    finally:
                        action_kind.reset(ttok)
                    t.update({"turn_id": tid, "miner": m.cfg.name,
                              "causality": causality,
                              "force_y": bool(y_gold)})
                    return t

                results = await asyncio.gather(*[one_miner(m) for m in miners])
                for r in results:
                    if r is not None:
                        out_f.write(json.dumps(r) + "\n")
                out_f.flush()
                print(f"[{ti + 1}/{len(turns)}] {tid} kind={kind} "
                      f"causality={causality:+.4f} "
                      f"scored {sum(1 for r in results if r)} miners",
                      flush=True)
            finally:
                action_kind.reset(tok)
        out_f.close()
        ref_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--miners", nargs="+", required=True,
                    help="name:port, e.g. king-genesis:8001")
    ap.add_argument("--n-turns", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-cache", required=True,
                    help="jsonl file of cached teacher rollouts, keyed by turn_id")
    ap.add_argument("--score-bank", action="store_true",
                    help="also compute per-pair L2_bank + miner bank_frac")
    ap.add_argument("--force-y-from-ref", action="store_true",
                    help="pin miner action to trajectory reference_turn "
                         "(force-only D_tau2 programmability probe)")
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
