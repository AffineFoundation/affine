"""Engy acceptance test: replay stored duel echoes through their endpoint.

For a sample of turns from a stored wvk-10 duel artifact, rebuild the exact
production echo prompts (evalsrv.chat rendering + tokenizer offsets), send
them to Engy's qwen3.8-27b-bf16 as pre-tokenized echo requests, and compare
per-byte logprobs against the values our vLLM fleet recorded at duel time.

Also counts spans that exceed Engy's 1,024-token per-request scoring cap.

Usage: python research/scripts/engy_parity.py [--turns 12] [--artifact PATH]
Writes research/results/engy_parity.{json,txt}.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import statistics
import sys
from pathlib import Path

import httpx
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "affine"))

from affine.config import load_config          # noqa: E402
from evalsrv.chat import force_text, thought_text  # noqa: E402
from evalsrv.corpus import CorpusSync          # noqa: E402
from evalsrv.dueling import sample_slice, turn_id  # noqa: E402

BASE = "https://api.engy.ai/v1"
KEY = "sk-engy-8dCc2yM9s2cGzoBFrEAxXuXYZtj1AX2JXxAHaYxhPj8"
MODEL = "qwen3.8-27b-bf16"
REPO = "Qwen/Qwen3.8-27B"
REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"  # Engy's pinned revision
SPAN_CAP = 1024
EMPTY = ""

REPO_ROOT = Path(__file__).resolve().parents[2]
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)


def spanize(full: str, span_start_byte_text: int, span_text: str):
    """(token_ids, n_prompt_tokens, n_span_tokens, span_bytes) for an echo."""
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    n_prompt = sum(1 for s, _ in enc["offset_mapping"]
                   if s < span_start_byte_text)
    return ids, n_prompt, len(ids) - n_prompt, len(span_text.encode())


async def engy_echo(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                    ids: list[int], n_prompt: int) -> list[float]:
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(
                    f"{BASE}/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json={"model": MODEL, "prompt": ids, "max_tokens": 1,
                          "temperature": 0, "echo": True, "logprobs": 1,
                          "logprob_start_len": n_prompt},
                    timeout=120.0)
                r.raise_for_status()
                d = r.json()
                return d["choices"][0]["logprobs"]["token_logprobs"]
            except Exception as e:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


async def score_term(client, sem, full: str, span_start: int, span: str,
                     caps: list) -> float | None:
    ids, n_prompt, n_span, n_bytes = spanize(full, span_start, span)
    # Engy/SGLang returns null for the token AT logprob_start_len (no logits
    # before it), so start one token early and drop that first null: the
    # remaining entries are exactly our span, matching the fleet's echo.
    if n_span + 1 > SPAN_CAP:
        caps.append(n_span)
        return None
    lps = await engy_echo(client, sem, ids, max(n_prompt - 1, 0))
    if len(lps) != n_span + 1 or lps[0] is not None:
        return None
    vals = [x for x in lps[1:] if x is not None]
    if len(vals) != n_span:
        return None
    return sum(vals) / max(n_bytes, 1)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()

    art_path = args.artifact or sorted(
        (REPO_ROOT / "affine/state/evals").glob("chal-*.json.gz"))[-1]
    art = json.load(gzip.open(art_path))
    print(f"artifact: {art_path}")
    sl = art["slice"]
    print(f"slice: seed={sl['seed']} n={sl['n']} epoch={sl['corpus_epoch']}")

    cfg = load_config()
    corpus = CorpusSync(cfg.dataset.corpus_base_url, cfg.dataset.manifest_key,
                        REPO_ROOT / "affine/state/corpus_cache",
                        lazy_chunks=True)
    if corpus.info()["manifest_sha256"] != sl["manifest_sha256"]:
        print("WARNING: local corpus manifest differs from artifact slice; "
              "prefixes may not replay")
    rows = corpus.load_index_rows()
    picked = sample_slice(rows, sl["n"], sl["seed"])

    # Pick valid rows spread over both sides.
    want = []
    for side in ("king_rows", "challenger_rows"):
        good = [r for r in art[side] if r.get("valid") and r.get("pairs")]
        step = max(1, len(good) // args.turns)
        want += [(side, r) for r in good[::step][:args.turns]]
    need_tids = {r["turn_id"] for _, r in want}

    # Materialize only the turns we need.
    by_tid = {}
    for idx_row in picked:
        t = corpus.materialize_turns([idx_row])[0]
        tid = turn_id(t)
        if tid in need_tids:
            by_tid[tid] = t["prefix"]
        if len(by_tid) == len(need_tids):
            break
    print(f"materialized {len(by_tid)}/{len(need_tids)} needed turns")

    trefs = art["teacher_refs"]
    if isinstance(trefs, dict):
        refs_by_tid = trefs
    else:
        refs_by_tid = {tid: (trefs[i] if i < len(trefs) else None)
                       for i, tid in enumerate(art["turn_ids"])}

    sem = asyncio.Semaphore(16)
    caps: list[int] = []
    results = []

    async with httpx.AsyncClient() as client:
        tasks = []
        meta = []

        def add(name, stored, full, span_start, span, tid, side):
            if stored is None:
                return
            tasks.append(score_term(client, sem, full, span_start, span, caps))
            meta.append((tid, side, name, stored))

        for side, row in want:
            tid = row["turn_id"]
            prefix = by_tid.get(tid)
            if prefix is None:
                continue
            pair = row["pairs"][0]
            z_a, y_a = pair["z_a"], pair["y_a"]
            refs = refs_by_tid.get(tid)
            ref0 = None
            if refs:
                ref0 = refs[0] if isinstance(refs, list) else None
            # Miner-side terms.
            if ref0:
                y_c, z_c = ref0["y"], ref0["z"]
                fx = force_text(REPO, REV, prefix, z_a, y_c)
                add("lpC_yc_za", pair.get("lpC_yc_za"), fx,
                    len(fx) - len(y_c), y_c, tid, side)
                fo = force_text(REPO, REV, prefix, z_c, y_c)
                add("lpC_yc_zc", pair.get("lpC_yc_zc"), fo,
                    len(fo) - len(y_c), y_c, tid, side)
                fe = force_text(REPO, REV, prefix, EMPTY, y_c)
                add("lpC_yc_e", pair.get("lpC_yc_e"), fe,
                    len(fe) - len(y_c), y_c, tid, side)
                ft = thought_text(REPO, REV, prefix, z_c)
                add("lpC_zc_x", pair.get("lpC_zc_x"), ft,
                    len(ft) - len(z_c), z_c, tid, side)
            fb = force_text(REPO, REV, prefix, z_a, y_a)
            add("lpC_ya_za", pair.get("lpC_ya_za"), fb,
                len(fb) - len(y_a), y_a, tid, side)
            fbe = force_text(REPO, REV, prefix, EMPTY, y_a)
            add("lpC_ya_e", pair.get("lpC_ya_e"), fbe,
                len(fbe) - len(y_a), y_a, tid, side)
            fg = thought_text(REPO, REV, prefix, z_a)
            add("lpC_za_x", pair.get("lpC_za_x"), fg,
                len(fg) - len(z_a), z_a, tid, side)

        print(f"scoring {len(tasks)} echoes through Engy "
              f"(concurrency 16)...")
        got = await asyncio.gather(*tasks, return_exceptions=True)

    errs = 0
    for (tid, side, name, stored), val in zip(meta, got):
        if isinstance(val, Exception):
            errs += 1
            print(f"ERR {side} {name} {tid[:40]}: {type(val).__name__}: "
                  f"{str(val)[:100]}")
            continue
        if val is None:
            continue
        results.append({"turn_id": tid, "side": side, "term": name,
                        "stored": stored, "engy": val,
                        "dev": abs(val - stored)})

    devs = [r["dev"] for r in results]
    out = {
        "artifact": str(art_path),
        "n_scored": len(results),
        "n_span_capped": len(caps),
        "capped_span_tokens": sorted(caps)[-5:],
        "n_errors": errs,
        "dev_per_byte": {
            "median": statistics.median(devs) if devs else None,
            "p90": (sorted(devs)[int(0.9 * len(devs))] if devs else None),
            "max": max(devs) if devs else None,
        },
        "per_term": {},
        "rows": results,
    }
    for term in sorted({r["term"] for r in results}):
        td = [r["dev"] for r in results if r["term"] == term]
        out["per_term"][term] = {"n": len(td),
                                 "median": statistics.median(td),
                                 "max": max(td)}

    res_dir = REPO_ROOT / "research/results"
    (res_dir / "engy_parity.json").write_text(json.dumps(out, indent=1))
    lines = [
        f"Engy parity replay — {MODEL} vs stored fleet echoes",
        f"artifact: {Path(str(art_path)).name}   n_scored={len(results)}   "
        f"span-capped(>{SPAN_CAP} tok)={len(caps)}   errors={errs}",
        f"abs per-byte deviation: median={out['dev_per_byte']['median']:.6f} "
        f"p90={out['dev_per_byte']['p90']:.6f} "
        f"max={out['dev_per_byte']['max']:.6f}"
        if devs else "no results",
        "",
        "term          n   median-dev   max-dev",
    ]
    for term, s in out["per_term"].items():
        lines.append(f"{term:12s} {s['n']:3d}   {s['median']:.6f}    "
                     f"{s['max']:.6f}")
    txt = "\n".join(lines)
    (res_dir / "engy_parity.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    asyncio.run(main())
