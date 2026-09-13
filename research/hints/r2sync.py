#!/usr/bin/env python
"""Upload a run directory to R2 `affine-data` under research/hints/<run_id>/.

Refuses any key outside `research/hints/`: the bucket-scoped DATA_R2_*
credential can also write the production corpus prefixes (traces/, corpus/,
views/, turns/), and this tool must never touch them. Files already present
with the same size and sha256 are skipped; a manifest.json (inputs, code
commit, models, seeds, cost) and a README are written next to the data.

  python r2sync.py --run-dir RUN --run-id e1-20260912a [--manifest extra.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config

BUCKET = "affine-data"
PREFIX = "research/hints/"
FORBIDDEN = ("traces/", "corpus/", "views/", "turns/")
GZIP_EXT = (".jsonl",)
README = """# Hinted-teacher explorations — run {run_id}

Research artifacts only. Nothing here is read by the validator, the eval pod,
the fold or the datagen pods. See manifest.json for inputs, code commit,
model revisions, seeds and cost.

Files
- turns.jsonl.gz        the probe turn set (prefix x, recorded reply, hindsight transcript, pivot text, stored duel rollouts)
- hints.jsonl.gz        every hint: generator, level, text, grounding gate, future-leak check, cost, latency
- hint_calls.jsonl.gz   every hint-generator call: exact messages sent + raw model text
- results.jsonl.gz      every teacher sample (raw text + split), every echo, every miner thought, per turn / condition
- analysis/             tables.md / tables.csv / tables.json / turn_conditions.jsonl (per-turn measurements)
- ledger.jsonl          GPU pods rented / released for this round, $/h, hours
- manifest.json         this run's inputs and settings

Re-run
1. Rent a teacher box: `python research/hints/pods.py rent --type h200-1x && python research/hints/pods.py wait`
   (vLLM 0.28.0, Qwen/Qwen3.8-27B, max_model_len 131072, echo-cache plugin; see pods.py).
2. Turn set: `python research/hints/turnset.py --out turns.jsonl --seed {seed}` (needs the epoch index,
   view chunks from data.affine.io, the trace mirror and the king-review side-tables; or reuse turns.jsonl.gz here).
3. Hints: `python research/hints/gen_hints.py --turns turns.jsonl --run-dir RUN --generators deepseek,self,pivot`
4. Probe: `python research/hints/run_probe.py --turns turns.jsonl --hints RUN/hints.jsonl --run-dir RUN --king`
5. Tables: `python research/hints/analyze.py --run-dir RUN`
6. Upload: `python research/hints/r2sync.py --run-dir RUN --run-id {run_id}`
Contract knobs used: temperature 0.8, max tokens 1792, k = 3 refs, tau 0.03, band_c 2, band_floor 0.002,
forfeit −0.1 — the live [duel] values on the day of the run; the scoring rule itself was not changed.
"""


def client():
    return boto3.client(
        "s3", endpoint_url=os.environ["DATA_R2_ENDPOINT"],
        aws_access_key_id=os.environ["DATA_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["DATA_R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
        region_name="auto")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(key: str) -> None:
    if not key.startswith(PREFIX) or any(key.startswith(p) for p in FORBIDDEN) or ".." in key:
        raise SystemExit(f"refusing to write outside {PREFIX}: {key}")


def stage(run_dir: Path, staging: Path) -> list[Path]:
    """Copy the run into a staging dir, gzipping the big JSONL files."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    out = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file() or p.suffix in (".log", ".tmp"):
            continue
        rel = p.relative_to(run_dir)
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in GZIP_EXT:
            dst = dst.with_suffix(dst.suffix + ".gz")
            with open(p, "rb") as src, gzip.open(dst, "wb", compresslevel=6) as gz:
                shutil.copyfileobj(src, gz)
        else:
            shutil.copy2(p, dst)
        out.append(dst)
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", help="JSON file with extra manifest fields (inputs, cost, notes)")
    ap.add_argument("--turns", help="turns.jsonl to include (copied into the run)")
    ap.add_argument("--ledger", help="pods ledger to include")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if args.turns and not (run_dir / "turns.jsonl").exists():
        shutil.copy2(args.turns, run_dir / "turns.jsonl")
    if args.ledger and Path(args.ledger).exists():
        shutil.copy2(args.ledger, run_dir / "ledger.jsonl")
    extra = json.load(open(args.manifest)) if args.manifest else {}
    manifest = {
        "run_id": args.run_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": git_commit(), "teacher": "Qwen/Qwen3.8-27B", "vllm_version": "0.28.0",
        "seed": args.seed, "files": {}, **extra,
    }
    staging = run_dir.parent / f"{run_dir.name}.staging"
    files = stage(run_dir, staging)
    (staging / "README").write_text(README.format(run_id=args.run_id, seed=args.seed))
    files.append(staging / "README")
    for p in files:
        manifest["files"][str(p.relative_to(staging))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    files.append(staging / "manifest.json")
    s3 = None if args.dry_run else client()
    for p in files:
        key = f"{PREFIX}{args.run_id}/{p.relative_to(staging)}"
        guard(key)
        if args.dry_run:
            print(f"would put {key} ({p.stat().st_size} B)")
            continue
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] == p.stat().st_size and head.get("Metadata", {}).get("sha256") == manifest["files"].get(str(p.relative_to(staging)), {}).get("sha256"):
                print(f"skip {key} (unchanged)")
                continue
        except s3.exceptions.ClientError:
            pass
        s3.upload_file(str(p), BUCKET, key, ExtraArgs={"Metadata": {"sha256": manifest["files"].get(str(p.relative_to(staging)), {}).get("sha256", "")}})
        print(f"put {key} ({p.stat().st_size} B)")
    print(f"manifest: {PREFIX}{args.run_id}/manifest.json", file=sys.stderr)


if __name__ == "__main__":
    main()
