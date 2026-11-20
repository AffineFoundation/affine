#!/usr/bin/env python3
"""Box-side Generator intake sync: atomically enrich closed-epoch intake with stable IDs and licenses.

Only the original {problem, answer, source_id, prior_p} values and their leading
order are preserved. New fields are task_id, license, king. A valid companion
.done makes the epoch idempotent; legacy JSONL is replaced only with --backfill.
This tool reads public HTTP data and writes local files; it has no deployment,
chain, credential or remote-delete capability.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import urllib.request

BASE = "https://affine.io/generator"
OUT = "/home/box/sn120-scratch/generator"
POOL_MAP = "/home/box/sn120-scratch/generator/ref/pool-license-map-v1.jsonl.gz"
UA = {"User-Agent": "sn120-generator-intake", "Cache-Control": "no-cache"}
KEYS = ("problem", "answer", "source_id", "prior_p")
FIELDS = KEYS + ("task_id", "license", "king")
EPOCH = re.compile(r"[0-9]{8}T[0-9]{4}Z\Z")


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def clean(line):
    d = json.loads(line)
    assert isinstance(d["problem"], str) and isinstance(d["answer"], str) and isinstance(d["source_id"], str)
    p = float(d["prior_p"])
    assert math.isfinite(p) and 0.0 <= p <= 1.0
    return json.dumps({"problem": d["problem"], "answer": d["answer"], "source_id": d["source_id"], "prior_p": p}, ensure_ascii=False)


def read_rows(body):
    return [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]


def load_pool_map(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return {}
    result = {}
    for row in rows:
        task_id = row["id"]
        if task_id in result:
            raise ValueError("duplicate pool-map id")
        result[task_id] = {"source_id": row["source_id"], "license": row.get("license")}
    return result


def enrich(intake, accepted, pool_map):
    if len(intake) != len(accepted):
        raise ValueError(f"intake/accepted count mismatch ({len(intake)} != {len(accepted)})")
    lines = []
    for original, claim in zip(intake, accepted):
        base = json.loads(clean(json.dumps(original, ensure_ascii=False)))
        if "task_id" in original and original["task_id"] != claim["task_id"]:
            raise ValueError("task_id mismatch")
        task_id = original.get("task_id", claim["task_id"])
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("invalid accepted task_id")
        mapped = pool_map.get(task_id)
        if mapped is not None and mapped["source_id"] != original["source_id"]:
            raise ValueError("pool-map source_id mismatch")
        base["task_id"] = task_id
        base["license"] = original.get("license", mapped.get("license") if mapped is not None else None)
        base["king"] = original.get("king", claim["king"])
        lines.append(json.dumps(base, ensure_ascii=False))
    return lines


def atomic_write(path, body, mode=0o644):
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_done(ep, jsonl_path, done_path):
    body = open(jsonl_path, "rb").read()
    lines = sum(bool(line.strip()) for line in body.decode("utf-8").splitlines())
    record = {"epoch": ep, "lines": lines, "sha256": hashlib.sha256(body).hexdigest(), "format": "generator-intake-v2"}
    atomic_write(done_path, (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


def valid_done(ep, jsonl_path, done_path):
    try:
        body = open(jsonl_path, "rb").read()
        with open(done_path, "rb") as stream:
            if len(stream.read()) > 65536:
                return False
        with open(done_path, encoding="utf-8") as stream:
            record = json.load(stream)
        lines = sum(bool(line.strip()) for line in body.decode("utf-8").splitlines())
        return record == {"epoch": ep, "lines": lines, "sha256": hashlib.sha256(body).hexdigest(), "format": "generator-intake-v2"}
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def complete_v2(path):
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not all(field in row for field in FIELDS) or list(row)[:4] != list(KEYS):
                    return False
        return True
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def sync_epoch(ep, *, out=OUT, base=BASE, pool_map=POOL_MAP, backfill=False, fetch=None):
    final = os.path.join(out, f"accepted-{ep}.jsonl")
    done = os.path.join(out, f"accepted-{ep}.done")
    os.makedirs(out, exist_ok=True)
    fetch = get if fetch is None else fetch
    if valid_done(ep, final, done):
        return "exists"
    if os.path.exists(final):
        if complete_v2(final):
            write_done(ep, final, done)
            return "done repaired"
        if not backfill:
            return "legacy (run --backfill)"
    body = fetch(f"{base}/epochs/{ep}/intake.jsonl")
    try:
        want = fetch(f"{base}/epochs/{ep}/intake.sha256").decode().split()[0]
        if hashlib.sha256(body).hexdigest() != want:
            return "sha-mismatch (skip; retry next run)"
    except Exception:
        return "no sha yet (skip)"
    accepted_body = fetch(f"{base}/epochs/{ep}/accepted.jsonl")
    intake, accepted = read_rows(body), read_rows(accepted_body)
    lines = enrich(intake, accepted, load_pool_map(pool_map))
    raw = "".join(line + "\n" for line in lines).encode("utf-8")
    atomic_write(final, raw)
    write_done(ep, final, done)
    return f"wrote {len(lines)}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--pool-map", default=POOL_MAP)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--epoch")
    return parser.parse_args(argv)


def main(argv=None, *, fetch=get):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.epoch:
        if not EPOCH.fullmatch(args.epoch):
            print("invalid --epoch", file=sys.stderr)
            return 2
        eps = [args.epoch]
    else:
        try:
            state = json.loads(fetch(f"{args.base}/state.json"))
        except Exception as e:
            print("state.json unavailable:", type(e).__name__)
            return 0
        eps = [e["epoch"] if isinstance(e, dict) else e for e in state.get("published_epochs", [])]
    for ep in eps:
        if not isinstance(ep, str) or not EPOCH.fullmatch(ep):
            print(ep, "invalid epoch (skip)")
            continue
        try:
            print(ep, sync_epoch(ep, out=args.out, base=args.base, pool_map=args.pool_map, backfill=args.backfill, fetch=fetch))
        except Exception as e:
            print(ep, "error", type(e).__name__, str(e)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
