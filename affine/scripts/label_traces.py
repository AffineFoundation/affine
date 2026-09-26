"""Label every rollout in a trace manifest with affine.corpus.labels.

    source .venv/bin/activate
    python affine/scripts/label_traces.py \\
        --manifest https://data.affine.io/traces/manifest.json \\
        --since 2026-09-09T19:00:00Z [--until ISO] \\
        [--policy-prefix king_,teacher_] [--source swelego,swesmith] \\
        --out /tmp/labels.jsonl.gz [--summary] [--procs 4] [--cache-dir DIR]

Streams chunk by chunk: fetch one `traces/chunks/*.jsonl.gz`, label its
envelopes, append one JSON line per rollout (rollout labels + `turns`) to
`--out`. Chunks are selected by the manifest's `created_at`
(`--since` / `--until`, ISO 8601, naive = UTC). `--manifest` may also be a
local manifest file; chunks are then read from `--chunk-dir` (default: the
manifest's directory) by basename.

Reads the public bucket first (`https://data.affine.io`). When the public
path is denied (401/403) and DATA_R2_ENDPOINT / DATA_R2_ACCESS_KEY_ID /
DATA_R2_SECRET_ACCESS_KEY are set, the same keys are read from the R2
bucket (`--r2-bucket`, default affine-data) with boto3. Env values are never
printed. `--summary` prints per-(source, harness) rollout-label tables and
`--summary-by policy` switches the grouping. `--reigns FILE` is a JSON
{king_digest12: reign} map for the `reign` field (the trace lacks it).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affine.corpus.labels import (LABELS_VERSION, format_summary,  # noqa: E402
                                  label_envelope, source_groups_from_toml,
                                  summarize)

PUBLIC_MANIFEST = "https://data.affine.io/traces/manifest.json"
USER_AGENT = f"affine-label-traces/{LABELS_VERSION}"
R2_ENV = ("DATA_R2_ENDPOINT", "DATA_R2_ACCESS_KEY_ID", "DATA_R2_SECRET_ACCESS_KEY")
DEFAULT_SOURCES_TOML = (Path(__file__).resolve().parents[2] / "rollouts"
                        / "rollouts" / "sources.toml")

_R2_CLIENT = None
_WORKER: dict = {}


def _iso(s: str | None) -> str | None:
    """Normalize an ISO timestamp to a UTC `YYYY-MM-DDTHH:MM:SS+00:00`
    string comparable with manifest `created_at`."""
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _r2_get(bucket: str, key: str) -> bytes:
    global _R2_CLIENT
    missing = [k for k in R2_ENV if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"public read denied and {', '.join(missing)} unset")
    if _R2_CLIENT is None:
        _R2_CLIENT = boto3.client(
            "s3", endpoint_url=os.environ["DATA_R2_ENDPOINT"],
            aws_access_key_id=os.environ["DATA_R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["DATA_R2_SECRET_ACCESS_KEY"],
            region_name="auto")
    return _R2_CLIENT.get_object(Bucket=bucket, Key=key)["Body"].read()


def fetch_key(key: str, *, base_url: str | None, chunk_dir: Path | None,
              cache_dir: Path | None, sha256: str | None, r2_bucket: str) -> bytes:
    name = os.path.basename(key)
    if cache_dir is not None:
        cached = cache_dir / name
        if cached.exists():
            body = cached.read_bytes()
            if not sha256 or hashlib.sha256(body).hexdigest() == sha256:
                return body
    if base_url is None:
        body = (chunk_dir / name).read_bytes()
    else:
        try:
            body = _http_get(base_url + key)
        except urllib.error.HTTPError as err:
            if err.code not in (401, 403):
                raise
            body = _r2_get(r2_bucket, key)
    if sha256 and hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError(f"sha256 mismatch for {key}")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / name).write_bytes(body)
    return body


def load_manifest(spec: str) -> tuple[dict, str | None, Path | None]:
    """(manifest, base_url or None, chunk_dir or None)."""
    if spec.startswith(("http://", "https://")):
        base = spec.split("/traces/", 1)[0] + "/"
        return json.loads(_http_get(spec)), base, None
    path = Path(spec)
    return json.loads(path.read_text()), None, path.parent


def _init_worker(cfg: dict) -> None:
    _WORKER.update(cfg)


def label_chunk(chunk: dict) -> tuple[str, list[str], list[str]]:
    """-> (key, json lines, error strings)."""
    cfg = _WORKER
    body = fetch_key(chunk["key"], base_url=cfg["base_url"], chunk_dir=cfg["chunk_dir"],
                     cache_dir=cfg["cache_dir"], sha256=chunk.get("sha256"),
                     r2_bucket=cfg["r2_bucket"])
    prefixes, sources = cfg["policy_prefixes"], cfg["sources"]
    lines, errors = [], []
    with gzip.open(io.BytesIO(body), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            e = json.loads(line)
            pid = str((e.get("policy") or {}).get("id") or "")
            if prefixes and not pid.startswith(prefixes):
                continue
            if sources and e.get("source") not in sources:
                continue
            try:
                rec = label_envelope(e, source_groups=cfg["source_groups"],
                                     reigns=cfg["reigns"])
            except Exception as ex:  # one bad trace must not stop the stream
                errors.append(f"{chunk['key']}:{i} {e.get('rollout_id')}: "
                              f"{type(ex).__name__}: {ex}")
                continue
            rec["chunk_key"] = chunk["key"]
            lines.append(json.dumps(rec, ensure_ascii=False))
    return chunk["key"], lines, errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", default=PUBLIC_MANIFEST)
    ap.add_argument("--since", help="chunk created_at >= (ISO 8601)")
    ap.add_argument("--until", help="chunk created_at <= (ISO 8601)")
    ap.add_argument("--policy-prefix", default="",
                    help="comma list; keep policies whose id starts with one")
    ap.add_argument("--source", default="", help="comma list of sources to keep")
    ap.add_argument("--out", required=True, help="labels .jsonl or .jsonl.gz")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--summary-by", default="source,harness",
                    help="rollout fields to group the summary by")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--cache-dir", help="keep fetched chunks here (sha-checked)")
    ap.add_argument("--chunk-dir", help="local chunk files for a local manifest")
    ap.add_argument("--r2-bucket", default="affine-data")
    ap.add_argument("--sources-toml", default=str(DEFAULT_SOURCES_TOML),
                    help="rollouts sources.toml for env_category ('' = none)")
    ap.add_argument("--reigns", help="JSON file {king_digest12: reign}")
    ap.add_argument("--limit-chunks", type=int, default=0, help="smoke runs")
    args = ap.parse_args()

    manifest, base_url, chunk_dir = load_manifest(args.manifest)
    if args.chunk_dir:
        chunk_dir = Path(args.chunk_dir)
    since, until = _iso(args.since), _iso(args.until)
    chunks = [c for c in manifest["chunks"]
              if (since is None or _iso(c["created_at"]) >= since)
              and (until is None or _iso(c["created_at"]) <= until)]
    chunks.sort(key=lambda c: c["created_at"])
    if args.limit_chunks:
        chunks = chunks[-args.limit_chunks:]
    source_groups = (source_groups_from_toml(args.sources_toml)
                     if args.sources_toml and Path(args.sources_toml).exists() else {})
    reigns = json.loads(Path(args.reigns).read_text()) if args.reigns else {}
    cfg = {
        "base_url": base_url, "chunk_dir": chunk_dir,
        "cache_dir": Path(args.cache_dir) if args.cache_dir else None,
        "r2_bucket": args.r2_bucket,
        "policy_prefixes": tuple(p for p in args.policy_prefix.split(",") if p),
        "sources": frozenset(s for s in args.source.split(",") if s),
        "source_groups": source_groups, "reigns": reigns,
    }
    print(f"manifest {manifest.get('published_at')}: {len(chunks)} chunks selected "
          f"of {len(manifest['chunks'])}; labels_version {LABELS_VERSION}",
          file=sys.stderr)

    opener = gzip.open if args.out.endswith(".gz") else open
    n_rollouts = n_errors = 0
    records_for_summary: list[dict] = []
    group_fields = tuple(f for f in args.summary_by.split(",") if f)
    with Pool(args.procs, initializer=_init_worker, initargs=(cfg,)) as pool, \
            opener(args.out, "wt", encoding="utf-8") as out:
        for done, (key, lines, errors) in enumerate(
                pool.imap_unordered(label_chunk, chunks), 1):
            for line in lines:
                out.write(line + "\n")
            n_rollouts += len(lines)
            n_errors += len(errors)
            for err in errors:
                print("ERROR", err, file=sys.stderr)
            if args.summary:
                for line in lines:
                    rec = json.loads(line)
                    rec.pop("turns", None)
                    records_for_summary.append(rec)
            if done % 50 == 0 or done == len(chunks):
                print(f"{done}/{len(chunks)} chunks, {n_rollouts} rollouts",
                      file=sys.stderr)
    print(f"wrote {n_rollouts} rollouts to {args.out} ({n_errors} label errors)",
          file=sys.stderr)
    if args.summary:
        table = summarize(records_for_summary,
                          key=lambda r: tuple(r.get(f) for f in group_fields))
        print(format_summary(table, key_names=group_fields))


if __name__ == "__main__":
    main()
