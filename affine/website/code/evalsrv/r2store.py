"""R2 checkpoint materialization for the eval pod.

`r2://bucket/prefix/` refs are served by vLLM from a local directory laid
out like an HF cache snapshot:

    $HF_HOME/hub/models--r2--<sha256(ref)[:16]>/snapshots/<model_digest>/

so the engine's disk accounting (keep set, prune, prefetch stall watchdog,
download credit) treats HF and R2 checkpoints identically. A snapshot is
usable only once `.affine_complete` exists: every file listed in the
prefix's signed manifest was downloaded, its size and sha256 matched, and
the manifest's model_digest equals the pinned revision. Anything else is an
IntegrityError — the checkpoint the miner committed to is not what is in
the bucket, which is a rejection, never an infra fault.

Credentials: read-only S3 pair in AFFINE_EVAL_R2_ACCESS_KEY_ID /
AFFINE_EVAL_R2_SECRET_ACCESS_KEY (+ AFFINE_EVAL_R2_ENDPOINT), written to the
pod's .eval_env by the provisioner. No Cloudflare management token ever
reaches a pod.

CLI (used by the engine's watchdogged prefetch child):
    python -m evalsrv.r2store fetch <r2-ref> <model_digest>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from affine import r2, r2protocol as proto

log = logging.getLogger("evalsrv.r2store")

HF_HOME = os.environ.get("HF_HOME", "/root/hf")
COMPLETE_MARKER = ".affine_complete"
MAX_MANIFEST_BYTES = 2 << 20
CHUNK = 8 << 20
DOWNLOAD_WORKERS = int(os.environ.get("AFFINE_R2_DOWNLOAD_WORKERS", "6"))


class IntegrityError(Exception):
    """Bucket content does not match the committed manifest/digest."""


class FetchError(RuntimeError):
    """Could not read the bucket (credentials, network, disk): an infra
    fault — the checkpoint may well be fine, retry later, never burn."""

    def __init__(self, repo: str, message: str):
        super().__init__(message)
        self.repo = repo


def is_r2(repo: str) -> bool:
    return proto.is_r2_ref(repo)


def cache_dir_name(repo: str) -> str:
    if is_r2(repo):
        return "models--r2--" + hashlib.sha256(repo.encode()).hexdigest()[:16]
    return "models--" + repo.replace("/", "--")


def repo_cache_dir(repo: str, hf_home: str | None = None) -> Path:
    return Path(hf_home or HF_HOME) / "hub" / cache_dir_name(repo)


def snapshot_dir(repo: str, revision: str, hf_home: str | None = None) -> Path:
    return repo_cache_dir(repo, hf_home) / "snapshots" / revision


def snapshot_ready(repo: str, revision: str, hf_home: str | None = None) -> bool:
    """HF: the snapshot dir exists (huggingface_hub writes it last). R2: the
    completion marker exists (written only after full verification)."""
    snap = snapshot_dir(repo, revision, hf_home)
    try:
        if is_r2(repo):
            return (snap / COMPLETE_MARKER).is_file()
        return snap.exists()
    except OSError:
        return False


def model_path(repo: str, revision: str | None) -> str:
    """What to hand vLLM / AutoTokenizer: the local snapshot for r2 refs,
    the repo id itself for HF."""
    if is_r2(repo) and revision:
        return str(snapshot_dir(repo, revision))
    return repo


def client():
    endpoint = (os.environ.get("AFFINE_EVAL_R2_ENDPOINT")
                or os.environ.get("R2_ENDPOINT") or "")
    ak = (os.environ.get("AFFINE_EVAL_R2_ACCESS_KEY_ID")
          or os.environ.get("R2_ACCESS_KEY_ID") or "")
    sk = (os.environ.get("AFFINE_EVAL_R2_SECRET_ACCESS_KEY")
          or os.environ.get("R2_SECRET_ACCESS_KEY") or "")
    if not (endpoint and ak and sk):
        raise RuntimeError("AFFINE_EVAL_R2_ENDPOINT / _ACCESS_KEY_ID / "
                           "_SECRET_ACCESS_KEY are not set on this pod")
    return r2.s3_client(endpoint, ak, sk, read_timeout=300.0)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(s3, bucket: str, key: str, dest: Path, size: int,
              sha256: str) -> None:
    tmp = dest.with_name(dest.name + ".incomplete")
    h = hashlib.sha256()
    n = 0
    resp = s3.get_object(Bucket=bucket, Key=key)
    clen = int(resp.get("ContentLength", -1))
    if clen != size:
        # The object in the bucket is not the one the manifest describes.
        raise IntegrityError(f"{dest.name}: object is {clen} bytes, manifest says {size}")
    with open(tmp, "wb") as f:
        for chunk in resp["Body"].iter_chunks(CHUNK):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    if n != size:
        # Object size was right, the stream was not: a dropped transfer.
        tmp.unlink(missing_ok=True)
        raise FetchError(key, f"{dest.name}: short read {n}/{size} bytes")
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise IntegrityError(f"{dest.name}: sha256 mismatch against manifest")
    tmp.replace(dest)


def fetch_snapshot(repo: str, revision: str, s3=None,
                   workers: int = DOWNLOAD_WORKERS) -> Path:
    """Materialize + verify an r2 ref at the pinned model_digest. Idempotent:
    a complete snapshot returns immediately; a partial one resumes (files
    already present are re-hashed, never trusted by size alone)."""
    if not is_r2(repo):
        raise ValueError(f"not an r2 ref: {repo}")
    bucket, prefix = proto.parse_r2_ref(repo)
    snap = snapshot_dir(repo, revision)
    if (snap / COMPLETE_MARKER).is_file():
        return snap
    try:
        return _fetch(repo, revision, bucket, prefix, snap, s3, workers)
    except IntegrityError:
        raise
    except ValueError as e:
        # get_bytes size cap: a manifest that grew past the cap after ready.
        raise IntegrityError(str(e)) from e
    except FetchError as e:
        raise FetchError(repo, str(e)) from e
    except ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            # A file the manifest lists is gone from the prefix: the bucket
            # no longer holds what was committed.
            raise IntegrityError(f"object missing from prefix: {e}") from e
        raise FetchError(repo, f"r2 {code or 'error'}: {e}") from e
    except (BotoCoreError, OSError, RuntimeError) as e:
        raise FetchError(repo, f"{type(e).__name__}: {e}") from e


def _fetch(repo: str, revision: str, bucket: str, prefix: str, snap: Path,
           s3, workers: int) -> Path:
    s3 = s3 or client()
    t0 = time.time()
    raw = r2.get_bytes(s3, bucket, prefix + "manifest.json", MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(raw)
        proto.validate_manifest_shape(manifest)
    except (ValueError, TypeError) as e:
        raise IntegrityError(f"manifest.json invalid: {e}") from e
    if manifest["model_digest"] != revision:
        raise IntegrityError(
            f"manifest model_digest {manifest['model_digest'][:12]} != pinned "
            f"{revision[:12]} (content changed after the ready signal)")
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "manifest.json").write_bytes(raw)

    todo: list[dict] = []
    for f in manifest["files"]:
        dest = snap / f["path"]
        if dest.is_file() and dest.stat().st_size == int(f["size"]) \
                and _sha256_file(dest) == f["sha256"]:
            continue
        todo.append(f)
    total = sum(int(f["size"]) for f in todo)
    log.info("fetching %s@%s: %d/%d files, %.1f GB", repo, revision[:12],
             len(todo), len(manifest["files"]), total / 1e9)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_download, s3, bucket, prefix + f["path"],
                            snap / f["path"], int(f["size"]), f["sha256"]): f
                for f in todo}
        for fut in as_completed(futs):
            fut.result()  # IntegrityError / FetchError / botocore errors propagate
    (snap / COMPLETE_MARKER).write_text(json.dumps({
        "repo": repo, "model_digest": revision,
        "files": len(manifest["files"]),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    log.info("snapshot %s@%s complete in %.0fs", repo, revision[:12],
             time.time() - t0)
    return snap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R2 checkpoint fetch (eval pod)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("repo")
    f.add_argument("revision")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        print(fetch_snapshot(args.repo, args.revision))
        return 0
    except IntegrityError as e:
        print(f"INTEGRITY: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
