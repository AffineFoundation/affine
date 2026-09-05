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
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from affine import r2, r2protocol as proto

log = logging.getLogger("evalsrv.r2store")

HF_HOME = os.environ.get("HF_HOME", "/root/hf")
COMPLETE_MARKER = ".affine_complete"
MAX_MANIFEST_BYTES = 2 << 20
CHUNK = 8 << 20
PART = 128 << 20           # ranged-GET unit; a stall costs at most one part
PART_RETRIES = 8
# 8 -> 16 (2026-09-05): one R2 stream from the eval pod runs ~50 MB/s (RTT
# bound); the NIC is 10 Gbps and the local-disk cache writes at 3 GB/s, so
# the stream count was the ceiling once HF_HOME left the encrypted volume.
DOWNLOAD_WORKERS = int(os.environ.get("AFFINE_R2_DOWNLOAD_WORKERS", "16"))
# Objects downloaded concurrently (each split into DOWNLOAD_WORKERS ranges).
FILE_WORKERS = int(os.environ.get("AFFINE_R2_FILE_WORKERS", "3"))


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
    # 20 s (was 60): a healthy stream delivers an 8 MB chunk in well under a
    # second, so a socket that is silent for 20 s is dead. Stalled parts were
    # the 17 MB/s shards (2026-09-05: a 50 GB file took 48 min) — each stall
    # idled a worker for the full timeout before the part was retried. Still
    # far inside the engine's 180 s no-progress watchdog.
    return r2.s3_client(endpoint, ak, sk, read_timeout=20.0)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parts(size: int) -> list[tuple[int, int]]:
    return [(a, min(a + PART, size)) for a in range(0, max(size, 1), PART)] \
        if size > 0 else [(0, 0)]


def _load_done(sidecar: Path) -> set[int]:
    try:
        return set(json.loads(sidecar.read_text()))
    except (OSError, ValueError):
        return set()


def incomplete_bytes(path: Path) -> int | None:
    """Bytes actually landed in a ranged `.incomplete` file, from its
    `.parts` sidecar. None when the file is not one of ours. The engine's
    stall watchdog needs this: the file is preallocated to full size, and
    the pod filesystem (fuseblk) does not report allocated blocks."""
    if not path.name.endswith(".incomplete"):
        return None
    sidecar = path.with_name(path.name[:-len(".incomplete")] + ".parts")
    if not sidecar.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    parts = _parts(size)
    return sum(b - a for i in _load_done(sidecar)
               if 0 <= i < len(parts) for a, b in (parts[i],))


def _fetch_part(s3, bucket: str, key: str, fd: int, a: int, b: int) -> None:
    """One ranged GET written in place. Retries with a fresh connection on
    any transport error; a stall inside a part therefore costs one part,
    not the file (found live 2026-09-04: single-stream GETs of 12-24 GB
    safetensors from the eval pod stalled after 1-14 GB and the whole file
    restarted from zero on every retry, so a 72 GB checkpoint never
    materialized)."""
    want = b - a
    last: Exception | None = None
    for attempt in range(PART_RETRIES):
        try:
            resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={a}-{b - 1}")
            off, n = a, 0
            for chunk in resp["Body"].iter_chunks(CHUNK):
                os.pwrite(fd, chunk, off)
                off += len(chunk)
                n += len(chunk)
            if n == want:
                return
            last = FetchError(key, f"short part {a}-{b}: {n}/{want} bytes")
        except ClientError as e:
            code = (e.response or {}).get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound", "InvalidRange"):
                raise
            last = e
        except (BotoCoreError, OSError) as e:
            last = e
        time.sleep(min(2.0 ** attempt, 30.0))
    raise FetchError(key, f"part {a}-{b} failed after {PART_RETRIES} attempts: {last}")


class FetchCancelled(Exception):
    """The caller's cancel event fired mid-download (the duel was superseded).
    Parts already landed stay on disk and resume next time."""


class _AnyEvent:
    """is_set() when either the caller's event or the local one is set."""

    def __init__(self, outer: threading.Event | None):
        self.outer = outer
        self.local = threading.Event()

    def is_set(self) -> bool:
        return self.local.is_set() or (self.outer is not None and self.outer.is_set())


def adopt_sibling_snapshot(repo: str, revision: str,
                           hf_home: str | None = None) -> bool:
    """Materialize repo@revision by hard-linking a verified snapshot of the
    SAME revision cached under another r2 ref, if one exists.

    The revision of an r2 ref is its model_digest, so equal revisions mean
    byte-identical files. A crowned challenger is re-served from the public
    bucket ref (r2://affine-models/models/sha256/<digest>/) whose cache dir
    differs from the private ref it just dueled under — without this the
    king was downloaded a second time (72 GB, 28 min on 2026-09-05). Links
    are instant on one filesystem and survive the sibling being pruned.
    Returns False (caller downloads) when no sibling or linking fails."""
    home = Path(hf_home or HF_HOME)
    want = snapshot_dir(repo, revision, hf_home)
    hub = home / "hub"
    if not hub.is_dir():
        return False
    for other in hub.glob("models--r2--*"):
        if other.name == cache_dir_name(repo) or other.name.endswith(".pruning"):
            continue
        src = other / "snapshots" / revision
        if not (src / COMPLETE_MARKER).is_file():
            continue
        try:
            marker = json.loads((src / COMPLETE_MARKER).read_text())
            if marker.get("model_digest") != revision:
                continue
            tmp = want.with_name(want.name + ".linking")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            for f in src.iterdir():
                if f.name == COMPLETE_MARKER or not f.is_file():
                    continue
                os.link(f, tmp / f.name)
            marker["adopted_from"] = other.name
            (tmp / COMPLETE_MARKER).write_text(json.dumps(marker))
            want.parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(want)
            log.info("adopted snapshot %s@%s from %s (hard links)",
                     repo, revision[:12], other.name)
            return True
        except OSError as e:
            log.warning("could not adopt %s from %s: %s", repo, other.name, e)
            shutil.rmtree(want.with_name(want.name + ".linking"), ignore_errors=True)
    return False


def _download(s3, bucket: str, key: str, dest: Path, size: int,
              sha256: str, pool: ThreadPoolExecutor,
              cancel: threading.Event | None = None) -> None:
    """Resumable ranged download: PART-sized ranges fetched concurrently into
    a preallocated `.incomplete` file, completed part indices recorded in a
    `.parts` sidecar so a killed downloader (engine stall watchdog, pod
    restart) resumes instead of restarting. The final sha256 pass over the
    assembled file is what admits it; sizes and sidecars are never trusted
    alone."""
    tmp = dest.with_name(dest.name + ".incomplete")
    sidecar = dest.with_name(dest.name + ".parts")
    head = s3.head_object(Bucket=bucket, Key=key)
    clen = int(head.get("ContentLength", -1))
    if clen != size:
        # The object in the bucket is not the one the manifest describes.
        raise IntegrityError(f"{dest.name}: object is {clen} bytes, manifest says {size}")
    parts = _parts(size)
    done = _load_done(sidecar) if tmp.is_file() and tmp.stat().st_size == size else set()
    lock = threading.Lock()

    def record(i: int):
        # Runs on the worker thread as each part lands, so parts that finish
        # while a sibling's failure is propagating are still on disk + recorded.
        def cb(fut):
            if not fut.cancelled() and fut.exception() is None:
                with lock:
                    done.add(i)
                    sidecar.write_text(json.dumps(sorted(done)))
        return cb

    fd = os.open(tmp, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, size)
        futs = []
        for i in range(len(parts)):
            if i in done:
                continue
            fut = pool.submit(_fetch_part, s3, bucket, key, fd, *parts[i])
            fut.add_done_callback(record(i))
            futs.append(fut)
        try:
            pending = set(futs)
            while pending:
                # Poll so a superseded duel stops downloading within seconds
                # instead of at the end of the file (a 72 GB checkpoint kept
                # a stale job alive for 50 min on 2026-09-05).
                if cancel is not None and cancel.is_set():
                    raise FetchCancelled(f"{dest.name}: cancelled")
                finished, pending = wait(pending, timeout=5.0)
                for fut in finished:
                    fut.result()
        except BaseException:
            for fut in futs:
                fut.cancel()
            # Running parts hold this fd: let them land (and be recorded)
            # before it is closed, never let a late pwrite hit a reused fd.
            wait(futs)
            raise
        os.fsync(fd)
    finally:
        os.close(fd)
    if _sha256_file(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise IntegrityError(f"{dest.name}: sha256 mismatch against manifest")
    tmp.replace(dest)
    sidecar.unlink(missing_ok=True)


def fetch_snapshot(repo: str, revision: str, s3=None,
                   workers: int = DOWNLOAD_WORKERS,
                   cancel: threading.Event | None = None) -> Path:
    """Materialize + verify an r2 ref at the pinned model_digest. Idempotent:
    a complete snapshot returns immediately; a partial one resumes (files
    already present are re-hashed, never trusted by size alone). A verified
    snapshot of the same digest under another ref is hard-linked instead of
    downloaded. `cancel` (set by the caller) raises FetchCancelled between
    parts; landed parts stay for the next attempt."""
    if not is_r2(repo):
        raise ValueError(f"not an r2 ref: {repo}")
    bucket, prefix = proto.parse_r2_ref(repo)
    snap = snapshot_dir(repo, revision)
    if (snap / COMPLETE_MARKER).is_file():
        return snap
    if adopt_sibling_snapshot(repo, revision):
        return snap
    try:
        return _fetch(repo, revision, bucket, prefix, snap, s3, workers, cancel)
    except (IntegrityError, FetchCancelled):
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
           s3, workers: int, cancel: threading.Event | None = None) -> Path:
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
    # FILE_WORKERS files at a time, each file's ranges in parallel on the
    # shared parts pool (file tasks never sit in the parts pool, so it cannot
    # be exhausted by waiters). Measured 2026-09-05 from the eval pod: one
    # object streams ~120 MB/s no matter how many ranges are in flight, while
    # two objects at once ran ~105 MB/s EACH — the cap is per object, the pod
    # path takes >2 Gbps. Serial files left half the link idle.
    # A local stop flag OR'd with the caller's cancel: a failure in one file
    # halts its siblings without touching the caller's event (which, for the
    # duel path, means "superseded" — not something a transport error may set).
    stop = _AnyEvent(cancel)

    def one_file(f: dict) -> None:
        if stop.is_set():
            raise FetchCancelled(f"{repo}: cancelled before {f['path']}")
        t1 = time.time()
        _download(s3, bucket, prefix + f["path"], snap / f["path"],
                  int(f["size"]), f["sha256"], pool, stop)
        dt = max(time.time() - t1, 1e-3)
        log.info("fetched %s (%.1f GB, %.0f MB/s)", f["path"],
                 int(f["size"]) / 1e9, int(f["size"]) / 1e6 / dt)

    # Largest files first so the tail of the run is short small files, not
    # one 50 GB shard streaming alone at the per-object cap.
    todo.sort(key=lambda f: -int(f["size"]))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
            ThreadPoolExecutor(max_workers=max(1, FILE_WORKERS)) as fpool:
        futs = [fpool.submit(one_file, f) for f in todo]
        try:
            for fut in as_completed(futs):
                fut.result()
        except BaseException:
            stop.local.set()
            for fut in futs:
                fut.cancel()
            wait(futs)
            raise
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
