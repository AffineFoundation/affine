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
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
from botocore.exceptions import BotoCoreError, ClientError

from affine import r2, r2protocol as proto

log = logging.getLogger("evalsrv.r2store")

HF_HOME = os.environ.get("HF_HOME", "/root/hf")
COMPLETE_MARKER = ".affine_complete"
MAX_MANIFEST_BYTES = 2 << 20
CHUNK = 8 << 20
PART = 128 << 20           # ranged-GET unit; a stall costs at most one part
# Per-part retry budget must stay UNDER the engine's 180 s no-progress
# watchdog: 8 attempts x 20 s read timeout + backoff capped at 30 s was ~250 s,
# so one bad part got the whole downloader killed before its own retries ran
# out (2026-09-06: two prefetches died at 71.8 of 71.9 GB). 5 x 20 s + 1+2+4+
# 8+8 s of backoff = 123 s worst case. With chunk-level progress reporting
# (see Progress) a part that is retrying while siblings land is not a stall.
PART_RETRIES = 5
PART_BACKOFF_CAP_S = 8.0
# Objects downloaded concurrently, each with its OWN range pool. The old
# single shared FIFO pool queued every part of the largest file first, so
# files effectively downloaded one at a time at R2's ~120 MB/s per-object
# cap (every 72 GB prefetch took 510-600 s). Measured 2026-09-05 from the
# eval pod: two objects at once run ~105 MB/s EACH; the pod path takes >2
# Gbps. 8 ranges per object saturate the per-object cap (one stream ~50
# MB/s, RTT bound).
FILE_WORKERS = int(os.environ.get("AFFINE_R2_FILE_WORKERS", "3"))
DOWNLOAD_WORKERS = int(os.environ.get("AFFINE_R2_DOWNLOAD_WORKERS", "8"))
# Heartbeat the engine's stall watchdog reads (see Progress). Written in the
# snapshot dir; counts every landed 8 MB chunk and every hashed chunk, so
# verification phases are visible as work and a genuine stall is detected at
# chunk granularity instead of 128 MB parts.
PROGRESS_FILE = ".progress"
PROGRESS_FLUSH_S = 2.0
# Per-file "verified" sidecar so a resumed download does not re-hash every
# completed shard (72 GB at ~1 GB/s = 70+ s of CPU per retry). Written only
# after a sha256 pass matched the manifest; trusted only while size+mtime
# match. Same trust level as the snapshot's completion marker.
VERIFIED_SUFFIX = ".sha256ok"
# Weight-content identity (2026-09-13, reign-13 byte-copy incident): the
# sha256 of every TENSOR's raw bytes, keyed by tensor name, independent of
# how the checkpoint is sharded or what the shard files are called. Two
# checkpoints with the same fingerprint serve the same weights even when
# every file hash differs (reign 13 was reign 12 re-saved from 16 shards
# into 2). Cached in the snapshot so the duel path pays the read once.
FINGERPRINT_FILE = ".affine_weights_fingerprint"
TENSOR_HASHES_FILE = ".affine_tensor_hashes.json"
FINGERPRINT_DOMAIN = b"affine-weights-fingerprint-v1\n"
# Element-level near-duplicate probe: how many hash-different tensors are
# read from both snapshots (challenger + king) per duel. 32 tensors of a
# 35B MoE ≈ 2–5 GB read, seconds on NVMe.
NEAR_DUP_SAMPLE_TENSORS = 32


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
    # Pool sized for every concurrent range plus slack; the boto default (10)
    # / old 32 dropped and re-opened connections under 3 files x 16 ranges
    # ("Connection pool is full, discarding connection", extra TLS setups).
    return r2.s3_client(endpoint, ak, sk, read_timeout=20.0,
                        max_pool_connections=FILE_WORKERS * DOWNLOAD_WORKERS + 8)


class Progress:
    """Monotonic work counter the downloader child publishes for the engine's
    stall watchdog. `work_bytes` grows with every landed 8 MB chunk and every
    hashed chunk; the watchdog kills the child only when it stops growing.
    Replaces inferring progress from on-disk sizes, which mis-fired: a
    preallocated `.incomplete` counted at full size until its first part
    landed, so the count *dropped* and never recovered within 180 s (every
    'attempt 0 failed after 195s' on 2026-09-05/06). Includes the pid so a
    stale file from a killed child is ignored by the watchdog."""

    def __init__(self, path: Path | None):
        self.path = path
        self.work_bytes = 0
        self.phase = "start"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, n: int, phase: str | None = None) -> None:
        with self._lock:
            self.work_bytes += n
            if phase:
                self.phase = phase

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def _write(self) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = {"pid": os.getpid(), "work_bytes": self.work_bytes,
                       "phase": self.phase, "ts": time.time()}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.path)
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.wait(PROGRESS_FLUSH_S):
            self._write()

    def __enter__(self) -> "Progress":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="r2-progress")
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._write()


def read_progress(snap: Path) -> dict | None:
    """The child's heartbeat, or None when absent/unreadable."""
    try:
        return json.loads((snap / PROGRESS_FILE).read_text())
    except (OSError, ValueError):
        return None


def _sha256_file(path: Path, progress: Progress | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            if progress is not None:
                progress.add(len(b))
    return h.hexdigest()


def _shard_tensor_hashes(path: Path, progress: Progress | None = None
                         ) -> dict[str, tuple[str, list[int], str, int]]:
    """{tensor name: (dtype, shape, sha256 of the tensor's raw bytes, nbytes)} for
    one safetensors file. Header layout per the safetensors spec: 8-byte
    little-endian header length, JSON header, then the byte buffer that
    `data_offsets` index into."""
    out: dict[str, tuple[str, list[int], str, int]] = {}
    with open(path, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        if hlen <= 0 or hlen > 256 << 20:
            raise IntegrityError(f"{path.name}: bad safetensors header length {hlen}")
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = (int(x) for x in meta["data_offsets"])
            f.seek(base + start)
            h = hashlib.sha256()
            left = end - start
            while left > 0:
                b = f.read(min(CHUNK, left))
                if not b:
                    raise IntegrityError(f"{path.name}: short read inside tensor {name}")
                h.update(b)
                left -= len(b)
                if progress is not None:
                    progress.add(len(b))
            out[name] = (str(meta["dtype"]), [int(s) for s in meta["shape"]],
                         h.hexdigest(), end - start)
    return out


def tensor_hashes(snap: Path, progress: Progress | None = None
                  ) -> dict[str, list]:
    """{tensor name: [dtype, shape, sha256, nbytes, shard]} for every
    *.safetensors under `snap`; cached in TENSOR_HASHES_FILE once computed
    (a complete snapshot never changes). A tensor name present in two
    shards is an integrity fault (the safetensors index would be
    ambiguous)."""
    side = snap / TENSOR_HASHES_FILE
    try:
        cached = json.loads(side.read_text())
        if isinstance(cached, dict) and cached:
            return cached
    except (OSError, ValueError):
        pass
    rows: dict[str, list] = {}
    for shard in sorted(snap.glob("*.safetensors")):
        for name, (dtype, shape, sha, nbytes) in _shard_tensor_hashes(shard, progress).items():
            if name in rows:
                raise IntegrityError(f"tensor {name} appears in two shards")
            rows[name] = [dtype, shape, sha, nbytes, shard.name]
    tmp = side.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, sort_keys=True))
    tmp.replace(side)
    return rows


def fingerprint_of(rows: dict[str, list]) -> str:
    """sha256(FINGERPRINT_DOMAIN + canonical JSON of the sorted rows
    [name, dtype, shape, tensor_sha256]). Shard count, shard names, header
    metadata, config / tokenizer bytes and the manifest do not enter it;
    only the tensors do."""
    canon = json.dumps(sorted((n, r[0], r[1], r[2]) for n, r in rows.items()),
                       separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(FINGERPRINT_DOMAIN + canon.encode()).hexdigest()


def compute_weights_fingerprint(snap: Path, progress: Progress | None = None
                                ) -> tuple[str, int]:
    """(fingerprint, n_tensors) of the checkpoint under `snap`."""
    rows = tensor_hashes(snap, progress)
    return fingerprint_of(rows), len(rows)


def weights_fingerprint(snap: Path) -> str | None:
    """Cached weight fingerprint of a complete snapshot; computed and cached
    on first use for snapshots fetched before this file existed. None when
    the snapshot is not complete (never fingerprint a partial download)."""
    if not (snap / COMPLETE_MARKER).is_file():
        return None
    side = snap / FINGERPRINT_FILE
    try:
        rec = json.loads(side.read_text())
        fp = str(rec.get("fingerprint") or "")
        if len(fp) == 64:
            return fp
    except (OSError, ValueError):
        pass
    t0 = time.time()
    fp, n = compute_weights_fingerprint(snap)
    _write_fingerprint(snap, fp, n)
    log.info("weights fingerprint of %s: %s (%d tensors, %.0fs)", snap.name[:12],
             fp[:16], n, time.time() - t0)
    return fp


def _write_fingerprint(snap: Path, fp: str, n_tensors: int) -> None:
    (snap / FINGERPRINT_FILE).write_text(json.dumps({
        "fingerprint": fp, "n_tensors": n_tensors,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))


def _read_tensor(snap: Path, row: list, name: str) -> "np.ndarray":
    """Raw bytes of one tensor as a flat uint8/uint16/uint32 view (element
    width from the safetensors dtype; unknown widths fall back to bytes)."""
    shard = snap / row[4]
    with open(shard, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(hlen))
        start, end = (int(x) for x in header[name]["data_offsets"])
        f.seek(8 + hlen + start)
        buf = f.read(end - start)
    width = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U32": 4, "I16": 2,
             "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1,
             "F8_E5M2": 1, "I64": 8, "F64": 8, "U64": 8}.get(row[0], 1)
    dt = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
    return np.frombuffer(buf, dtype=dt)


def near_duplicate_stats(chall: Path, king: Path, seed: str,
                         sample_tensors: int = NEAR_DUP_SAMPLE_TENSORS) -> dict:
    """How much of the challenger IS the king, tensor by tensor.

    Hash level (free, both tables cached): the fraction of tensors, and of
    weight bytes, that are byte-identical to the king. Element level (a
    seeded sample of up to `sample_tensors` of the tensors whose hashes
    differ, read from both snapshots): the fraction of elements that
    actually changed. A trained model changes most elements of every tensor
    it touched; a copy with a few flipped values (reign-12 → chal-00466:
    26 elements out of 3.5e10, one per shard so every file hash moved)
    changes ~1e-6 of them. The sample is seeded by the duel's block hash so
    a copier cannot know which tensors will be read."""
    ch, kh = tensor_hashes(chall), tensor_hashes(king)
    common = [n for n in ch if n in kh and ch[n][0] == kh[n][0] and ch[n][1] == kh[n][1]]
    stats = {"n_tensors": len(ch), "n_king_tensors": len(kh), "n_comparable": len(common),
             "identical_tensors": 0, "identical_bytes_frac": 0.0,
             "sampled_tensors": 0, "sampled_elements": 0, "sampled_changed_elements": 0,
             "sampled_changed_frac": None}
    if not common:
        return stats
    total_bytes = sum(int(ch[n][3]) for n in ch) or 1
    same = [n for n in common if ch[n][2] == kh[n][2]]
    differ = [n for n in common if ch[n][2] != kh[n][2]]
    stats["identical_tensors"] = len(same)
    stats["identical_bytes_frac"] = sum(int(ch[n][3]) for n in same) / total_bytes
    if not differ:
        stats["sampled_changed_frac"] = 0.0
        return stats
    rng = random.Random(seed)
    picked = differ if len(differ) <= sample_tensors else rng.sample(differ, sample_tensors)
    changed = elements = 0
    for name in picked:
        a, b = _read_tensor(chall, ch[name], name), _read_tensor(king, kh[name], name)
        if a.shape != b.shape:
            continue
        elements += int(a.size)
        changed += int(np.count_nonzero(a != b))
    stats.update(sampled_tensors=len(picked), sampled_elements=elements,
                 sampled_changed_elements=changed,
                 sampled_changed_frac=(changed / elements) if elements else None)
    return stats


def near_duplicate_reason(stats: dict, max_identical_bytes_frac: float,
                          min_changed_frac: float) -> str | None:
    """Why `stats` mean 'this is the king with cosmetic edits', or None.

    Two independent tests, either one rejects:
      * identical_bytes_frac > max_identical_bytes_frac — most of the weight
        bytes ARE the king's (1.0 = exact copy, disabled with >= 1.0);
      * sampled_changed_frac < min_changed_frac — the tensors that do differ
        differ in a vanishing fraction of their elements (0 disables).
    """
    ib = float(stats.get("identical_bytes_frac") or 0.0)
    if max_identical_bytes_frac < 1.0 and ib > max_identical_bytes_frac:
        return (f"{ib:.4%} of the weight bytes are byte-identical to the king "
                f"({stats.get('identical_tensors')}/{stats.get('n_tensors')} tensors)")
    cf = stats.get("sampled_changed_frac")
    if min_changed_frac > 0 and cf is not None and stats.get("sampled_elements", 0) > 0 \
            and cf < min_changed_frac:
        return (f"only {cf:.2e} of the elements differ from the king in the "
                f"{stats.get('sampled_tensors')} sampled non-identical tensors "
                f"({stats.get('sampled_changed_elements')} of {stats.get('sampled_elements')})")
    return None


def _verified_sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + VERIFIED_SUFFIX)


def _is_verified(dest: Path, size: int, sha256: str,
                 progress: Progress | None = None) -> bool:
    """True when `dest` is a complete file matching the manifest. Cheap path:
    a verified sidecar we wrote earlier whose size+mtime still match. Slow
    path: hash it (and write the sidecar for next time)."""
    try:
        st = dest.stat()
    except OSError:
        return False
    if not dest.is_file() or st.st_size != size:
        return False
    side = _verified_sidecar(dest)
    try:
        rec = json.loads(side.read_text())
        if (rec.get("sha256") == sha256 and rec.get("size") == size
                and rec.get("mtime_ns") == st.st_mtime_ns):
            return True
    except (OSError, ValueError):
        pass
    if _sha256_file(dest, progress) != sha256:
        return False
    _write_verified(dest, sha256)
    return True


def _write_verified(dest: Path, sha256: str) -> None:
    try:
        st = dest.stat()
        _verified_sidecar(dest).write_text(json.dumps(
            {"sha256": sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}))
    except OSError:
        pass


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
    if not any(part.startswith("models--r2--") for part in path.parts):
        # huggingface_hub's own `.incomplete` blobs grow as they are written;
        # st_size is their progress.
        return None
    sidecar = path.with_name(path.name[:-len(".incomplete")] + ".parts")
    if not sidecar.is_file():
        # Ours, preallocated to full size, no part recorded yet: nothing has
        # landed. Reporting st_size here made the watchdog's first sample the
        # whole checkpoint, after which real progress could never exceed it.
        return 0
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    parts = _parts(size)
    return sum(b - a for i in _load_done(sidecar)
               if 0 <= i < len(parts) for a, b in (parts[i],))


def _fetch_part(s3, bucket: str, key: str, fd: int, a: int, b: int,
                progress: Progress | None = None) -> None:
    """One ranged GET written in place. Retries with a fresh connection on
    any transport error; a stall inside a part therefore costs one part,
    not the file (found live 2026-09-04: single-stream GETs of 12-24 GB
    safetensors from the eval pod stalled after 1-14 GB and the whole file
    restarted from zero on every retry, so a 72 GB checkpoint never
    materialized). Every landed chunk is reported to `progress`; a retried
    part re-reports its bytes, which only ever makes the watchdog more
    lenient, never less."""
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
                if progress is not None:
                    progress.add(len(chunk))
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
        time.sleep(min(2.0 ** attempt, PART_BACKOFF_CAP_S))
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
                if f.name in (COMPLETE_MARKER, PROGRESS_FILE) or not f.is_file():
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
              sha256: str, workers: int,
              cancel: threading.Event | None = None,
              progress: Progress | None = None) -> None:
    """Resumable ranged download: PART-sized ranges fetched concurrently (on
    this file's own pool of `workers` streams) into a preallocated
    `.incomplete` file, completed part indices recorded in a `.parts`
    sidecar so a killed downloader (engine stall watchdog, pod restart)
    resumes instead of restarting. The final sha256 pass over the assembled
    file is what admits it; sizes and sidecars are never trusted alone."""
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
        if not sidecar.is_file():
            # Exists from the first byte so the watchdog's disk fallback
            # never mistakes the preallocated size for landed bytes.
            sidecar.write_text(json.dumps(sorted(done)))
        futs = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i in range(len(parts)):
                if i in done:
                    continue
                fut = pool.submit(_fetch_part, s3, bucket, key, fd, *parts[i],
                                  progress)
                fut.add_done_callback(record(i))
                futs.append(fut)
            try:
                pending = set(futs)
                while pending:
                    # Poll so a superseded duel stops downloading within
                    # seconds instead of at the end of the file (a 72 GB
                    # checkpoint kept a stale job alive for 50 min, 2026-09-05).
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
    if progress is not None:
        progress.set_phase(f"hashing {dest.name}")
    if _sha256_file(tmp, progress) != sha256:
        tmp.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise IntegrityError(f"{dest.name}: sha256 mismatch against manifest")
    tmp.replace(dest)
    sidecar.unlink(missing_ok=True)
    _write_verified(dest, sha256)


def fetch_snapshot(repo: str, revision: str, s3=None,
                   workers: int = DOWNLOAD_WORKERS,
                   cancel: threading.Event | None = None) -> Path:
    """Materialize + verify an r2 ref at the pinned model_digest. Idempotent:
    a complete snapshot returns immediately; a partial one resumes (files
    already present are re-hashed unless our own verified sidecar still
    matches them; never trusted by size alone). `workers` is the number of
    parallel ranges PER FILE (FILE_WORKERS files at a time). A verified
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

    with Progress(snap / PROGRESS_FILE) as progress:
        progress.set_phase("verifying cached files")
        todo: list[dict] = []
        for f in manifest["files"]:
            if _is_verified(snap / f["path"], int(f["size"]), f["sha256"],
                            progress):
                continue
            todo.append(f)
        total = sum(int(f["size"]) for f in todo)
        log.info("fetching %s@%s: %d/%d files, %.1f GB", repo, revision[:12],
                 len(todo), len(manifest["files"]), total / 1e9)
        progress.set_phase("downloading")
        # FILE_WORKERS files at a time, each on its own pool of `workers`
        # ranges — objects genuinely stream in parallel (see the constants'
        # note on the old shared FIFO pool). A local stop flag OR'd with the
        # caller's cancel: a failure in one file halts its siblings without
        # touching the caller's event (which, for the duel path, means
        # "superseded" — not something a transport error may set).
        stop = _AnyEvent(cancel)

        def one_file(f: dict) -> None:
            if stop.is_set():
                raise FetchCancelled(f"{repo}: cancelled before {f['path']}")
            t1 = time.time()
            _download(s3, bucket, prefix + f["path"], snap / f["path"],
                      int(f["size"]), f["sha256"], workers, stop, progress)
            dt = max(time.time() - t1, 1e-3)
            log.info("fetched %s (%.1f GB, %.0f MB/s)", f["path"],
                     int(f["size"]) / 1e9, int(f["size"]) / 1e6 / dt)

        # Largest files first so the tail of the run is short small files,
        # not one 50 GB shard streaming alone at the per-object cap.
        todo.sort(key=lambda f: -int(f["size"]))
        with ThreadPoolExecutor(max_workers=max(1, FILE_WORKERS)) as fpool:
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
        # One more read pass over the verified shards (NVMe, ~1 GB/s): the
        # weight fingerprint the duel's identity gate compares against the
        # king and every past challenger. Done here, at prefetch time, so
        # the duel itself never waits for it.
        progress.set_phase("fingerprinting")
        fp, n_tensors = compute_weights_fingerprint(snap, progress)
        progress.set_phase("complete")
    (snap / COMPLETE_MARKER).write_text(json.dumps({
        "repo": repo, "model_digest": revision,
        "files": len(manifest["files"]),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights_fingerprint": fp, "n_tensors": n_tensors,
    }))
    _write_fingerprint(snap, fp, n_tensors)
    (snap / PROGRESS_FILE).unlink(missing_ok=True)
    log.info("snapshot %s@%s complete in %.0fs (%.0f MB/s over the run)",
             repo, revision[:12], time.time() - t0,
             total / 1e6 / max(time.time() - t0, 1e-3))
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
