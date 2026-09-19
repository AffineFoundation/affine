"""Public dataset browser backing /api/v1/dataset*.

The dash box keeps a local cache of the public corpus (manifest + Parquet
turn index, chunks fetched lazily on first view) via the same fail-closed
CorpusSync the eval pods use. Everything served here is already public on
the bucket — this module only makes it browsable.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import posixpath
import shutil
import threading
from collections import Counter, OrderedDict
from pathlib import Path

from evalsrv.corpus import CorpusSync, CorpusVerificationError

from ..config import Config

log = logging.getLogger("affine.dash.corpus")

REFRESH_TTL_S = 900          # manifest pointer re-check cadence
MAX_PAGE = 200
TOP_REPOS = 25
# Materialized turns are immutable per manifest; a crawler walking turn ids
# re-fetches the same popular ones, and each miss costs a chunk decode.
TURN_LRU = 256
# Longest a /turn request waits for the chunk lock before answering 503.
# Bounded waiting keeps the sync-handler thread pool (40 tokens) free for
# the static site and the other endpoints when a crawler floods /turn.
CHUNK_WAIT_S = 3.0

# n_prefix_chars buckets for the length histogram (chars of prompt context).
LEN_BUCKETS = [
    (0, 2_000, "<2k"),
    (2_000, 5_000, "2–5k"),
    (5_000, 10_000, "5–10k"),
    (10_000, 20_000, "10–20k"),
    (20_000, 50_000, "20–50k"),
    (50_000, None, "50k+"),
]


def _slim_row(row: dict) -> dict:
    stratum = str(row.get("stratum") or "")
    return {
        "turn_id": row.get("turn_id"),
        "traj_id": row.get("traj_id"),
        "turn_idx": row.get("turn_idx"),
        "repo": stratum.split("|", 1)[0] if stratum else "",
        "phase": row.get("phase") or "",
        "source": row.get("source") or "",
        "language": row.get("language") or "",
        "n_prefix_chars": row.get("n_prefix_chars"),
    }


class LazyCorpusSync(CorpusSync):
    """Dash-only CorpusSync: one rollout per request, no per-request gunzip.

    Extends PR #7's `CorpusSync._traj_at_lazy` (decode one raw line instead
    of every rollout of the chunk), kept inside dash/ so the evalsrv module
    the eval pods run is untouched. Live load on 2026-09-13 showed the
    remaining cost: a crawler walking random turn ids at ~7/s over ~500
    chunks misses any small in-memory chunk cache almost every time, and a
    full-chunk gunzip (~66 MB, ~0.3 s) per miss serialised behind one lock
    still saturated the process (15-30 s TTFB site-wide). So each chunk is
    inflated ONCE to `chunks_raw/<name>.jsonl` next to the verified .gz
    (immutable, sha-named) and served by byte offset: a turn is a seek +
    one readline + one json.loads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_dir = self.chunks_dir.parent / "chunks_raw"
        self._raw_offsets: dict[str, tuple[Path, list[int]]] = {}

    def _raw_chunk(self, chunk_key: str) -> tuple[Path, list[int]]:
        ent = self._raw_offsets.get(chunk_key)
        if ent is not None:
            return ent
        self.ensure_chunk(chunk_key)
        name = posixpath.basename(chunk_key)
        gz_path = self.chunks_dir / name
        raw_path = self.raw_dir / (name[:-3] if name.endswith(".gz") else name)
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
            with gzip.open(gz_path, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(tmp, raw_path)
        offsets: list[int] = []
        pos = 0
        with open(raw_path, "rb") as f:
            for line in f:
                if line.strip():
                    offsets.append(pos)
                pos += len(line)
        ent = (raw_path, offsets)
        self._raw_offsets[chunk_key] = ent
        return ent

    def _traj_at(self, chunk_key: str, traj_line: int) -> dict:
        raw_path, offsets = self._raw_chunk(chunk_key)
        if traj_line < 0 or traj_line >= len(offsets):
            raise CorpusVerificationError(
                f"traj_line {traj_line} OOB in {chunk_key}")
        with open(raw_path, "rb") as f:
            f.seek(offsets[traj_line])
            return json.loads(f.readline())


class DatasetBusy(Exception):
    """/turn could not get the chunk lock within CHUNK_WAIT_S."""


class DatasetView:
    """Thread-safe reader over the synced corpus for the dataset endpoints."""

    def __init__(self, cfg: Config):
        # `_lock` guards the index-derived caches (stats, turn pages);
        # `_chunk_lock` guards chunk fetch/decode for /turn. They are separate
        # so a slow chunk decode never stalls /api/v1/dataset or /turns
        # (2026-09-11: 5–12 s TTFB across the site under turn-id crawling).
        # refresh() takes both, in that order; nothing else nests them.
        self._lock = threading.Lock()
        self._chunk_lock = threading.Lock()
        self._sync = LazyCorpusSync(
            cfg.dataset.corpus_base_url,
            cfg.dataset.manifest_key,
            cfg.state_dir / "corpus_cache",
            lazy_chunks=True,
        )
        self._stats_sha = ""          # manifest sha the caches were built for
        self._stats: dict | None = None
        self._by_turn_id: dict[str, dict] = {}
        self._turn_lru: OrderedDict[str, dict] = OrderedDict()

    # -- refresh -----------------------------------------------------------------
    def refresh(self) -> None:
        """Re-check the manifest pointer; rebuild derived caches on change.
        Called from a background task — endpoints never hit the network for
        the index (only /turn may fetch one chunk)."""
        with self._lock, self._chunk_lock:
            self._sync.refresh()
            if self._sync.ready and self._sync.manifest_sha256 != self._stats_sha:
                try:
                    self._rebuild()
                    self._turn_lru.clear()
                except Exception as exc:
                    log.warning("dataset cache rebuild failed: %s", exc)

    def _rebuild(self) -> None:
        rows = self._sync.load_index_rows()
        self._by_turn_id = {str(r["turn_id"]): r for r in rows}

        by_source: Counter[str] = Counter()
        by_language: Counter[str] = Counter()
        by_phase: Counter[str] = Counter()
        by_repo: Counter[str] = Counter()
        len_hist = [0] * len(LEN_BUCKETS)
        trajs: set[str] = set()
        for r in rows:
            by_source[str(r.get("source") or "unknown")] += 1
            by_language[str(r.get("language") or "unknown")] += 1
            by_phase[str(r.get("phase") or "other")] += 1
            stratum = str(r.get("stratum") or "")
            by_repo[stratum.split("|", 1)[0] or "unknown"] += 1
            trajs.add(str(r.get("traj_id")))
            n = int(r.get("n_prefix_chars") or 0)
            for i, (lo, hi, _) in enumerate(LEN_BUCKETS):
                if n >= lo and (hi is None or n < hi):
                    len_hist[i] += 1
                    break

        manifest = self._sync.manifest or {}
        active = [s for s in manifest.get("shards", []) if s.get("active")]
        top = by_repo.most_common(TOP_REPOS)
        rest = sum(by_repo.values()) - sum(n for _, n in top)

        self._stats = {
            "corpus_epoch": int(manifest.get("corpus_epoch") or 0),
            "schema_version": self._sync.schema_version,
            "manifest_sha256": self._sync.manifest_sha256,
            "n_turns": len(rows),
            "n_trajectories": len(trajs),
            "n_chunks": sum(1 for s in active
                            if (s.get("format") or "") in ("traj_v1", "view_v4")),
            "view_spec": manifest.get("view_spec") or "",
            "mix": {
                "source": dict(by_source.most_common()),
                "language": dict(by_language.most_common()),
                "phase": dict(by_phase.most_common()),
            },
            "repos": (
                [{"repo": k, "n_turns": n} for k, n in top]
                + ([{"repo": "(other)", "n_turns": rest}] if rest else [])),
            "prefix_chars_hist": [
                {"bucket": label, "n_turns": len_hist[i]}
                for i, (_, _, label) in enumerate(LEN_BUCKETS)],
        }
        self._stats_sha = self._sync.manifest_sha256
        log.info("dataset cache built: epoch=%s turns=%d trajs=%d",
                 self._stats["corpus_epoch"], len(rows), len(trajs))

    # -- reads -------------------------------------------------------------------
    def stats(self) -> dict | None:
        with self._lock:
            if self._stats is None:
                return None
            return {**self._stats, "synced_at": self._sync.synced_at,
                    "stale": self._sync.stale}

    def turns_page(self, *, source: str = "", language: str = "",
                   phase: str = "", repo: str = "", q: str = "",
                   limit: int = 50, cursor: int = 0) -> dict | None:
        limit = max(1, min(int(limit), MAX_PAGE))
        cursor = max(0, int(cursor))
        needle = q.strip().lower()
        with self._lock:
            if self._stats is None:
                return None
            rows = self._sync.load_index_rows()
            matched = []
            for r in rows:
                if source and str(r.get("source") or "") != source:
                    continue
                if language and str(r.get("language") or "") != language:
                    continue
                if phase and str(r.get("phase") or "") != phase:
                    continue
                stratum = str(r.get("stratum") or "")
                if repo and stratum.split("|", 1)[0] != repo:
                    continue
                if needle and needle not in str(r.get("turn_id", "")).lower():
                    continue
                matched.append(r)
            page = [_slim_row(r) for r in matched[cursor:cursor + limit]]
            next_cursor = cursor + limit
            return {
                "total": len(matched),
                "cursor": cursor,
                "next_cursor": next_cursor if next_cursor < len(matched) else None,
                "items": page,
            }

    def turn_detail(self, turn_id: str) -> dict | None:
        """Materialize one turn (prompt prefix + reference action) from its
        chunk. May download + verify that single chunk on first view.
        Raises DatasetBusy if the chunk lock is contended for CHUNK_WAIT_S."""
        with self._lock:
            row = self._by_turn_id.get(turn_id)
        if row is None:
            return None
        if not self._chunk_lock.acquire(timeout=CHUNK_WAIT_S):
            raise DatasetBusy(turn_id)
        try:
            hit = self._turn_lru.get(turn_id)
            if hit is not None:
                self._turn_lru.move_to_end(turn_id)
                return hit
            try:
                turn = self._sync.materialize_turns([row])[0]
            except (CorpusVerificationError, OSError, ValueError,
                    KeyError, StopIteration) as exc:
                log.warning("turn materialize failed %s: %s", turn_id, exc)
                return None
            detail = self._turn_payload(turn_id, turn)
            self._turn_lru[turn_id] = detail
            while len(self._turn_lru) > TURN_LRU:
                self._turn_lru.popitem(last=False)
        finally:
            self._chunk_lock.release()
        return detail

    @staticmethod
    def _turn_payload(turn_id: str, turn: dict) -> dict:
        return {
            "turn_id": turn_id,
            "traj_id": turn.get("traj_id"),
            "turn_idx": turn.get("turn_idx"),
            "repo": turn.get("repo") or "",
            "instance_id": turn.get("instance_id") or "",
            "model": turn.get("model") or "",
            "phase": turn.get("phase") or "",
            "source": turn.get("source") or "",
            "language": turn.get("language") or "",
            "action_kind": turn.get("action_kind") or "",
            "generated_at": turn.get("generated_at") or "",
            "n_prefix_chars": turn.get("n_prefix_chars"),
            "prefix": turn.get("prefix") or [],
            "reference_turn": turn.get("reference_turn") or "",
        }
