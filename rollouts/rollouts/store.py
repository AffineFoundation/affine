"""Append-only trace store: the system of record for rollouts.

Layout (corpus-v2 discipline — immutable content-named chunks behind a
manifest):

    <root>/chunks/<tag>-<sha12>.jsonl.gz   one envelope per line
    <root>/manifest.json                   every chunk, with sha + counts

One chunk per appended batch (batches are the natural write unit and the
current trace_archive granularity). Chunks are never rewritten; the
manifest is replaced atomically. A crash between chunk write and manifest
update is healed on the next append (rescan picks up orphan chunks).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from pathlib import Path

from rollouts.schema import utc_now_iso

log = logging.getLogger("rollouts.store")

MANIFEST_SCHEMA = "rollouts-traces-v1"


class TraceStore:
    def __init__(self, root: Path):
        self.root = root
        self.chunk_dir = root / "chunks"
        self.manifest_path = root / "manifest.json"

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"schema": MANIFEST_SCHEMA, "chunks": []}

    def _write_manifest(self, manifest: dict) -> None:
        manifest["schema"] = MANIFEST_SCHEMA
        manifest["updated_at"] = utc_now_iso()
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        tmp.replace(self.manifest_path)

    def append_batch(self, envelopes: list[dict], tag: str) -> str | None:
        """Write one immutable chunk for a batch of envelopes; returns the
        chunk key (chunks/<name>) or None when there is nothing to write."""
        if not envelopes:
            return None
        payload = "".join(
            json.dumps(e, ensure_ascii=False) + "\n" for e in envelopes
        ).encode("utf-8")
        sha = hashlib.sha256(payload).hexdigest()
        name = f"{tag}-{sha[:12]}.jsonl.gz"
        key = f"chunks/{name}"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        path = self.chunk_dir / name
        if not path.exists():   # same content re-appended: chunk is identical
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wb") as f:
                f.write(payload)
            tmp.replace(path)
        manifest = self._load_manifest()
        entries = [c for c in manifest["chunks"] if c.get("key") != key]
        entries.append({
            "key": key,
            "sha256": sha,
            "n_rollouts": len(envelopes),
            "sources": sorted({e["source"] for e in envelopes}),
            "created_at": utc_now_iso(),
        })
        manifest["chunks"] = entries
        self._write_manifest(manifest)
        log.info("stored %s (%d rollouts, sha %s)", key, len(envelopes),
                 sha[:12])
        return key

    def iter_envelopes(self, chunk_key: str | None = None):
        """Yield envelopes from one chunk or from every manifested chunk."""
        keys = ([chunk_key] if chunk_key else
                [c["key"] for c in self._load_manifest()["chunks"]])
        for key in keys:
            path = self.root / key
            if not path.exists():
                log.warning("manifested chunk missing on disk: %s", key)
                continue
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def unmirrored_chunks(self) -> list[dict]:
        """Chunks not yet marked as mirrored to the remote trace dataset."""
        return [c for c in self._load_manifest()["chunks"]
                if not c.get("mirrored_at")]

    def mark_mirrored(self, keys: list[str]) -> None:
        manifest = self._load_manifest()
        now = utc_now_iso()
        for c in manifest["chunks"]:
            if c["key"] in keys:
                c["mirrored_at"] = now
        self._write_manifest(manifest)
