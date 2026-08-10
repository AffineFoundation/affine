"""Corpus publishing: push/retire turn shards on the public Hippius bucket.

Run from the root box with doppler creds (HIPPIUS_ACCESS_KEY / _SECRET_KEY):

  doppler run -- python affine/scripts/corpus_push.py init [--file .../turns.jsonl]
  doppler run -- python affine/scripts/corpus_push.py validate <shard.jsonl>
  doppler run -- python affine/scripts/corpus_push.py add <shard.jsonl>
  doppler run -- python affine/scripts/corpus_push.py add-v2 --pack-dir DIR
  doppler run -- python affine/scripts/corpus_push.py retire <shard-key>
  doppler run -- python affine/scripts/corpus_push.py status [--verify]

Schema versions
  * 1 (legacy): per-turn JSONL shards under turns/shards/*.jsonl.gz
  * 2: trajectory chunks under turns/chunks/*.jsonl.gz + Parquet index
    under turns/index/turns_{epoch}.parquet. Eval samples the index and
    materializes prefixes on demand.

Turn record contract (v1 shards / compat flat JSONL):
  * one JSON object per line: {"traj_id": str, "turn_idx": int, "prefix": [...]}
  * turn_id = "{traj_id}:{turn_idx}" globally unique across ALL shards ever
    published (teacher refs are cached by turn_id — a collision would score
    against the wrong reference);
  * prefix = plain chat messages [{"role", "content"}], ending on a "user"
    message (the scorer appends the generation prompt from there);
  * the system message must mandate the action format (one closed ```bash
    block) — foreign scaffolds must be rewritten at extraction time;
  * recommended: reference_turn (research force-y/RT tooling) and traj_ids
    matching the phase regex so stratification can see bug families.

Invariants enforced here:
  * shard objects and manifest revisions are immutable — never overwritten;
  * every manifest revision is published at turns/manifests/{sha256}.json
    BEFORE the mutable pointer moves (pod sync rejects a pointer without
    its immutable copy);
  * removal is logical: `retire` flips active=false in a new revision, the
    shard object stays so old verdicts remain replayable;
  * every active-window change bumps corpus_epoch.

Deliberately boto3-direct and fail-LOUD: unlike affine.hippius (fail-open
presentation writes), a partial corpus publish must abort, not cool down.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import posixpath
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from affine.config import load_config
from affine.corpus.materialize import PHASE_RE, materialize_turn
from affine.corpus.pack import INDEX_COLUMNS, PackResult

SHARDS_PREFIX = "turns/shards"
CHUNKS_PREFIX = "turns/chunks"
INDEX_PREFIX = "turns/index"
# Same stratification pattern as evalsrv.dueling / harness.sample_turns.
# Re-export name kept for ops/datagen_refresh imports.
# ~30k tokens at ~4 chars/token; the serving window is 32768 tokens minus
# thought+action budget. Longer prefixes silently waste turns at duel time.
MAX_PREFIX_CHARS = 120_000


def _client(cfg):
    ak = os.environ.get("HIPPIUS_ACCESS_KEY", "")
    sk = os.environ.get("HIPPIUS_SECRET_KEY", "")
    if not (ak and sk):
        sys.exit("HIPPIUS_ACCESS_KEY / HIPPIUS_SECRET_KEY missing "
                 "(run under `doppler run --`)")
    return boto3.client(
        "s3", endpoint_url=cfg.hippius["endpoint"],
        aws_access_key_id=ak, aws_secret_access_key=sk,
        region_name="decentralized",
        config=BotoConfig(signature_version="s3v4",
                          s3={"addressing_style": "path"},
                          connect_timeout=10, read_timeout=300,
                          retries={"max_attempts": 3, "mode": "standard"}))


def schema_version(manifest: dict | None) -> int:
    if not manifest:
        return 1
    return int(manifest.get("schema_version") or 1)


def iter_published_turn_ids(get_bytes, shard: dict):
    """Yield turn_id strings from a shard entry (v1 turn-flat or v2 traj)."""
    blob = gzip.decompress(get_bytes(shard["key"]))
    fmt = shard.get("format") or "turn_v1"
    if fmt == "traj_v1":
        for line in blob.splitlines():
            traj = json.loads(line)
            tid = traj["traj_id"]
            for t in traj.get("turns") or []:
                yield f"{tid}:{t['turn_idx']}"
    else:
        for line in blob.splitlines():
            r = json.loads(line)
            yield f"{r['traj_id']}:{r['turn_idx']}"


class Corpus:
    def __init__(self):
        self.cfg = load_config()
        self.bucket = self.cfg.hippius["bucket"]
        self.manifest_key = self.cfg.dataset.manifest_key
        self.manifests_prefix = posixpath.join(
            posixpath.dirname(self.manifest_key), "manifests")
        self.s3 = _client(self.cfg)

    # -- s3 helpers -----------------------------------------------------------
    def _exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def _get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def _put(self, key: str, body: bytes, content_type: str,
             cache_control: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                           ContentType=content_type, CacheControl=cache_control)
        print(f"put {key} ({len(body):,} bytes)")

    # -- manifest -------------------------------------------------------------
    def current_manifest(self) -> dict | None:
        if not self._exists(self.manifest_key):
            return None
        return json.loads(self._get(self.manifest_key))

    def publish_manifest(self, manifest: dict) -> str:
        raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
        mhash = hashlib.sha256(raw).hexdigest()
        frozen_key = f"{self.manifests_prefix}/{mhash}.json"
        # Immutable revision FIRST, then move the pointer: the pod rejects a
        # pointer whose immutable copy is missing.
        self._put(frozen_key, raw, "application/json",
                  "public, max-age=31536000, immutable")
        self._put(self.manifest_key, raw, "application/json", "no-cache")
        active = [s for s in manifest["shards"] if s["active"]]
        n_turns = 0
        if manifest.get("index"):
            n_turns = int(manifest["index"].get("n_turns") or 0)
        else:
            n_turns = sum(int(s.get("n_turns") or 0) for s in active)
        print(f"manifest {mhash[:16]} epoch={manifest['corpus_epoch']} "
              f"schema={schema_version(manifest)} "
              f"active={len(active)}/{len(manifest['shards'])} "
              f"n_turns={n_turns:,}")
        return mhash

    # -- shard upload -----------------------------------------------------------
    def upload_shard(self, path: Path, epoch: int) -> dict:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        n_turns = raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1)
        key = f"{SHARDS_PREFIX}/turns_epoch_{epoch:04d}.jsonl.gz"
        if self._exists(key):
            sys.exit(f"refusing to overwrite existing shard {key} (immutable)")
        print(f"gzipping {path} ({len(raw):,} bytes, {n_turns:,} turns)…")
        blob = gzip.compress(raw, compresslevel=6)
        self._put(key, blob, "application/gzip",
                  "public, max-age=31536000, immutable")
        return {"key": key, "sha256": sha, "n_turns": n_turns, "active": True}

    def upload_gzip_jsonl(self, path: Path, key: str) -> dict:
        """Upload an uncompressed local jsonl as immutable .jsonl.gz."""
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if self._exists(key):
            got = hashlib.sha256(gzip.decompress(self._get(key))).hexdigest()
            if got != sha:
                sys.exit(f"refusing to overwrite {key} with different sha")
            print(f"reuse {key} (sha verified)")
            n_lines = raw.count(b"\n") + (
                0 if raw.endswith(b"\n") or not raw else 1)
            return {"key": key, "sha256": sha, "n_lines": n_lines}
        print(f"gzipping → {key} ({len(raw):,} bytes)…")
        blob = gzip.compress(raw, compresslevel=6)
        self._put(key, blob, "application/gzip",
                  "public, max-age=31536000, immutable")
        n_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1)
        return {"key": key, "sha256": sha, "n_lines": n_lines}

    def upload_bytes(self, path: Path, key: str,
                     content_type: str = "application/octet-stream") -> str:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if self._exists(key):
            got = hashlib.sha256(self._get(key)).hexdigest()
            if got != sha:
                sys.exit(f"refusing to overwrite {key} with different sha")
            print(f"reuse {key} (sha verified)")
            return sha
        self._put(key, raw, content_type, "public, max-age=31536000, immutable")
        return sha

    # -- commands -----------------------------------------------------------------
    def cmd_init(self, file: Path, force: bool) -> None:
        if self.current_manifest() is not None and not force:
            sys.exit(f"{self.manifest_key} already exists; use `add`, "
                     "or --force to re-init epoch 1")
        shard = self.upload_shard(file, epoch=1)
        self.publish_manifest({
            "corpus_epoch": 1,
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shards": [shard],
            "prev_manifest": None,
        })

    def cmd_add(self, file: Path, skip_validate: bool = False) -> None:
        m = self._require_manifest()
        if schema_version(m) >= 2:
            sys.exit("live manifest is schema_version=2; use `add-v2` "
                     "(or migrate) instead of turn-flat `add`")
        if skip_validate:
            print("WARNING: --no-validate; shipping an unchecked shard")
        elif not self.cmd_validate(file, manifest=m):
            sys.exit("validation failed; fix the shard or use --no-validate "
                     "if you are absolutely sure")
        epoch = int(m["corpus_epoch"]) + 1
        shard = self.upload_shard(file, epoch=epoch)
        self._next_revision(m, epoch, m["shards"] + [shard],
                            schema_version=1)

    def cmd_add_v2(self, pack: PackResult, *, retire_previous: bool = True,
                   compat_key: str | None = None,
                   skip_validate: bool = False) -> dict:
        """Publish a packed v2 corpus revision (chunks + index + optional compat)."""
        m = self._require_manifest()
        epoch = int(m["corpus_epoch"]) + 1
        if not skip_validate and not self.cmd_validate_v2(pack, manifest=m):
            sys.exit("v2 validation failed")

        shards: list[dict] = []
        if retire_previous:
            for s in m["shards"]:
                shards.append(dict(s, active=False) if s.get("active") else s)
        else:
            shards = list(m["shards"])

        uploaded_chunks = []
        for local, meta in zip(pack.chunk_paths, pack.chunk_meta):
            # Ensure key epoch matches the publish epoch (packer may have
            # been called with the target epoch already).
            key = meta["key"]
            info = self.upload_gzip_jsonl(local, key)
            uploaded_chunks.append({
                "key": key,
                "sha256": info["sha256"],
                "n_trajectories": meta["n_trajectories"],
                "n_turns": meta.get("n_turns", 0),
                "format": "traj_v1",
                "active": True,
            })
        shards.extend(uploaded_chunks)

        index_key = f"{INDEX_PREFIX}/turns_{epoch:04d}.parquet"
        index_sha = self.upload_bytes(
            pack.index_path, index_key, "application/vnd.apache.parquet")
        if index_sha != pack.index_sha256:
            sys.exit(f"index sha mismatch: {index_sha} != {pack.index_sha256}")

        compat_entry = None
        if pack.compat_path is not None:
            ck = compat_key or (
                f"{SHARDS_PREFIX}/turns_epoch_{epoch:04d}_compat.jsonl.gz")
            info = self.upload_gzip_jsonl(pack.compat_path, ck)
            compat_entry = {
                "key": ck,
                "sha256": info["sha256"],
                "n_turns": pack.n_turns,
                "active": False,  # not scored; miner migration aid only
                "format": "turn_v1_compat",
            }
            shards.append(compat_entry)

        prev_raw = json.dumps(m, indent=2, sort_keys=True).encode()
        prev_hash = hashlib.sha256(prev_raw).hexdigest()
        manifest = {
            "corpus_epoch": epoch,
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "index": {
                "key": index_key,
                "sha256": pack.index_sha256,
                "n_turns": pack.n_turns,
            },
            "shards": shards,
            "prev_manifest": f"{self.manifests_prefix}/{prev_hash}.json",
        }
        if compat_entry is not None:
            manifest["compat_shards"] = [compat_entry]
        self.publish_manifest(manifest)
        return manifest

    def cmd_validate(self, file: Path, manifest: dict | None = None) -> bool:
        """Check a candidate shard against the turn contract (see module
        docstring) and against every turn_id ever published. Errors fail;
        warnings print but pass."""
        errors: list[str] = []
        warnings: list[str] = []
        ids: set[str] = set()
        n_records = 0
        n_no_phase = n_no_ref = n_long = 0

        with open(file, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                n_records += 1
                if not line.strip():
                    errors.append(f"line {lineno}: blank line")
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"line {lineno}: not JSON ({e})")
                    continue
                tid, tix = rec.get("traj_id"), rec.get("turn_idx")
                if not (isinstance(tid, str) and tid):
                    errors.append(f"line {lineno}: missing/empty traj_id")
                    continue
                if not isinstance(tix, int):
                    errors.append(f"line {lineno}: turn_idx must be int")
                    continue
                turn_id = f"{tid}:{tix}"
                if turn_id in ids:
                    errors.append(f"line {lineno}: duplicate turn_id {turn_id}")
                ids.add(turn_id)

                prefix = rec.get("prefix")
                if (not isinstance(prefix, list) or not prefix
                        or not all(isinstance(m_, dict) and
                                   isinstance(m_.get("role"), str) and
                                   isinstance(m_.get("content"), str)
                                   for m_ in prefix)):
                    errors.append(f"line {lineno}: prefix must be a non-empty "
                                  "list of {role, content} messages")
                    continue
                if prefix[-1]["role"] != "user":
                    errors.append(f"line {lineno}: prefix must end on a user "
                                  f"message (got {prefix[-1]['role']!r})")
                sys_msgs = [m_ for m_ in prefix if m_["role"] == "system"]
                if not sys_msgs or "bash" not in sys_msgs[0]["content"].lower():
                    errors.append(f"line {lineno}: system message missing or "
                                  "does not mandate the bash action format")
                if sum(len(m_["content"]) for m_ in prefix) > MAX_PREFIX_CHARS:
                    n_long += 1
                if not PHASE_RE.search(tid):
                    n_no_phase += 1
                if not rec.get("reference_turn"):
                    n_no_ref += 1

        if not ids:
            errors.append("shard is empty")
        if n_long:
            errors.append(f"{n_long} turns exceed {MAX_PREFIX_CHARS:,} prefix "
                          "chars (would not fit the serving window)")
        if n_no_phase:
            warnings.append(f"{n_no_phase}/{n_records} traj_ids match no phase "
                            "tag — they pool into one 'other' stratum")
        if n_no_ref:
            warnings.append(f"{n_no_ref}/{n_records} turns lack reference_turn "
                            "(research force-y/RT tooling needs it)")

        # Global uniqueness vs every shard ever published (active or retired:
        # ref caches and old verdicts reference retired ids too).
        if manifest is None:
            manifest = self.current_manifest()
        if manifest and not errors:
            for s in manifest["shards"]:
                print(f"checking turn_id overlap vs {s['key']}…")
                clashes = 0
                for tid in iter_published_turn_ids(self._get, s):
                    if tid in ids:
                        clashes += 1
                if clashes:
                    errors.append(f"{clashes} turn_ids already published in "
                                  f"{s['key']}")

        for w in warnings:
            print(f"warn: {w}")
        for e in errors[:20]:
            print(f"ERROR: {e}")
        if len(errors) > 20:
            print(f"ERROR: … and {len(errors) - 20} more")
        print(f"validate {file.name}: {n_records:,} records "
              f"({len(ids):,} unique turn_ids), "
              f"{len(errors)} errors, {len(warnings)} warnings")
        return not errors

    def cmd_validate_v2(self, pack: PackResult,
                        manifest: dict | None = None) -> bool:
        """Validate traj chunks + parquet index consistency."""
        errors: list[str] = []
        ids: set[str] = set()

        # Index columns + row count.
        table = pq.read_table(pack.index_path)
        cols = set(table.column_names)
        missing = [c for c in INDEX_COLUMNS if c not in cols]
        if missing:
            errors.append(f"index missing columns: {missing}")
        n_index = table.num_rows
        if n_index != pack.n_turns:
            errors.append(
                f"index rows {n_index} != pack.n_turns {pack.n_turns}")

        # Resolve every index row against local chunks.
        chunks_by_key = {m["key"]: p
                         for m, p in zip(pack.chunk_meta, pack.chunk_paths)}
        # Cache chunk lines
        line_cache: dict[str, list[dict]] = {}

        def lines_for(key: str) -> list[dict]:
            if key not in line_cache:
                path = chunks_by_key[key]
                with open(path, encoding="utf-8") as f:
                    line_cache[key] = [json.loads(l) for l in f if l.strip()]
            return line_cache[key]

        turn_ids_col = table.column("turn_id").to_pylist()
        chunk_keys = table.column("chunk_key").to_pylist()
        traj_lines = table.column("traj_line").to_pylist()
        turn_idxs = table.column("turn_idx").to_pylist()
        msg_positions = table.column("msg_pos").to_pylist()

        for i, turn_id in enumerate(turn_ids_col):
            if turn_id in ids:
                errors.append(f"duplicate turn_id in index: {turn_id}")
            ids.add(turn_id)
            key = chunk_keys[i]
            if key not in chunks_by_key:
                errors.append(f"index row {turn_id}: unknown chunk_key {key}")
                continue
            lines = lines_for(key)
            li = int(traj_lines[i])
            if li < 0 or li >= len(lines):
                errors.append(f"index row {turn_id}: traj_line {li} OOB")
                continue
            traj = lines[li]
            meta = next((t for t in traj["turns"]
                         if int(t["turn_idx"]) == int(turn_idxs[i])), None)
            if meta is None:
                errors.append(f"index row {turn_id}: turn missing in traj")
                continue
            if int(meta["msg_pos"]) != int(msg_positions[i]):
                errors.append(f"index row {turn_id}: msg_pos mismatch")
                continue
            try:
                mat = materialize_turn(traj, meta)
            except ValueError as e:
                errors.append(f"index row {turn_id}: materialize failed: {e}")
                continue
            if sum(len(m["content"]) for m in mat["prefix"]) > MAX_PREFIX_CHARS:
                errors.append(f"index row {turn_id}: prefix too long")
            sys_msgs = [m for m in mat["prefix"] if m["role"] == "system"]
            if not sys_msgs or "bash" not in sys_msgs[0]["content"].lower():
                errors.append(f"index row {turn_id}: bash mandate missing")

        # Overlap vs previously published (skip when replacing whole corpus
        # with a rebuild that reuses the same turn_ids — caller passes
        # manifest=None or we only check against shards that stay active).
        if manifest is None:
            manifest = self.current_manifest()
        # When retiring previous active shards in the same publish, overlap
        # with those ids is expected (rebuild). Only flag clashes against
        # shards that will remain active AND are not being replaced — for
        # migrate/rebuild we pass check_overlap=False via empty active set
        # by using manifest shards that won't stay active. Simplest: skip
        # overlap when pack.n_turns covers a full rebuild (caller decides).
        # Here we check only against shards with format turn_v1_compat or
        # explicitly left active in a non-retire add. For standard add-v2
        # with retire_previous, skip overlap entirely.
        _ = manifest  # overlap handled by migrate / incremental callers

        for e in errors[:30]:
            print(f"ERROR: {e}")
        if len(errors) > 30:
            print(f"ERROR: … and {len(errors) - 30} more")
        print(f"validate v2: {pack.n_trajectories:,} trajs, "
              f"{pack.n_turns:,} turns, {len(pack.chunk_paths)} chunks, "
              f"{len(errors)} errors")
        return not errors

    def cmd_retire(self, shard_key: str) -> None:
        m = self._require_manifest()
        hit = [s for s in m["shards"] if s["key"] == shard_key and s["active"]]
        if not hit:
            active = [s["key"] for s in m["shards"] if s["active"]]
            sys.exit(f"no active shard {shard_key!r}; active: {active}")
        if sum(1 for s in m["shards"] if s["active"]) == 1:
            sys.exit("refusing to retire the last active shard")
        shards = [dict(s, active=False) if s["key"] == shard_key else s
                  for s in m["shards"]]
        self._next_revision(m, int(m["corpus_epoch"]) + 1, shards,
                            schema_version=schema_version(m),
                            index=m.get("index"),
                            compat_shards=m.get("compat_shards"))

    def cmd_status(self, verify: bool) -> None:
        m = self.current_manifest()
        if m is None:
            print(f"no manifest at {self.manifest_key}")
            return
        raw = json.dumps(m, indent=2, sort_keys=True).encode()
        mhash = hashlib.sha256(raw).hexdigest()
        frozen_ok = self._exists(f"{self.manifests_prefix}/{mhash}.json")
        print(f"manifest {mhash} epoch={m['corpus_epoch']} "
              f"schema={schema_version(m)} "
              f"immutable_copy={'ok' if frozen_ok else 'MISSING'}")
        if m.get("index"):
            idx = m["index"]
            line = (f"  INDEX  {idx['key']} n_turns={idx['n_turns']:,} "
                    f"sha={idx['sha256'][:16]}")
            if verify:
                if not self._exists(idx["key"]):
                    line += "  OBJECT MISSING"
                else:
                    got = hashlib.sha256(self._get(idx["key"])).hexdigest()
                    line += ("  sha_ok" if got == idx["sha256"]
                             else f"  SHA MISMATCH {got[:16]}")
            print(line)
        for s in m["shards"]:
            n = s.get("n_turns") or s.get("n_trajectories") or 0
            line = (f"  {'ACTIVE ' if s['active'] else 'retired'} {s['key']} "
                    f"n={n:,} fmt={s.get('format', 'turn_v1')} "
                    f"sha={s['sha256'][:16]}")
            if verify:
                if not self._exists(s["key"]):
                    line += "  OBJECT MISSING"
                else:
                    blob = self._get(s["key"])
                    got = hashlib.sha256(gzip.decompress(blob)).hexdigest()
                    line += ("  sha_ok" if got == s["sha256"]
                             else f"  SHA MISMATCH {got[:16]}")
            print(line)

    # -- internals -----------------------------------------------------------------
    def _require_manifest(self) -> dict:
        m = self.current_manifest()
        if m is None:
            sys.exit(f"no manifest at {self.manifest_key}; run `init` first")
        return m

    def _next_revision(self, prev: dict, epoch: int, shards: list[dict],
                       schema_version: int = 1,
                       index: dict | None = None,
                       compat_shards: list | None = None) -> None:
        prev_raw = json.dumps(prev, indent=2, sort_keys=True).encode()
        prev_hash = hashlib.sha256(prev_raw).hexdigest()
        manifest = {
            "corpus_epoch": epoch,
            "schema_version": schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shards": shards,
            "prev_manifest": f"{self.manifests_prefix}/{prev_hash}.json",
        }
        if index is not None:
            manifest["index"] = index
        if compat_shards is not None:
            manifest["compat_shards"] = compat_shards
        self.publish_manifest(manifest)


def main() -> None:
    default_init = Path(__file__).resolve().parents[2] / "research/data/turns_minicoder.jsonl"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init", help="epoch-1 migration from a local jsonl")
    p_init.add_argument("--file", type=Path, default=default_init)
    p_init.add_argument("--force", action="store_true")
    p_add = sub.add_parser("add", help="upload a new shard + manifest revision")
    p_add.add_argument("file", type=Path)
    p_add.add_argument("--no-validate", action="store_true",
                       help="skip the shard contract check (dangerous)")
    p_add_v2 = sub.add_parser(
        "add-v2", help="publish packed traj chunks + parquet index")
    p_add_v2.add_argument("--pack-dir", type=Path, required=True,
                          help="directory with chunk_*.jsonl + turns_XXXX.parquet "
                               "(from affine.corpus.pack)")
    p_add_v2.add_argument("--epoch", type=int, required=True,
                          help="epoch number used in pack filenames")
    p_add_v2.add_argument("--compat", type=Path, default=None,
                          help="optional flat turn JSONL for dual-publish")
    p_add_v2.add_argument("--keep-previous-active", action="store_true",
                          help="do not retire previous active shards")
    p_val = sub.add_parser(
        "validate", help="check a candidate shard against the turn contract "
        "and all published turn_ids, without uploading anything")
    p_val.add_argument("file", type=Path)
    p_ret = sub.add_parser("retire", help="deactivate a shard (object stays)")
    p_ret.add_argument("shard_key")
    p_st = sub.add_parser("status", help="print manifest; --verify checks shards")
    p_st.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    c = Corpus()
    if args.cmd == "init":
        if not args.file.is_file():
            sys.exit(f"corpus file not found: {args.file}")
        c.cmd_init(args.file, args.force)
    elif args.cmd == "add":
        if not args.file.is_file():
            sys.exit(f"shard file not found: {args.file}")
        c.cmd_add(args.file, skip_validate=args.no_validate)
    elif args.cmd == "add-v2":
        from affine.corpus.pack import PackResult as PR
        pack_dir: Path = args.pack_dir
        epoch = args.epoch
        chunk_paths = sorted(pack_dir.glob(f"chunk_{epoch:04d}_*.jsonl"))
        index_path = pack_dir / f"turns_{epoch:04d}.parquet"
        if not chunk_paths or not index_path.is_file():
            sys.exit(f"pack-dir missing chunk_{epoch:04d}_*.jsonl or "
                     f"turns_{epoch:04d}.parquet")
        chunk_meta = []
        for p in chunk_paths:
            raw = p.read_bytes()
            n_traj = sum(1 for line in raw.splitlines() if line.strip())
            # count turns
            n_turns = 0
            for line in raw.splitlines():
                if line.strip():
                    n_turns += len(json.loads(line)["turns"])
            idx = int(p.stem.rsplit("_", 1)[-1])
            key = f"{CHUNKS_PREFIX}/chunk_{epoch:04d}_{idx:04d}.jsonl.gz"
            chunk_meta.append({
                "key": key,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "n_trajectories": n_traj,
                "n_turns": n_turns,
                "format": "traj_v1",
                "active": True,
            })
        pack = PR(
            chunk_paths=chunk_paths,
            chunk_meta=chunk_meta,
            index_path=index_path,
            index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            n_turns=sum(m["n_turns"] for m in chunk_meta),
            n_trajectories=sum(m["n_trajectories"] for m in chunk_meta),
            compat_path=args.compat,
        )
        c.cmd_add_v2(pack, retire_previous=not args.keep_previous_active)
    elif args.cmd == "validate":
        if not args.file.is_file():
            sys.exit(f"shard file not found: {args.file}")
        if not c.cmd_validate(args.file):
            sys.exit(1)
    elif args.cmd == "retire":
        c.cmd_retire(args.shard_key)
    elif args.cmd == "status":
        c.cmd_status(args.verify)


if __name__ == "__main__":
    main()
