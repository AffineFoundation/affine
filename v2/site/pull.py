"""The public pull: one run brings the public files of the object store to the website.

    python v2/site/pull.py --site-dir <dir> --public-url <URL> [--v1-history <file>]

The store comes from the ``SN120_R2_*`` variables. A variable that is not set
is taken from its plain name: ``R2_ENDPOINT`` (a URL or a host name; without
it, ``R2_ACCOUNT_ID``), ``R2_BUCKET``, ``R2_ACCESS_KEY_ID``,
``R2_SECRET_ACCESS_KEY`` (``sn120_v2.r2_names``). A read-only token is enough:
this command never writes to the store.

  --site-dir     the directory that the web server serves. It must exist.
  --public-url   the address under which a browser reaches that directory.
  --v1-history   the frozen v1 history from ``v1_archive.py``, for ``v1.html``.
  --ttl-s        seconds a signed read link works (default 86400, limit 604800).
  --keep-rounds  round directories and task tables to keep on disk (default 200).
  --max-file-mb  an object under ``public/rounds/`` above this size is not copied.
  --max-rollout-mb  a public rollout above this size is not copied (default 512).
  --keep-rollout-rounds  newest mirrored gate rounds to keep; 0 keeps all (default 0).

The prefix ``public/`` of the store can hold objects of other software. This
command reads only keys that ``sn120_v2.layout`` names (``layout_key``). It
lists only ``public/heads/``, ``public/rounds/``, and the two live rollout prefixes, never ``public/``, and
from a list it takes only a key that a function of ``layout`` builds.

One run does this, in this order:

1. Copies ``public/mining.json`` and ``public/llms.txt`` to the path of their
   key, and also to ``mining.json`` and ``llms.txt`` at the root of the site.
   The two files are taken only as a pair of the same snapshot. The crowned-only
   ``public/winner-stats.json`` is then checked against that snapshot and copied
   byte-for-byte to both its canonical key and root ``winner-stats.json``.
2. Copies the head pointer (``layout.CURRENT``), and writes ``LINKS``: signed
   read links for the weight files of the current head, with the expiry time
   and the sha256 and size of each file. Weight files are never copied.
   When the store has a head chain of that head (``layout.HEAD_CHAIN``: the
   box keeps the folded weight files), the chain file is copied, and ``LINKS``
   has the links of the newest checkpoint and of the LoRAs after it.
3. Copies ``public/heads/``, ``public/rounds/``, the task tables of
   ``public/irt/`` that mining.json names, the public reserve and rotation
   artifacts, and ``public/benchmarks.json``.
4. Removes its own old round files, rotation decisions and task tables above
   ``--keep-rounds``. Content-addressed active sets, states and shards stay.
5. Builds the site with ``build.py`` from the copied files: the v1 history,
   then the data file (it takes the reference models of its benchmark table
   from the v1 history), then the pages.

Every file is written to a temporary name and renamed. A file at a key that
never changes is copied once. ``INDEX`` lists every file under ``public/``
that this command wrote: it overwrites and deletes only those. A file at one
of its paths that it did not write is left as it is, and the run fails. The
root files that it replaces are ``ROOT_COPIES`` and the files of ``build.py``.

A step that fails keeps the last good files, the other steps still run, and
the exit code is 1. No signed link is ever printed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build
from sn120_v2 import layout
from sn120_v2.errors import ContractError, V2Error
from sn120_v2.headchain import chain_links, check_chain, heads_of
from sn120_v2.r2 import MAX_PRESIGN_S, R2Store
from sn120_v2.public import validate_winner_stats, winner_enrichment, winner_stats_from_snapshot
from sn120_v2.publisher import refuse_private
from sn120_v2.r2_names import store_env
from sn120_v2.types import HeadManifest

LINKS = "public/read_links.json"  # not an object of the store: this command makes it
INDEX = "public/pull_index.json"  # the files under public/ that this command wrote
ROOT_COPIES = {"llms.txt": layout.LLMS_TXT, "mining.json": layout.MINING_JSON, **layout.PUBLIC_ALIASES}
LINKS_SCHEMA = "sn120-v2-read-links-1"
INDEX_SCHEMA = "sn120-v2-pull-index-1"
HEADS, ROUNDS, IRT = (make("x").partition("x")[0] for make in (layout.head_manifest, layout.round_file, layout.irt_snapshot))
POOL = "public/pool/"
POOL_ACTIVE, POOL_ROTATION, POOL_STATE, POOL_RANKING = (make("x").partition("x")[0] for make in (
    layout.active_pool, layout.pool_rotation, layout.pool_rotation_state, layout.pool_ranking,
))
POOL_SHARDS = "public/pool/reserve/shards/"
ROLLOUT_INDEX = layout.ROLLOUT_INDEX  # the one adoptable hand-written path
BACKFILL_INDEX = "public/rollouts/backfill_index.json"  # never read from the store, never owned
ROLLOUT_INDEX_SCHEMA = "sn120-v2-rollouts-index-2"
ROUND_FILES = (layout.round_file, layout.round_scores, layout.winner_batch)
FIXED = (layout.MINING_JSON, layout.LLMS_TXT, layout.WINNER_STATS, layout.CURRENT, layout.BENCHMARKS, layout.HEAD_CHAIN,
         layout.RESERVE_MANIFEST, layout.RESERVE_WITHDRAWN, layout.RETIRED_POOL, layout.RETIRED_TASKS, layout.POOL_SUMMARY)
RECHECK_ROUNDS = 2  # the newest round files are read again: a restart of the controller may write one twice
PAIR_TRIES, PAIR_WAIT_S = 3, 1.0
STALE_TMP_S = 3600
PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROUND_ID = re.compile(r"r[0-9]{6}\Z")
FAILURES = (V2Error, KeyError, OSError, ValueError, TypeError, AttributeError)
SCHEMA_SOURCE = Path(__file__).parent / "static" / "rollouts" / "SCHEMA.md"


class Refused(ValueError):
    """The run must not do this. The message says what the operator has to do."""


def layout_key(key: str) -> bool:
    """True when ``key`` is exactly a key that ``layout`` builds for a small public file of v2. Nothing else is read."""
    part = key.rsplit("/", 1)[-1].removesuffix(".json")
    try:
        if key.startswith(HEADS):
            return key == layout.head_manifest(part)
        if key.startswith(IRT):
            return key == layout.irt_snapshot(part)
        if key.startswith(ROUNDS):
            return key in {make(key[len(ROUNDS):].split("/")[0]) for make in ROUND_FILES}
        if key.startswith(layout.ROLLOUT_BENCH_PREFIX):
            parts = key[len(layout.ROLLOUT_BENCH_PREFIX):].split("/")
            if len(parts) != 2:
                return False
            model, filename = parts
            if filename.endswith(".jsonl.gz"):
                return key == layout.public_rollout_bench(model, filename[:-9])
            if filename.endswith(".meta.json"):
                return key == layout.public_rollout_bench_meta(model, filename[:-10])
            return False
        if key.startswith(layout.ROLLOUT_GATE_PREFIX):
            parts = key[len(layout.ROLLOUT_GATE_PREFIX):].split("/")
            if not ROUND_ID.fullmatch(parts[0]):
                return False
            if len(parts) == 3 and parts[1] == "entries" and parts[2].endswith(".jsonl.gz"):
                number = parts[2][:-9]
                if not number.isdecimal() or str(int(number)) != number:
                    return False
                return key == layout.public_rollout_gate_entry(parts[0], int(number))
            if len(parts) != 2:
                return False
            if parts[1] == "meta.json":
                return key == layout.public_rollout_gate_meta(parts[0])
            for role in ("head", "winner"):
                if parts[1] == f"{role}.jsonl.gz":
                    return key == layout.public_rollout_gate(parts[0], role)
            return False
        if key.startswith(POOL):
            if key in (layout.RESERVE_MANIFEST, layout.RESERVE_WITHDRAWN, layout.RETIRED_POOL, layout.RETIRED_TASKS, layout.POOL_SUMMARY):
                return True
            if key.startswith(POOL_ACTIVE):
                return bool(SHA256.fullmatch(part)) and key == layout.active_pool(part)
            if key.startswith(POOL_ROTATION):
                return bool(ROUND_ID.fullmatch(part)) and key == layout.pool_rotation(part)
            if key.startswith(POOL_STATE):
                return bool(SHA256.fullmatch(part)) and key == layout.pool_rotation_state(part)
            if key.startswith(POOL_RANKING):
                return bool(ROUND_ID.fullmatch(part)) and key == layout.pool_ranking(part)
            if key.startswith(POOL_SHARDS):
                stem = key[len(POOL_SHARDS):].removesuffix(".jsonl")
                name, separator, sha = stem.rpartition(".")
                return bool(separator and SHA256.fullmatch(sha)) and key == layout.reserve_shard(name, sha)
            return False
    except ContractError:
        return False
    return key in FIXED


def mine(key: str) -> bool:
    """True for a path that this command can write under public/: a key of ``layout``, or one of its own two files."""
    return key in (LINKS, INDEX, ROLLOUT_INDEX) or layout_key(key)


class Pull:
    """One run. ``store`` is any ``ObjectStore`` that can also sign read links (``Presigner``)."""

    def __init__(self, store: Any, site: Path, now: int, keep_rounds: int = 200, max_bytes: int = 128 << 20,
                 max_rollout_bytes: int = 512 << 20, keep_rollout_rounds: int = 0, public_url: str = "") -> None:
        self.store, self.site, self.now, self.keep, self.max_bytes = store, site, now, keep_rounds, max_bytes
        self.public_url = public_url.rstrip("/")
        self.max_rollout_bytes, self.keep_rollout_rounds = max_rollout_bytes, keep_rollout_rounds
        self.owned = self._load_index()
        self.saved = set(self.owned)
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.copied = self.same = self.trimmed = 0

    # ------------------------------------------------------------ the index: what this command may touch

    def _load_index(self) -> set[str]:
        path = self.site / INDEX
        if not path.exists():
            return set()
        try:
            doc = json.loads(path.read_bytes())
            if doc["schema"] != INDEX_SCHEMA:
                raise ValueError("another schema")
            return {key for key in doc["files"] if isinstance(key, str) and mine(key)}
        except FAILURES as error:
            raise Refused(f"{INDEX} is not the index of this command ({type(error).__name__}). Without it the command does not "
                          f"know which files it wrote. Move the directory public/ of the site away and run again") from None

    def _save_index(self) -> None:
        if self.owned != self.saved:
            doc = {"schema": INDEX_SCHEMA, "updated_at": self.now, "files": sorted(self.owned)}
            (self.site / "public").mkdir(exist_ok=True)
            build.write(self.site / INDEX, json.dumps(doc, indent=1).encode())
            self.saved = set(self.owned)

    def put(self, key: str, data: bytes) -> None:
        """Write one file at the path of its key. A file that this command did not write is never replaced."""
        if key == ROLLOUT_INDEX:
            self.owned.add(key)
            self._save_index()
            path = self.site / key
            if path.is_symlink():
                raise Refused(f"{key} is a symlink; leave the hand-written rollout index in place and remove the link")
            path.parent.mkdir(parents=True, exist_ok=True)
            same = path.is_file() and path.read_bytes() == data
            build.write(path, data)
            if same:
                self.same += 1
            else:
                self.copied += 1
            return

        if not mine(key):
            raise Refused(f"{key!r} is not a key of the v2 layout")
        path = self.site / key
        if path.is_symlink() or path.exists():
            same = path.is_file() and not path.is_symlink() and path.read_bytes() == data
            if not same and key not in self.owned:
                raise Refused(f"{key} is in the site directory and this command did not write it; it is left as it is. Move it away")
        else:
            same = False
        if key not in self.owned:  # the claim is saved before the file is written: a run that dies here still owns the file
            self.owned.add(key)
            self._save_index()
        if same:
            self.same += 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        build.write(path, data)
        self.copied += 1

    def put_file(self, key: str, expected_sha: str) -> None:
        """Stream one immutable large file into the site, then check its content."""
        if not layout_key(key) or not SHA256.fullmatch(expected_sha):
            raise Refused(f"refusing unrecognised rollout key or sha256: {key}")
        path = self.site / key
        if key in self.owned and path.is_file() and not path.is_symlink():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() == expected_sha:
                self.same += 1
                return
        if path.exists() and key not in self.owned:
            raise Refused(f"{key} is in the site directory and this command did not write it; it is left as it is")
        if key not in self.owned:
            self.owned.add(key)
            self._save_index()
        self.store.get_file(key, path)  # store writes atomically; no whole rollout in RAM
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            raise ValueError(f"{key} differs from its published sha256")
        self.copied += 1

    def refresh(self, key: str) -> bytes:
        """Copy an object that can change: read it on every run."""
        data = self.store.get(key)
        self.put(key, data)
        return data

    def copy_once(self, key: str) -> None:
        """Copy an object whose bytes never change. A file that is on the disk is not read again."""
        if key in self.owned and (self.site / key).is_file():
            self.same += 1
            return
        size = self.store.size(key)
        if size > self.max_bytes:
            self.notes.append(f"{key} has {size} bytes, above --max-file-mb: not copied")
            return
        self.put(key, self.store.get(key))

    def remove(self, key: str) -> None:
        """Delete one file that this command wrote, and its directory when that is empty."""
        path = self.site / key
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
        self.owned.discard(key)
        self.trimmed += 1

    # ------------------------------------------------------------ the steps

    def contract(self) -> None:
        """mining.json and llms.txt, as a pair of one snapshot, at their keys and at the root of the site."""
        for attempt in range(PAIR_TRIES):
            mining, llms = self.store.get(layout.MINING_JSON), self.store.get(layout.LLMS_TXT)
            digest = json.loads(mining).get("snapshot_digest")
            if isinstance(digest, str) and digest and digest.encode() in llms:
                break
            if attempt + 1 < PAIR_TRIES:
                time.sleep(PAIR_WAIT_S)  # the publisher writes one file after the other
        else:
            raise Refused(f"{layout.LLMS_TXT} and {layout.MINING_JSON} are not of the same snapshot; the last good pair stays")
        for name, data in (("mining.json", mining), ("llms.txt", llms)):
            self.put(ROOT_COPIES[name], data)
            build.write(self.site / name, data)

    def winner_stats(self) -> None:
        """Crowned-only history, checked before its first mirror. A legacy store gets the safe public projection."""
        mining = json.loads((self.site / layout.MINING_JSON).read_bytes())
        try:
            data = self.store.get(layout.WINNER_STATS)
        except KeyError:
            # An older publisher has no separate file yet. Do not stall its public site; never
            # invent batch counts or train knobs which are absent from the public snapshot.
            data = json.dumps(winner_stats_from_snapshot(mining, self.public_url)).encode()
        else:
            checked = json.loads(data)
            validate_winner_stats(mining, checked, self.public_url)
            before = json.dumps(checked).encode()
            self._check_enrichment(mining, checked)
            refuse_private(checked, "winner-stats")
            if json.dumps(checked).encode() != before:
                data = json.dumps(checked).encode()
        self.put(layout.WINNER_STATS, data)
        build.write(self.site / layout.WINNER_STATS_ALIAS, data)

    def _check_enrichment(self, mining: dict, stats: dict, *, required: bool = False) -> None:
        """Check optional fields in place; a disagreement never discards core-validated stats.

        A first pass before mirroring the reserve defers. The post-pool pass strips only
        fields it cannot reproduce, so a newer publisher reserve cannot stall the site.
        """
        from sn120_v2.envs.reserve import load_reserve
        from sn120_v2.public import ENRICHMENT_FIELDS

        if not any(set(row) & ENRICHMENT_FIELDS for row in stats["winners"]):
            return
        root = self.site / layout.RESERVE_MANIFEST.rsplit("/", 1)[0]
        try:
            _, catalog = load_reserve(root, with_catalog=True, allow_retired=True)
        except V2Error as exc:
            # The public reserve on the site uses content-addressed shard names that
            # load_reserve may not be able to read. Do not discard producer-validated
            # enrichment when the catalog simply cannot be loaded; only strip when
            # the catalog is readable and a field cannot be reproduced.
            self.notes.append(
                "winner enrichment check deferred until the public reserve catalog is readable "
                f"({type(exc).__name__})"
            )
            return
        for row in stats["winners"]:
            fields = set(row) & ENRICHMENT_FIELDS
            if not fields:
                continue
            try:
                derived = winner_enrichment(mining, row, self.store, catalog)
                bad = {name for name in fields if name not in derived or row[name] != derived[name]}
                if bad:
                    self._strip_enrichment(row, fields=bad, reason="does not match the public crowned batch and crown-round IRT")
            except FAILURES as exc:
                self._strip_enrichment(row, fields=fields, reason=f"{type(exc).__name__}: {exc}")

    def _strip_enrichment(self, row: dict, *, fields: set[str], reason: str) -> None:
        for name in fields:
            row.pop(name, None)
        self.notes.append(f"winner enrichment stripped for {row['round_id']} ({', '.join(sorted(fields))}): {reason}")

    def head(self, ttl_s: int) -> None:
        """The head pointer, and the signed read links for the weight files of that head."""
        pointer = self.store.get(layout.CURRENT)
        manifest = HeadManifest.from_json(pointer)  # a pointer that does not hash to its ids is refused
        chain = None
        try:
            raw = self.store.get(layout.HEAD_CHAIN)
            doc = json.loads(raw)
            check_chain(doc)
            if doc["head_version_id"] == manifest.head.version_id:  # a chain of another head is not used: the links below then need the full files
                chain = doc
        except KeyError:
            pass  # a store whose box uploads each head in full has no chain
        links = read_links(self.store, manifest, ttl_s, self.now, chain)
        self.put(layout.CURRENT, pointer)
        if chain is not None:
            self.put(layout.HEAD_CHAIN, raw)
        elif layout.HEAD_CHAIN in self.owned:
            self.remove(layout.HEAD_CHAIN)  # the copy is of an older head, and the links above are of the full files
        self.put(LINKS, json.dumps(links, indent=1).encode())

    def heads(self) -> None:
        """Head manifests. A file is taken only when it is a head manifest with the id of its key."""
        for key in self.store.list(HEADS):
            if not layout_key(key) or (key in self.owned and (self.site / key).is_file()):
                continue
            data = self.store.get(key)
            try:
                if layout.head_manifest(HeadManifest.from_json(data).head.version_id) != key:
                    raise ValueError("the id of the head is not the id of the key")
            except (V2Error, ValueError, KeyError, TypeError, AttributeError) as error:
                self.notes.append(f"{key} is not a v2 head manifest ({type(error).__name__}): not copied")
                continue
            self.put(key, data)

    def named(self) -> tuple[set[str], set[str]]:
        """The round ids and the task table ids that the copied mining.json names."""
        try:
            doc = json.loads((self.site / layout.MINING_JSON).read_bytes())
        except (OSError, ValueError):
            return set(), set()
        rounds = [*(doc.get("history") or []), *([doc["round"]] if doc.get("round") else [])]
        ids = {str(row.get("round_id")) for row in [*rounds, *(doc.get("kings") or [])]}
        tables = {str(row.get("irt_snapshot_id") or "") for row in rounds} | {str(doc.get("irt_snapshot_id") or "")}
        return ids, {table for table in tables if PART.fullmatch(table)}

    def rounds(self) -> None:
        """Round files: every file of the newest rounds and of the rounds that mining.json names."""
        found: dict[str, list[str]] = {}
        for key in self.store.list(ROUNDS):
            if layout_key(key):
                found.setdefault(key.split("/")[2], []).append(key)
        newest = sorted(found, reverse=True)  # a round id is "r" and a number of fixed width: the text order is the time order
        keep = set(newest[: self.keep]) | (self.named()[0] & set(found))
        for round_id in sorted(keep):
            for key in found[round_id]:
                if round_id in newest[:RECHECK_ROUNDS] and key != layout.winner_batch(round_id):  # a batch never changes
                    self.refresh(key)
                else:
                    self.copy_once(key)
        for key in sorted(self.owned):
            if key.startswith(ROUNDS) and key.split("/")[2] not in keep:
                self.remove(key)

    def pool(self) -> None:
        """Public reserve and the rotation artifacts named by mining.json rounds."""
        if self.store.exists(layout.RESERVE_MANIFEST):
            manifest = self.refresh(layout.RESERVE_MANIFEST)
            self.refresh(layout.RESERVE_WITHDRAWN)
            for line in manifest.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "shard" not in row:
                    continue
                key = layout.reserve_shard(row["shard"], row["sha256"])
                data = self.store.get(key)
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    self.notes.append(f"{key} does not match the reserve manifest sha256: not copied")
                    continue
                if len(data) <= self.max_bytes:
                    self.put(key, data)
                else:
                    self.notes.append(f"{key} has {len(data)} bytes, above --max-file-mb: not copied")
        named = self.named()[0]
        views = {}
        for round_id in named:
            try:
                view = json.loads((self.site / layout.round_file(round_id)).read_bytes())
            except (OSError, ValueError):
                if self.store.exists(layout.round_file(round_id)):
                    view = json.loads(self.store.get(layout.round_file(round_id)))
                else:
                    # open round: round.json is written only at close; mining.json carries its pool block now
                    try:
                        doc = json.loads((self.site / layout.MINING_JSON).read_bytes())
                    except (OSError, ValueError):
                        continue
                    view = doc.get("round") if isinstance(doc.get("round"), dict) and str(doc["round"].get("round_id")) == round_id else None
                    if view is None:
                        continue
            if isinstance(view.get("pool"), dict):
                views[round_id] = view
        keep = set(sorted(named, reverse=True)[:self.keep])
        for round_id in sorted(views.keys() & keep):
            key = layout.pool_rotation(round_id)
            try:
                if round_id in sorted(named, reverse=True)[:RECHECK_ROUNDS]:
                    raw = self.refresh(key)
                else:
                    self.copy_once(key)
                    raw = self.store.get(key)
                rotation = json.loads(raw)
                block = views[round_id]["pool"]
                for name in ("active_sha", "prev_active_sha", "initial_active_sha"):
                    if rotation.get(name) or block.get(name):
                        self.copy_once(layout.active_pool(rotation.get(name) or block[name]))
                for name in ("state_sha", "prev_state_sha"):
                    if rotation.get(name) or block.get(name):
                        self.copy_once(layout.pool_rotation_state(rotation.get(name) or block[name]))
                if rotation.get("rule_version") == "affine-pool-rotation-v2":
                    key = layout.pool_ranking(round_id)
                    if round_id in sorted(named, reverse=True)[:RECHECK_ROUNDS]:
                        self.refresh(key)
                    else:
                        self.copy_once(key)
            except (KeyError, ValueError, TypeError, AttributeError) as error:
                self.notes.append(f"{key} cannot be mirrored ({type(error).__name__}): not copied")
        for key in sorted(self.owned):
            if key.startswith((POOL_ROTATION, POOL_RANKING)) and key.rsplit("/", 1)[-1].removesuffix(".json") not in keep:
                self.remove(key)
        for key in (layout.RETIRED_POOL, layout.RETIRED_TASKS):
            if self.store.exists(key):
                self.refresh(key)
        if self.store.exists(layout.POOL_SUMMARY):
            self.refresh(layout.POOL_SUMMARY)

    def tables(self) -> None:
        """Task tables: the ones that mining.json names. Older ones stay until there are more than ``--keep-rounds``."""
        named = self.named()[1]
        for table in sorted(named):
            try:
                self.copy_once(layout.irt_snapshot(table))
            except KeyError:
                self.notes.append(f"{layout.irt_snapshot(table)} is named in mining.json and is not in the store yet")
        old = [key for key in self.owned if key.startswith(IRT) and key.split("/")[-1].removesuffix(".json") not in named]
        age = {key: (self.site / key).stat().st_mtime if (self.site / key).is_file() else 0.0 for key in old}
        for key in sorted(old, key=lambda key: (-age[key], key))[max(self.keep - len(named), 0):]:
            self.remove(key)

    def rollouts(self) -> None:
        """Mirror exactly the live layout, verify bytes, and write its index after every included object."""
        bench, gate = [], []
        found_gate = set()
        for key in self.store.list(layout.ROLLOUT_BENCH_PREFIX):
            if not layout_key(key) or not key.endswith(".meta.json"):
                continue
            meta = json.loads(self.store.get(key))
            model, bid = key[len(layout.ROLLOUT_BENCH_PREFIX):].split("/")
            bid = bid[:-10]
            data_key = layout.public_rollout_bench(model, bid)
            if meta.get("schema") != "sn120-v2-rollouts-bench-1" or meta.get("key") != data_key:
                raise ValueError(f"{key} is not a live bench rollout meta")
            if meta.get("bytes", self.max_rollout_bytes + 1) > self.max_rollout_bytes or self.store.size(data_key) > self.max_rollout_bytes:
                self.notes.append(f"{data_key} is above --max-rollout-mb: not copied")
                continue
            self.put_file(data_key, meta["sha256"])
            self.put(key, self.store.get(key))
            bench.append(meta)
        for key in self.store.list(layout.ROLLOUT_GATE_PREFIX):
            if not layout_key(key) or not key.endswith("/meta.json"):
                continue
            round_id = key[len(layout.ROLLOUT_GATE_PREFIX):].split("/")[0]
            found_gate.add(round_id)
        newest = sorted(found_gate, reverse=True)
        keep_gate = set(newest[:self.keep_rollout_rounds]) if self.keep_rollout_rounds else found_gate
        for round_id in sorted(keep_gate):
            meta_key = layout.public_rollout_gate_meta(round_id)
            meta = json.loads(self.store.get(meta_key))
            if meta.get("schema") not in ("sn120-v2-rollouts-gate-1", "sn120-v2-rollouts-gate-2") or meta.get("round_id") != round_id:
                raise ValueError(f"{meta_key} is not a live gate rollout meta")
            included = []
            for file_meta in meta.get("files") or []:
                role = file_meta.get("role")
                if role == "entry" and meta.get("schema") == "sn120-v2-rollouts-gate-2":
                    if file_meta.get("missing"):
                        included.append(file_meta)
                        continue
                    key = layout.public_rollout_gate_entry(round_id, file_meta["queue_pos"])
                elif role in ("head", "winner"):
                    key = layout.public_rollout_gate(round_id, role)
                else:
                    raise ValueError(f"{meta_key} names an invalid rollout file")
                if file_meta.get("key") != key:
                    raise ValueError(f"{meta_key} names an invalid rollout file")
                if file_meta.get("bytes", self.max_rollout_bytes + 1) > self.max_rollout_bytes or self.store.size(key) > self.max_rollout_bytes:
                    self.notes.append(f"{key} is above --max-rollout-mb: not copied")
                    continue
                self.put_file(key, file_meta["sha256"])
                included.append(file_meta)
            self.put(meta_key, self.store.get(meta_key))
            gate.append(meta | {"files": included})
        for key in sorted(self.owned):
            if key.startswith(layout.ROLLOUT_GATE_PREFIX) and key.split("/")[3] not in keep_gate:
                self.remove(key)
        try:
            backfill = json.loads((self.site / BACKFILL_INDEX).read_bytes()) if (self.site / BACKFILL_INDEX).is_file() else None
        except ValueError:
            backfill = None
            self.notes.append(f"{BACKFILL_INDEX} is not valid JSON: kept unchanged; index backfill is null")
        doc = dict(schema=ROLLOUT_INDEX_SCHEMA, backfill=backfill, bench=bench, gate=gate, schema_doc="SCHEMA.md")
        self.put(ROLLOUT_INDEX, json.dumps(doc, indent=1, sort_keys=True).encode())

    def benchmarks(self) -> None:
        try:
            self.refresh(layout.BENCHMARKS)
        except KeyError:
            pass  # the benchmark service does not run at launch: no file is the normal state

    def clean(self) -> None:
        """Remove temporary files of a run that was killed. Only names that ``build.write`` makes, only under public/."""
        for path in (self.site / "public").rglob(".tmp-*") if (self.site / "public").is_dir() else ():
            if path.is_file() and not path.is_symlink() and self.now - path.stat().st_mtime > STALE_TMP_S:
                path.unlink(missing_ok=True)

    def step(self, name: str, run: Callable[[], None]) -> None:
        """Run one step. A failure is recorded, the files of that step stay as they were, and the next step runs."""
        try:
            run()
        except FAILURES as error:
            text = f"missing in the store: {error}" if isinstance(error, KeyError) else f"{type(error).__name__}: {error}"
            self.problems.append(f"{name}: {text}")
        finally:
            try:
                self._save_index()
            except OSError as error:
                self.problems.append(f"index: {type(error).__name__}: {error}")


def read_links(store: Any, manifest: HeadManifest, ttl_s: int, now: int, chain: dict[str, Any] | None = None) -> dict[str, Any]:
    """Signed read links for every weight file of one head. Each file is looked up in the store, so no link is dead.

    With ``chain`` (the head chain of this head): the links of the newest
    checkpoint and of the LoRAs of each step after it. ``base`` and ``master``
    are then null unless that checkpoint is the base of the head itself.
    """

    def file_set(set_id: str, files: dict[str, str], key_of: Callable[[str, str], str]) -> dict[str, Any]:
        rows = []
        for name, sha256 in sorted(files.items()):
            key = key_of(set_id, name)
            rows.append({"name": name, "key": key, "sha256": sha256, "size": store.size(key), "url": store.presign_get(key, ttl_s)})
        return {"id": set_id, "bytes": sum(row["size"] for row in rows), "files": rows}

    head = manifest.head
    master_note = {"note": "The fp32 master. It is needed only to check a fold. A miner does not need it. Keep it out of the directory of the base."}
    if chain is None:
        base = file_set(head.base_hash, manifest.base_files, layout.public_base_file)
        master = file_set(head.base_fp32_master_digest, manifest.master_files, layout.public_master_file) | master_note
        rebuild = {}
    else:
        rebuild = chain_links(store, chain, ttl_s, target=heads_of(chain)[-1]["version_id"])
        point = rebuild["checkpoints"][0]
        here = point["base"]["id"] == head.base_hash
        base, master = (point["base"], point["master"] | master_note) if here else (None, None)
        rebuild["rebuild"] = (
            "The box keeps the folded weight files. checkpoints has the newest head whose full files are in the store. "
            "chain_loras has the LoRAs of each fold after it, in order. base and master are null when the head is not that checkpoint. "
            "Then make the head: python -m sn120_v2.headfold rebuild --chain <mining.json> --links <this file> --work <directory>"
        )
    return {
        "schema": LINKS_SCHEMA,
        "generated_at": now,
        "expires_at": now + ttl_s,  # every link was signed after ``now``: none ends before this time
        "ttl_s": ttl_s,
        "head_version_id": head.version_id,
        "how": "GET each url before expires_at, then check the size and the sha256 of the file. The id of a file set is the "
               "sha256 of the canonical JSON of {file name: sha256}. This file is written again on every run with new links.",
        "base": base,
        "loras": [
            file_set(ref.lora_hash, manifest.lora_files[ref.lora_hash], layout.public_lora_file) | {"round_id": ref.round_id}
            for ref in head.loras
        ],
        "master": master,
    } | rebuild


def run(store: Any, site: Path, public_url: str, ttl_s: int, keep_rounds: int, max_bytes: int, v1_history: str | None, now: int,
        max_rollout_bytes: int = 512 << 20, keep_rollout_rounds: int = 0) -> int:
    def link(key: str) -> str:
        return f"{public_url.rstrip('/')}/{key}"

    def ready() -> None:
        if not (site / layout.MINING_JSON).is_file():  # until v2 data is here, the pages that are in the directory stay
            raise Refused(f"no {layout.MINING_JSON} in the site directory yet; the site is not built")

    def site_data() -> None:
        ready()
        build.build_data(str(site), site, link, quiet_missing=True)

    def v1_page_data() -> None:
        ready()
        build.install_v1(v1_history, site)

    def site_pages() -> None:  # after the data: a new page never waits for its data file
        ready()
        build.build_static(site, link)

    pull = Pull(store, site, now, keep_rounds, max_bytes, max_rollout_bytes, keep_rollout_rounds, public_url)
    pull.step("clean", pull.clean)
    pull.step("llms.txt and mining.json", pull.contract)
    pull.step("winner stats", pull.winner_stats)
    pull.step("head and read links", lambda: pull.head(ttl_s))
    pull.step("heads", pull.heads)
    pull.step("rounds", pull.rounds)
    pull.step("task tables", pull.tables)
    pull.step("pool", pull.pool)

    def enriched_fields() -> None:
        stats = json.loads((site / layout.WINNER_STATS).read_bytes())
        before = json.dumps(stats).encode()
        pull._check_enrichment(json.loads((site / layout.MINING_JSON).read_bytes()), stats, required=True)
        data = json.dumps(stats).encode()
        if data != before:
            pull.put(layout.WINNER_STATS, data)
            build.write(site / layout.WINNER_STATS_ALIAS, data)

    pull.step("winner stats enriched fields", enriched_fields)
    pull.step("benchmarks", pull.benchmarks)
    pull.step("rollouts", pull.rollouts)
    if (site / ROLLOUT_INDEX).is_file():
        # The schema lives beside the live index even with a read-only bucket.
        # An existing hand-written SCHEMA.md is left untouched by this first pass.
        (site / "public/rollouts").mkdir(parents=True, exist_ok=True)
        build.write(site / "public/rollouts/SCHEMA.md", SCHEMA_SOURCE.read_bytes())
    if v1_history:
        pull.step("v1 history", v1_page_data)
    pull.step("site data", site_data)
    pull.step("site pages", site_pages)
    # A contract that changed while the long pull ran still gets mirrored before this locked run exits.
    pull.step("llms.txt and mining.json final refresh", pull.contract)
    for note in pull.notes:
        print(f"pull: note: {note}", file=sys.stderr)
    for problem in pull.problems:
        print(f"pull: FAILED {problem}", file=sys.stderr)
    state = "the files of each failed step are the last good ones" if pull.problems else "site built"
    print(f"pull: wrote {pull.copied}, unchanged {pull.same}, removed {pull.trimmed}, failed steps {len(pull.problems)}; {state}")
    return 1 if pull.problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--site-dir", required=True, type=Path, help="the directory that the web server serves; it must exist")
    parser.add_argument("--public-url", required=True, help="the address under which a browser reaches that directory")
    parser.add_argument("--v1-history", help="the frozen v1 history from v1_archive.py: a file or an http(s) URL")
    parser.add_argument("--ttl-s", type=int, default=86400, help="seconds a signed read link works")
    parser.add_argument("--keep-rounds", type=int, default=200, help="round directories and task tables to keep on disk")
    parser.add_argument("--max-file-mb", type=int, default=128, help="an object under public/rounds/ above this size is not copied")
    parser.add_argument("--max-rollout-mb", type=int, default=512, help="one rollout object above this size is not copied")
    parser.add_argument("--keep-rollout-rounds", type=int, default=0, help="newest mirrored gate rounds to keep; 0 keeps all")
    args = parser.parse_args(argv)
    if not args.public_url.startswith(("http://", "https://")):
        parser.error("--public-url must be an http(s) address")
    if not 0 < args.ttl_s <= MAX_PRESIGN_S:
        parser.error(f"--ttl-s must be 1..{MAX_PRESIGN_S}")
    if args.keep_rollout_rounds < 0 or args.max_rollout_mb < 1:
        parser.error("--max-rollout-mb must be at least 1 and --keep-rollout-rounds must be at least 0")
    if args.keep_rounds < RECHECK_ROUNDS or args.max_file_mb < 1:
        parser.error(f"--keep-rounds must be at least {RECHECK_ROUNDS} and --max-file-mb at least 1")
    site = args.site_dir
    if not site.is_dir():
        print(f"pull: FAILED {site} is not a directory; nothing written", file=sys.stderr)
        return 1
    lock = os.open(site, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # two runs at once could each remove what the other wrote
        store = R2Store.from_env(store_env(os.environ))
        return run(store, site, args.public_url, args.ttl_s, args.keep_rounds, args.max_file_mb << 20, args.v1_history, int(time.time()),
                   args.max_rollout_mb << 20, args.keep_rollout_rounds)
    except BlockingIOError:
        print("pull: FAILED another pull runs on this directory; nothing written", file=sys.stderr)
    except (V2Error, Refused) as error:  # the store did not open, or the index is not ours
        print(f"pull: FAILED {error}; nothing written, the last good files stay", file=sys.stderr)
    finally:
        os.close(lock)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
