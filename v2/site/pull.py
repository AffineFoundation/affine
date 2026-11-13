"""The public pull: one run brings the public files of the object store to the website.

    python v2/site/pull.py --site-dir <dir> --public-url <URL> [--v1-history <file>]

The store comes from the ``SN120_R2_*`` variables (``R2Store.from_env``). A
read-only token is enough: this command never writes to the store.

  --site-dir     the directory that the web server serves. It must exist.
  --public-url   the address under which a browser reaches that directory.
  --v1-history   the frozen v1 history from ``v1_archive.py``, for ``v1.html``.
  --ttl-s        seconds a signed read link works (default 86400, limit 604800).
  --keep-rounds  round directories and task tables to keep on disk (default 200).
  --max-file-mb  an object under ``public/rounds/`` above this size is not copied.

The prefix ``public/`` of the store can hold objects of other software. This
command reads only keys that ``sn120_v2.layout`` names (``layout_key``). It
lists only ``public/heads/`` and ``public/rounds/``, never ``public/``, and
from a list it takes only a key that a function of ``layout`` builds.

One run does this, in this order:

1. Copies ``public/mining.json`` and ``public/llms.txt`` to the path of their
   key, and also to ``mining.json`` and ``llms.txt`` at the root of the site.
   The two files are taken only as a pair of the same snapshot.
2. Copies the head pointer (``layout.CURRENT``), and writes ``LINKS``: signed
   read links for the weight files of the current head, with the expiry time
   and the sha256 and size of each file. Weight files are never copied.
3. Copies ``public/heads/``, ``public/rounds/``, the task tables of
   ``public/irt/`` that mining.json names, and ``public/benchmarks.json``.
4. Removes its own old round files and task tables above ``--keep-rounds``.
5. Builds the site with ``build.py`` from the copied files: the data file,
   then the v1 history, then the pages.

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
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import build
from sn120_v2 import layout
from sn120_v2.errors import ContractError, V2Error
from sn120_v2.r2 import MAX_PRESIGN_S, R2Store
from sn120_v2.types import HeadManifest

LINKS = "public/read_links.json"  # not an object of the store: this command makes it
INDEX = "public/pull_index.json"  # the files under public/ that this command wrote
ROOT_COPIES = {"llms.txt": layout.LLMS_TXT, "mining.json": layout.MINING_JSON}
LINKS_SCHEMA = "sn120-v2-read-links-1"
INDEX_SCHEMA = "sn120-v2-pull-index-1"
HEADS, ROUNDS, IRT = (make("x").partition("x")[0] for make in (layout.head_manifest, layout.round_file, layout.irt_snapshot))
ROUND_FILES = (layout.round_file, layout.round_scores, layout.winner_batch)
FIXED = (layout.MINING_JSON, layout.LLMS_TXT, layout.CURRENT, layout.BENCHMARKS)
RECHECK_ROUNDS = 2  # the newest round files are read again: a restart of the controller may write one twice
PAIR_TRIES, PAIR_WAIT_S = 3, 1.0
STALE_TMP_S = 3600
PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
FAILURES = (V2Error, KeyError, OSError, ValueError, TypeError, AttributeError)


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
    except ContractError:
        return False
    return key in FIXED


def mine(key: str) -> bool:
    """True for a path that this command can write under public/: a key of ``layout``, or one of its own two files."""
    return key in (LINKS, INDEX) or layout_key(key)


class Pull:
    """One run. ``store`` is any ``ObjectStore`` that can also sign read links (``Presigner``)."""

    def __init__(self, store: Any, site: Path, now: int, keep_rounds: int = 200, max_bytes: int = 128 << 20) -> None:
        self.store, self.site, self.now, self.keep, self.max_bytes = store, site, now, keep_rounds, max_bytes
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

    def head(self, ttl_s: int) -> None:
        """The head pointer, and the signed read links for the weight files of that head."""
        pointer = self.store.get(layout.CURRENT)
        manifest = HeadManifest.from_json(pointer)  # a pointer that does not hash to its ids is refused
        links = read_links(self.store, manifest, ttl_s, self.now)
        self.put(layout.CURRENT, pointer)
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


def read_links(store: Any, manifest: HeadManifest, ttl_s: int, now: int) -> dict[str, Any]:
    """Signed read links for every weight file of one head. Each file is looked up in the store, so no link is dead."""

    def file_set(set_id: str, files: dict[str, str], key_of: Callable[[str, str], str]) -> dict[str, Any]:
        rows = []
        for name, sha256 in sorted(files.items()):
            key = key_of(set_id, name)
            rows.append({"name": name, "key": key, "sha256": sha256, "size": store.size(key), "url": store.presign_get(key, ttl_s)})
        return {"id": set_id, "bytes": sum(row["size"] for row in rows), "files": rows}

    head = manifest.head
    return {
        "schema": LINKS_SCHEMA,
        "generated_at": now,
        "expires_at": now + ttl_s,  # every link was signed after ``now``: none ends before this time
        "ttl_s": ttl_s,
        "head_version_id": head.version_id,
        "how": "GET each url before expires_at, then check the size and the sha256 of the file. The id of a file set is the "
               "sha256 of the canonical JSON of {file name: sha256}. This file is written again on every run with new links.",
        "base": file_set(head.base_hash, manifest.base_files, layout.public_base_file),
        "loras": [
            file_set(ref.lora_hash, manifest.lora_files[ref.lora_hash], layout.public_lora_file) | {"round_id": ref.round_id}
            for ref in head.loras
        ],
        "master": file_set(head.base_fp32_master_digest, manifest.master_files, layout.public_master_file)
        | {"note": "The fp32 master. It is needed only to check a fold. A miner does not need it. Keep it out of the directory of the base."},
    }


def run(store: Any, site: Path, public_url: str, ttl_s: int, keep_rounds: int, max_bytes: int, v1_history: str | None, now: int) -> int:
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

    pull = Pull(store, site, now, keep_rounds, max_bytes)
    pull.step("clean", pull.clean)
    pull.step("llms.txt and mining.json", pull.contract)
    pull.step("head and read links", lambda: pull.head(ttl_s))
    pull.step("heads", pull.heads)
    pull.step("rounds", pull.rounds)
    pull.step("task tables", pull.tables)
    pull.step("benchmarks", pull.benchmarks)
    pull.step("site data", site_data)
    if v1_history:
        pull.step("v1 history", v1_page_data)
    pull.step("site pages", site_pages)
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
    args = parser.parse_args(argv)
    if not args.public_url.startswith(("http://", "https://")):
        parser.error("--public-url must be an http(s) address")
    if not 0 < args.ttl_s <= MAX_PRESIGN_S:
        parser.error(f"--ttl-s must be 1..{MAX_PRESIGN_S}")
    if args.keep_rounds < RECHECK_ROUNDS or args.max_file_mb < 1:
        parser.error(f"--keep-rounds must be at least {RECHECK_ROUNDS} and --max-file-mb at least 1")
    site = args.site_dir
    if not site.is_dir():
        print(f"pull: FAILED {site} is not a directory; nothing written", file=sys.stderr)
        return 1
    lock = os.open(site, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # two runs at once could each remove what the other wrote
        store = R2Store.from_env()
        return run(store, site, args.public_url, args.ttl_s, args.keep_rounds, args.max_file_mb << 20, args.v1_history, int(time.time()))
    except BlockingIOError:
        print("pull: FAILED another pull runs on this directory; nothing written", file=sys.stderr)
    except (V2Error, Refused) as error:  # the store did not open, or the index is not ours
        print(f"pull: FAILED {error}; nothing written, the last good files stay", file=sys.stderr)
    finally:
        os.close(lock)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
