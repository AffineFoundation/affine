"""The site reset at the cutover: v2's llms.txt alone, v1's files out, and the way back.

    python v2/site/reset.py llms --site-dir <site directory>
    python v2/site/reset.py out  --site-dir <site directory> --to <new directory> [--scores] [--assets] [--dry-run]
    python v2/site/reset.py back --site-dir <site directory> --from <copy of the v1 site> [--dry-run]

The public pull (``pull.py``) is the site switch: its first good run replaces ``llms.txt``, ``mining.json`` and
``index.html`` at the root. This command does the three things around that switch.

``llms`` copies the store's ``llms.txt`` to the root of the site, and nothing else: the new contract can go up
before the site. The store comes from the same variables as for the pull (``SN120_R2_*``, or the plain ``R2_*``
names); a read-only token is enough. The file is taken only when it is of the same snapshot as the store's ``mining.json``, as the pull takes it, so
the pull later finds the same bytes and writes nothing.

``out`` moves every v1 entry out of the site directory into ``--to``. It runs after the pull wrote the v2 site,
and it refuses when the v2 site is not there. It never deletes and never replaces: each entry is renamed, so
``--to`` must be on the same file system. What stays is named in ``KEEP`` and ``FF_KEEP``. Two kinds of v1 files
stay until a flag says otherwise. ``scores.json``: v1's weight writer reads it; ``--scores`` moves it when that
writer is stopped. A ``.js`` or ``.css`` file: the web server lets the CDN keep a "not found" answer of such an
address for a day, and that would break the way back; ``--assets`` moves them when the way back is not needed.

``back`` copies every file of a full copy of the v1 site (``cp -a``, made before the first run of the pull) into
the site directory again, each one under a temporary name and then renamed, ``index.html`` last. It leaves the
link files and the v2 files as they are. Stop the pull's timer first, or the next run replaces the files again.

No path has a default. Exit code 0: done. 1: something could not be read, moved or copied. 2: refused, nothing changed.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from build import ASSETS, DATA, V1_DATA, write
from sn120_v2 import layout
from sn120_v2.errors import V2Error
from sn120_v2.r2 import R2Store
from sn120_v2.r2_names import store_env

V2_SITE = (*ASSETS, DATA, V1_DATA)
KEEP = frozenset({
    *V2_SITE,
    "llms.txt", "mining.json",  # the pull replaces them in place; the two addresses never go away
    "public",  # the pull
    "ff",  # the upload link job; see FF_KEEP
    "llms-page.html", "network.css",  # the web server renders /llms.txt for a browser with these two
    "robots.txt",
    "favicon-96x96.png", "affine-mark.png",  # the CDN may keep a "not found" answer of these names for 7 days
})
FF_KEEP = frozenset({"links", "links.json"})  # the upload link job writes them on every run
SCORES = "scores.json"  # v1's weight writer reads it
ASSET_ENDS = (".js", ".css")


def refuse(text: str) -> int:
    print(f"reset: refused: {text}", file=sys.stderr)
    return 2


def v2_site_is_there(site: Path) -> bool:
    page = site / "index.html"
    return all((site / name).is_file() for name in ASSETS) and "site.js?v=" in page.read_text(errors="replace")


def llms(site: Path) -> int:
    if not site.is_dir():
        return refuse(f"{site} is not a directory")
    try:
        store = R2Store.from_env(store_env(os.environ))
        text, mining = store.get(layout.LLMS_TXT), store.get(layout.MINING_JSON)
        digest = json.loads(mining).get("snapshot_digest")
    except KeyError as error:
        print(f"reset: llms.txt not copied: missing in the store: {error}; the file in the site directory stays", file=sys.stderr)
        return 1
    except (V2Error, ValueError, AttributeError) as error:
        print(f"reset: llms.txt not copied: {type(error).__name__}: {error}; the file in the site directory stays", file=sys.stderr)
        return 1
    if not isinstance(digest, str) or not digest or digest.encode() not in text:
        print(f"reset: llms.txt not copied: {layout.LLMS_TXT} and {layout.MINING_JSON} are not of the same snapshot; "
              "the file in the site directory stays. Run again in a moment", file=sys.stderr)
        return 1
    write(site / "llms.txt", text)
    print(f"reset llms: {site / 'llms.txt'} is the store's {layout.LLMS_TXT}: {len(text)} bytes, sha256 {hashlib.sha256(text).hexdigest()}")
    return 0


def to_move(site: Path, scores: bool, assets: bool) -> tuple[list[str], list[str]]:
    """(paths to move, paths that stay), relative to the site directory."""
    move, stay = [], []
    for entry in sorted(os.listdir(site)):
        if entry == "ff" and (site / entry).is_dir():
            for inner in sorted(os.listdir(site / entry)):
                (stay if inner in FF_KEEP or inner.startswith(".") or inner.endswith(".part") else move).append(f"ff/{inner}")
        elif entry in KEEP or entry.startswith(".") or (entry == SCORES and not scores) or (entry.endswith(ASSET_ENDS) and not assets):
            stay.append(entry)
        else:
            move.append(entry)
    return move, stay


def out(site: Path, to: Path, scores: bool, assets: bool, dry_run: bool) -> int:
    if not site.is_dir():
        return refuse(f"{site} is not a directory")
    if not v2_site_is_there(site):
        return refuse(f"the v2 site is not in {site}: the pull must write it first, then v1 moves out")
    if to.resolve().is_relative_to(site.resolve()):
        return refuse(f"{to} is inside the site directory: the web server would serve it")
    if not to.parent.is_dir() or os.stat(to.parent).st_dev != os.stat(site).st_dev:
        return refuse(f"{to.parent} must be a directory on the file system of {site}: an entry is renamed, never copied")
    move, stay = to_move(site, scores, assets)
    taken = [path for path in move if (to / path).exists() or (to / path).is_symlink()]
    if taken:
        return refuse(f"{to} already holds {', '.join(taken)}: give a new directory")
    failed = 0
    for path in move:
        if dry_run:
            print(f"would move {path}")
            continue
        try:
            (to / path).parent.mkdir(parents=True, exist_ok=True)
            os.rename(site / path, to / path)
        except OSError as error:
            failed += 1
            print(f"reset: not moved: {path}: {error}", file=sys.stderr)
    print(f"reset out: {'would move' if dry_run else 'moved'} {len(move) - failed} of {len(move)} entries to {to}; "
          f"{len(stay)} stay: {' '.join(stay)}")
    return 1 if failed else 0


def back(site: Path, copy: Path, dry_run: bool) -> int:
    if not site.is_dir():
        return refuse(f"{site} is not a directory")
    if not (copy / "index.html").is_file() or v2_site_is_there(copy):
        return refuse(f"{copy} is not a copy of the v1 site: it has no index.html, or its index.html is the v2 page")
    files = []
    for folder, _, names in os.walk(copy):
        for name in names:
            path = (Path(folder) / name).relative_to(copy)
            if path.parts[0] == "ff" and len(path.parts) > 1 and path.parts[1] in FF_KEEP:
                continue  # an old link file is never put back: the link job that runs writes the full set
            files.append(path)
    files.sort(key=lambda path: (path.as_posix() == "index.html", path.as_posix()))  # the page last, after its files
    copied = failed = 0
    for path in files:
        source, target = copy / path, site / path
        if not source.is_file() or source.is_symlink():
            continue
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            continue
        if dry_run:
            print(f"would copy {path}")
            copied += 1
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-reset-")
            os.close(handle)
            shutil.copy2(source, tmp)
            os.replace(tmp, target)
            copied += 1
        except OSError as error:
            failed += 1
            print(f"reset: not copied: {path}: {error}", file=sys.stderr)
    from_v1 = {path.as_posix() for path in files}
    left = [name for name in V2_SITE if name not in from_v1 and (site / name).exists()]
    print(f"reset back: {'would copy' if dry_run else 'copied'} {copied} files from {copy}, {failed} failed; "
          f"these v2 files stay, and no v1 page links them: {' '.join(left)}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("llms", help="copy the store's llms.txt alone to the root of the site")
    one = commands.add_parser("out", help="move v1's entries out of the site directory, after the pull wrote the v2 site")
    one.add_argument("--to", required=True, type=Path, help="a directory outside the site directory, on the same file system")
    one.add_argument("--scores", action="store_true", help="also move scores.json (when v1's weight writer is stopped)")
    one.add_argument("--assets", action="store_true", help="also move v1's .js and .css files (when the way back is not needed)")
    two = commands.add_parser("back", help="put the files of a full copy of the v1 site into the site directory again")
    two.add_argument("--from", dest="copy", required=True, type=Path, help="the full copy of the v1 site (cp -a)")
    for command in (first, one, two):
        command.add_argument("--site-dir", required=True, type=Path, help="the directory that the web server serves")
    for command in (one, two):
        command.add_argument("--dry-run", action="store_true", help="print what would change; change nothing")
    args = parser.parse_args(argv)
    if args.command == "llms":
        return llms(args.site_dir)
    if args.command == "out":
        return out(args.site_dir, args.to, args.scores, args.assets, args.dry_run)
    return back(args.site_dir, args.copy, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
