"""Build the SN120 v2 status site: static files plus one small ``site-data.json``.

The site reads only public files. This builder reads ``public/mining.json``
(the publisher's snapshot) and ``public/benchmarks.json`` (the benchmark
service's file); it never calls the controller. ``view`` and ``bench_view``
are the only code that knows the shape of those two files. A missing or bad
benchmark file leaves the benchmark table empty and nothing else.

    python v2/site/build.py --source <dir or URL> --public-url <URL> --out <dir> [--v1-history <file or URL>] [--every 60]

  --source      root of the object store: a local directory or an http(s) URL.
  --public-url  URL under which a browser reaches that same root (for file links).
  --out         directory to write the site into.
  --v1-history  the frozen v1 history that ``v1_archive.py`` wrote at cutover. It
                is copied to ``v1-data.json`` for the page ``v1.html``.
  --every       rebuild the data every N seconds; without it, build once and exit.

The first three are required: there is no default host path. Files are written
to a temporary name and then renamed, so a reader never sees half a file. A
failed read keeps the last good ``site-data.json`` and exits non-zero (or, with
``--every``, tries again). The builder writes a fixed set of files and no copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from sn120_v2 import layout
from sn120_v2.public import SCHEMA
from sn120_v2.state import APPLIED
from sn120_v2.types import WINNER

STATIC = Path(__file__).parent / "static"
# Order matters: a file is stamped with the hashes of the files before it.
ASSETS = ("network-haffer.ttf", "network-mono.ttf", "favicon.svg", "site.js", "site.css", "index.html", "v1.html")
TEXT = (".css", ".html")
DATA = "site-data.json"
V1_DATA = "v1-data.json"
V1_SCHEMA = "sn120-v1-archive-1"
NOTED: dict[str, str] = {}  # the last note printed per topic
FLAGS = ("front", "demoted", "held")


def fetch(location: str) -> bytes:
    """One file, from a local path or over http(s)."""
    if location.startswith(("http://", "https://")):
        request = urllib.request.Request(location, headers={"User-Agent": "sn120-v2-site"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    return Path(location).read_bytes()


def read(source: str, key: str) -> bytes:
    """One object under a root: the object store or the v1 site, as a directory or an http(s) URL."""
    return fetch(f"{source.rstrip('/')}/{key}")


def write(path: Path, data: bytes) -> None:
    """Atomic write: a reader sees the old file or the new file, never a part of one. The same bytes are not written again."""
    if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
        return
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def view(snap: dict[str, Any], bench: dict[str, Any] | None, link: Callable[[str], str]) -> dict[str, Any]:
    """The public snapshot (``sn120_v2.public.snapshot``), cut down to what the page shows.

    Every output field is named here, so an unknown or private snapshot field
    cannot reach the page. A batch link and a LoRA hash are set for an applied
    winner only. Caps and the winner bar come from the snapshot's ``Params``.
    """
    if snap.get("schema") != SCHEMA:
        raise ValueError(f"unknown snapshot schema {snap.get('schema')!r}; this builder reads {SCHEMA!r}")
    params, live = snap["params"], snap.get("round") or {}
    head_id = ((snap.get("head") or {}).get("head") or {}).get("version_id", "")
    closed = snap["history"][::-1]  # newest first
    with_file = {record["round_id"] for record in closed} | {king["round_id"] for king in snap["kings"] if king["head_version_id"]}

    def round_link(round_id: str) -> str | None:
        return link(layout.round_file(round_id)) if round_id in with_file else None

    history, rounds, lora_of = [], [], {}
    for record in closed:
        round_id = record["round_id"]
        for entry in record["entries"]:
            won = record["outcome"] == APPLIED and entry["status"] == WINNER
            if won:
                lora_of[round_id] = entry.get("lora_hash")
            history.append({
                "round_id": round_id, "uid": entry["uid"], "hotkey": entry["hotkey"], "status": entry["status"],
                "delta": entry.get("delta"), "se": entry.get("se"), "z": entry.get("z"), "cap_rate": entry.get("cap_rate"),
                "look": entry.get("eliminated_at_look"), "sha": entry["batch_sha256"],
                "batch": link(layout.winner_batch(round_id)) if won else None, "lora": lora_of[round_id] if won else None,
                "at": record["closed_at"],
            })
        rounds.append({
            "round_id": round_id, "file_url": round_link(round_id), "env": snap["env"], "head": record["head_version_id"],
            "seats": record["seats"], "skipped": record["skipped"], "tasks": record["n_tasks"],
            "null_pass": record.get("null_pass_rate"), "null_cap": record.get("null_cap_rate"),
            "outcome": record["outcome"], "winner_uid": record.get("winner_uid"), "at": record["closed_at"],
        })

    kings = []  # the snapshot gives the payee window, newest first; a king carried over from v1 has no v2 head and no v2 files
    for rank, king in enumerate(snap["kings"], 1):
        round_id, v2 = king["round_id"], bool(king["head_version_id"])
        kings.append({
            "rank": rank, "uid": king["uid"], "hotkey": king["hotkey"], "round_id": round_id, "file_url": round_link(round_id),
            "at": king["applied_at"], "z": king["z"] if v2 else None, "head": king["head_version_id"],
            "batch": link(layout.winner_batch(round_id)) if v2 else None, "lora": lora_of.get(round_id), "imported": not v2,
        })

    # Seats in the open round come first, then the waiting line. A frozen batch passed every check at round open.
    seats = [(seat, True, ["leased"]) for seat in live.get("entries", [])]
    seats += [(seat, seat["file_valid"], [flag for flag in FLAGS if seat[flag]]) for seat in snap["queue"]["next"]]
    queue = [{
        "line": line, "uid": seat["uid"], "hotkey": seat["hotkey"], "registration_block": seat["registration_block"],
        "file_valid": valid, "flags": flags,
    } for line, (seat, valid, flags) in enumerate(seats, 1)]

    heads = [{
        "head": head["version_id"], "current": head["version_id"] == head_id, "round_id": head["round_id"],
        "file_url": round_link(head["round_id"]), "folded": head["folded"], "at": head["applied_at"],
        "manifest": link(layout.head_manifest(head["version_id"])) if head["version_id"] == head_id else None,
    } for head in snap["heads"][::-1]]

    return {
        "generated_at": float(snap["generated_at"]), "netuid": snap["netuid"], "env": snap["env"], "phase": snap["phase"],
        "head": head_id, "z_bar": params["z_bar"], "max_output_tokens": params["max_output_tokens"],
        "kings_window": params["kings_window"],
        "round": {"id": live.get("round_id"), "seats": live.get("seats"), "cap": params["batch_cap"],
                  "tasks": live.get("n_tasks") or params["n_tasks"], "opened_at": live.get("opened_at")},
        "kings": kings, "queue": {"rows": queue, "total": len(live.get("entries", [])) + snap["queue"]["length"]},
        "history": history, "rounds": rounds, "heads": heads, "bench": bench,
    }


def bench_view(bench: dict[str, Any]) -> dict[str, Any]:
    """``benchmarks.json`` of the benchmark service, cut down to one table: kings first, newest ``crowned_at`` first.

    ``partial`` marks a result that lost samples (``complete=false``). Aggregates only; the node record is not copied.
    """
    columns = sorted((b for b in bench["benchmarks"] if b.get("enabled")), key=lambda b: (b["suite"], b.get("order") or 0))
    cells: dict[str, dict[str, Any]] = {}
    for result in bench["results"]:  # the file orders the newest protocol of one (model, benchmark) last
        cells.setdefault(result["model_id"], {})[result["benchmark_id"]] = {
            "score": result["score"], "se": result.get("se"), "partial": result.get("complete") is False,
            "samples": result.get("samples"), "expected": result.get("samples_expected"), "cap_rate": result.get("cap_rate"),
        }
    rows = []
    for model in sorted(bench["models"], key=lambda m: (m.get("kind") != "king", -(m.get("crowned_at") or 0))):
        row = cells.get(model["model_id"], {})
        counted = [row[b["id"]] for b in columns if b.get("in_avg", True) and b["id"] in row]
        rows.append({
            "label": model.get("label") or model["model_id"], "kind": model.get("kind"), "status": model.get("status"),
            "round_id": model.get("round_id"), "head": model.get("head_version_id"), "at": model.get("crowned_at"),
            "avg": sum(cell["score"] for cell in counted) / len(counted) if counted else None,
            "avg_partial": any(cell["partial"] for cell in counted), "cells": row,
        })
    return {
        "updated_at": bench["updated_at"], "rows": rows,
        "columns": [{"id": b["id"], "label": b["label"], "in_avg": b.get("in_avg", True)} for b in columns],
    }


def build_static(out: Path, link: Callable[[str], str]) -> None:
    """Copy the static files. Every asset link gets ``?v=<hash of the file>``, so a new build is never served stale."""
    versions: dict[str, str] = {}
    for name in ASSETS:
        data = (STATIC / name).read_bytes()
        if name.endswith(TEXT):
            text = re.sub(r"\{\{asset:([\w.-]+)\}\}", lambda m: f"{m[1]}?v={versions[m[1]]}", data.decode())
            data = text.replace("{{llms}}", link(layout.LLMS_TXT)).replace("{{mining}}", link(layout.MINING_JSON)).encode()
        versions[name] = hashlib.sha256(data).hexdigest()[:8]
        write(out / name, data)


def build_data(source: str, out: Path, link: Callable[[str], str], quiet_missing: bool = False) -> None:
    snap = json.loads(read(source, layout.MINING_JSON))
    bench = None
    try:  # the benchmark table is informational: without it the rest of the page is still built
        bench = bench_view(json.loads(read(source, layout.BENCHMARKS)))
        json.dumps(bench, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        bench = None
        note = f"site: no benchmark table: {type(error).__name__}: {error}"
        quiet = quiet_missing and isinstance(error, FileNotFoundError)  # no file yet is the state at launch
        if note != NOTED.get("bench") and not quiet:  # say it once, not on every rebuild
            print(note, file=sys.stderr)
        NOTED["bench"] = note
    write(out / DATA, json.dumps(view(snap, bench, link), separators=(",", ":"), allow_nan=False).encode())


def install_v1(location: str, out: Path) -> None:
    """Copy the frozen v1 history into the site. Anything that is not such an archive is refused."""
    data = fetch(location)
    if json.loads(data).get("schema") != V1_SCHEMA:
        raise ValueError(f"{location} is not a v1 archive ({V1_SCHEMA}); make one with v1_archive.py")
    write(out / V1_DATA, data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", required=True, help="root of the object store: a directory or an http(s) URL")
    parser.add_argument("--public-url", required=True, help="URL under which a browser reaches that same root")
    parser.add_argument("--out", required=True, type=Path, help="directory to write the site into")
    parser.add_argument("--v1-history", help="the frozen v1 history from v1_archive.py: a file or an http(s) URL")
    parser.add_argument("--every", type=float, default=0, help="rebuild the data every N seconds (0 = build once)")
    args = parser.parse_args(argv)

    def link(key: str) -> str:
        return f"{args.public_url.rstrip('/')}/{key}"

    args.out.mkdir(parents=True, exist_ok=True)
    build_static(args.out, link)
    if args.v1_history:
        try:
            install_v1(args.v1_history, args.out)
        except Exception as error:  # noqa: BLE001 - a bad archive is never installed; the operator must see it
            print(f"site: v1 history not installed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    while True:
        try:
            build_data(args.source, args.out, link)
            failed = False
        except Exception as error:  # noqa: BLE001 - any bad read or bad snapshot keeps the last good data
            print(f"site: kept the last good {DATA}: {type(error).__name__}: {error}", file=sys.stderr)
            failed = True
        if not args.every:
            return 1 if failed else 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
