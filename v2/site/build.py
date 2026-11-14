"""Build the SN120 v2 status site: static files plus one small ``site-data.json``.

The site reads only public files. This builder reads ``public/mining.json``
(the publisher's snapshot) and ``public/benchmarks.json`` (the benchmark
service's file); it never calls the controller. ``view`` and ``bench_view``
are the only code that knows the shape of those two files. The benchmark table
has one row for the genesis head and one for every king, with or without the
benchmark file: a missing or bad file leaves those rows with no score, and
nothing else. The reference models (models of other labs, scored by v1) come
from the frozen v1 history that is installed in the site (``v1-data.json``),
so they do not depend on the benchmark file or on a file of v1.

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
from typing import Any, Callable, Sequence

from sn120_v2 import layout
from sn120_v2.headlog import GENESIS_ROUND
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
# The state of the benchmark run of one row. The benchmark file of v1 says "evaluating" for a run in progress.
SCORED, QUEUED, RUNNING, FAILED, NO_RUN = "scored", "queued", "running", "failed", "none"
RUN_STATE = {"scored": SCORED, "done": SCORED, "queued": QUEUED, "running": RUNNING, "evaluating": RUNNING, "failed": FAILED}
KING, GENESIS = "king", "genesis"


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
        round_id, winner_pass = record["round_id"], None
        best_z = max((entry["z"] for entry in record["entries"] if entry.get("z") is not None), default=None)
        for entry in record["entries"]:
            won = record["outcome"] == APPLIED and entry["status"] == WINNER
            if won:
                lora_of[round_id], winner_pass, best_z = entry.get("lora_hash"), entry.get("pass_rate"), entry.get("z")
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
            # for the charts: the open time, the pass rate of the applied winner, its z (with no winner: the highest z), and
            # the number of entries with a z. An entry whose training failed has none. ``reason`` is the validator's own text.
            "opened_at": record["opened_at"], "winner_pass": winner_pass, "best_z": best_z,
            "with_z": sum(entry.get("z") is not None for entry in record["entries"]), "reason": record.get("reason") or None,
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


def bench_view(bench: dict[str, Any] | None, heads: Sequence[dict[str, Any]] = (), kept: dict[str, Any] | None = None) -> dict[str, Any]:
    """The benchmark table of the v1 site: one row for every king, the reference models, and the genesis head.

    ``bench`` is ``benchmarks.json`` of the benchmark service, or None when there is no such file. ``heads`` is the
    head list of the snapshot (the newest heads, oldest first). A king is one applied round. A king or a genesis head
    that the file does not name still has its row, made from ``heads``, with the state ``none``.

    ``kept`` is a table that this function made before: the benchmark part of the frozen v1 history. Its rows of a kind
    other than king and genesis (the reference models) are added, with ``v1`` set, unless ``bench`` has a row of that
    kind with the same label. A king or a genesis row of ``kept`` is a model of v1 and is never added. When ``bench``
    has no column, the columns of ``kept`` are the columns.

    A row says the state of its run: ``scored``, ``queued``, ``running``, ``failed`` or ``none``. ``avg`` is the mean
    of the ``in_avg`` columns, and only a row that has all of them gets one: a mean over fewer benchmarks cannot be
    compared. ``partial`` marks a result that lost samples (``complete=false``). Aggregates only; the node record is
    not copied.
    """
    bench = bench or {"benchmarks": [], "results": [], "models": [], "updated_at": None}
    enabled = sorted((b for b in bench["benchmarks"] if b.get("enabled")), key=lambda b: (b["suite"], b.get("order") or 0))
    columns = [{"id": b["id"], "label": b["label"], "in_avg": b.get("in_avg", True), "suite": b.get("suite")}
               for b in enabled or (kept or {}).get("columns", ())]
    counted = [b["id"] for b in columns if b["in_avg"]]
    cells: dict[str, dict[str, Any]] = {}
    for result in bench["results"]:  # the file orders the newest protocol of one (model, benchmark) last
        cells.setdefault(result["model_id"], {})[result["benchmark_id"]] = {
            "score": result["score"], "se": result.get("se"), "partial": result.get("complete") is False,
            "samples": result.get("samples"), "expected": result.get("samples_expected"), "cap_rate": result.get("cap_rate"),
        }

    def row(label: str, kind: Any, status: str, round_id: Any, head: Any, at: Any, scores: dict[str, Any]) -> dict[str, Any]:
        if kind == KING and round_id and head:  # one name for a king, with or without the benchmark file
            label = f"{round_id} {head[:8]}"
        if kind == GENESIS and label.lower().startswith(GENESIS):  # the service writes "genesis <model name>"
            label = "Genesis" + label[len(GENESIS):]
        full = bool(counted) and all(name in scores for name in counted)
        return {
            "label": label, "kind": kind, "status": status, "round_id": round_id, "head": head, "at": at,
            "avg": sum(scores[name]["score"] for name in counted) / len(counted) if full else None,
            "avg_partial": any(scores[name]["partial"] for name in counted if name in scores), "cells": scores,
        }

    rows = [row(
        model.get("label") or model["model_id"], model.get("kind"), RUN_STATE.get(model.get("status"), model.get("status") or NO_RUN),
        model.get("round_id"), model.get("head_version_id"), model.get("crowned_at"), cells.get(model["model_id"], {}),
    ) for model in bench["models"]]

    by_round: dict[str, dict[str, Any]] = {}  # the head that is served for a benchmark: the folded head of the round
    for head in heads:
        if head["round_id"] not in by_round or head["folded"]:
            by_round[head["round_id"]] = head
    named = {name for made in rows for name in (made["round_id"], made["head"]) if name}
    genesis = next((made for made in rows if made["kind"] == GENESIS), None)
    for round_id, head in by_round.items():
        if round_id == GENESIS_ROUND and genesis:  # the file may give the genesis row with no head id and no time
            genesis |= {"round_id": round_id, "head": genesis["head"] or head["version_id"], "at": genesis["at"] or head["applied_at"]}
        elif round_id == GENESIS_ROUND:
            rows.append(row("Genesis", GENESIS, NO_RUN, round_id, head["version_id"], head["applied_at"], {}))
        elif round_id not in named and not any(other["version_id"] in named for other in heads if other["round_id"] == round_id):
            rows.append(row(round_id, KING, NO_RUN, round_id, head["version_id"], head["applied_at"], {}))

    order = {KING: 0, GENESIS: 2}  # kings, newest first; then the other models, best average first; the genesis head is last
    have = {made["label"] for made in rows if made["kind"] not in order}
    for old in (kept or {}).get("rows", ()):
        if old.get("kind") not in order and old["label"] not in have:
            state = RUN_STATE.get(old.get("status"), old.get("status") or NO_RUN)
            rows.append(row(old["label"], old.get("kind"), state, None, None, None, old["cells"]) | {"v1": True})
    rows.sort(key=lambda made: (order.get(made["kind"], 1), -(made["at"] or 0) if made["kind"] == KING else -(made["avg"] or 0)))
    return {"updated_at": bench["updated_at"], "rows": rows, "columns": columns}


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


def tried(topic: str, what: str, make: Callable[[], Any], quiet: Callable[[Exception], bool] = lambda error: False) -> Any:
    """``make()``, or None when the input is missing or bad. The reason is printed once, not on every rebuild."""
    try:
        made = make()
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        note = f"site: {what}: {type(error).__name__}: {error}"
        if note != NOTED.get(topic) and not quiet(error):
            print(note, file=sys.stderr)
        NOTED[topic] = note
        return None
    NOTED.pop(topic, None)
    return made


def build_data(source: str, out: Path, link: Callable[[str], str], quiet_missing: bool = False) -> None:
    """Write the data file. The benchmark file and the v1 history are informational: without one of them, or with a bad
    one, the page is still built, and the genesis head and each king still have their rows."""
    snap = json.loads(read(source, layout.MINING_JSON))
    heads = snap["heads"] if snap.get("schema") == SCHEMA else []  # ``view`` refuses any other snapshot

    def usable(bench: dict[str, Any] | None, kept: dict[str, Any] | None) -> dict[str, Any] | None:
        json.dumps(bench_view(bench, heads, kept), allow_nan=False)  # an input that gives no table is not used
        return bench or kept

    def v1_models() -> dict[str, Any]:
        doc = json.loads((out / V1_DATA).read_bytes())  # the frozen v1 history, as ``install_v1`` copied it
        if doc.get("schema") != V1_SCHEMA:
            raise ValueError(f"{V1_DATA} is not a v1 archive ({V1_SCHEMA})")
        return usable(None, doc["bench"])

    def missing(error: Exception) -> bool:
        return isinstance(error, FileNotFoundError)

    bench = tried("bench", "no benchmark file, so no benchmark score", lambda: usable(json.loads(read(source, layout.BENCHMARKS)), None),
                  lambda error: quiet_missing and missing(error))  # no file yet is the state at launch
    kept = tried("kept", "no reference model from the v1 history", v1_models, missing)  # a site can have no v1 history
    table = bench_view(bench, heads, kept)
    write(out / DATA, json.dumps(view(snap, table, link), separators=(",", ":"), allow_nan=False).encode())


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
