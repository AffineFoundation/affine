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
import urllib.error
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
ON_SITE: dict[str, bool] = {}  # per http(s) location: is the file there. A head manifest is written once and is not removed
FLAGS = ("front", "demoted", "held")
# The state of the benchmark run of one row. The benchmark file of v1 says "evaluating" for a run in progress.
SCORED, QUEUED, RUNNING, FAILED, NO_RUN = "scored", "queued", "running", "failed", "none"
FULL = ":full"  # the end of the column id of a benchmark's full run: the score of a longer run of the same model
RUN_STATE = {"scored": SCORED, "done": SCORED, "queued": QUEUED, "running": RUNNING, "evaluating": RUNNING, "failed": FAILED}
KING, GENESIS = "king", "genesis"
V1_KING = "v1-king"  # a row of the benchmark file: a king of v1 that the benchmark service of v2 scores. It won no round of v2


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


def exists(source: str, key: str) -> bool:
    """True when the object is under the root. Over http(s), a yes or a 404 is kept for this process; a failed read is asked again."""
    location = f"{source.rstrip('/')}/{key}"
    if not location.startswith(("http://", "https://")):
        return Path(location).is_file()
    if location not in ON_SITE:
        try:
            fetch(location)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                return False
            ON_SITE[location] = False
        except OSError:
            return False
        else:
            ON_SITE[location] = True
    return ON_SITE[location]


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


def view(snap: dict[str, Any], bench: dict[str, Any] | None, link: Callable[[str], str],
         on_site: Callable[[str], bool] = lambda key: False) -> dict[str, Any]:
    """The public snapshot (``sn120_v2.public.snapshot``), cut down to what the page shows.

    Every output field is named here, so an unknown or private snapshot field
    cannot reach the page. A batch link and a LoRA hash are set for an applied
    winner only. Caps and the winner bar come from the snapshot's ``Params``.
    ``on_site`` says if a public object is there to link. A head gets the link
    to its manifest when it is the current head or when its manifest is there:
    the validator writes the manifest of a head only while that head is current.
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
            status = entry.get("status")  # a round with no result (the box failed, or the operator ended it) gives no entry a status
            won = record["outcome"] == APPLIED and status == WINNER
            if won:
                lora_of[round_id], winner_pass, best_z = entry.get("lora_hash"), entry.get("pass_rate"), entry.get("z")
            history.append({
                "round_id": round_id, "uid": entry["uid"], "hotkey": entry["hotkey"], "status": status,
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

    def manifest_link(version_id: str) -> str | None:
        key = layout.head_manifest(version_id)
        return link(key) if version_id == head_id or on_site(key) else None

    heads = [{
        "head": head["version_id"], "current": head["version_id"] == head_id, "round_id": head["round_id"],
        "file_url": round_link(head["round_id"]), "folded": head["folded"], "at": head["applied_at"],
        "manifest": manifest_link(head["version_id"]),
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


def whole(value: Any) -> int | None:
    """``value`` when it is a whole number above 0, or None."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def number(value: Any) -> bool:
    """True when ``value`` is a number. True and False are not numbers here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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

    ``rollouts`` of a cell is its count of samples for each problem. A result of ``bench`` names it. A cell of ``kept``
    has only its sample count, so its ``rollouts`` is that count divided by the problem count of the same benchmark in
    ``bench``, and it is None when ``bench`` gives no single problem count or the division leaves a rest. A column
    names that single problem count as ``problems``.

    A result with ``derived_from`` was computed from a longer run of the same model, and its cell is ``derived``. With
    ``score_full`` it also gives the score of that longer run: one more cell, under the id of the benchmark with
    ``FULL`` at its end. The column of these cells stands directly after its benchmark, is never in the average, and
    is there only when a row has such a cell; ``head`` is its short name for the head of the table. ``pod`` of a row
    is the state of its machine and the time of that state. ``waiting_for`` says what a row waits for, and ``stopped``
    why the benchmark service starts no run of it. ``profile`` of a cell is the settings profile of that result, and
    ``profile`` of a row the one that the file gives the model: the sampling of a run.

    A row of the kind ``V1_KING`` has the label "v1 king", its ``uid`` and its ``v1_epoch``, and no round, no head and
    no time: these are of v2, and it won no round of v2. It is not a king here and not a reference model.
    """
    bench = bench or {"benchmarks": [], "results": [], "models": [], "updated_at": None}
    # Disabled, in_avg=false benchmarks keep their published history as a display-only column.
    enabled = sorted((b for b in bench["benchmarks"] if b.get("enabled") or b.get("in_avg") is False),
                     key=lambda b: (b["suite"], b.get("order") or 0))
    columns = [{"id": b["id"], "label": b["label"], "in_avg": b.get("in_avg", True), "suite": b.get("suite")}
               for b in enabled or (kept or {}).get("columns", ())]
    counted = [b["id"] for b in columns if b["in_avg"]]
    cells: dict[str, dict[str, Any]] = {}
    problems: dict[str, set[int | None]] = {}  # per benchmark: the problem count of each result, None when it does not say
    for result in bench["results"]:  # the file orders the newest protocol of one (model, benchmark) last
        bench_id, each, expected = result["benchmark_id"], whole(result.get("rollouts")), whole(result.get("samples_expected"))
        count = expected // each if each and expected and expected % each == 0 else whole(result.get("n"))
        problems.setdefault(bench_id, set()).add(count)
        scores = cells.setdefault(result["model_id"], {})
        scores[bench_id] = {
            "score": result["score"], "se": result.get("se"), "partial": result.get("complete") is False,
            "samples": result.get("samples"), "expected": result.get("samples_expected"), "cap_rate": result.get("cap_rate"),
            "rollouts": each, "derived": bool(result.get("derived_from")), "profile": result.get("profile"),
        }
        scores.pop(bench_id + FULL, None)
        if number(result.get("score_full")):
            total = whole(result.get("samples_full"))
            scores[bench_id + FULL] = {
                "score": result["score_full"], "se": result.get("se_full"), "partial": False, "samples": total, "expected": total,
                "cap_rate": result.get("cap_rate"),  # the benchmark service keeps the cap rate of the full run in a computed result
                "rollouts": total // count if total and count and total % count == 0 else None, "profile": result.get("profile"),
            }

    def per_problem(name: str, cell: dict[str, Any]) -> int | None:
        if whole(cell.get("rollouts")):
            return cell["rollouts"]
        count, expected = next(iter(problems[name])) if len(problems.get(name, ())) == 1 else None, whole(cell.get("expected"))
        return expected // count if count and expected and expected % count == 0 else None

    def row(label: str, kind: Any, status: str, round_id: Any, head: Any, at: Any, scores: dict[str, Any]) -> dict[str, Any]:
        if kind == KING and round_id and head:  # one name for a king, with or without the benchmark file
            label = f"{round_id} {head[:8]}"
        if kind == GENESIS and label.lower().startswith(GENESIS):  # the service writes "genesis <model name>"
            label = "Genesis" + label[len(GENESIS):]
        if kind == V1_KING:
            label, round_id, head, at = "v1 king", None, None, None
        full = bool(counted) and all(name in scores for name in counted)
        return {
            "label": label, "kind": kind, "status": status, "round_id": round_id, "head": head, "at": at,
            "avg": sum(scores[name]["score"] for name in counted) / len(counted) if full else None,
            "avg_partial": any(scores[name]["partial"] for name in counted if name in scores), "cells": scores,
        }

    def more(model: dict[str, Any]) -> dict[str, Any]:
        found = {name: model[name] for name in ("waiting_for", "stopped", "profile") if isinstance(model.get(name), str)}
        if model.get("kind") == V1_KING:
            found |= {name: model[name] for name in ("uid", "v1_epoch") if number(model.get(name))}
        pod = model.get("pod")
        if isinstance(pod, dict) and isinstance(pod.get("state"), str):
            found["pod"] = {"state": pod["state"], "since": pod["since"] if number(pod.get("since")) else None}
        return found

    rows = [row(
        model.get("label") or model["model_id"], model.get("kind"), RUN_STATE.get(model.get("status"), model.get("status") or NO_RUN),
        model.get("round_id"), model.get("head_version_id"), model.get("crowned_at"), cells.get(model["model_id"], {}),
    ) | more(model) for model in bench["models"]]

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

    order = {KING: 0, GENESIS: 2}  # kings, newest first; then the other models, best average first (the page orders them by size); the genesis head is last
    have = {made["label"] for made in rows if made["kind"] not in order}
    for old in (kept or {}).get("rows", ()):
        if old.get("kind") not in order and old["label"] not in have:
            state = RUN_STATE.get(old.get("status"), old.get("status") or NO_RUN)
            scores = {name: cell | {"rollouts": per_problem(name, cell)} for name, cell in old["cells"].items()}
            rows.append(row(old["label"], old.get("kind"), state, None, None, None, scores) | {"v1": True})
    rows.sort(key=lambda made: (order.get(made["kind"], 1), -(made["at"] or 0) if made["kind"] == KING else -(made["avg"] or 0)))
    longer = {name for made in rows for name in made["cells"] if name.endswith(FULL)}
    columns = [column | {"problems": next(iter(problems[column["id"]])) if len(problems.get(column["id"], ())) == 1 else None}
               for column in columns]
    columns = [column for base in columns for column in (base, {
        "id": base["id"] + FULL, "label": f"{base['label']}, full run", "head": "full run", "in_avg": False, "suite": base["suite"], "full_of": base["id"],
    }) if column is base or column["id"] in longer]
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
    data = view(snap, table, link, lambda key: exists(source, key))
    write(out / DATA, json.dumps(data, separators=(",", ":"), allow_nan=False).encode())


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
