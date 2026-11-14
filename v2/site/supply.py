"""The miner supply record: one run counts the new miner files of the object store and writes the public counts.

    python v2/site/supply.py --state-dir <dir> --out <site dir>/miner-supply.json
    python v2/site/supply.py --out <site dir>/miner-supply.json --report

The store comes from the ``SN120_R2_*`` variables, or from the plain names
(``sn120_v2.r2_names``), as for the public pull. A read-only token is enough.
This command never writes to the store and never downloads a miner file: it
uses list calls, and one read of ``public/mining.json``.

  --state-dir  a private directory of this command. It holds the record: hotkeys and times. Never serve it.
  --out        the public file. It holds counts only: no hotkey and no time of one file.
  --report     no run: print the public file in words, with each time in UTC and in Costa Rica time.

What one run does:

1. Reads ``public/mining.json``: the validator's public snapshot.
2. Lists ``private/ff/v1/miners/``. For each hotkey the list gives the upload
   time of the file (``LastModified``), its size and its ETag. The ETag is a
   mark of the version of the file and nothing more.
3. Lists the frozen copies of each round that the snapshot names
   (``private/rounds/<round>/batches/``). A closed round is listed one time.
4. Records each new file: a hotkey with an ETag that the record does not have
   for that hotkey. The time of the file is its ``LastModified``. The same
   ETag with a newer ``LastModified`` is the same file and counts one time.
   A frozen copy of a round whose ETag the record does not have for that
   hotkey is a file too: its miner replaced it before a run saw it. The store
   no longer has its upload time, so its time is the time the round opened.
5. Gives a file the validator's verdict. Only the validator says "valid":
   a. A round froze the file: the frozen copy of that hotkey has the same
      ETag as the miner's file, and the public entry of the round names the
      hotkey with its ``batch_sha256``. ``admission.freeze`` writes a frozen
      copy only for bytes that pass every check, and both objects are written
      with one PUT, so equal ETags are equal bytes.
   b. The last file check: ``queue.files`` of the snapshot (every hotkey that
      waits), or ``queue.next`` (only the first seats) when that key is not
      there. A verdict is given to a file only when the file was stored at
      least ``SKEW_S`` seconds before the check and is still the stored file
      in the list that this run read after the snapshot.
   A file that the validator once called valid stays counted. A file with a
   "not valid" verdict can become valid at a later check.
6. Writes the record, then the public file. Each write is a rename.

Clocks. "now" is the ``Date`` header of the store's answer to the list call,
so "now" and the upload times come from one clock. Without that header, "now"
is the clock of this host; the public file says which clock gave "now". The
time of a file check is the validator's clock.

What the record cannot know, and the public file says so:
* The store keeps one file per hotkey. A file that no round took and that was
  replaced before the first run is not counted: a window that starts before
  ``counting_since`` is marked ``complete: false``, and its counts are "at
  least". A file that a round took is always counted: the folder of the round
  keeps it.
* The same holds for a time with no run (``gaps``), and for a file that a
  miner replaced before the next run saw it (one run each minute).
* The validator checks the files between rounds, never inside a round. A file
  that came during a round has no verdict until that round is over.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from sn120_v2 import layout
from sn120_v2.errors import ContractError, V2Error
from sn120_v2.public import SCHEMA as SNAPSHOT_SCHEMA
from sn120_v2.r2_names import PREFIX, store_env

SCHEMA = "sn120-v2-miner-supply-1"
STATE_SCHEMA = "sn120-v2-supply-state-1"
STATE = "supply-state.json"
MINERS = layout.miner_object("HOTKEY").partition("HOTKEY")[0]
HOUR = 3600
WINDOWS_H = (1, 6, 24)
HOURS_SHOWN = 24
SKEW_S = 10  # a verdict is given to a file only when the file is at least this much older than the check
GAP_S = 180  # no run for longer than this is a gap: a file that came and went in it was not seen
KEEP_S = 8 * 24 * HOUR  # a file that is no longer the stored file leaves the record after this time
VALID, NOT_VALID, NOT_JUDGED = "valid", "not_valid", "not_judged"
KINDS = ("validated", "no_verdict", "not_valid", "replaced")
COSTA_RICA = timezone(timedelta(hours=-6))  # Costa Rica has no daylight saving time
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
ROUND_WORDS = "about 90 minutes, about 137 minutes with a king"
AT_ROUND_WORDS = ("A file that a round took and that its miner replaced before the record saw it is counted at the time the round "
                  "took it, because the store no longer has its upload time (at_round_time in each row). A file that came into "
                  "more than one round counts one time.")
Proof = tuple[str, str, str, str, int | None]  # hotkey, ETag of the frozen copy, batch_sha256, round, the time the round opened


def labels(scope: dict[str, Any]) -> dict[str, str]:
    """What each count is, in the operator's three names, with the time of the validator's last file check."""
    checked = scope["checked_at"]
    last = "The validator has published no file check yet." if checked is None else f"Last file check: {both(checked)}."
    reach = ("The verdicts cover each hotkey of the waiting line." if scope["whole_line"] else
             f"Only the first {scope['seats']} of {scope['line']} seats of the waiting line have a published verdict; "
             "the files of the other seats show as no verdict yet.")
    return {
        "uploads": "Uploads per hour from R2 LastModified: new miner files, by the upload time in the store's list. "
                   f"The same bytes stored again count one time. {AT_ROUND_WORDS}",
        "validated": "Validated so far: uploads that the validator called valid, at a file check or when a round took the file. "
                     "Each file that a round took is in this count, also when its miner replaced it since. "
                     "The count of a past hour can still grow, because a verdict can come later than the upload.",
        "no_verdict": "Uploads with no verdict yet: the file is still the stored file, and no published file check after its "
                      "upload gives a verdict for it. The validator checks files when it tries to open a round and at each "
                      "publish between rounds, never inside a round. So an upload during a round has no verdict until that "
                      f"round is over ({ROUND_WORDS}). {last} {reach}",
        "not_valid": "Not valid at the last check: the validator's last file check after the upload did not call the file valid.",
        "replaced": "Replaced before a verdict: the miner stored another file before a file check gave a verdict.",
        "waiting": f"Valid files waiting now: hotkeys in the waiting line with a valid file at the last file check. {last} {reach}",
    }


class Refused(ValueError):
    """The run must not go on. The message says what the operator has to do."""


@dataclass(frozen=True)
class Stored:
    """One object, as a list call shows it."""

    key: str
    etag: str
    modified: int  # LastModified, in seconds
    size: int


class Listing(Protocol):
    date: int | None  # the clock of the store at its last answer to a list call

    def list(self, prefix: str) -> list[Stored]: ...

    def get(self, key: str) -> bytes: ...


class S3Listing:
    """Read-only S3 calls on one bucket: ListObjectsV2, and GetObject for the public snapshot."""

    def __init__(self, settings: dict[str, str]) -> None:
        self._bucket = settings["SN120_R2_BUCKET"]
        prefix = settings.get(PREFIX, "").strip("/")
        self._prefix = f"{prefix}/" if prefix else ""
        self._s3 = boto3.client(
            "s3", endpoint_url=settings["SN120_R2_ENDPOINT"], region_name="auto",
            aws_access_key_id=settings["SN120_R2_ACCESS_KEY_ID"], aws_secret_access_key=settings["SN120_R2_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 4, "mode": "standard"},
                          connect_timeout=10, read_timeout=30),
        )
        self.date: int | None = None

    def list(self, prefix: str) -> list[Stored]:
        found = []
        for page in self._s3.get_paginator("list_objects_v2").paginate(Bucket=self._bucket, Prefix=self._prefix + prefix):
            try:
                self.date = int(parsedate_to_datetime(page["ResponseMetadata"]["HTTPHeaders"]["date"]).timestamp())
            except (KeyError, TypeError, ValueError):
                pass
            for row in page.get("Contents", []):
                key = row["Key"][len(self._prefix):]
                found.append(Stored(key, row["ETag"].strip('"'), int(row["LastModified"].timestamp()), int(row["Size"])))
        return found

    def get(self, key: str) -> bytes:
        return self._s3.get_object(Bucket=self._bucket, Key=self._prefix + key)["Body"].read()


# ------------------------------------------------------------ times in words


def utc_text(t: float) -> str:
    return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def costa_rica(t: float) -> str:
    """The time in Costa Rica (UTC-6, all year), as "6:43 AM Costa Rica time, 14 Nov"."""
    local = datetime.fromtimestamp(int(t), COSTA_RICA)
    return f"{local.hour % 12 or 12}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'} Costa Rica time, {local.day} {MONTHS[local.month - 1]}"


def both(t: float) -> str:
    return f"{utc_text(t)} ({costa_rica(t)})"


# ------------------------------------------------------------ what the store and the snapshot say


def miner_files(listing: Listing) -> dict[str, Stored]:
    """The stored file of each hotkey. Only a key that ``layout.miner_object`` builds is a miner file."""
    found = {}
    for stored in listing.list(MINERS):
        hotkey = stored.key[len(MINERS):].split("/")[0]
        try:
            ours = layout.miner_object(hotkey) == stored.key
        except ContractError:
            ours = False
        if ours:
            found[hotkey] = stored
    return found


def frozen_etags(listing: Listing, round_id: str, hotkeys: Iterable[str]) -> dict[str, str]:
    """The ETag of the frozen copy of each named hotkey in one round."""
    wanted = {layout.frozen_batch(round_id, hotkey): hotkey for hotkey in hotkeys}
    if not wanted:
        return {}
    prefix = layout.frozen_batch(round_id, "HOTKEY").partition("HOTKEY")[0]
    return {wanted[stored.key]: stored.etag for stored in listing.list(prefix) if stored.key in wanted}


def round_proofs(listing: Listing, cache: dict[str, dict[str, str]], snap: dict[str, Any]) -> tuple[list[Proof], dict[str, dict[str, str]]]:
    """One ``Proof`` for each file that a round of the snapshot froze, the oldest round first, and the new cache.

    The frozen copies of a round are written when the round opens and never
    change, so a closed round is listed one time. A round that the snapshot
    no longer names leaves the cache.
    """
    proofs, kept = [], {}
    for row in [*(snap.get("history") or []), *([snap["round"]] if snap.get("round") else [])]:
        round_id = str(row["round_id"])
        named = {str(entry["hotkey"]): str(entry["batch_sha256"]) for entry in row.get("entries") or [] if entry.get("batch_sha256")}
        closed = row.get("closed_at") is not None
        etags = cache.get(round_id) if closed else None
        if etags is None:
            etags = frozen_etags(listing, round_id, named)
        if closed:
            kept[round_id] = etags
        opened = row.get("opened_at")
        opened = int(opened) if isinstance(opened, (int, float)) else None
        proofs += [(hotkey, etags[hotkey], sha, round_id, opened) for hotkey, sha in named.items() if hotkey in etags]
    return sorted(proofs, key=lambda proof: (proof[4] is None, proof[4] or 0)), kept


def file_check(snap: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """The verdict of the last file check for each hotkey that waits, and the scope of that table.

    ``queue.files.verdicts`` has one row for each hotkey of the line, with three verdicts.
    Without it, ``queue.next`` covers the first seats only, and its
    ``file_valid: false`` can also mean "not judged".
    """
    queue = snap["queue"]
    scope = {"checked_at": queue.get("files_checked_at"), "line": int(queue["length"])}
    table = queue.get("files")
    rows = table.get("verdicts") if isinstance(table, dict) else None
    if isinstance(rows, list) and all(
        isinstance(row, dict) and isinstance(row.get("hotkey"), str) and row.get("file_verdict") in (VALID, NOT_VALID, NOT_JUDGED) for row in rows
    ):
        verdicts = {row["hotkey"]: row["file_verdict"] for row in rows}
        return verdicts, scope | {"source": "queue.files", "whole_line": True, "hotkeys": len(verdicts)}
    verdicts = {}
    for seat in queue["next"]:  # a hotkey can have more than one seat: the file belongs to the hotkey
        verdicts[str(seat["hotkey"])] = VALID if seat["file_valid"] is True else NOT_VALID
    return verdicts, scope | {"source": "queue.next", "whole_line": len(queue["next"]) >= int(queue["length"]), "hotkeys": len(verdicts),
                              "seats": len(queue["next"])}


# ------------------------------------------------------------ the record


def new_state(now: int) -> dict[str, Any]:
    return {"schema": STATE_SCHEMA, "since": now, "last_poll": None, "gaps": [], "files": {}, "rounds": {}}


def record(state: dict[str, Any], files: dict[str, Stored], proofs: list[Proof], verdicts: dict[str, str],
           checked_at: int | None, now: int) -> None:
    """Bring the record up to this run: new files, files that went away, files that only a round's folder has, then the validator's verdicts."""
    if state["last_poll"] is not None and now - state["last_poll"] > GAP_S:
        state["gaps"].append([state["last_poll"], now])
    state["last_poll"] = now
    known = state["files"]
    for hotkey, stored in files.items():
        entries = known.setdefault(hotkey, [])
        same = next((entry for entry in entries if entry["etag"] == stored.etag), None)
        for entry in entries:
            if entry is not same and entry["ended_at"] is None:
                entry["ended_at"] = now
        if same is None:
            entries.append({"etag": stored.etag, "uploaded_at": stored.modified, "stored_at": stored.modified, "size": stored.size,
                            "seen_at": now, "ended_at": None, "verdict": None, "proof": None, "sha": None, "decided_at": None})
        else:  # the same file: it counts one time, at its first upload time. ``stored_at`` is the time of the newest store
            same["ended_at"], same["stored_at"] = None, max(same["stored_at"], stored.modified)
    for hotkey, entries in known.items():
        if hotkey not in files:
            for entry in entries:
                if entry["ended_at"] is None:
                    entry["ended_at"] = now

    def valid(entry: dict[str, Any], proof: str) -> None:
        if entry["verdict"] != VALID:
            entry.update(verdict=VALID, proof=proof, decided_at=now)

    for hotkey, etag, sha, round_id, opened_at in proofs:
        entries = known.setdefault(hotkey, [])
        # An ETag with "-" is not the MD5 of the bytes (a multipart upload), so it cannot say that the frozen copy is
        # another file. Then the frozen copy is not counted by itself: a count that is too low, never one file two times.
        in_record = opened_at is None or any(entry["etag"] == etag or ("-" in entry["etag"] and entry["uploaded_at"] <= opened_at)
                                             for entry in entries)
        if not in_record and now - KEEP_S <= opened_at <= now:
            # The miner replaced this file before a run saw it. The frozen copy is the record of it: one file, at the
            # time of the first round that took it. It is not the stored file, so it starts as ended.
            entries.append({"etag": etag, "uploaded_at": opened_at, "stored_at": opened_at, "size": None, "seen_at": now,
                            "ended_at": now, "verdict": None, "proof": None, "sha": None, "decided_at": None, "time_from": "round"})
        for entry in entries:
            if entry["etag"] == etag:
                valid(entry, f"round {round_id}")
                entry["sha"] = sha
    for hotkey, verdict in verdicts.items():
        entry = next((entry for entry in known.get(hotkey, []) if entry["ended_at"] is None), None)
        if entry is None or checked_at is None or checked_at < entry["stored_at"] + SKEW_S:
            continue  # the file came after the check, or too near to it: it waits for the next check
        if verdict == VALID:
            valid(entry, "file check")
        elif verdict == NOT_VALID and entry["verdict"] is None:
            entry.update(verdict=NOT_VALID, proof="file check", decided_at=now)

    for entries in known.values():  # where the validator's record gives batch_sha256, a file counts by it
        seen: set[str] = set()
        for entry in sorted(entries, key=lambda entry: entry["uploaded_at"]):
            entry["same_bytes"] = bool(entry["sha"]) and entry["sha"] in seen
            seen.add(entry["sha"] or "")

    state["gaps"] = [gap for gap in state["gaps"] if gap[1] >= now - KEEP_S]
    for hotkey in list(known):
        known[hotkey] = [entry for entry in known[hotkey] if entry["ended_at"] is None or entry["ended_at"] >= now - KEEP_S]
        if not known[hotkey]:
            del known[hotkey]


def kind(entry: dict[str, Any]) -> str:
    if entry["verdict"] == VALID:
        return "validated"
    if entry["verdict"] == NOT_VALID:
        return "not_valid"
    return "no_verdict" if entry["ended_at"] is None else "replaced"


def counts(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    found = {"uploads": 0, **{name: 0 for name in KINDS}, "at_round_time": 0}
    for entry in entries:
        found["uploads"] += 1
        found[kind(entry)] += 1
        found["at_round_time"] += entry.get("time_from") == "round"  # of the uploads: the files with the round's time
    return found


def public(state: dict[str, Any], verdicts: dict[str, str], scope: dict[str, Any], snap: dict[str, Any], now: int,
           clock: dict[str, Any]) -> dict[str, Any]:
    """The public file: counts only. No hotkey, no ETag and no time of one file is in it."""
    entries = [entry for group in state["files"].values() for entry in group if not entry.get("same_bytes")]

    def between(start: int, end: int) -> dict[str, Any]:
        """Uploads with ``start < upload time <= end``. ``complete`` is false when the record does not cover all of that time."""
        covered = start >= state["since"] and not any(gap[0] < end and gap[1] > start for gap in state["gaps"])
        return counts(entry for entry in entries if start < min(entry["uploaded_at"], now) <= end) | {"complete": covered}

    windows = []
    for hours in WINDOWS_H:
        found = between(now - hours * HOUR, now)
        windows.append({"hours": hours, "from": now - hours * HOUR, **found,
                        "uploads_per_hour": round(found["uploads"] / hours, 2), "validated_per_hour": round(found["validated"] / hours, 2)})
    this_hour = now - now % HOUR
    series = [{"t": start, **between(start - 1, min(start + HOUR - 1, now)), "open": start == this_hour}
              for start in range(this_hour - (HOURS_SHOWN - 1) * HOUR, this_hour + 1, HOUR)]
    return {
        "schema": SCHEMA,
        "generated_at": now,
        "clock": clock,
        "counting_since": state["since"],
        "gaps": [gap for gap in state["gaps"] if gap[1] > now - HOURS_SHOWN * HOUR],
        "labels": labels(scope),
        "windows": windows,
        "hours": series,
        "waiting": {"valid": sum(verdict == VALID for verdict in verdicts.values()), **scope},
        "snapshot_at": snap.get("generated_at"),
        "phase": snap.get("phase"),
        "method": "A count goes to the hour of the file's upload time (LastModified of the store's list). Only the validator "
                  f"says valid: a round froze the file, or the last file check after the upload called it valid. {AT_ROUND_WORDS} "
                  "That time is opened_at of the round in the public snapshot, the validator's clock. The validator "
                  "checks files when it tries to open a round and at each publish between rounds, never inside a round, so an "
                  f"upload during a round has no verdict until that round is over ({ROUND_WORDS}). "
                  + ("No file check is published yet. " if scope["checked_at"] is None else f"Last file check: {both(scope['checked_at'])}. ")
                  + "A file of a hotkey with no seat in the waiting line gets no check. A window or an hour with complete=false "
                  "began before counting_since or meets a gap with no run: its counts are 'at least', because a file that "
                  "no round took and that was replaced in that time is not in them.",
    }


def poll(listing: Listing, state: dict[str, Any] | None, host_now: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One run on a record (None: the first run). Returns the record and the public file.

    The snapshot is read before the list: a file in the list that is older
    than the check was the stored file at that check.
    """
    snap = json.loads(listing.get(layout.MINING_JSON))
    if not isinstance(snap, dict) or snap.get("schema") != SNAPSHOT_SCHEMA:
        raise Refused(f"{layout.MINING_JSON} is not a snapshot of schema {SNAPSHOT_SCHEMA}")
    files = miner_files(listing)
    now = listing.date if listing.date is not None else host_now
    clock = {
        "now": "store" if listing.date is not None else "host",
        "host_minus_store_s": host_now - listing.date if listing.date is not None else None,
        "words": "generated_at and each window end: the Date header of the store's answer to the list call (now=store), or the "
                 "clock of the host that runs the recorder (now=host). Upload times: the store's clock (LastModified). "
                 "checked_at: the validator's clock.",
    }
    state = new_state(now) if state is None else state
    proofs, state["rounds"] = round_proofs(listing, state["rounds"], snap)
    verdicts, scope = file_check(snap)
    record(state, files, proofs, verdicts, scope["checked_at"], now)
    return state, public(state, verdicts, scope, snap, now, clock)


# ------------------------------------------------------------ files on this host


def write(path: Path, data: bytes, mode: int) -> None:
    """Atomic write: a reader sees the old file or the new file, never a part of one."""
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_bytes())
        if state["schema"] != STATE_SCHEMA or not isinstance(state["files"], dict) or not isinstance(state["since"], int):
            raise ValueError("another schema")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Refused(f"{path} is not the record of this command ({type(error).__name__}). Move it away; the next run starts "
                      f"a new record, and its counts start at that time") from None
    return state


def report(doc: dict[str, Any]) -> list[str]:
    """The public file in words, for the operator. Times in UTC and in Costa Rica time."""
    if doc.get("schema") != SCHEMA:
        raise Refused(f"the file is not of schema {SCHEMA}")
    clock, waiting = doc["clock"], doc["waiting"]
    ahead = clock["host_minus_store_s"]
    lines = [
        f"made {both(doc['generated_at'])}; 'now' is the clock of the {clock['now']}"
        + ("" if ahead is None else f" (the host's clock is {ahead:+d} s from it)"),
        f"counting since {both(doc['counting_since'])}; gaps with no run: {len(doc['gaps'])}",
    ]

    def numbers(row: dict[str, Any]) -> str:
        return (f"uploads {row['uploads']} | validated so far {row['validated']} | no verdict yet {row['no_verdict']} | "
                f"not valid at the last check {row['not_valid']} | replaced before a verdict {row['replaced']}"
                + (f" | of the uploads, {row['at_round_time']} have the time of their round" if row.get("at_round_time") else ""))

    for row in doc["windows"]:
        cover = "complete" if row["complete"] else "AT LEAST: the record does not cover all of this window"
        lines.append(f"last {row['hours']:>2} h: {numbers(row)} | per hour: uploads {row['uploads_per_hour']:.2f}, "
                     f"validated {row['validated_per_hour']:.2f} | {cover}")
    lines.append("by hour, by the upload time (LastModified, or the time of the round for a file that only a round's folder has); "
                 "each hour in UTC, and in Costa Rica time in the brackets:")
    for row in doc["hours"]:
        cover = "this hour is not over" if row["open"] else "complete" if row["complete"] else "at least"
        lines.append(f"  {utc_text(row['t'])[5:16]} UTC ({costa_rica(row['t']).split(' Costa')[0]}): uploads {row['uploads']:>3} | validated "
                     f"{row['validated']:>3} | no verdict {row['no_verdict']:>3} | not valid {row['not_valid']:>3} | replaced {row['replaced']:>3} | {cover}")
    checked = "no file check yet" if waiting["checked_at"] is None else f"at the validator's file check of {both(waiting['checked_at'])}"
    reach = (f"all {waiting['hotkeys']} hotkeys of the line" if waiting["whole_line"]
             else f"only the first {waiting.get('seats', waiting['hotkeys'])} of {waiting['line']} seats have a published verdict")
    lines.append(f"valid files waiting now: {waiting['valid']} ({reach}; source {waiting['source']}), {checked}; phase {doc.get('phase')}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--state-dir", type=Path, help="the private directory of the record; it is made with mode 700")
    parser.add_argument("--out", required=True, type=Path, help="the public file with the counts, in the directory that the web server serves")
    parser.add_argument("--report", action="store_true", help="no run: print the public file in words")
    args = parser.parse_args(argv)
    if args.report:
        try:
            print("\n".join(f"supply: {line}" for line in report(json.loads(args.out.read_bytes()))))
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"supply: FAILED {args.out} cannot be read as the public file: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        return 0
    if args.state_dir is None:
        parser.error("--state-dir is needed for a run")
    if not args.out.parent.is_dir():
        print(f"supply: FAILED {args.out.parent} is not a directory; nothing written", file=sys.stderr)
        return 1
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = os.open(args.state_dir, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, doc = poll(S3Listing(store_env(os.environ)), load_state(args.state_dir / STATE), int(time.time()))
        write(args.state_dir / STATE, json.dumps(state, separators=(",", ":")).encode(), 0o600)
        write(args.out, json.dumps(doc, separators=(",", ":"), allow_nan=False).encode(), 0o644)
    except BlockingIOError:
        print("supply: FAILED another run uses this record; nothing written", file=sys.stderr)
        return 1
    except ClientError as error:  # the text of a store error can name the endpoint: only its code is printed
        print(f"supply: FAILED the store refused a call ({error.response.get('Error', {}).get('Code', 'no code')}); nothing written", file=sys.stderr)
        return 1
    except BotoCoreError as error:
        print(f"supply: FAILED the store did not answer ({type(error).__name__}); nothing written", file=sys.stderr)
        return 1
    except (V2Error, Refused) as error:
        print(f"supply: FAILED {error}; nothing written", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"supply: FAILED the snapshot or the record has another form ({type(error).__name__}: {error}); nothing written", file=sys.stderr)
        return 1
    finally:
        os.close(lock)
    day = doc["windows"][-1]
    print(f"supply: {day['uploads']} uploads in 24 h, {day['validated']} validated so far, {day['no_verdict']} with no verdict yet; "
          f"{doc['waiting']['valid']} valid files wait; now {utc_text(doc['generated_at'])}, clock of the {doc['clock']['now']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
