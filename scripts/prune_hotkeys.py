#!/usr/bin/env python3
"""Filter JSON/JSONL results by hotkey, optionally duplicating entries.

The script reads a JSON Lines file (defaults to
`affine/results/block-hotkey.jsonl`) or a JSON array and rewrites it so that
only entries whose hotkey (or nested field) appears in the provided allow-list
remain. It can scale the remaining entries up or down to a fixed count and
force their score to a chosen value. A timestamped backup is written before the
original file is replaced (unless `--dry-run` is used).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import shutil
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


def load_allow_list(values: Iterable[str], file: Path | None) -> Set[str]:
    allow = set(values)
    if file:
        allow.update(
            line.strip()
            for line in file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    if not allow:
        raise SystemExit("No hotkeys supplied; pass at least one via --keep or --keep-file")
    return allow


def backup_file(path: Path) -> Path:
    stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".{stamp}.bak")
    shutil.copy2(path, backup)
    return backup


def read_records(path: Path) -> Tuple[List[dict], int, str]:
    """Return records, skipped_count, format."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as src:
            data = json.load(src)
        if not isinstance(data, list):
            raise SystemExit(f"Unsupported JSON structure in {path}: expected list at top level")
        return data, 0, "json"

    skipped = 0
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            records.append(obj)
    return records, skipped, "jsonl"


def resolve_field(payload: dict, field: str):
    keys = field.split(".") if field else []
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def filter_records(records: Sequence[dict], allow: Set[str], field: str) -> List[dict]:
    filtered = []
    for record in records:
        value = resolve_field(record, field)
        if value in allow:
            filtered.append(copy.deepcopy(record))
    return filtered


def adjust_scores(records: List[dict], score: float | None) -> None:
    if score is None:
        return
    for record in records:
        evaluation = record.setdefault("evaluation", {})
        evaluation["score"] = score


def scale_records(records: List[dict], target: int) -> List[dict]:
    if target <= 0:
        raise SystemExit("--target-count must be positive")
    if not records:
        raise SystemExit("No records remain after filtering; cannot scale")
    if len(records) == target:
        return records
    if len(records) > target:
        return [copy.deepcopy(rec) for rec in islice(records, target)]

    result: List[dict] = []
    idx = 0
    while len(result) < target:
        result.append(copy.deepcopy(records[idx % len(records)]))
        idx += 1
    return result


def write_records(path: Path, records: Sequence[dict], fmt: str, dry_run: bool) -> None:
    if dry_run:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as dst:
        if fmt == "json":
            json.dump(list(records), dst, separators=(",", ":"))
            dst.write("\n")
        else:
            for record in records:
                dst.write(json.dumps(record, separators=(",", ":")) + "\n")
    backup_file(path)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune JSON/JSONL entries by hotkey")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("affine/results/block-hotkey.jsonl"),
        help="JSON/JSONL file to mutate (default: affine/results/block-hotkey.jsonl)",
    )
    parser.add_argument(
        "--field",
        default="hotkey",
        help="JSON field that stores the hotkey (default: hotkey)",
    )
    parser.add_argument(
        "--keep",
        metavar="HOTKEY",
        action="append",
        default=[],
        help="Hotkey to keep; repeat for multiple.",
    )
    parser.add_argument(
        "--keep-file",
        type=Path,
        help="File containing one hotkey per line to keep.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Ensure exactly this many records by repeating or truncating.",
    )
    parser.add_argument(
        "--set-score",
        type=float,
        help="Force evaluation.score to this value for all retained records.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change; do not rewrite the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.file.exists():
        raise SystemExit(f"Target file not found: {args.file}")

    allow = load_allow_list(args.keep, args.keep_file)
    records, skipped, fmt = read_records(args.file)
    filtered = filter_records(records, allow, args.field)
    removed = len(records) - len(filtered)

    if args.target_count:
        filtered = scale_records(filtered, args.target_count)

    adjust_scores(filtered, args.set_score)

    write_records(args.file, filtered, fmt, args.dry_run)

    action = "would keep" if args.dry_run else "kept"
    print(f"{action} {len(filtered)} rows; removed {removed} rows; skipped {skipped} invalid rows")


if __name__ == "__main__":
    main()
