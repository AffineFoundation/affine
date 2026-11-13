#!/usr/bin/env python3
"""Mechanical check: which lines does new v2 code share with legacy trees?

v2 is written from the ground up. "Legacy" is v1 (this repository outside
``v2/``) and the v2 prototype. They are references for behaviour and formats
only, so a shared non-trivial line is a finding: explain it or rewrite it.

Each new line is put in at most one class:

  EXACT    the whole line also appears in a legacy file, and the line is
           non-trivial (at least ``--min-chars`` characters and three distinct
           names that are not Python keywords or builtins).
  NEAR     one single legacy line shares at least ``--near-ratio`` of this
           line's token windows. A token window is a run of ``--ngram``
           consecutive tokens. This catches a line that was adapted, not
           copied whole.
  SCATTER  some token windows appear in legacy lines, but no single legacy
           line comes close. This is ordinary shared syntax. It is counted,
           and listed only with ``--verbose``.

Lines are compared after trimming, collapsing spaces and using one quote
style. Import lines are skipped. Paths given with ``--exempt`` are not
checked: use it only for the two workstreams that may reuse v1 (the benchmark
service and the website).

Example (run from the repository root):

  python v2/tools/legacy_overlap.py --new v2 --legacy . --exclude v2 \
      --exempt v2/bench --exempt v2/site --legacy /path/to/unpacked/prototype
"""

from __future__ import annotations

import argparse
import builtins
import keyword
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|\S")
TEXT_SUFFIXES = frozenset(
    ".py .sh .js .cjs .mjs .ts .tsx .json .toml .cfg .ini .yaml .yml .md .txt .service .timer .html .css".split()
)
SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"})
COMMON_WORDS = frozenset(keyword.kwlist) | frozenset(dir(builtins)) | {"self", "cls"}
MAX_FILE_BYTES = 4 << 20
MAX_LINE_CHARS = 2000
MIN_NEAR_WINDOWS = 3


@dataclass
class NewLine:
    path: str
    number: int
    text: str
    n_windows: int
    exact_count: int = 0
    exact_example: str = ""
    seen_windows: set[int] = field(default_factory=set)
    best_count: int = 0
    best_example: str = ""


def normalize(line: str) -> str:
    return " ".join(line.replace("'", '"').split())


def is_import(norm: str) -> bool:
    return norm.startswith("import ") or (norm.startswith("from ") and " import " in norm)


def is_non_trivial(norm: str, tokens: list[str], min_chars: int) -> bool:
    if len(norm) < min_chars:
        return False
    names = {t for t in tokens if t[0].isalpha() or t[0] == "_"}
    return len(names - COMMON_WORDS) >= 3


def windows(tokens: list[str], n: int) -> list[str]:
    return ["\x1f".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def text_files(root: Path, skipped: list[Path]) -> list[Path]:
    if root.is_file():
        return [root]
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if SKIP_DIRS.intersection(path.parts):
            continue
        if any(path.is_relative_to(skip) for skip in skipped):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        found.append(path)
    return found


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def scan(new_roots, legacy_roots, excluded, exempt, ngram, min_chars):
    lines: list[NewLine] = []
    by_text: dict[str, list[int]] = {}
    by_window: dict[str, list[tuple[int, int]]] = {}
    new_files = [p for root in new_roots for p in text_files(root, exempt)]
    for path in new_files:
        for number, raw in enumerate(read_lines(path), 1):
            norm = normalize(raw)
            if not norm or is_import(norm) or len(norm) > MAX_LINE_CHARS:
                continue
            tokens = TOKEN.findall(norm)
            wins = windows(tokens, ngram)
            index = len(lines)
            lines.append(NewLine(str(path), number, raw.strip(), len(wins)))
            if is_non_trivial(norm, tokens, min_chars):
                by_text.setdefault(norm, []).append(index)
            for position, win in enumerate(wins):
                by_window.setdefault(win, []).append((index, position))

    new_set = set(new_files)
    legacy_files = [p for root in legacy_roots for p in text_files(root, excluded) if p not in new_set]
    for path in legacy_files:
        for number, raw in enumerate(read_lines(path), 1):
            if len(raw) > MAX_LINE_CHARS:
                continue
            norm = normalize(raw)
            if not norm:
                continue
            where = f"{path}:{number}"
            for index in by_text.get(norm, ()):
                lines[index].exact_count += 1
                lines[index].exact_example = lines[index].exact_example or where
            tokens = TOKEN.findall(norm)
            if len(tokens) < ngram:
                continue
            shared: dict[int, set[int]] = {}
            for win in windows(tokens, ngram):
                for index, position in by_window.get(win, ()):
                    shared.setdefault(index, set()).add(position)
            for index, positions in shared.items():
                line = lines[index]
                line.seen_windows |= positions
                if len(positions) > line.best_count:
                    line.best_count, line.best_example = len(positions), where
    return lines, len(new_files), len(legacy_files)


def classify(line: NewLine, near_ratio: float) -> str:
    if line.exact_count:
        return "EXACT"
    if line.n_windows >= MIN_NEAR_WINDOWS and line.best_count >= near_ratio * line.n_windows:
        return "NEAR"
    return "SCATTER" if line.seen_windows else ""


def report(lines, n_new, n_legacy, args) -> dict[str, int]:
    print("legacy-overlap report")
    print(f"new:      {', '.join(map(str, args.new))}  ({n_new} files, {len(lines)} compared lines)")
    print(f"legacy:   {', '.join(map(str, args.legacy))}  ({n_legacy} files)")
    print(f"excluded: {', '.join(map(str, args.exclude)) or '-'}")
    print(f"exempt:   {', '.join(map(str, args.exempt)) or '-'}")
    print(f"settings: ngram={args.ngram} tokens, near_ratio={args.near_ratio}, min_chars={args.min_chars}")
    counts = {"EXACT": 0, "NEAR": 0, "SCATTER": 0}
    shown = ("EXACT", "NEAR", "SCATTER") if args.verbose else ("EXACT", "NEAR")
    current = None
    for line in sorted(lines, key=lambda item: (item.path, item.number)):
        kind = classify(line, args.near_ratio)
        if not kind:
            continue
        counts[kind] += 1
        if kind not in shown:
            continue
        if line.path != current:
            current = line.path
            print(f"\n== {current}")
        if kind == "EXACT":
            detail = f"in {line.exact_count} legacy line(s), e.g. {line.exact_example}"
        elif kind == "NEAR":
            detail = f"{line.best_count}/{line.n_windows} windows of one legacy line: {line.best_example}"
        else:
            detail = f"{len(line.seen_windows)}/{line.n_windows} windows, best single line {line.best_count}"
        print(f"  L{line.number} {kind} {detail}")
        print(f"      {line.text}")
    print(
        f"\nsummary: EXACT={counts['EXACT']} NEAR={counts['NEAR']} "
        f"SCATTER={counts['SCATTER']} (scattered shared syntax; list with --verbose)"
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new", type=Path, action="append", required=True, help="new v2 file or directory")
    parser.add_argument("--legacy", type=Path, action="append", required=True, help="legacy file or directory")
    parser.add_argument("--exclude", type=Path, action="append", default=[], help="path to skip inside a legacy root")
    parser.add_argument("--exempt", type=Path, action="append", default=[], help="path to skip inside a new root")
    parser.add_argument("--ngram", type=int, default=8, help="tokens per window")
    parser.add_argument("--near-ratio", type=float, default=0.6, help="share of windows in one legacy line for NEAR")
    parser.add_argument("--min-chars", type=int, default=24, help="shortest line that can be EXACT")
    parser.add_argument("--verbose", action="store_true", help="also list SCATTER lines")
    parser.add_argument("--fail-on", choices=("none", "exact", "near"), default="none")
    args = parser.parse_args(argv)
    for path in args.new + args.legacy:
        if not path.exists():
            parser.error(f"path does not exist: {path}")
    resolve = lambda paths: [p.resolve() for p in paths]
    lines, n_new, n_legacy = scan(
        resolve(args.new), resolve(args.legacy), resolve(args.exclude), resolve(args.exempt), args.ngram, args.min_chars
    )
    counts = report(lines, n_new, n_legacy, args)
    if args.fail_on == "exact" and counts["EXACT"]:
        return 1
    if args.fail_on == "near" and (counts["EXACT"] or counts["NEAR"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
