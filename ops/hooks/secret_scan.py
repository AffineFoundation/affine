#!/usr/bin/env python3
"""Token-shaped-literal scan for the affine repo (stdlib only).

Refuses commits / pushes that add a line matching a secret-shaped pattern,
and gives affine-pipeline-health a whole-tree scan. Born from the Engy key
that sat in research/scripts/engy_parity.py from 2026-09-09 to 2026-09-20.

    secret_scan.py --staged            # pre-commit: added lines of the index
    secret_scan.py --range A..B        # pre-push: added lines of those commits
    secret_scan.py --commits SHA...    # same, explicit commit list
    secret_scan.py --tree              # every tracked file (health check)
    secret_scan.py --text FILE         # one file / '-' = stdin (self-test)

Exit 0 = clean, 1 = hit(s), 2 = usage / git error. Output never prints the
matched secret: the first 6 characters and the length, nothing more.

Allowlist: ops/hooks/secret_scan_allow.txt — one path glob per line
(fnmatch, `**` = any depth), `#` comments; `commit:<full sha>` lines name
historical commits whose leak is already known and rotated (skipped in
--commits mode only, so re-pushing history to a new remote is not blocked
by a key that is already public). Dataset dumps with fake keys in task
prompts live there. A file, not a pragma: a `# noscan` comment next to
a literal would be the wrong reflex.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ALLOW = HERE / "secret_scan_allow.txt"

# (name, regex). Keep the shapes narrow: a false refusal costs a minute,
# a miss costs a key.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai/engy sk-", re.compile(r"sk-[A-Za-z0-9_-]{16,}")),
    ("anthropic sk-ant-", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("huggingface hf_", re.compile(r"\bhf_[A-Za-z0-9]{25,}")),
    ("github ghp_", re.compile(r"\bghp_[A-Za-z0-9]{30,}")),
    ("github pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("slack xox", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{20,}")),
    ("aws AKIA", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("1password op_", re.compile(r"\bop_[A-Za-z0-9]{40,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}")),
    ("assignment", re.compile(
        r"(?i:\b(api[_-]?key|secret|token|password|passwd|bearer)\b[\"' ]*[:=][ ]*[\"'][A-Za-z0-9_\-]{24,}[\"'])")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
]
# One pass over a whole text (a 19 MB dataset dump takes seconds, not
# minutes): the alternation of every shape; the per-shape name is resolved
# on the (rare) match.
COMBINED = re.compile("|".join(f"(?:{rx.pattern})" for _, rx in PATTERNS))
# Text we never treat as a hit (documentation of the shapes themselves).
IGNORE_LINE = re.compile(r"<REDACTED>|sk-\.\.\.|sk-…|\$\{?[A-Z_]+\}?|op://")
SKIP_SUFFIXES = (".lock", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".gz", ".zip",
                 ".parquet", ".safetensors", ".bin", ".pt", ".ico", ".woff", ".woff2")


def load_allow() -> list[str]:
    try:
        return [l.strip() for l in ALLOW.read_text().split("\n")
                if l.strip() and not l.lstrip().startswith("#") and not l.startswith("commit:")]
    except OSError:
        return []


def known_commits() -> set[str]:
    try:
        return {l.strip()[len("commit:"):].split()[0] for l in ALLOW.read_text().split("\n")
                if l.startswith("commit:")}
    except OSError:
        return set()


def allowed(path: str, globs: list[str]) -> bool:
    if path.endswith(SKIP_SUFFIXES):
        return True
    for g in globs:
        # fnmatch's `*` already crosses `/`; `**/` additionally matches "no dir".
        if fnmatch.fnmatch(path, g) or (g.startswith("**/") and fnmatch.fnmatch(path, g[3:])):
            return True
    return False


def git(*args: str) -> str:
    # diffs may carry non-UTF-8 bytes (binary-ish data files): never crash on them
    p = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(2)
    return p.stdout


def redact(s: str) -> str:
    return f"{s[:6]}…({len(s)} chars)"


def _name_of(token: str) -> str:
    for name, rx in PATTERNS:
        if rx.search(token):
            return name
    return "secret-shaped"


def scan_text(path: str, text: str, hits: list[tuple[str, int | None, str, str]], lineno_from: int = 1):
    """Whole-text scan; one hit per offending line."""
    seen_lines: set[int] = set()
    for m in COMBINED.finditer(text):
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.end())
        line = text[ls:le if le != -1 else None]
        if IGNORE_LINE.search(line):
            continue
        ln = lineno_from + text.count("\n", 0, ls)
        if ln in seen_lines:
            continue
        seen_lines.add(ln)
        hits.append((path, ln, _name_of(m.group(0)), redact(m.group(0))))


def scan_lines(path: str, lines, hits: list[tuple[str, int | None, str, str]], lineno_from: int | None = None):
    scan_text(path, "\n".join(lines), hits, lineno_from if lineno_from is not None else 1)


def scan_diff(diff_text: str, globs: list[str]) -> list[tuple[str, int | None, str, str]]:
    """Added lines of a unified diff (-U0), attributed to their file."""
    hits: list[tuple[str, int | None, str, str]] = []
    path = ""
    skip = False
    lineno = 0
    for raw in diff_text.split("\n"):
        if raw.startswith("+++ "):
            path = raw[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            skip = path == "/dev/null" or allowed(path, globs)
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
            continue
        if skip or not raw.startswith("+") or raw.startswith("+++"):
            if raw.startswith(" ") or raw.startswith("+"):
                lineno += 1
            continue
        scan_lines(path, [raw[1:]], hits, lineno)
        lineno += 1
    return hits


def scan_tree(globs: list[str]) -> list[tuple[str, int | None, str, str]]:
    hits: list[tuple[str, int | None, str, str]] = []
    for path in git("ls-files", "-z").split("\0"):
        if not path or allowed(path, globs):
            continue
        try:
            text = (REPO / path).read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if "\0" in text[:4096]:
            continue
        scan_text(path, text, hits, 1)
    return hits


def report(hits, what: str) -> int:
    if not hits:
        return 0
    sys.stderr.write(f"secret_scan: REFUSED — {len(hits)} secret-shaped literal(s) in {what}:\n")
    for path, ln, name, red in hits[:40]:
        where = f"{path}:{ln}" if ln is not None else path
        sys.stderr.write(f"  {where}  [{name}]  {red}\n")
    sys.stderr.write("Move the value to the environment / the Arbos vault (op://Arbos/...). "
                     "Dataset false positives: add the path glob to ops/hooks/secret_scan_allow.txt. "
                     "Bypass only with an operator directive: git commit/push --no-verify.\n")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--range", metavar="A..B")
    g.add_argument("--commits", nargs="+", metavar="SHA")
    g.add_argument("--tree", action="store_true")
    g.add_argument("--text", metavar="FILE")
    args = ap.parse_args()
    globs = load_allow()
    if args.staged:
        return report(scan_diff(git("diff", "--cached", "-U0", "--no-color", "--diff-filter=AM"), globs), "the staged changes")
    if args.range:
        return report(scan_diff(git("diff", "-U0", "--no-color", "--diff-filter=AM", args.range), globs), f"commits {args.range}")
    if args.commits:
        hits = []
        known = known_commits()
        for sha in args.commits:
            full = git("rev-parse", sha).strip()
            if full in known:
                continue
            hits += scan_diff(git("show", "--format=", "-U0", "--no-color", "--diff-filter=AM", sha), globs)
        return report(hits, f"{len(args.commits)} commit(s)")
    if args.tree:
        hits = scan_tree(globs)
        rc = report(hits, "the tracked tree")
        if rc == 0:
            print(f"secret_scan: tree clean ({len(git('ls-files', '-z').split(chr(0))) - 1} tracked files, "
                  f"{len(globs)} allowlist globs)")
        return rc
    if args.text:
        text = sys.stdin.read() if args.text == "-" else Path(args.text).read_text(errors="ignore")
        hits = []
        scan_text(args.text, text, hits, 1)
        return report(hits, args.text)
    return 2


if __name__ == "__main__":
    os.chdir(REPO)
    sys.exit(main())
