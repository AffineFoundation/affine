"""Page files: every PAGE any of our pagers posts also lands on disk.

Why (2026-09-21): pipeline-health paged the private Arbos Discord channel
at 05:32 UTC about the env-backfill driver pod and nobody read it until
09:00 — the channel has no reader at night. Discord is now one of three
sinks; the other two are files a report can pick up:

  affine/state/health/PAGE-<utc ts>-<key>.txt   one line, one page; renamed
                                                 to .resolved when the
                                                 condition clears
  affine/state/health/pages.md                   append-only ledger
                                                 (`- [ ]` opened, `- [x]`
                                                 resolved)
  affine/state/health/pages_open.json            {key: {...}} still open —
                                                 the 3-hourly benchsuite
                                                 report lists these

If the Project Agent Store is mounted (STORE_PAGES) the ledger line is
appended there too; on the validator box it is not, so the box-local files
are the source of truth and the report carries them into the store.

    from pagefile import page, resolve, open_pages
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIR = REPO / "affine" / "state" / "health"
LEDGER = DIR / "pages.md"
OPEN = DIR / "pages_open.json"
STORE_PAGES = Path(os.environ.get(
    "AFFINE_STORE_PAGES",
    "/cursor/stores/bc-eea690a7-3595-4417-a14f-9edbe51d567b/internal/pipeline-health/pages.md"))


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _slug(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:60]


def _load_open() -> dict:
    try:
        return json.loads(OPEN.read_text())
    except (OSError, ValueError):
        return {}


def _save_open(d: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = OPEN.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    tmp.replace(OPEN)


def _ledger(line: str) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text("# Pages (all pagers) — `- [ ]` open, `- [x]` resolved. Written by ops/health/pagefile.py.\n\n")
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        if STORE_PAGES.parent.is_dir():
            with STORE_PAGES.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass


def page(key: str, text: str, *, source: str = "pipeline-health") -> Path | None:
    """Record one page. Re-paging an already-open key only refreshes its
    `last` stamp (no second file). Returns the PAGE file path (or None)."""
    key = f"{source}:{key}"
    d = _load_open()
    now = _iso()
    if key in d:
        d[key]["last"] = now
        d[key]["text"] = text
        d[key]["count"] = int(d[key].get("count", 1)) + 1
        _save_open(d)
        return Path(d[key]["file"])
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / f"PAGE-{now.replace(':', '')}-{_slug(key)}.txt"
    path.write_text(f"{now} PAGE {key}: {text}\n")
    d[key] = {"file": str(path), "at": now, "last": now, "text": text, "count": 1}
    _save_open(d)
    _ledger(f"- [ ] {now} PAGE **{key}** — {text}")
    return path


def resolve(key: str, *, source: str = "pipeline-health", note: str = "") -> bool:
    key = f"{source}:{key}"
    d = _load_open()
    rec = d.pop(key, None)
    if rec is None:
        return False
    _save_open(d)
    p = Path(rec["file"])
    try:
        if p.exists():
            p.rename(p.with_suffix(".resolved"))
    except OSError:
        pass
    _ledger(f"- [x] {_iso()} resolved **{key}** (opened {rec['at']}){' — ' + note if note else ''}")
    return True


def open_pages() -> dict:
    return _load_open()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "page":
        print(page(sys.argv[2], " ".join(sys.argv[3:]) or "(manual)", source="cli"))
    elif len(sys.argv) >= 3 and sys.argv[1] == "resolve":
        print(resolve(sys.argv[2], source="cli"))
    else:
        d = open_pages()
        print(f"{len(d)} open page(s)")
        for k, v in sorted(d.items(), key=lambda kv: kv[1]["at"]):
            print(f"  {v['at']}  {k}  x{v.get('count', 1)}  {v['text'][:120]}")
