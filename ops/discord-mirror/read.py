#!/usr/bin/env python
"""Read the local Discord archive written by mirror.py. Plain-text output.

  python read.py channels                          mirrored channels + counts
  python read.py tail [CHANNEL] [-n 50]            newest messages (oldest first)
  python read.py search QUERY [--since ISO] [--until ISO] [--channel C] [-n 50]
  python read.py export --since ISO [--until ISO] [--channel C] [--format markdown|jsonl] [-o FILE]
  python read.py stats [--since ISO] [--channel C]  messages per channel per day, top authors

CHANNEL is a channel id, a thread id, or a case-insensitive substring of the
channel name ("affine"). Threads are shown under their parent channel; pass a
thread id to read one thread. --since/--until accept ISO dates ("2026-09-01",
"2026-09-01T12:00Z") or relative "14d" / "36h".
QUERY uses SQLite FTS5 syntax when available ("stuck eval", '"weight version"',
'miner OR miners', 'author_name:unconst'); falls back to LIKE otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sqlite3
import sys
import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "channels.toml"
RELATIVE = re.compile(r"^(\d+)([dhm])$")


def load_config(config: Path = CONFIG) -> tuple[Path, dict[str, list[str]]]:
    """-> (data_dir, {channel id: [aliases]})."""
    raw = tomllib.loads(config.read_text()) if config.exists() else {}
    d = Path((raw.get("mirror") or {}).get("data_dir") or "affine/state/discord")
    aliases = {str(c["id"]): [str(a) for a in (c.get("aliases") or [])] for c in raw.get("channels") or []}
    return (d if d.is_absolute() else REPO / d), aliases


def fold(name: str) -> str:
    """Channel names use Tifinagh 'ⴷ' for 'a' (ⴷffine); fold it so 'affine' matches."""
    return (name or "").lower().replace("ⴷ", "a")


def parse_when(value: str | None) -> str | None:
    """ISO date/datetime or relative ('14d', '6h', '30m') -> UTC ISO string."""
    if not value:
        return None
    m = RELATIVE.match(value.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
        return (datetime.now(timezone.utc) - delta).isoformat(timespec="seconds")
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Archive:
    def __init__(self, path: Path, aliases: dict[str, list[str]] | None = None):
        if not path.exists():
            raise SystemExit(f"read.py: no archive at {path} (has mirror.py run?)")
        self.aliases = aliases or {}
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.has_fts = bool(self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone())
        self.channels = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM channels")}

    def name_of(self, cid: str | None) -> str:
        c = self.channels.get(cid or "")
        return c["name"] if c else str(cid)

    def resolve(self, spec: str | None) -> tuple[list[str], str | None]:
        """CHANNEL spec -> (parent channel ids, thread id or None)."""
        if not spec:
            return [c for c, r in self.channels.items() if r["kind"] == "channel"], None
        if spec in self.channels:
            r = self.channels[spec]
            return ([r["parent_id"]], spec) if r["kind"] == "thread" else ([spec], None)
        want = fold(spec)
        hits = [c for c, r in self.channels.items() if r["kind"] == "channel"
                and (want in fold(r["name"]) or any(want in fold(a) for a in self.aliases.get(c, [])))]
        if not hits:
            raise SystemExit(f"read.py: no mirrored channel matches {spec!r}; try `read.py channels`")
        return hits, None

    def query(self, channels: list[str], thread: str | None, since: str | None, until: str | None,
              text: str | None, limit: int | None, newest_first: bool) -> list[sqlite3.Row]:
        where, args = ["m.channel_id IN (%s)" % ",".join("?" * len(channels))], list(channels)
        if thread:
            where.append("m.thread_id = ?"); args.append(thread)
        if since:
            where.append("m.timestamp >= ?"); args.append(since)
        if until:
            where.append("m.timestamp <= ?"); args.append(until)
        join = ""
        if text:
            if self.has_fts:
                join = "JOIN messages_fts f ON f.rowid = m.id"
                where.append("messages_fts MATCH ?"); args.append(text)
            else:
                where.append("m.content LIKE ?"); args.append(f"%{text}%")
        sql = (f"SELECT m.* FROM messages m {join} WHERE {' AND '.join(where)} "
               f"ORDER BY m.id {'DESC' if newest_first else 'ASC'}")
        if limit:
            sql += f" LIMIT {int(limit)}"
        try:
            return self.db.execute(sql, args).fetchall()
        except sqlite3.OperationalError as exc:
            raise SystemExit(f"read.py: bad query ({exc}); quote phrases: '\"weight version\"'")


def fmt_line(a: Archive, r: sqlite3.Row, show_channel: bool) -> str:
    ts = (r["timestamp"] or "")[:16].replace("T", " ")
    who = r["author_name"] or r["author_id"]
    if r["author_display"] and r["author_display"] != who:
        who = f"{who} ({r['author_display']})"
    where = ""
    if show_channel:
        where = f" #{a.name_of(r['channel_id'])}"
    if r["thread_id"]:
        where += f" [thread {a.name_of(r['thread_id'])}]"
    tags = []
    if r["reply_to"]:
        tags.append(f"reply:{r['reply_to']}")
    if r["edited_timestamp"]:
        tags.append("edited")
    if r["deleted_at"]:
        tags.append("DELETED")
    atts = json.loads(r["attachments"] or "[]")
    if atts:
        tags.append("att:" + ",".join(x.get("filename") or "?" for x in atts))
    embeds = json.loads(r["embeds"] or "[]")
    if embeds:
        tags.append(f"embeds:{len(embeds)}")
    reacts = json.loads(r["reactions"] or "[]")
    if reacts:
        tags.append("react:" + " ".join(f"{x['emoji']}x{x['count']}" for x in reacts if x.get("emoji")))
    body = (r["content"] or "").replace("\r", "")
    if not body and embeds:
        body = " | ".join(
            " — ".join(filter(None, [e.get("title") or "", (e.get("description") or "")[:200]])) for e in embeds)
    head = f"{ts}{where} <{who}>" + (f" {{{', '.join(tags)}}}" if tags else "")
    lines = body.split("\n")
    if len(lines) == 1:
        return f"{head} {body}"
    return head + "\n    " + "\n    ".join(lines)


# -- commands ----------------------------------------------------------------------
def cmd_channels(a: Archive) -> None:
    counts = {}
    for r in a.db.execute("SELECT channel_id, thread_id, count(*) n, max(timestamp) t FROM messages GROUP BY 1,2"):
        counts[(r["channel_id"], r["thread_id"])] = (r["n"], r["t"])
    for cid, c in sorted(a.channels.items(), key=lambda kv: (kv[1]["kind"] != "channel", kv[1]["name"] or "")):
        if c["kind"] != "channel":
            continue
        n, t = counts.get((cid, None), (0, None))
        threads = [t for t in a.channels.values() if t["parent_id"] == cid]
        tn = sum(counts.get((cid, t["id"]), (0,))[0] for t in threads)
        print(f"{cid:<20} {n:>7} msgs  +{tn} in {len(threads)} threads  latest={(t or '-')[:19]}  "
              f"backfilled={c['backfilled']}  #{c['name']}" + (f"  ERROR {c['error']}" if c["error"] else ""))
        for t in threads:
            n2, t2 = counts.get((cid, t["id"]), (0, None))
            print(f"    thread {t['id']:<20} {n2:>5} msgs  latest={(t2 or '-')[:19]}  {t['name']}")


def cmd_tail(a: Archive, args) -> None:
    channels, thread = a.resolve(args.channel)
    rows = a.query(channels, thread, None, None, None, args.n, newest_first=True)
    for r in reversed(rows):
        print(fmt_line(a, r, show_channel=len(channels) > 1))


def cmd_search(a: Archive, args) -> None:
    channels, thread = a.resolve(args.channel)
    rows = a.query(channels, thread, parse_when(args.since), parse_when(args.until), args.query, args.n, newest_first=True)
    print(f"{len(rows)} hit(s) for {args.query!r}" + (" (newest first, limited)" if len(rows) == args.n else ""))
    for r in rows:
        print(fmt_line(a, r, show_channel=True))


def cmd_export(a: Archive, args) -> None:
    channels, thread = a.resolve(args.channel)
    rows = a.query(channels, thread, parse_when(args.since), parse_when(args.until), None, None, newest_first=False)
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        if args.format == "jsonl":
            for r in rows:
                d = dict(r)
                d["id"] = str(d["id"])
                for k in ("attachments", "embeds", "reactions"):
                    d[k] = json.loads(d[k] or "[]")
                d["channel_name"] = a.name_of(d["channel_id"])
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
        else:
            out.write(f"# Discord export — {len(rows)} messages\n")
            out.write(f"channels: {', '.join('#' + a.name_of(c) + ' (' + c + ')' for c in channels)}"
                      + (f"; thread {a.name_of(thread)}" if thread else "") + "\n")
            out.write(f"since: {parse_when(args.since) or '-'}  until: {parse_when(args.until) or 'now'}\n")
            day = None
            for r in rows:
                d = (r["timestamp"] or "")[:10]
                if d != day:
                    day = d
                    out.write(f"\n## {day}\n\n")
                out.write(fmt_line(a, r, show_channel=len(channels) > 1) + "\n")
    finally:
        if args.output:
            out.close()
            print(f"wrote {len(rows)} messages to {args.output}")


def cmd_stats(a: Archive, args) -> None:
    channels, thread = a.resolve(args.channel)
    rows = a.query(channels, thread, parse_when(args.since), None, None, None, newest_first=False)
    per_day: dict[str, Counter] = defaultdict(Counter)
    authors, bots = Counter(), Counter()
    for r in rows:
        key = a.name_of(r["channel_id"])
        per_day[(r["timestamp"] or "")[:10]][key] += 1
        (bots if r["author_bot"] else authors)[r["author_name"] or r["author_id"]] += 1
    print(f"{len(rows)} messages, {len(authors)} human authors, {len(bots)} bots"
          + (f", since {parse_when(args.since)}" if args.since else ""))
    names = sorted({n for c in per_day.values() for n in c})
    print("\nmessages per channel per day")
    print(f"{'day':<10} " + " ".join(f"{n[:14]:>14}" for n in names) + f" {'total':>6}")
    for day in sorted(per_day):
        c = per_day[day]
        print(f"{day:<10} " + " ".join(f"{c[n]:>14}" for n in names) + f" {sum(c.values()):>6}")
    print("\ntop authors (humans)")
    for name, n in authors.most_common(args.top):
        print(f"  {n:>6}  {name}")
    if bots:
        print("\nbots")
        for name, n in bots.most_common(5):
            print(f"  {n:>6}  {name}")


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `read.py ... | head` exits quietly
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG), help="channels.toml (for data_dir)")
    ap.add_argument("--db", help="sqlite path (overrides config)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("channels")
    p = sub.add_parser("tail"); p.add_argument("channel", nargs="?"); p.add_argument("-n", type=int, default=50)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("--since"); p.add_argument("--until")
    p.add_argument("--channel"); p.add_argument("-n", type=int, default=50)
    p = sub.add_parser("export"); p.add_argument("--since", required=True); p.add_argument("--until")
    p.add_argument("--channel"); p.add_argument("--format", choices=["markdown", "jsonl"], default="markdown")
    p.add_argument("-o", "--output")
    p = sub.add_parser("stats"); p.add_argument("--since"); p.add_argument("--channel"); p.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    ddir, aliases = load_config(Path(args.config))
    a = Archive(Path(args.db) if args.db else ddir / "discord.sqlite", aliases)
    {"channels": lambda: cmd_channels(a), "tail": lambda: cmd_tail(a, args), "search": lambda: cmd_search(a, args),
     "export": lambda: cmd_export(a, args), "stats": lambda: cmd_stats(a, args)}[args.cmd]()


if __name__ == "__main__":
    main()
