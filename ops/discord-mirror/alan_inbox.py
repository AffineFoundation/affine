#!/usr/bin/env python
"""Alan's inbox: collect his Discord messages from the mirror and keep the
ideas file up to date.

Standing duty (Jacob, 2026-09-12): record Alan's ideas out of the inbox so we
can talk about them. Alan = channels.toml [alan] (user id, posting target).

Reads the mirror's SQLite archive (mirror.py; channels, threads and the DM
channel with the bot). Selects, inside the window:
  * every message Alan wrote anywhere the mirror can see, and
  * every non-bot message in the posting channel (the DM with the bot, or the
    channel [alan].target names) — replies to our updates land there.
Flags "ideas" with a loose heuristic (keywords + length + questions; it
over-includes on purpose) and writes:
  --md    the ideas file (one entry per idea: date, quote, link, status) plus
          an "Inbox" section with every message, newest first. Statuses
          (new / discussed / adopted / dropped) edited in that file are kept
          across runs — the entry's message id is the key.
  --raw-dir  alan_messages.jsonl (every selected message), alan_ideas.json
          (the flagged ones with status), run.json (window, counts).

  python alan_inbox.py --since 60d --md OUT.md --raw-dir DIR
  python alan_inbox.py --since 60d --dry-run          # counts only
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from read import Archive, load_config, parse_when  # noqa: E402

CONFIG = HERE / "channels.toml"
STATUSES = ("new", "discussed", "adopted", "dropped")
MIN_IDEA_CHARS = 120        # long messages are ideas until proven otherwise
MIN_QUESTION_CHARS = 25     # short questions ("when?") are not ideas
IDEA_WORDS = re.compile(
    r"\b(idea|propos\w*|suggest\w*|should|could we|what if|why not|how about|maybe we|"
    r"consider|instead|better|improv\w*|threshold|algorithm|mechanism|anti-?copy|distill\w*|"
    r"teacher|dataset|eval\w*|duel|scor\w*|incentive|emission\w*|variance|calibrat\w*|"
    r"benchmark|env\b|envs\b|environment\w*|harness|reward|weight\w*|merge\w*|fine-?tun\w*|"
    r"exploit\w*|bug|fix|plan|design|analysis|we (?:can|could|need|want)|i think|"
    r"my (?:view|take|opinion))\b", re.I)
SKIP = re.compile(r"^\s*(af|gm|ok|okay|yes|no|thanks|thank you|lol|haha|\+1|👍)\W*$", re.I)
ENTRY_RE = re.compile(r"<!--\s*id:(\d+)\s+status:(\w+)\s*-->")


def load_alan(config: Path = CONFIG) -> dict:
    raw = tomllib.loads(config.read_text()) if config.exists() else {}
    a = raw.get("alan") or {}
    if not a.get("user_id"):
        raise SystemExit("alan_inbox: channels.toml has no [alan].user_id")
    return {"user_id": str(a["user_id"]), "username": str(a.get("username") or ""),
            "display": str(a.get("display") or ""), "target": str(a.get("target") or "dm")}


def is_idea(content: str) -> tuple[bool, str]:
    """-> (flag, reason). Over-inclusive on purpose."""
    text = (content or "").strip()
    if not text or SKIP.match(text):
        return False, ""
    if len(text) >= MIN_IDEA_CHARS:
        return True, f"long ({len(text)} chars)"
    if "?" in text and len(text) >= MIN_QUESTION_CHARS:
        return True, "question"
    m = IDEA_WORDS.search(text)
    if m and len(text) >= 40:
        return True, f"keyword '{m.group(0).lower()}'"
    return False, ""


def message_link(a: Archive, r: sqlite3.Row) -> str:
    cid = r["thread_id"] or r["channel_id"]
    guild = r["guild_id"] or (a.channels.get(r["channel_id"]) or {}).get("guild_id") or ""
    return f"https://discord.com/channels/{guild or '@me'}/{cid}/{r['id']}"


def posting_channels(a: Archive, alan: dict) -> list[str]:
    if alan["target"] != "dm":
        return [alan["target"]] if alan["target"] in a.channels else []
    return [cid for cid, c in a.channels.items()
            if c["kind"] == "dm" and (alan["username"] and alan["username"] in (c["name"] or ""))]


def select(a: Archive, alan: dict, since: str | None, until: str | None) -> list[sqlite3.Row]:
    where, args = ["(m.author_id = ?", ], [alan["user_id"]]
    posting = posting_channels(a, alan)
    if posting:
        where[0] += " OR (m.author_bot = 0 AND m.channel_id IN (%s))" % ",".join("?" * len(posting))
        args += posting
    where[0] += ")"
    if since:
        where.append("m.timestamp >= ?"); args.append(since)
    if until:
        where.append("m.timestamp <= ?"); args.append(until)
    sql = f"SELECT m.* FROM messages m WHERE {' AND '.join(where)} ORDER BY m.id DESC"
    return a.db.execute(sql, args).fetchall()


def prior_statuses(md: Path) -> dict[str, str]:
    if not md.exists():
        return {}
    return {mid: st for mid, st in ENTRY_RE.findall(md.read_text(encoding="utf-8")) if st in STATUSES}


def row_dict(a: Archive, r: sqlite3.Row) -> dict:
    d = dict(r)
    d["id"] = str(d["id"])
    for k in ("attachments", "embeds", "reactions"):
        d[k] = json.loads(d[k] or "[]")
    d["channel_name"] = a.name_of(d["channel_id"])
    d["thread_name"] = a.name_of(d["thread_id"]) if d["thread_id"] else None
    d["link"] = message_link(a, r)
    return d


def fmt_ts(ts: str) -> str:
    return (ts or "")[:16].replace("T", " ") + " UTC"


def where_of(d: dict) -> str:
    w = f"#{d['channel_name']}"
    if d["thread_name"]:
        w += f" / thread {d['thread_name']}"
    return w


def render_md(alan: dict, msgs: list[dict], ideas: list[dict], since: str | None, until: str | None) -> str:
    by_status = Counter(i["status"] for i in ideas)
    out = [
        "# Alan — ideas and inbox",
        "",
        f"Alan = `{alan['username']}` (display {alan['display']}, Discord user id `{alan['user_id']}`). "
        "Built by `ops/discord-mirror/alan_inbox.py` from the Discord mirror; re-run to refresh. "
        "Edit the `status:` word of an idea (new / discussed / adopted / dropped) — it survives re-runs.",
        "",
        f"Window: {since or 'archive start'} → {until or 'now'}. Messages: {len(msgs)}. "
        f"Ideas flagged: {len(ideas)} (" + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())) + ").",
        "",
        "Flagging is a loose heuristic (length ≥ 120 chars, or a question, or a keyword like "
        "propose / threshold / distill / eval); it over-includes on purpose. Anything Alan sends "
        "to the bot (DM or a reply in the posting channel) shows up here after the next run.",
        "",
        "## Ideas",
        "",
    ]
    if not ideas:
        out += ["_none flagged in this window_", ""]
    for i in ideas:
        head = re.sub(r"\s+", " ", i["content"]).strip()
        head = head[:90] + ("…" if len(head) > 90 else "")
        out += [
            f"### {i['timestamp'][:10]} — {head}",
            "",
            f"<!-- id:{i['id']} status:{i['status']} -->",
            f"- date: {fmt_ts(i['timestamp'])}",
            f"- where: {where_of(i)} · [link]({i['link']})" + (f" · reply to `{i['reply_to']}`" if i.get("reply_to") else ""),
            f"- flagged: {i['reason']}",
            f"- status: {i['status']}",
            "- quote:",
            "",
        ]
        out += ["  > " + line for line in (i["content"] or "").replace("\r", "").split("\n")]
        if i["attachments"]:
            out += ["", "  attachments: " + ", ".join(f"[{x.get('filename')}]({x.get('url')})" for x in i["attachments"])]
        out.append("")
    out += ["## Inbox", "", "Every message in the window, newest first. `[idea]` = flagged above.", ""]
    if not msgs:
        out += ["_no messages in this window_", ""]
    for d in msgs:
        body = re.sub(r"\s+", " ", (d["content"] or "")).strip()
        if not body and d["attachments"]:
            body = "(attachment: " + ", ".join(x.get("filename") or "?" for x in d["attachments"]) + ")"
        if not body and d["embeds"]:
            body = "(embed)"
        body = body[:300] + ("…" if len(body) > 300 else "")
        tag = " `[idea]`" if d.get("idea") else ""
        who = "" if d["author_id"] == alan["user_id"] else f" <{d['author_name']}>"
        out.append(f"- {fmt_ts(d['timestamp'])} · {where_of(d)} · [link]({d['link']}){who}{tag} — {body}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--db", help="sqlite path (overrides config)")
    ap.add_argument("--since", default="60d", help="ISO date or relative ('60d'); default 60d")
    ap.add_argument("--until")
    ap.add_argument("--md", help="ideas markdown to write/update")
    ap.add_argument("--raw-dir", help="directory for alan_messages.jsonl / alan_ideas.json / run.json")
    ap.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = ap.parse_args()

    alan = load_alan(Path(args.config))
    ddir, aliases = load_config(Path(args.config))
    a = Archive(Path(args.db) if args.db else ddir / "discord.sqlite", aliases)
    since, until = parse_when(args.since), parse_when(args.until)
    rows = select(a, alan, since, until)
    msgs = [row_dict(a, r) for r in rows]
    prior = prior_statuses(Path(args.md)) if args.md else {}
    ideas = []
    for d in msgs:
        flag, reason = is_idea(d["content"])
        d["idea"] = flag
        if flag:
            ideas.append({**d, "reason": reason, "status": prior.get(d["id"], "new")})
    posting = posting_channels(a, alan)
    print(f"alan_inbox: {len(msgs)} message(s) by/for {alan['username']} since {since or '-'}; "
          f"{len(ideas)} flagged as ideas; posting channel(s) mirrored: {posting or 'none yet'}")
    if args.dry_run:
        for i in ideas[:10]:
            preview = re.sub(r"\s+", " ", i["content"])[:100]
            print(f"  {i['timestamp'][:16]} [{i['reason']}] {preview}")
        return
    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_md(alan, msgs, ideas, since, until), encoding="utf-8")
        print(f"wrote {md}")
    if args.raw_dir:
        raw = Path(args.raw_dir)
        raw.mkdir(parents=True, exist_ok=True)
        with (raw / "alan_messages.jsonl").open("w", encoding="utf-8") as f:
            for d in msgs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        (raw / "alan_ideas.json").write_text(json.dumps(ideas, indent=1, ensure_ascii=False), encoding="utf-8")
        (raw / "run.json").write_text(json.dumps({
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "since": since, "until": until,
            "alan": alan, "messages": len(msgs), "ideas": len(ideas), "posting_channels": posting,
            "status_counts": dict(Counter(i["status"] for i in ideas)),
        }, indent=1), encoding="utf-8")
        print(f"wrote {raw}/alan_messages.jsonl, alan_ideas.json, run.json")


if __name__ == "__main__":
    main()
