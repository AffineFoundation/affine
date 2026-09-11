#!/usr/bin/env python
"""Discord mirror: archive the SN120 channels locally so agents can read them.

Polls the Discord REST API with the "Arbos" bot token (no gateway, no
websockets). Channels come from channels.toml plus auto-discovery (any text
channel whose name or guild name matches [mirror].discover_patterns).

Per channel (and per thread under a mirrored channel):
  * first run: full history, newest -> oldest (GET /channels/{id}/messages
    ?before=<oldest_id>&limit=100), progress persisted so a restart resumes;
  * then every poll_interval_s: GET ...?after=<last_id>&limit=100 until the
    page is short, plus a re-fetch of the newest `recent_window` messages so
    edits / reactions / deletions inside that window are recorded.

Storage under [mirror].data_dir:
  discord.sqlite   messages + channels tables, FTS5 index for read.py search
  raw/<id>.jsonl   append-only mirror of every message object as fetched
                   (one line per fetch: {"fetched_at", "channel_id",
                   "thread_id", "message"}); edits append again
  status.json      per-channel counters, last poll, blockers (intent, 403s)

Message Content intent: Discord blanks `content` for messages the bot did not
write unless the app has the privileged "Message Content Intent". The mirror
checks the application flags at start and watches for the symptom (many
non-bot messages with empty content and no attachments/embeds); both are
logged and written to status.json["blockers"].

Rate limits: X-RateLimit-Remaining / Reset-After are honoured per route;
429 sleeps retry_after; 5xx / network errors back off 1 -> 60 s.

  python mirror.py run          # the service (pm2 affine-discord-mirror)
  python mirror.py once         # one full cycle, then exit
  python mirror.py discover     # list guilds / channels the bot can see
  python mirror.py status       # what is mirrored, counts, blockers
Never prints the token.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "channels.toml"
VALIDATOR_ENV = Path.home() / ".affine-validator.env"
REPO_ENV = REPO / ".env"

API = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (https://affine.io, 1.0) affine-discord-mirror"
PAGE = 100
TEXT_CHANNEL_TYPES = {0, 5, 15, 16}         # text, announcement, forum, media
THREAD_TYPES = {10, 11, 12}                  # news, public, private threads
FLAG_MESSAGE_CONTENT = 1 << 18               # GATEWAY_MESSAGE_CONTENT
FLAG_MESSAGE_CONTENT_LIMITED = 1 << 19       # granted to bots in < 100 guilds
INTENT_MIN_SAMPLE = 20                       # non-bot messages before judging
INTENT_EMPTY_RATIO = 0.9
BACKOFF_MAX_S = 60.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"{now_iso()} {msg}", flush=True)


def env_file_value(name: str) -> str:
    """Token lookup: process env, then ~/.affine-validator.env, then repo .env.
    Same order as ops/king-datagen/kingctl.py. Missing -> ""."""
    if os.environ.get(name):
        return os.environ[name]
    for path in (VALIDATOR_ENV, REPO_ENV):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# -- config ----------------------------------------------------------------------
@dataclass
class ChannelCfg:
    id: str
    guild_id: str
    name: str
    note: str = ""


@dataclass
class Config:
    data_dir: Path
    poll_interval_s: float
    thread_scan_interval_s: float
    recent_window: int
    token_env: str
    discover_patterns: list[str]
    channels: list[ChannelCfg] = field(default_factory=list)


def load_config(path: Path = CONFIG) -> Config:
    raw = tomllib.loads(path.read_text())
    m = raw.get("mirror") or {}
    data_dir = Path(m.get("data_dir") or "affine/state/discord")
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    return Config(
        data_dir=data_dir,
        poll_interval_s=float(m.get("poll_interval_s", 60)),
        thread_scan_interval_s=float(m.get("thread_scan_interval_s", 600)),
        recent_window=int(m.get("recent_window", 50)),
        token_env=str(m.get("token_env") or "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR"),
        discover_patterns=[str(p).lower() for p in (m.get("discover_patterns") or [])],
        channels=[
            ChannelCfg(id=str(c["id"]), guild_id=str(c.get("guild_id") or ""),
                       name=str(c.get("name") or c["id"]), note=str(c.get("note") or ""))
            for c in raw.get("channels") or []
        ],
    )


# -- REST client -----------------------------------------------------------------
class DiscordError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


class Discord:
    """Minimal REST client with rate-limit bookkeeping and retries."""

    def __init__(self, token: str):
        if not token:
            raise SystemExit("mirror: no Discord bot token (env / ~/.affine-validator.env / repo .env)")
        self.token = token
        self.requests = 0
        self.rate_sleeps = 0.0

    def get(self, route: str, **params) -> dict | list:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = API + route + (f"?{query}" if query else "")
        backoff = 1.0
        while True:
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bot {self.token}", "User-Agent": USER_AGENT})
            self.requests += 1
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.load(resp)
                    self._honour_bucket(resp.headers)
                    return data
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code == 429:
                    delay = self._retry_after(exc.headers, body)
                    scope = "global" if exc.headers.get("X-RateLimit-Global") else "route"
                    log(f"http: 429 ({scope}) on {route}; sleeping {delay:.1f}s")
                    self._sleep(delay)
                    continue
                if exc.code >= 500:
                    log(f"http: {exc.code} on {route}; retry in {backoff:.0f}s")
                    self._sleep(backoff)
                    backoff = min(BACKOFF_MAX_S, backoff * 2)
                    continue
                raise DiscordError(exc.code, body) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                log(f"http: {type(exc).__name__} on {route}: {exc}; retry in {backoff:.0f}s")
                self._sleep(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)

    def _honour_bucket(self, headers) -> None:
        try:
            remaining = int(headers.get("X-RateLimit-Remaining", "1"))
            reset_after = float(headers.get("X-RateLimit-Reset-After", "0"))
        except ValueError:
            return
        if remaining <= 0 and reset_after > 0:
            self._sleep(reset_after + 0.05)

    @staticmethod
    def _retry_after(headers, body: str) -> float:
        try:
            return max(0.5, min(3600.0, float(json.loads(body).get("retry_after"))))
        except (ValueError, TypeError, AttributeError):
            pass
        try:
            return max(0.5, min(3600.0, float(headers.get("Retry-After", "5"))))
        except ValueError:
            return 5.0

    def _sleep(self, seconds: float) -> None:
        self.rate_sleeps += seconds
        time.sleep(seconds)


# -- storage ---------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,           -- channel or thread id
    guild_id TEXT,
    name TEXT,
    kind TEXT,                     -- 'channel' | 'thread'
    parent_id TEXT,                -- thread -> parent channel id
    archived INTEGER DEFAULT 0,
    last_id INTEGER,               -- newest message id seen (poll cursor)
    oldest_id INTEGER,             -- backfill cursor (before=)
    backfilled INTEGER DEFAULT 0,
    error TEXT,                    -- last permanent error (403/404), else NULL
    first_seen TEXT,
    last_polled TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,        -- snowflake; rowid alias so FTS rowids are stable
    channel_id TEXT NOT NULL,      -- parent channel (thread messages keep the parent here)
    guild_id TEXT,
    author_id TEXT,
    author_name TEXT,
    author_display TEXT,
    author_bot INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL,
    edited_timestamp TEXT,
    content TEXT,
    reply_to TEXT,
    attachments TEXT,              -- json list
    embeds TEXT,                   -- json list
    reactions TEXT,                -- json list
    thread_id TEXT,                -- NULL for top-level channel messages
    message_type INTEGER,
    deleted_at TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS messages_channel_ts ON messages(channel_id, timestamp);
CREATE INDEX IF NOT EXISTS messages_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS messages_author ON messages(author_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, author_name, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content, author_name) VALUES (new.id, new.content, new.author_name);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content, author_name)
  VALUES ('delete', old.id, old.content, old.author_name);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content, author_name)
  VALUES ('delete', old.id, old.content, old.author_name);
  INSERT INTO messages_fts(rowid, content, author_name) VALUES (new.id, new.content, new.author_name);
END;
"""

UPSERT = """
INSERT INTO messages (id, channel_id, guild_id, author_id, author_name, author_display, author_bot,
    timestamp, edited_timestamp, content, reply_to, attachments, embeds, reactions, thread_id,
    message_type, deleted_at, fetched_at)
VALUES (:id, :channel_id, :guild_id, :author_id, :author_name, :author_display, :author_bot,
    :timestamp, :edited_timestamp, :content, :reply_to, :attachments, :embeds, :reactions, :thread_id,
    :message_type, NULL, :fetched_at)
ON CONFLICT(id) DO UPDATE SET
    edited_timestamp = excluded.edited_timestamp,
    content = excluded.content,
    attachments = excluded.attachments,
    embeds = excluded.embeds,
    reactions = excluded.reactions,
    author_display = excluded.author_display,
    deleted_at = NULL,
    fetched_at = excluded.fetched_at
WHERE edited_timestamp IS NOT excluded.edited_timestamp
   OR content IS NOT excluded.content
   OR reactions IS NOT excluded.reactions
   OR attachments IS NOT excluded.attachments
   OR embeds IS NOT excluded.embeds
   OR deleted_at IS NOT NULL
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    try:
        db.executescript(FTS_SCHEMA)
    except sqlite3.OperationalError as exc:
        log(f"db: FTS5 unavailable ({exc}); read.py search falls back to LIKE")
    return db


def normalize(m: dict, channel_id: str, guild_id: str, thread_id: str | None) -> dict:
    author = m.get("author") or {}
    member = m.get("member") or {}
    ref = m.get("message_reference") or {}
    attachments = [{
        "id": a.get("id"), "filename": a.get("filename"), "url": a.get("url"),
        "size": a.get("size"), "content_type": a.get("content_type"),
    } for a in m.get("attachments") or []]
    reactions = [{
        "emoji": (r.get("emoji") or {}).get("name") or (r.get("emoji") or {}).get("id"),
        "count": r.get("count"),
    } for r in m.get("reactions") or []]
    return {
        "id": int(m["id"]),
        "channel_id": channel_id,
        "guild_id": guild_id,
        "author_id": author.get("id"),
        "author_name": author.get("username"),
        "author_display": member.get("nick") or author.get("global_name") or author.get("username"),
        "author_bot": int(bool(author.get("bot"))),
        "timestamp": m.get("timestamp"),
        "edited_timestamp": m.get("edited_timestamp"),
        "content": m.get("content") or "",
        "reply_to": ref.get("message_id"),
        "attachments": json.dumps(attachments, ensure_ascii=False),
        "embeds": json.dumps(m.get("embeds") or [], ensure_ascii=False),
        "reactions": json.dumps(reactions, ensure_ascii=False),
        "thread_id": thread_id,
        "message_type": m.get("type"),
        "fetched_at": now_iso(),
    }


def looks_blank(m: dict) -> bool:
    """A non-bot message with no content and no other payload — the Message
    Content intent symptom (Discord blanks content, not attachments)."""
    return not (m.get("content") or m.get("attachments") or m.get("embeds")
                or m.get("sticker_items") or m.get("components") or m.get("poll"))


# -- the mirror ------------------------------------------------------------------
@dataclass
class Source:
    """One thing we poll: a channel or a thread."""
    id: str
    guild_id: str
    name: str
    kind: str                     # 'channel' | 'thread'
    parent_id: str | None = None  # threads: the mirrored channel they hang off

    @property
    def channel_id(self) -> str:
        return self.parent_id if self.kind == "thread" else self.id

    @property
    def thread_id(self) -> str | None:
        return self.id if self.kind == "thread" else None


class Mirror:
    def __init__(self, cfg: Config, api: Discord):
        self.cfg = cfg
        self.api = api
        self.data_dir = cfg.data_dir
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.db = open_db(self.data_dir / "discord.sqlite")
        self.sources: dict[str, Source] = {}
        self.last_thread_scan = 0.0
        self.blockers: dict[str, str] = {}
        self.intent_seen: dict[str, list[int]] = {}   # channel -> [non_bot, blank]
        self.cycle_stats: dict[str, int] = {}
        self.me: dict = {}

    # -- discovery -------------------------------------------------------------
    def check_application(self) -> None:
        self.me = self.api.get("/users/@me")
        log(f"bot: {self.me.get('username')} ({self.me.get('id')})")
        try:
            app = self.api.get("/oauth2/applications/@me")
        except DiscordError as exc:
            log(f"app: could not read application flags ({exc})")
            return
        flags = int(app.get("flags") or 0)
        if flags & FLAG_MESSAGE_CONTENT:
            log("app: Message Content Intent = enabled (privileged)")
            self.blockers.pop("message_content_intent", None)
        elif flags & FLAG_MESSAGE_CONTENT_LIMITED:
            log("app: Message Content Intent = limited (bot in < 100 guilds; content readable)")
            self.blockers.pop("message_content_intent", None)
        else:
            msg = ("Message Content Intent is OFF for this application: Discord will blank "
                   "`content` on messages the bot did not write. Enable it in the Discord "
                   "Developer Portal -> Bot -> Privileged Gateway Intents.")
            log("BLOCKER app: " + msg)
            self.blockers["message_content_intent"] = msg

    def discover(self) -> list[dict]:
        """Every text channel the bot can see, flagged when it matches the patterns."""
        rows = []
        for g in self.api.get("/users/@me/guilds"):
            try:
                channels = self.api.get(f"/guilds/{g['id']}/channels")
            except DiscordError as exc:
                log(f"discover: guild {g['name']} ({g['id']}): {exc}")
                continue
            names = {c["id"]: c["name"] for c in channels}
            for c in channels:
                if c.get("type") not in TEXT_CHANNEL_TYPES:
                    continue
                hay = f"{g['name']} {c['name']}".lower()
                rows.append({
                    "guild_id": g["id"], "guild": g["name"], "id": c["id"], "name": c["name"],
                    "type": c["type"], "category": names.get(c.get("parent_id"), ""),
                    "match": any(p in hay for p in self.cfg.discover_patterns),
                })
        return rows

    def ensure_sources(self) -> None:
        configured = {c.id: c for c in self.cfg.channels}
        for c in configured.values():
            self._add_source(Source(c.id, c.guild_id, c.name, "channel"))
        if self.cfg.discover_patterns:
            for row in self.discover():
                if row["match"] and row["id"] not in self.sources:
                    log(f"discover: adding {row['guild']} / #{row['name']} ({row['id']})")
                    self._add_source(Source(row["id"], row["guild_id"], row["name"], "channel"))
        for row in self.db.execute("SELECT * FROM channels"):
            if row["id"] not in self.sources and row["kind"] == "thread" and row["parent_id"] in self.sources:
                self._add_source(Source(row["id"], row["guild_id"], row["name"], "thread", row["parent_id"]))

    def _add_source(self, s: Source) -> None:
        self.sources[s.id] = s
        self.db.execute(
            "INSERT INTO channels (id, guild_id, name, kind, parent_id, first_seen) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, guild_id=excluded.guild_id, parent_id=excluded.parent_id",
            (s.id, s.guild_id, s.name, s.kind, s.parent_id, now_iso()))
        self.db.commit()

    def scan_threads(self) -> None:
        """Active threads per guild + archived public/private threads per channel."""
        parents = {s.id: s for s in self.sources.values() if s.kind == "channel"}
        found = 0
        for gid in {s.guild_id for s in parents.values() if s.guild_id}:
            try:
                for t in self.api.get(f"/guilds/{gid}/threads/active").get("threads", []):
                    found += self._add_thread(t, parents)
            except DiscordError as exc:
                log(f"threads: guild {gid} active list: {exc}")
        for p in parents.values():
            for kind in ("public", "private"):
                before = None
                while True:
                    try:
                        page = self.api.get(f"/channels/{p.id}/threads/archived/{kind}", before=before, limit=100)
                    except DiscordError as exc:
                        if exc.status not in (403, 404):
                            log(f"threads: #{p.name} archived/{kind}: {exc}")
                        break
                    threads = page.get("threads", [])
                    for t in threads:
                        found += self._add_thread(t, parents, archived=True)
                    if not page.get("has_more") or not threads:
                        break
                    before = (threads[-1].get("thread_metadata") or {}).get("archive_timestamp")
                    if not before:
                        break
        self.last_thread_scan = time.monotonic()
        if found:
            log(f"threads: {found} new thread(s) added")

    def _add_thread(self, t: dict, parents: dict[str, Source], archived: bool = False) -> int:
        parent = parents.get(str(t.get("parent_id")))
        if parent is None or t.get("type") not in THREAD_TYPES:
            return 0
        new = t["id"] not in self.sources
        self._add_source(Source(t["id"], parent.guild_id, t.get("name") or t["id"], "thread", parent.id))
        self.db.execute("UPDATE channels SET archived=? WHERE id=?",
                        (int(archived or bool((t.get("thread_metadata") or {}).get("archived"))), t["id"]))
        self.db.commit()
        return int(new)

    # -- fetching ----------------------------------------------------------------
    def _fetch(self, s: Source, limit: int = PAGE, **params) -> list[dict] | None:
        try:
            msgs = self.api.get(f"/channels/{s.id}/messages", limit=limit, **params)
        except DiscordError as exc:
            if exc.status in (403, 404):
                msg = f"#{s.name} ({s.id}): HTTP {exc.status} — bot cannot read this channel"
                if self.blockers.get(f"channel:{s.id}") != msg:
                    log("BLOCKER " + msg)
                self.blockers[f"channel:{s.id}"] = msg
                self.db.execute("UPDATE channels SET error=? WHERE id=?", (msg, s.id))
                self.db.commit()
                return None
            log(f"fetch #{s.name}: {exc}")
            return None
        self.blockers.pop(f"channel:{s.id}", None)
        return msgs

    def _store(self, s: Source, msgs: list[dict]) -> int:
        if not msgs:
            return 0
        rows = [normalize(m, s.channel_id, s.guild_id, s.thread_id) for m in msgs]
        with self.db:
            cur = self.db.executemany(UPSERT, rows)
            changed = cur.rowcount if cur.rowcount is not None else 0
        with (self.raw_dir / f"{s.channel_id}.jsonl").open("a", encoding="utf-8") as f:
            for m in msgs:
                f.write(json.dumps({"fetched_at": rows[0]["fetched_at"], "channel_id": s.channel_id,
                                    "thread_id": s.thread_id, "message": m}, ensure_ascii=False) + "\n")
        seen = self.intent_seen.setdefault(s.channel_id, [0, 0])
        for m in msgs:
            if not (m.get("author") or {}).get("bot"):
                seen[0] += 1
                seen[1] += int(looks_blank(m))
        return max(changed, 0)

    def backfill(self, s: Source) -> None:
        row = self.db.execute("SELECT last_id, oldest_id FROM channels WHERE id=?", (s.id,)).fetchone()
        before = row["oldest_id"]
        total = 0
        while True:
            msgs = self._fetch(s, before=before)
            if msgs is None:
                return
            if before is None and msgs and row["last_id"] is None:
                self.db.execute("UPDATE channels SET last_id=? WHERE id=?", (int(msgs[0]["id"]), s.id))
            total += self._store(s, msgs)
            if msgs:
                before = int(msgs[-1]["id"])
                self.db.execute("UPDATE channels SET oldest_id=?, last_polled=? WHERE id=?", (before, now_iso(), s.id))
                self.db.commit()
            if len(msgs) < PAGE:
                break
        self.db.execute("UPDATE channels SET backfilled=1, last_polled=? WHERE id=?", (now_iso(), s.id))
        self.db.commit()
        n = self.db.execute("SELECT count(*) FROM messages WHERE channel_id=? AND thread_id IS ?",
                            (s.channel_id, s.thread_id)).fetchone()[0]
        log(f"backfill #{s.name} ({s.kind}): complete, {n} messages stored ({total} new/changed this run)")
        self.cycle_stats["backfilled"] = self.cycle_stats.get("backfilled", 0) + 1

    def poll(self, s: Source) -> None:
        row = self.db.execute("SELECT last_id FROM channels WHERE id=?", (s.id,)).fetchone()
        after = row["last_id"] or 0
        new = 0
        while True:
            msgs = self._fetch(s, after=after)
            if msgs is None:
                return
            if msgs:
                msgs.sort(key=lambda m: int(m["id"]))
                new += self._store(s, msgs)
                after = int(msgs[-1]["id"])
                self.db.execute("UPDATE channels SET last_id=? WHERE id=?", (after, s.id))
                self.db.commit()
            if len(msgs) < PAGE:
                break
        recent = self._fetch(s, limit=min(PAGE, self.cfg.recent_window)) if self.cfg.recent_window else None
        if recent is not None:
            changed = self._store(s, recent)
            self._mark_deleted(s, recent)
            new += changed
        self.db.execute("UPDATE channels SET last_polled=? WHERE id=?", (now_iso(), s.id))
        self.db.commit()
        if new:
            log(f"poll #{s.name}: {new} new/changed")
        self.cycle_stats["new"] = self.cycle_stats.get("new", 0) + new

    def _mark_deleted(self, s: Source, recent: list[dict]) -> None:
        """Stored messages newer than the oldest of the recent window that Discord
        no longer returns were deleted."""
        if not recent:
            return
        ids = {int(m["id"]) for m in recent}
        floor = min(ids)
        rows = self.db.execute(
            "SELECT id FROM messages WHERE channel_id=? AND thread_id IS ? AND id>=? AND deleted_at IS NULL",
            (s.channel_id, s.thread_id, floor)).fetchall()
        gone = [r["id"] for r in rows if r["id"] not in ids]
        if gone:
            with self.db:
                self.db.executemany("UPDATE messages SET deleted_at=? WHERE id=?", [(now_iso(), i) for i in gone])
            log(f"poll #{s.name}: {len(gone)} message(s) deleted upstream")

    # -- health --------------------------------------------------------------------
    def check_intent_symptom(self) -> None:
        for cid, (non_bot, blank) in self.intent_seen.items():
            key = f"content_blank:{cid}"
            if non_bot >= INTENT_MIN_SAMPLE and blank / non_bot >= INTENT_EMPTY_RATIO:
                msg = (f"channel {cid}: {blank}/{non_bot} non-bot messages have empty content — "
                       "the bot lacks the Message Content Intent (Developer Portal -> Bot -> "
                       "Privileged Gateway Intents) or read permission on message content.")
                if self.blockers.get(key) != msg:
                    log("BLOCKER " + msg)
                self.blockers[key] = msg
            else:
                self.blockers.pop(key, None)

    def write_status(self) -> None:
        rows = []
        for r in self.db.execute("SELECT * FROM channels ORDER BY kind, name"):
            n, latest = self.db.execute(
                "SELECT count(*), max(timestamp) FROM messages WHERE channel_id=? AND thread_id IS ?",
                (r["parent_id"] if r["kind"] == "thread" else r["id"], r["id"] if r["kind"] == "thread" else None),
            ).fetchone()
            rows.append({"id": r["id"], "name": r["name"], "kind": r["kind"], "parent_id": r["parent_id"],
                         "guild_id": r["guild_id"], "messages": n, "latest": latest,
                         "backfilled": bool(r["backfilled"]), "archived": bool(r["archived"]),
                         "last_polled": r["last_polled"], "error": r["error"]})
        status = {
            "updated_at": now_iso(), "bot": self.me.get("username"), "bot_id": self.me.get("id"),
            "total_messages": self.db.execute("SELECT count(*) FROM messages").fetchone()[0],
            "requests": self.api.requests, "rate_limit_sleep_s": round(self.api.rate_sleeps, 1),
            "blockers": self.blockers, "channels": rows,
        }
        tmp = self.data_dir / "status.json.tmp"
        tmp.write_text(json.dumps(status, indent=1, ensure_ascii=False))
        tmp.replace(self.data_dir / "status.json")

    # -- loop ------------------------------------------------------------------------
    def cycle(self) -> None:
        self.cycle_stats = {}
        if time.monotonic() - self.last_thread_scan >= self.cfg.thread_scan_interval_s:
            self.ensure_sources()
            self.scan_threads()
        for s in list(self.sources.values()):
            row = self.db.execute("SELECT backfilled, error FROM channels WHERE id=?", (s.id,)).fetchone()
            if not row["backfilled"]:
                self.backfill(s)
            else:
                self.poll(s)
        self.check_intent_symptom()
        self.write_status()

    def run(self, once: bool = False) -> None:
        self.check_application()
        self.ensure_sources()
        log(f"mirror: {len(self.sources)} source(s), data_dir={self.data_dir}, "
            f"poll every {self.cfg.poll_interval_s:.0f}s")
        while True:
            t0 = time.monotonic()
            try:
                self.cycle()
            except Exception as exc:  # keep the service alive; pm2 restarts on a hard crash
                log(f"cycle: {type(exc).__name__}: {exc}")
            if once:
                return
            total = self.db.execute("SELECT count(*) FROM messages").fetchone()[0]
            took = time.monotonic() - t0
            if self.cycle_stats.get("new") or self.cycle_stats.get("backfilled") or took > 5:
                log(f"cycle: {took:.1f}s, {self.cycle_stats.get('new', 0)} new/changed, total {total}, "
                    f"{len(self.blockers)} blocker(s)")
            time.sleep(max(1.0, self.cfg.poll_interval_s - took))


# -- CLI ---------------------------------------------------------------------------
def cmd_discover(cfg: Config, api: Discord) -> None:
    m = Mirror(cfg, api)
    m.check_application()
    rows = m.discover()
    print(f"{'match':5} {'guild':<24} {'channel id':<20} name  [category]")
    for r in sorted(rows, key=lambda r: (not r["match"], r["guild"], r["name"])):
        flag = "*" if r["match"] else " "
        print(f"  {flag}   {r['guild'][:24]:<24} {r['id']:<20} #{r['name']}  [{r['category']}]")
    print(f"\n{sum(r['match'] for r in rows)} matching / {len(rows)} text channels in {len({r['guild_id'] for r in rows})} guilds")


def cmd_status(cfg: Config) -> None:
    path = cfg.data_dir / "status.json"
    if not path.exists():
        print(f"no status yet ({path})")
        return
    st = json.loads(path.read_text())
    print(f"updated {st['updated_at']}  bot={st.get('bot')}  total={st['total_messages']}  "
          f"requests={st['requests']}  rl_sleep={st['rate_limit_sleep_s']}s")
    for c in st["channels"]:
        tag = "thread" if c["kind"] == "thread" else "chan  "
        print(f"  {tag} {c['id']:<20} {c['messages']:>7}  backfilled={int(c['backfilled'])}  "
              f"latest={(c['latest'] or '-')[:19]}  {c['name']}" + (f"  ERROR {c['error']}" if c["error"] else ""))
    if st["blockers"]:
        print("\nBLOCKERS:")
        for k, v in st["blockers"].items():
            print(f"  - {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "once", "discover", "status"])
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    if args.cmd == "status":
        cmd_status(cfg)
        return
    api = Discord(env_file_value(cfg.token_env))
    if args.cmd == "discover":
        cmd_discover(cfg, api)
        return
    Mirror(cfg, api).run(once=(args.cmd == "once"))


if __name__ == "__main__":
    main()
