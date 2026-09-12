#!/usr/bin/env python
"""Post the daily update to Alan with the Arbos bot.

Target = channels.toml [alan].target: "dm" opens the bot's DM channel with
[alan].user_id (POST /users/@me/channels; idempotent) and posts there; a
channel id posts to that channel. Nothing else is ever a target — posting to
Alan is the only approved destination (Jacob, 2026-09-12).

Text longer than Discord's 2000-char limit is split on paragraph breaks.
Every post is appended to <raw-dir>/posts.jsonl (channel, message ids, link,
text) so the store copy can be reconciled.

  python alan_post.py --file update.md [--dry-run] [--raw-dir DIR]
Never prints the token.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mirror import Discord, DiscordError, env_file_value, load_config  # noqa: E402

CONFIG = HERE / "channels.toml"
LIMIT = 2000


def chunks(text: str, limit: int = LIMIT) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for para in text.split("\n\n"):
        cand = (cur + "\n\n" + para) if cur else para
        if len(cand) <= limit:
            cur = cand
            continue
        if cur:
            out.append(cur)
        while len(para) > limit:            # a single paragraph over the limit: hard split on a newline
            cut = para.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            out.append(para[:cut])
            para = para[cut:].lstrip("\n")
        cur = para
    if cur:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--file", required=True, help="text to post (markdown; Discord renders a subset)")
    ap.add_argument("--raw-dir", help="where posts.jsonl is appended (default <data_dir>/alan)")
    ap.add_argument("--dry-run", action="store_true", help="resolve the target and show the chunks; post nothing")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    alan = (tomllib.loads(Path(args.config).read_text()).get("alan") or {})
    uid, target = str(alan.get("user_id") or ""), str(alan.get("target") or "dm")
    if not uid:
        raise SystemExit("alan_post: channels.toml has no [alan].user_id")
    text = Path(args.file).read_text(encoding="utf-8")
    parts = chunks(text)
    api = Discord(env_file_value(cfg.token_env))
    me = api.get("/users/@me")

    if target == "dm":
        ch = api.post("/users/@me/channels", {"recipient_id": uid})
        channel_id, guild = str(ch["id"]), "@me"
        who = ",".join(r.get("username") or r.get("id") for r in ch.get("recipients") or [])
        print(f"target: DM with {who} (channel {channel_id}) as bot {me.get('username')}")
    else:
        channel_id, guild = target, None
        ch = api.get(f"/channels/{channel_id}")
        guild = str(ch.get("guild_id") or "@me")
        print(f"target: channel #{ch.get('name')} ({channel_id}) as bot {me.get('username')}")
    print(f"{len(parts)} message(s), {sum(map(len, parts))} chars")
    if args.dry_run:
        for i, p in enumerate(parts, 1):
            print(f"--- part {i} ({len(p)} chars)\n{p}")
        return

    posted = []
    for p in parts:
        try:
            m = api.post(f"/channels/{channel_id}/messages", {"content": p})
        except DiscordError as exc:
            hint = " (user blocks DMs from server members / the bot?)" if exc.status == 403 else ""
            raise SystemExit(f"alan_post: post failed: {exc}{hint}; {len(posted)} part(s) already sent")
        posted.append({"id": m["id"], "link": f"https://discord.com/channels/{guild}/{channel_id}/{m['id']}"})
        print(f"posted {m['id']} -> {posted[-1]['link']}")

    raw = Path(args.raw_dir) if args.raw_dir else cfg.data_dir / "alan"
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "posts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "posted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bot": me.get("username"),
            "target": target, "channel_id": channel_id, "recipient": uid, "messages": posted,
            "source_file": str(Path(args.file).resolve()), "text": text,
        }, ensure_ascii=False) + "\n")
    print(f"recorded in {raw / 'posts.jsonl'}")


if __name__ == "__main__":
    main()
