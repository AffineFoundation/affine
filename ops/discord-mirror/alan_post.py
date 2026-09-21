#!/usr/bin/env python
"""Post the daily update to Alan.

Poster = channels.toml [alan].poster:
  "user"  (since 2026-09-13, Jacob's directive) — post as Jacob's own account
          `consttt` with a user token into the existing Jacob<->Alan DM
          ([alan].target = that DM channel id). The token is read from
          --token-stdin (piped from 1Password at post time; never stored on
          the box) or, if absent, from the env var [alan].user_token_env.
          WARNING: automating a user account is a "self-bot" under Discord's
          Terms of Service; the account can be actioned. Keep volume to the
          one daily message and nothing else.
  "bot"   — post as the Arbos bot; target "dm" opens the bot's DM with
          [alan].user_id (POST /users/@me/channels; idempotent), or a channel
          id. The bot keeps this role for reading/mirroring in every mode.

Nothing else is ever a target — posting to Alan is the only approved
destination (Jacob, 2026-09-12).

Text longer than Discord's 2000-char limit is split on paragraph breaks.
Every post is appended to <raw-dir>/posts.jsonl (poster, channel, message
ids, link, text) so the store copy can be reconciled.

  python alan_post.py --file update.md --dry-run                  # resolve target, show chunks, post nothing
  op item get <id> --fields notesPlain --reveal | python alan_post.py --file update.md --token-stdin
Never prints a token.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mirror import Discord, DiscordError, env_file_value, load_config  # noqa: E402

CONFIG = HERE / "channels.toml"
LIMIT = 2000


def strip_stamp(text: str) -> str:
    """Drop the HTML comment(s) the store copies carry at the top (posting stamp)."""
    return re.sub(r"\A(\s*<!--.*?-->\s*)+", "", text, flags=re.S)


def chunks(text: str, limit: int = LIMIT) -> list[str]:
    text = strip_stamp(text).strip()
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


def resolve_target(api: Discord, alan: dict, poster: str) -> tuple[str, str, str]:
    """-> (channel_id, guild-or-@me for the link, human description)."""
    uid, target = str(alan.get("user_id") or ""), str(alan.get("target") or "dm")
    if target == "dm":
        if poster != "bot":
            raise SystemExit("alan_post: target 'dm' is only valid for poster = 'bot'; set [alan].target to the DM channel id")
        ch = api.post("/users/@me/channels", {"recipient_id": uid})
        who = ",".join(r.get("username") or r.get("id") for r in ch.get("recipients") or [])
        return str(ch["id"]), "@me", f"bot DM with {who}"
    ch = api.get(f"/channels/{target}")
    recipients = [r.get("username") or r.get("id") for r in ch.get("recipients") or []]
    if ch.get("type") in (1, 3):
        if uid and uid not in {r.get("id") for r in ch.get("recipients") or []}:
            raise SystemExit(f"alan_post: channel {target} is a DM but Alan ({uid}) is not a recipient — refusing")
        return target, "@me", f"{'group ' if ch.get('type') == 3 else ''}DM with {','.join(recipients)}"
    return target, str(ch.get("guild_id") or "@me"), f"channel #{ch.get('name')}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--file", required=True, help="text to post (markdown; Discord renders a subset)")
    ap.add_argument("--raw-dir", help="where posts.jsonl is appended (default <data_dir>/alan)")
    ap.add_argument("--token-stdin", action="store_true", help="read the poster token from stdin (one line)")
    ap.add_argument("--dry-run", action="store_true", help="resolve the target and show the chunks; post nothing")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    alan = (tomllib.loads(Path(args.config).read_text()).get("alan") or {})
    if not alan.get("user_id"):
        raise SystemExit("alan_post: channels.toml has no [alan].user_id")
    poster = str(alan.get("poster") or "bot")
    if poster not in ("bot", "user"):
        raise SystemExit(f"alan_post: [alan].poster must be 'bot' or 'user', not {poster!r}")
    text = Path(args.file).read_text(encoding="utf-8")
    parts = chunks(text)

    if args.token_stdin:
        token = sys.stdin.readline().strip()
    elif poster == "user":
        token = env_file_value(str(alan.get("user_token_env") or "DISCORD_USER_TOKEN_CONST"))
    else:
        token = env_file_value(cfg.token_env)
    api = Discord(token, bot=(poster == "bot"))
    me = api.get("/users/@me")
    if poster == "user" and me.get("bot"):
        raise SystemExit("alan_post: poster = 'user' but the token belongs to a bot")
    channel_id, guild, desc = resolve_target(api, alan, poster)
    print(f"poster: {poster} = {me.get('username')} ({me.get('id')})  target: {desc} (channel {channel_id})")
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
            hint = " (user blocks DMs / no access to this channel?)" if exc.status == 403 else ""
            raise SystemExit(f"alan_post: post failed: {exc}{hint}; {len(posted)} part(s) already sent")
        posted.append({"id": m["id"], "link": f"https://discord.com/channels/{guild}/{channel_id}/{m['id']}"})
        print(f"posted {m['id']} -> {posted[-1]['link']}")

    raw = Path(args.raw_dir) if args.raw_dir else cfg.data_dir / "alan"
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "posts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "posted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "poster": poster,
            "as": me.get("username"), "as_id": me.get("id"), "target": desc, "channel_id": channel_id,
            "recipient": str(alan.get("user_id")), "messages": posted,
            "source_file": str(Path(args.file).resolve()), "text": text,
        }, ensure_ascii=False) + "\n")
    print(f"recorded in {raw / 'posts.jsonl'}")


if __name__ == "__main__":
    main()
