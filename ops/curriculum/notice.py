#!/usr/bin/env python
"""Public notice for the curriculum apply fold (Jacob pre-approved the
community post 2026-09-14 23:39 UTC "+ community"; plan §8.3 default = yes;
coordinator 2026-09-16 17:14 UTC).

Idempotent, fail-closed. Posts `apply_notice.md` of the applied snapshot to
the public SN120 channel only when ALL of:
  * affine/state/curriculum/latest.json says mode = apply;
  * the live corpus manifest carries curriculum.mode = apply with the SAME
    weights_sha256 (i.e. the 16:00 UTC fold has published under the vector);
  * no marker affine/state/curriculum/notice_posted.json exists.
Writes the marker with the message link, posts one private line.

    pm2 start ops/curriculum/notice.sh --name affine-curriculum-notice --interpreter bash \
        --cron-restart "30 16,17,18,19 * * *" --no-autorestart
    python ops/curriculum/notice.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA_BASE, LATEST_PATH, STATE_DIR, env_value, fetch_bytes, log, write_json  # noqa: E402
from publish import discord_line  # noqa: E402

PUBLIC_CHANNEL = "1381987595881414656"
GUILD = "799672011265015819"
MARKER = STATE_DIR / "notice_posted.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-base", default=DATA_BASE)
    args = ap.parse_args()
    if MARKER.is_file():
        log(f"notice already posted: {json.loads(MARKER.read_text()).get('link')}")
        return
    if not LATEST_PATH.is_file():
        log("no latest.json; nothing to do")
        return
    latest = json.loads(LATEST_PATH.read_text())
    if latest.get("mode") != "apply":
        log(f"mode is {latest.get('mode')}; no notice")
        return
    manifest = json.loads(fetch_bytes(f"{args.data_base}/corpus/manifest.json"))
    block = manifest.get("curriculum") or {}
    if block.get("mode") != "apply" or block.get("weights_sha256") != latest["weights_sha256"]:
        log(f"manifest epoch {manifest.get('corpus_epoch')} curriculum block {block} does not yet carry the applied "
            f"vector {latest['weights_sha256'][:12]}; waiting for the fold")
        return
    snapshot = Path(latest["local_snapshot"])
    text = (snapshot / "apply_notice.md").read_text(encoding="utf-8")
    text = text.replace(f"/curriculum/{latest['for_epoch']}/", f"/curriculum/{manifest['corpus_epoch']}/")
    text += (f"\nEffective corpus epoch {manifest['corpus_epoch']} (manifest sha256 `{block.get('manifest_sha256', '')[:12]}…`, "
             f"weights `{latest['weights_sha256'][:12]}…`, ledger `{latest['ledger_sha256'][:12]}…`).")
    if args.dry_run:
        print(text)
        return
    token = env_value("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if not token:
        raise SystemExit("no Discord bot token")
    r = httpx.post(f"https://discord.com/api/v10/channels/{PUBLIC_CHANNEL}/messages",
                   headers={"Authorization": f"Bot {token}"}, json={"content": text[:1990]}, timeout=30)
    if r.status_code >= 300:
        raise SystemExit(f"discord HTTP {r.status_code}: {r.text[:200]}")
    mid = r.json().get("id")
    link = f"https://discord.com/channels/{GUILD}/{PUBLIC_CHANNEL}/{mid}"
    write_json({"posted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "message_id": mid,
                "link": link, "weights_sha256": latest["weights_sha256"], "corpus_epoch": manifest["corpus_epoch"]},
               MARKER)
    discord_line(f"curriculum apply notice posted to the public channel: {link} (epoch {manifest['corpus_epoch']}, "
                 f"weights {latest['weights_sha256'][:12]})")
    print(link)


if __name__ == "__main__":
    main()
