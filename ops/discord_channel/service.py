"""SN120 Discord reader and explicit panel-send command. Credentials stay here."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

import websockets

ROOT = Path(__file__).resolve().parents[2]
PANEL = 'panels/discord-sn120.html'
CHANNEL = '1381987595881414656'
GUILD = '799672011265015819'
GATEWAY = os.environ.get('SN120_DISCORD_BOARD_WS', 'ws://127.0.0.1:24585/api/board/ws')
API = 'https://discord.com/api/v10'
SNOWFLAKE = re.compile(r'^[0-9]{15,22}$')
NONCE = re.compile(r'^[0-9]{16,25}$')


def now():
    return datetime.now(timezone.utc).isoformat()


class DiscordError(Exception):
    pass


class Discord:
    def __init__(self, token=None):
        self.token = token or os.environ.get('DISCORD_BOT_TOKEN_ARBOS_BITTENSOR') or os.environ.get('DISCORD_BOT_TOKEN')
        if not self.token:
            self.token = subprocess.check_output([str(ROOT / 'ralphs/secret.sh'), 'DISCORD_BOT_TOKEN'], text=True, timeout=30).strip()
        self.blocked_until = 0.0

    def request(self, method, route, body=None):
        if time.monotonic() < self.blocked_until:
            raise DiscordError('Discord rate limit: waiting before the next request.')
        headers = {'Authorization': 'Bot ' + self.token, 'User-Agent': 'DiscordBot (https://affine.io, 1.0)'}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(API + route, headers=headers, data=data, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                try:
                    delay = max(1, min(3600, float(json.loads(exc.read()).get('retry_after', 5))))
                except (ValueError, TypeError):
                    delay = 5
                self.blocked_until = time.monotonic() + delay
                raise DiscordError(f'Discord rate limit; retry after {delay:.0f}s.') from None
            labels = {401: 'Bot credential rejected', 403: 'Bot lacks permission', 404: 'Channel or message not found'}
            raise DiscordError(f'{labels.get(exc.code, "Discord request failed")} (HTTP {exc.code}).') from None
        except (OSError, ValueError):
            raise DiscordError('Discord network/response error. Delivery may be unknown if this was a send.') from None

    def channel(self, channel_id):
        if not isinstance(channel_id, str) or not SNOWFLAKE.fullmatch(channel_id):
            raise DiscordError('Invalid channel ID.')
        item = self.request('GET', f'/channels/{channel_id}')
        if item.get('guild_id') != GUILD or not (
            channel_id == CHANNEL or (item.get('parent_id') == CHANNEL and item.get('type') in (10, 11, 12))
        ):
            raise DiscordError('Only the SN120 Affine channel and its threads are allowed.')
        return item


def message(item):
    author = item.get('author', {})
    member = item.get('member') or {}
    reply = item.get('referenced_message') or {}
    return {
        'id': item['id'], 'channel_id': item['channel_id'],
        'author': member.get('nick') or author.get('global_name') or author.get('username', 'Unknown'),
        'username': author.get('username', ''), 'author_id': author.get('id', ''), 'bot': bool(author.get('bot')),
        'content': item.get('content', ''), 'timestamp': item.get('timestamp'), 'edited_at': item.get('edited_timestamp'),
        'type': item.get('type', 0),
        'reply': {'id': reply.get('id'), 'author': (reply.get('author') or {}).get('global_name') or (reply.get('author') or {}).get('username', ''), 'content': reply.get('content', '')[:300]} if reply else None,
        'attachments': [{'filename': a.get('filename', 'Attachment'), 'url': a.get('url', ''), 'size': a.get('size', 0)} for a in item.get('attachments', [])],
        'embeds': [{'title': e.get('title', ''), 'description': e.get('description', '')[:1200], 'url': e.get('url', '')} for e in item.get('embeds', [])],
        'reactions': [{'emoji': (r.get('emoji') or {}).get('name', '?'), 'count': r.get('count', 0)} for r in item.get('reactions', [])],
        'thread': {'id': item['thread']['id'], 'name': item['thread'].get('name', 'Thread')} if item.get('thread') else None,
    }


class Document:
    async def __aenter__(self):
        self.ws = await websockets.connect(GATEWAY, max_size=4 * 1024 * 1024, open_timeout=10)
        await self.ws.send(json.dumps({'type': 'state_sub', 'panel': PANEL}))
        self.doc = await self.read()
        return self

    async def __aexit__(self, *_):
        await self.ws.close()

    async def read(self):
        while True:
            event = json.loads(await asyncio.wait_for(self.ws.recv(), 10))
            if event.get('type') == 'state' and event.get('panel') == PANEL:
                return event.get('doc') or {}

    async def patch(self, patch):
        await self.ws.send(json.dumps({'type': 'state_set', 'panel': PANEL, 'patch': patch}))
        while True:
            self.doc = await self.read()
            if all(self.doc.get(k) == v for k, v in patch.items() if v is not None) and all(k not in self.doc for k, v in patch.items() if v is None):
                return self.doc


@contextmanager
def lock(name):
    directory = ROOT / 'panels/data'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('.discord-' + name + '.lock')).open('w') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DiscordError('Another Discord ' + name + ' process is already active.') from None
        yield


async def collect(once=False):
    client = Discord()
    identity = None
    threads = []
    threads_at = 0.0
    with lock('reader'):
        while True:
            try:
                async with Document() as state:
                    try:
                        if identity is None:
                            bot = await asyncio.to_thread(client.request, 'GET', '/users/@me')
                            identity = {'id': bot['id'], 'name': bot.get('global_name') or bot['username']}
                        selected = (state.doc.get('view') or {}).get('channel_id', CHANNEL)
                        channel = await asyncio.to_thread(client.channel, selected)
                        if time.monotonic() - threads_at > 60:
                            try:
                                active = await asyncio.to_thread(client.request, 'GET', f'/guilds/{GUILD}/threads/active')
                                threads = [{'id': t['id'], 'name': t['name']} for t in active.get('threads', []) if t.get('parent_id') == CHANNEL]
                                threads_at = time.monotonic()
                            except DiscordError:
                                pass
                        items = await asyncio.to_thread(client.request, 'GET', f'/channels/{selected}/messages?limit=75')
                        feed = {'channel_id': selected, 'name': channel['name'], 'guild_id': GUILD, 'bot': identity,
                                'messages': {m['id']: message(m) for m in reversed(items)},
                                'threads': {t['id']: t for t in threads}, 'updated_at': now(),
                                'status': 'live', 'error': '', 'poll_seconds': 5,
                                'archived': bool((channel.get('thread_metadata') or {}).get('archived'))}
                        await state.patch({'feed': feed})
                        print(f'Discord live: channel={selected} messages={len(items)}', flush=True)
                    except DiscordError as exc:
                        feed = dict(state.doc.get('feed') or {})
                        feed.update(status='error', error=str(exc), attempted_at=now())
                        await state.patch({'feed': feed})
                        print(str(exc), flush=True)
            except Exception as exc:
                print('Board connection unavailable: ' + type(exc).__name__, flush=True)
                if once:
                    raise SystemExit(1) from None
            if once:
                return
            await asyncio.sleep(5)


def send_body(request):
    content = request.get('content')
    request_id = request.get('id', '')
    if not isinstance(request_id, str) or not NONCE.fullmatch(request_id):
        raise DiscordError('Invalid send request ID.')
    if not isinstance(content, str) or not content.strip() or len(content.encode('utf-16-le')) // 2 > 2000:
        raise DiscordError('Message must contain 1–2000 characters.')
    body = {'content': content, 'allowed_mentions': {'parse': [], 'replied_user': False}, 'nonce': request_id, 'enforce_nonce': True}
    reply_id = request.get('reply_id')
    if reply_id:
        if not isinstance(reply_id, str) or not SNOWFLAKE.fullmatch(reply_id):
            raise DiscordError('Invalid reply target.')
        body['message_reference'] = {'message_id': reply_id, 'channel_id': request['channel_id'], 'fail_if_not_exists': True}
    return body


async def send(request_id):
    with lock('sender'):
        async with Document() as state:
            request = state.doc.get('send_request') or {}
            delivery = state.doc.get('delivery') or {}
            if request.get('id') != request_id:
                raise DiscordError('Request is no longer the explicit pending send in this panel.')
            if delivery.get('request_id') == request_id and delivery.get('status') == 'sent':
                print('Already sent: ' + delivery['message_id'])
                return
            if delivery.get('request_id') == request_id and delivery.get('status') in ('sending', 'unknown'):
                raise DiscordError('Delivery uncertain. Inspect Discord before creating a new send; automatic retry refused.')
            client = Discord()
            try:
                body = send_body(request)
                channel_id = request.get('channel_id')
                await asyncio.to_thread(client.channel, channel_id)
                await state.patch({'delivery': {'request_id': request_id, 'status': 'sending', 'updated_at': now()}})
                try:
                    posted = await asyncio.to_thread(client.request, 'POST', f'/channels/{channel_id}/messages', body)
                except Exception:
                    await state.patch({'delivery': {'request_id': request_id, 'status': 'unknown', 'updated_at': now(), 'error': 'Send not confirmed. Check Discord before retrying to avoid duplicates.'}})
                    raise
                await state.patch({'delivery': {'request_id': request_id, 'status': 'sent', 'message_id': posted['id'], 'channel_id': channel_id, 'updated_at': now()}})
                print(json.dumps({'status': 'sent', 'message_id': posted['id'], 'channel_id': channel_id}))
            except DiscordError as exc:
                if (state.doc.get('delivery') or {}).get('status') not in ('sending', 'unknown'):
                    await state.patch({'delivery': {'request_id': request_id, 'status': 'failed', 'updated_at': now(), 'error': str(exc)}})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    reader = sub.add_parser('read')
    reader.add_argument('--once', action='store_true')
    sender = sub.add_parser('send')
    sender.add_argument('request_id')
    args = parser.parse_args()
    try:
        asyncio.run(collect(args.once) if args.command == 'read' else send(args.request_id))
    except DiscordError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == '__main__':
    main()
