"""Sealed upload links and the one-time activation.

A miner uploads its batch to one private object, ``layout.miner_object(hotkey)``.
It gets there through a link: a presigned PUT address that only the owner of the
hotkey can read. This module issues those links. It keeps the v1 formats, so a
miner tool written for v1 keeps working:

Activation (one time, on chain). The miner writes this text as a commitment on
the subnet: ``affine2|activate|<hotkey>|<signature>``. The signature is Ed25519,
by the hotkey, over ``activation_message(hotkey, netuid)``. It proves that the
hotkey is an Ed25519 key, so a link can be sealed to it.

Link file (public, one per hotkey, at ``ff/links/<hotkey>.json``)::

    {"version": "affine-ff-link-v1", "hotkey", "issued_at", "expires_at",
     "sealed_to": <public key, hex>, "sealed": <base64 of a libsodium sealed box>}

The sealed box holds JSON with ``put_url``, ``put_headers``, ``get_url``,
``object_key``, ``hotkey``, ``issued_at`` and ``expires_at``. Only the hotkey's
secret key opens it. ``ff/links.json`` lists the hotkeys that have a link now.

Who gets a link: every hotkey with a seat in the queue (its entry is not
evaluated yet), except the burn UID, that has a valid activation. The miner may
upload again as often as it likes; the object is overwritten each time.

There is no inbound service. ``python -m sn120_v2.links issue`` runs on a timer
and writes static files: into a local web directory (``--site-dir``) or into the
store under ``public/`` (``--to-store``). The operator gives the public address
of the link files (``--links-url``); this module names no host and derives no
address. A presigned address is a secret: it is never printed by ``issue`` and
never stored outside the sealed box.

The job needs three things: read-only chain access, store credentials that can
sign a PUT, and the queue. The queue comes from the controller's journal: the
local file in ``--state-dir``, or with ``--journal-from-store`` the copy that
the store holds, so the job can run on a host without the controller.

Commands:
``issue``     the operator's job: write the link files (see ``--help``)
``activate``  the miner's one-time chain write (dry run without ``--execute``)
``open``      the miner opens its link and prints the upload address
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import layout
from .admission import FORMATS
from .chain import SubtensorClient
from .errors import ConfigError, ContractError, InfraError, V2Error
from .journal import JOURNAL_FILE
from .seams import ObjectStore, Presigner
from .state import State, load, load_from_store
from .store import LocalStore

try:
    from nacl.exceptions import CryptoError
    from nacl.public import SealedBox
    from nacl.signing import SigningKey, VerifyKey
except ImportError:  # PyNaCl is the optional "links" extra: the control plane imports this module without it
    SealedBox = None

try:
    from .r2 import R2Store
except ImportError:  # boto3 is the optional "r2" extra
    R2Store = None

try:
    import bittensor
except ImportError:  # the SDK is the optional "chain" extra
    bittensor = None

LINK_VERSION = "affine-ff-link-v1"
LINK_TTL_S = 24 * 3600  # how long one link works
REISSUE_S = 30 * 60  # how often the job writes fresh links
MAX_TTL_S = 7 * 24 * 3600  # the store signs no link for longer than this
LINKS_DIR = "ff/links/"  # one file per hotkey, under the site root
INDEX = "ff/links.json"
PUBLIC_PREFIX = "public/"  # where the files go when the site is served from the object store
ACTIVATION_MARK = "affine2|activate|"
REVEAL_IN = "180s"  # the chain shows the commitment after this delay
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _links_url(text: str) -> str:
    """The public address of the link files, checked. It is always given, never derived from another setting."""
    url = text.strip().rstrip("/")
    if not url.startswith(("https://", "http://")) or url.endswith(".json"):
        raise ConfigError(f"the public address of the link files must be like https://<site>/ff/links, got {text!r}")
    return url


def _need_nacl() -> None:
    if SealedBox is None:
        raise ConfigError("PyNaCl is not installed: install the 'links' extra")


def public_key(hotkey: str) -> bytes:
    """The 32-byte public key inside an SS58 address. ``ContractError`` when the address is not valid."""
    number = 0
    for char in hotkey:
        digit = _B58.find(char)
        if digit < 0:
            raise ContractError(f"hotkey {hotkey!r} is not an SS58 address")
        number = number * 58 + digit
    if number >> (35 * 8):
        raise ContractError(f"hotkey {hotkey!r} is not an SS58 address")
    raw = number.to_bytes(35, "big")  # 1 prefix byte, 32 key bytes, 2 checksum bytes
    if hashlib.blake2b(b"SS58PRE" + raw[:33], digest_size=64).digest()[:2] != raw[33:]:
        raise ContractError(f"hotkey {hotkey!r} has a wrong SS58 checksum")
    return raw[1:33]


# ---- activation ----


def activation_message(hotkey: str, netuid: int) -> bytes:
    """The bytes a hotkey signs one time. The same bytes as in v1, so a v1 activation stays valid."""
    body = json.dumps({"hotkey": hotkey, "netuid": netuid}, sort_keys=True, separators=(",", ":")).encode()
    request_id = hashlib.sha256(b"affine-registration-v1\0" + body).hexdigest()
    return f"affine-activate|v1|{netuid}|{hotkey}|{request_id}".encode()


def activation_text(hotkey: str, signature: bytes) -> str:
    return f"{ACTIVATION_MARK}{hotkey}|{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"


def valid_activation(text: str, hotkey: str, netuid: int) -> bool:
    """True when ``text`` is the activation of this hotkey on this subnet, with a good Ed25519 signature."""
    _need_nacl()
    parts = text.split("|")
    if len(parts) != 4 or "|".join(parts[:3]) + "|" != ACTIVATION_MARK + hotkey + "|":
        return False
    try:
        signature = base64.urlsafe_b64decode(parts[3] + "=" * (-len(parts[3]) % 4))
        VerifyKey(public_key(hotkey)).verify(activation_message(hotkey, netuid), signature)
    except (ValueError, ContractError, CryptoError):  # bad base64, bad address, bad signature
        return False
    return True


def _commitment_text(value: Any) -> str:
    """The text of one revealed commitment. The chain puts a SCALE length prefix (1 to 4 bytes) before it."""
    try:
        if isinstance(value, str):
            raw = bytes.fromhex(value[2:]) if value.startswith("0x") else value.encode("utf-8", "replace")
        elif isinstance(value, (bytes, bytearray, list, tuple)):
            raw = bytes(value)
        else:
            return ""
    except (TypeError, ValueError):
        return ""
    start = raw.find(ACTIVATION_MARK.encode())
    return raw[start:].decode("ascii", "replace") if 0 <= start <= 4 else ""


def activated(rows: Iterable[tuple[Any, Any]], hotkeys: Iterable[str], netuid: int) -> set[str]:
    """The hotkeys among ``hotkeys`` that have a valid activation in ``rows``.

    ``rows`` is the chain map ``Commitments.RevealedCommitments[netuid]``:
    pairs of (hotkey, list of (commitment, block)). Any valid entry counts, old or new.
    """
    wanted, found = set(hotkeys), set()
    for key, entries in rows:
        hotkey = str(key)
        if hotkey not in wanted:
            continue
        for entry in entries or ():
            if valid_activation(_commitment_text(entry[0] if isinstance(entry, (tuple, list)) else entry), hotkey, netuid):
                found.add(hotkey)
                break
    return found


def read_activated(client: SubtensorClient, hotkeys: Iterable[str], netuid: int) -> set[str]:
    """Read the activations from the chain. Read-only: it needs no wallet.

    A failed or empty read raises ``InfraError``. The caller then writes nothing,
    and the links issued before stay valid until they expire.
    """
    try:
        rows = list(client.chain.query_map(client.sdk.storage.Commitments.RevealedCommitments, [netuid]))
    except Exception as exc:  # the SDK raises many types; each one means "no read"
        raise InfraError(f"cannot read the activations from the chain: {type(exc).__name__}: {exc}") from exc
    if not rows:
        raise InfraError("the chain returned no revealed commitment; refusing to treat every hotkey as not activated")
    return activated(rows, hotkeys, netuid)


def _compact(data: bytes) -> bytes:
    """``data`` with its SCALE compact length before it, as the chain stores a byte string."""
    for mode, width in ((0, 1), (1, 2), (2, 4)):
        if len(data) < 1 << (8 * width - 2):
            return ((len(data) << 2) | mode).to_bytes(width, "little") + data
    raise ContractError("the commitment is too long")


def activate(wallet: Any, sdk: Any, netuid: int, network: str, execute: bool) -> dict[str, Any]:
    """Make the activation of the wallet's hotkey. With ``execute``, write it to the chain (the miner's own write).

    It never registers a hotkey and never spends more than the fee of one commitment.
    """
    hotkey = wallet.hotkey.ss58_address
    text = activation_text(hotkey, bytes(wallet.hotkey.sign(activation_message(hotkey, netuid))))
    if not valid_activation(text, hotkey, netuid):
        raise ConfigError(f"hotkey {hotkey} is not an Ed25519 key; a link can be sealed only to an Ed25519 hotkey")
    if not execute:
        return {"status": "dry_run", "netuid": netuid, "payload": text}
    chain = sdk.Subtensor(network)
    uid = chain.query(sdk.storage.SubtensorModule.Uids, [netuid, hotkey])
    if uid is None:
        raise ConfigError(f"hotkey {hotkey} is not registered on subnet {netuid}; register it first (this command never does)")
    locked = sdk.timelock.encrypt(_compact(text.encode()), reveal_in=REVEAL_IN)
    info = {"fields": [{"TimelockEncrypted": {"encrypted": locked.ciphertext, "reveal_round": locked.reveal_round}}]}
    result = chain.submit_call(sdk.calls.Commitments.set_commitment(netuid, info), wallet, signer="hotkey")
    result.raise_for_failure()
    return {"status": "submitted", "netuid": netuid, "uid": int(uid), "block_hash": str(result.block_hash)}


# ---- links ----


def seal(link: dict[str, Any], hotkey: str) -> str:
    """Encrypt ``link`` so that only the secret key of ``hotkey`` can read it (libsodium sealed box, base64)."""
    _need_nacl()
    try:
        box = SealedBox(VerifyKey(public_key(hotkey)).to_curve25519_public_key())
    except (ValueError, CryptoError) as exc:
        raise ContractError(f"hotkey {hotkey} is not an Ed25519 key: {exc}") from exc
    return base64.b64encode(box.encrypt(json.dumps(link, sort_keys=True, separators=(",", ":")).encode())).decode()


def read_seed(path: str | Path) -> bytes:
    """The 32-byte Ed25519 seed of a hotkey: a file of 64 hex characters, or a bittensor hotkey file that is not encrypted."""
    try:
        text = Path(path).read_text().strip()
        if text.startswith("{"):
            text = str(json.loads(text).get("secretSeed", ""))
        seed = bytes.fromhex(text.removeprefix("0x"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read the key file {path}: {type(exc).__name__}") from exc
    if len(seed) != 32:
        raise ConfigError(f"the key file {path} must hold the 32-byte Ed25519 seed as 64 hex characters")
    return seed


def open_link(doc: dict[str, Any], hotkey: str, seed: bytes, now: int) -> dict[str, Any]:
    """Open a link file with the hotkey's seed. Returns the fields inside, with ``put_url`` and ``put_headers``."""
    _need_nacl()
    key = SigningKey(seed)
    if bytes(key.verify_key) != public_key(hotkey):
        raise ConfigError(f"the key file is not the Ed25519 key of hotkey {hotkey}")
    try:
        link = json.loads(SealedBox(key.to_curve25519_private_key()).decrypt(base64.b64decode(doc["sealed"])))
    except (KeyError, TypeError, ValueError, CryptoError) as exc:
        raise ContractError(f"cannot open the link file of {hotkey}: {type(exc).__name__}") from exc
    if link.get("hotkey") != hotkey:
        raise ContractError("the link is for another hotkey")
    if int(link.get("expires_at", 0)) < now + 60:
        raise ContractError("the link is expired; a fresh one is issued on a timer, read it again later")
    return link


def eligible(state: State, burn_uid: int) -> list[str]:
    """Hotkeys that may upload now: a seat in the queue (not evaluated yet), not the burn UID."""
    seats = (seat.reg for seat in state.queue.values())
    return sorted({reg.hotkey for reg in seats if reg.uid != burn_uid and reg.hotkey not in state.served_hotkeys})


def link_files(
    hotkeys: Iterable[str], presigner: Presigner, now: int, links_url: str, ttl_s: int = LINK_TTL_S, reissue_s: int = REISSUE_S
) -> dict[str, bytes]:
    """Build one sealed link file per hotkey, and the index. Returns site path -> file content. Writes nothing.

    ``links_url`` is the public address under which a miner reads the files: ``<links_url>/<hotkey>.json``.
    A hotkey that is not an Ed25519 key gets no file. The index lists the hotkeys that got one.
    """
    link_file = f"{_links_url(links_url)}/<hotkey>.json"
    if not 0 < reissue_s < ttl_s <= MAX_TTL_S:
        raise ConfigError(f"need 0 < reissue ({reissue_s}) < ttl ({ttl_s}) <= {MAX_TTL_S} seconds")
    as_json = lambda value: (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    stamp = {"version": LINK_VERSION, "issued_at": now, "expires_at": now + ttl_s}
    files = {}
    for hotkey in sorted(set(hotkeys)):
        try:
            sealed_to = public_key(hotkey).hex()
            key = layout.miner_object(hotkey)
            put_url, put_headers = presigner.presign_put(key, ttl_s)
            link = stamp | {
                "hotkey": hotkey,
                "object_key": key,
                "put_url": put_url,
                "put_headers": put_headers,
                "get_url": presigner.presign_get(key, ttl_s),
                "format": " | ".join(FORMATS),
                "spec": "llms.txt and mining.json of the subnet (batch file)",
            }
            outer = stamp | {"hotkey": hotkey, "sealed_to": sealed_to, "sealed": seal(link, hotkey)}
        except ContractError:
            continue  # not an Ed25519 hotkey: nobody could open a link sealed to it, so it gets none
        files[f"{LINKS_DIR}{hotkey}.json"] = as_json(outer)
    names = sorted(Path(name).stem for name in files)
    local = time.strftime("%Y-%m-%d %H:%M:%S CR (UTC-6)", time.gmtime(now - 6 * 3600))
    index = {"issued_at_cr": local, "hotkeys": names, "reissue_every_seconds": reissue_s, "link_file": link_file}
    files[INDEX] = as_json(stamp | index)
    return files


def write_files(files: dict[str, bytes], sink: ObjectStore, prefix: str = "") -> int:
    """Write the link files, then remove the link file of each hotkey that is not in ``files``. Returns how many were removed."""
    for name in sorted(files, key=lambda name: name == INDEX):  # the index goes last
        sink.put(prefix + name, files[name])
    keep = {prefix + name for name in files}
    stale = [key for key in sink.list(prefix + LINKS_DIR) if key.endswith(".json") and key not in keep]
    for key in stale:
        sink.delete(key)
    return len(stale)


def upload_entry(links_url: str, ttl_s: int = LINK_TTL_S, reissue_s: int = REISSUE_S) -> dict[str, Any]:
    """The ``upload`` entry of the public snapshot. Public words only: no address that can write.

    ``links_url`` is the public address of the link files, as given to ``issue --links-url``.
    """
    links_url = _links_url(links_url)
    return {
        "sealed_links": f"{links_url}/<hotkey>.json",
        "index": f"{links_url}.json",
        "version": LINK_VERSION,
        "who": "every registered SN120 hotkey (not the burn UID) with the one-time Ed25519 activation, until its queue entry is evaluated",
        "sealed_to": "the Ed25519 public key of the hotkey (libsodium sealed box, base64); open it with the hotkey's secret seed",
        "inside": "JSON with put_url, put_headers, get_url, object_key, hotkey, issued_at, expires_at",
        "method": "PUT",
        "content_type": "application/octet-stream",
        "link_ttl_seconds": ttl_s,
        "reissue_every_seconds": reissue_s,
        "replace": "upload again as often as you like until your entry is evaluated; each upload replaces the file before",
        "activation": "python -m sn120_v2.links activate --wallet NAME --hotkey NAME --execute (one time; an activation made for v1 stays valid)",
        "open": f"python -m sn120_v2.links open --hotkey HOTKEY --key SEED_FILE --links-url {links_url}",
    }


# ---- commands ----


def _chain_activated(network: str, hotkeys: Iterable[str], netuid: int) -> set[str]:
    try:
        client = SubtensorClient.connect(netuid, network)  # no wallet: it can only read
    except V2Error:
        raise
    except Exception as exc:  # the SDK raises many types; each one means "no connection"
        raise InfraError(f"cannot connect to the chain: {type(exc).__name__}: {exc}") from exc
    try:
        return read_activated(client, hotkeys, netuid)
    finally:
        with contextlib.suppress(Exception):
            client.chain.close()


def _issue_once(args: argparse.Namespace) -> int:
    if R2Store is None:
        raise ConfigError("boto3 is not installed: install the 'r2' extra")
    store = R2Store.from_env()
    journal = Path(args.state_dir) / JOURNAL_FILE
    try:
        state = load_from_store(store, journal) if args.journal_from_store else load(journal)
    except OSError as exc:
        raise ConfigError(f"cannot read the journal in {args.state_dir}: {type(exc).__name__}") from exc
    if state.genesis is None:  # an empty journal would look like "nobody waits" and remove every link file
        raise ConfigError("the journal has no genesis: is this the right state directory and the right store?")
    waiting = eligible(state, args.burn_uid)
    hotkeys = waiting
    if not args.no_activation_check:
        ready = _chain_activated(args.network, waiting, args.netuid)
        hotkeys = [hotkey for hotkey in waiting if hotkey in ready]
    files = link_files(hotkeys, store, int(time.time()), args.links_url, args.ttl_s, args.reissue_s)
    sink, prefix = (LocalStore(args.site_dir), "") if args.site_dir else (store, PUBLIC_PREFIX)
    removed = write_files(files, sink, prefix)
    issued = len(files) - 1  # every file but the index
    skipped = f"not activated {len(waiting) - len(hotkeys)}, not Ed25519 {len(hotkeys) - issued}"
    print(f"links: issued {issued} of {len(waiting)} waiting hotkeys ({skipped}), removed {removed}, valid {args.ttl_s} s")
    return 0


def _issue(args: argparse.Namespace) -> int:
    if not args.network and not args.no_activation_check:
        raise ConfigError("give --network (the chain to read the activations from), or --no-activation-check")
    while True:
        try:
            code = _issue_once(args)
        except V2Error as exc:
            if not args.loop:
                raise
            code = 1
            print(f"no link written: {exc}", file=sys.stderr, flush=True)
        if not args.loop:
            return code
        sys.stdout.flush()
        time.sleep(args.reissue_s)


def _activate(args: argparse.Namespace) -> int:
    if bittensor is None:
        raise ConfigError("the bittensor SDK is not installed: install the 'chain' extra")
    wallet = bittensor.Wallet(name=args.wallet, hotkey=args.hotkey, path=str(Path(args.wallet_path).expanduser()))
    print(json.dumps(activate(wallet, bittensor, args.netuid, args.network, args.execute)))
    return 0


def _open(args: argparse.Namespace) -> int:
    if not args.file and not args.links_url:
        raise ConfigError("give --links-url (the public address of the link files, from llms.txt), or --file")
    address = "" if args.file else f"{_links_url(args.links_url)}/{args.hotkey}.json"
    try:
        if args.file:
            doc = json.loads(Path(args.file).read_text())
        else:
            with urllib.request.urlopen(urllib.request.Request(address, headers={"User-Agent": "sn120-v2-links/1"}), timeout=60) as reply:
                doc = json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        hint = "no link for this hotkey: it needs a registration, the activation, and an entry that is not evaluated yet"
        raise InfraError(hint if exc.code == 404 else f"{address} answered {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise InfraError(f"cannot read the link file {args.file or address}: {type(exc).__name__}") from exc
    link = open_link(doc, args.hotkey, read_seed(args.key), int(time.time()))
    print(json.dumps(link, indent=1, sort_keys=True) if args.json else link["put_url"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.links", description="Sealed upload links and the one-time activation.")
    commands = parser.add_subparsers(dest="command", required=True)

    issue = commands.add_parser("issue", help="operator: write a sealed link file for each hotkey that may upload")
    issue.add_argument("--state-dir", required=True, help="the directory that holds the controller's journal (it is read, never written)")
    issue.add_argument(
        "--journal-from-store",
        action="store_true",
        help="read the journal copy in the object store; --state-dir is then only a cache, and never the controller's own directory",
    )
    issue.add_argument("--links-url", required=True, help="the public address of the link files, like https://<site>/ff/links (never derived)")
    where = issue.add_mutually_exclusive_group(required=True)
    where.add_argument("--site-dir", help=f"write {LINKS_DIR}<hotkey>.json and {INDEX} under this local directory: the root of the public website")
    where.add_argument("--to-store", action="store_true", help=f"write the same two paths under {PUBLIC_PREFIX} in the object store instead")
    issue.add_argument("--network", default="", help="the chain to read the activations from: finney, or the address of a node")
    issue.add_argument("--no-activation-check", action="store_true", help="issue to every waiting hotkey; use only when the chain cannot be read")
    issue.add_argument("--netuid", type=int, default=120, help="the subnet (default 120)")
    issue.add_argument("--burn-uid", type=int, default=0, help="the UID that never gets a link (default 0)")
    issue.add_argument("--ttl-s", type=int, default=LINK_TTL_S, help=f"seconds one link works (default {LINK_TTL_S}, at most {MAX_TTL_S})")
    issue.add_argument("--reissue-s", type=int, default=REISSUE_S, help=f"seconds between two issues (default {REISSUE_S})")
    issue.add_argument("--loop", action="store_true", help="issue again every --reissue-s seconds, and keep going after an error")
    issue.set_defaults(run=_issue)

    act = commands.add_parser("activate", help="miner: make the one-time activation (a dry run without --execute)")
    act.add_argument("--wallet", required=True, help="the bittensor wallet name")
    act.add_argument("--hotkey", required=True, help="the hotkey name in that wallet (an Ed25519 key)")
    act.add_argument("--wallet-path", default="~/.bittensor/wallets", help="the directory of the wallets")
    act.add_argument("--network", default="finney", help="the chain: finney, or the address of a node")
    act.add_argument("--netuid", type=int, default=120, help="the subnet (default 120)")
    act.add_argument("--execute", action="store_true", help="write the activation to the chain; without it, only print it")
    act.set_defaults(run=_activate)

    opened = commands.add_parser("open", help="miner: open your link and print the upload address (a secret: do not share it)")
    opened.add_argument("--hotkey", required=True, help="your hotkey, as an SS58 address")
    opened.add_argument("--key", required=True, help="a file with the hotkey's Ed25519 seed (64 hex characters), or the bittensor hotkey file")
    opened.add_argument("--links-url", default="", help="the public address of the link files, as llms.txt gives it")
    opened.add_argument("--file", help="read the link file from this path instead of the public address")
    opened.add_argument("--json", action="store_true", help="print every field of the link, not only the upload address")
    opened.set_defaults(run=_open)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except V2Error as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
