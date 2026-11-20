#!/usr/bin/env python3
"""Generator (SN120 mechanism 1) miner kit.

Run from the ``v2`` directory: ``python miner/generator_miner.py --help``.
The signing, schema, clock and pinned-pool checks are the validator's own
primitives, not a second implementation. Only the key-file and transport
conveniences live here. A sealed upload URL is read into memory, never saved
or logged; the explicit ``open-link --show-url`` opt-in is the sole exception.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import getpass
import hashlib
import importlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from nacl.public import SealedBox
from nacl.signing import SigningKey

from sn120_v2.canonical import canonical_json, file_sha256, sha256_hex
from sn120_v2.errors import ConfigError, ContractError, InfraError, V2Error
from sn120_v2.generator import epoch, pool, protocol
from sn120_v2.generator.constants import (
    MAX_TASKS_PER_SUBMISSION,
    NETUID,
    POOL_CACHE_ENV,
    POOL_MANIFEST_SHA256,
    POOL_ROWS,
    POOL_VERSION,
    POOL_NAME,
)

SITE = "https://affine.io"
LINKS = SITE + "/generator/links"
STATE_URL = SITE + "/generator/state.json"
RESULTS = SITE + "/generator/epochs/{epoch}/results.json"
ACCEPTED_ALL = SITE + "/generator/accepted-all.jsonl"
LINK_VERSION = "affine-generator-link-v1"
MAX_PUBLIC_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_BYTES = protocol.MAX_SUBMISSION_BYTES
USER_AGENT = "affine-generator-miner/1"
_REDACTED = "[redacted]"
_SECRET_QUERY = re.compile(r"(?i)([?&](?:X-Amz-)?(?:Signature|Credential|Security-Token|Token|Expires|SignedHeaders)=)[^&\s]+")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_EPOCH = re.compile(r"[0-9]{8}T[0-9]{4}Z")


def safe_text(value: object) -> str:
    """Make an exception safe for a terminal without guessing a secret value."""
    text = " ".join(str(value).split())
    return _SECRET_QUERY.sub(lambda match: match.group(1) + _REDACTED, text)


def _no_secrets(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, _REDACTED)
    return safe_text(text)


def fail(message: str) -> None:
    raise ConfigError(message)


def utc_stamp(value: int | float | str) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("a link timestamp must be a Unix timestamp") from exc
    if number < 0:
        raise ContractError("a link timestamp cannot be negative")
    return dt.datetime.fromtimestamp(number, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_bytes(source: str | Path, *, maximum: int = MAX_PUBLIC_BYTES) -> bytes:
    """Read a public object or explicit local fixture, never an object-store secret."""
    if isinstance(source, Path) or "://" not in str(source):
        try:
            with Path(source).expanduser().open("rb") as stream:
                raw = stream.read(maximum + 1)
        except OSError as exc:
            raise InfraError(f"cannot read {source}: {safe_text(exc)}") from exc
    else:
        url = str(source)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            fail("public downloads must use an HTTPS URL without credentials")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:  # noqa: S310 - public HTTPS object
                raw = reply.read(maximum + 1)
        except urllib.error.HTTPError as exc:
            raise InfraError(f"public object returned HTTP {exc.code}: {url}") from exc
        except (OSError, ValueError) as exc:
            raise InfraError(f"cannot read public object {url}: {safe_text(exc)}") from exc
    if len(raw) > maximum:
        raise ContractError(f"object exceeds the {maximum}-byte safety limit")
    return raw


def read_json(source: str | Path, *, maximum: int = MAX_PUBLIC_BYTES) -> Any:
    try:
        return json.loads(read_bytes(source, maximum=maximum))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"expected UTF-8 JSON: {safe_text(exc)}") from exc


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Publish only after the complete batch is valid; no secret can be in data."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(data) + b"\n")
    temporary.replace(path)
    print(f"wrote {path} ({len(data['task_ids'])} tasks, epoch {data['epoch']})")


# ---- wallet hotkeys, without the bittensor SDK ----


def read_seed_file(path: Path) -> bytes:
    """Read the exact secretSeed in an unencrypted bittensor hotkey file."""
    try:
        document = json.loads(path.expanduser().read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read hotkey file {path}: {safe_text(exc)}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("secretSeed"), str):
        fail(f"{path} is not a bittensor hotkey JSON file with a secretSeed")
    if document.get("encrypted") is True:
        fail("this hotkey file is encrypted; use --wallet/--hotkey to unlock it with bittensor")
    try:
        seed = bytes.fromhex(document["secretSeed"].removeprefix("0x"))
    except ValueError as exc:
        raise ConfigError(f"hotkey file {path} has an invalid secretSeed encoding") from exc
    if len(seed) != 32:
        fail(f"hotkey file {path} must hold a 32-byte Ed25519 secretSeed")
    return seed


def load_wallet_key(wallet_path: Path, name: str, hotkey: str, password: str | None) -> tuple[str, SigningKey]:
    """Use bittensor only to unlock an encrypted wallet; it is not a kit dependency."""
    try:
        import bittensor  # type: ignore[import-not-found]  # optional encrypted-wallet path
    except ImportError as exc:
        raise ConfigError("encrypted wallets need the bittensor SDK, or pass --key-file containing secretSeed") from exc
    try:
        wallet = bittensor.Wallet(name=name, hotkey=hotkey, path=str(wallet_path.expanduser()))
        keypair = wallet.hotkey
        if keypair is None:
            fail(f"wallet {name}/{hotkey} has no hotkey")
        if password is not None:
            keypair.decrypt(password)
        elif getattr(keypair, "is_encrypted", False):
            keypair.decrypt(getpass.getpass("Hotkey password: "))
        address = keypair.ss58_address
        seed = bytes.fromhex(keypair.private_key.removeprefix("0x"))
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"cannot unlock wallet {name}/{hotkey}: {safe_text(exc)}") from exc
    if len(seed) != 32:
        fail("Generator needs a 32-byte Ed25519 hotkey, not an sr25519 hotkey")
    return address, SigningKey(seed)


def signing_key(args: argparse.Namespace) -> tuple[str, SigningKey]:
    if args.key_file:
        address, seed = args.hotkey, read_seed_file(args.key_file)
        key = SigningKey(seed)
    elif args.seed:
        try:
            seed = bytes.fromhex(args.seed.removeprefix("0x"))
        except ValueError as exc:
            raise ConfigError("--seed must be a 32-byte Ed25519 seed in hexadecimal") from exc
        if len(seed) != 32:
            fail("--seed must encode exactly 32 bytes")
        address, key = args.hotkey, SigningKey(seed)
    elif args.wallet or args.hotkey_name:
        if not args.wallet or not args.hotkey_name:
            fail("give both --wallet and --hotkey-name, or use --key-file/--seed")
        address, key = load_wallet_key(args.wallet_path, args.wallet, args.hotkey_name, args.password)
    else:
        fail("identify the hotkey with --key-file, --seed, or --wallet plus --hotkey-name")
    if not address:
        fail("give --hotkey SS58 when using --key-file or --seed")
    if args.hotkey and address != args.hotkey:
        fail("the wallet hotkey and --hotkey do not match")
    if bytes(key.verify_key) != protocol.public_key(address):
        fail("the secret key does not match the Ed25519 hotkey")
    return address, key


def add_key_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hotkey", default="", help="your SS58 Ed25519 hotkey (required for --key-file/--seed)")
    parser.add_argument("--key-file", type=Path, help="unencrypted bittensor hotkey JSON containing secretSeed")
    parser.add_argument("--seed", help="32-byte Ed25519 seed in hex (prefer --key-file to avoid shell history)")
    parser.add_argument("--wallet", help="bittensor wallet name (SDK needed only for this option)")
    parser.add_argument("--hotkey-name", help="hotkey name inside that wallet")
    parser.add_argument("--wallet-path", type=Path, default=Path("~/.bittensor/wallets"))
    parser.add_argument("--password", help=argparse.SUPPRESS)  # prompt instead; retained for programmatic unlockers


# ---- sealed links and upload ----


def _linked_epoch(link: dict[str, Any]) -> str:
    value = link.get("epoch")
    if value is not None:
        return epoch.epoch_id(value)
    key = link.get("object_key", "")
    if isinstance(key, str):
        match = re.search(r"(?:^|/)sub/([^/]+)/", key)
        if match:
            return epoch.epoch_id(urllib.parse.unquote(match.group(1)))
    for field in ("put_url", "get_url"):
        url = link.get(field, "")
        if isinstance(url, str):
            match = re.search(r"(?:^|/)sub/([^/]+)/", urllib.parse.urlparse(url).path)
            if match:
                return epoch.epoch_id(urllib.parse.unquote(match.group(1)))
    fail("the sealed link has no epoch or recognizable Generator object key")
    raise AssertionError


def open_sealed_link(
    document: dict[str, Any], hotkey: str, key: SigningKey, now: int, field: str = "current", *, peek: bool = False
) -> dict[str, Any]:
    """Open one current/next Generator box, without exposing its capability."""
    if not isinstance(document, dict) or document.get("version") != LINK_VERSION:
        fail(f"expected a {LINK_VERSION} link")
    if document.get("hotkey") != hotkey or document.get("sealed_to") != bytes(key.verify_key).hex():
        fail("the link is not sealed to this hotkey")
    if field not in ("current", "next") or field not in document:
        fail(f"the public link has no {field} sealed box; run again when the issuer publishes it")
    try:
        sealed = base64.b64decode(document[field], validate=True)
        plaintext = SealedBox(key.to_curve25519_private_key()).decrypt(sealed)
        link = json.loads(plaintext)
    except Exception as exc:
        # Deliberately do not include backend text, which can quote ciphertext.
        raise ContractError(f"cannot open this hotkey's sealed Generator link ({type(exc).__name__})") from exc
    if not isinstance(link, dict) or link.get("hotkey") != hotkey:
        fail("the sealed link belongs to another hotkey")
    if link.get("version") != LINK_VERSION:
        fail("the sealed link has the wrong Generator version")
    for name in ("put_url", "put_headers", "expires_at", "not_before", "deadline", "object_key", "epoch"):
        if name not in link:
            fail(f"the sealed link has no {name}")
    if not isinstance(link["put_url"], str) or not isinstance(link["put_headers"], dict):
        fail("the sealed link has an invalid PUT URL or headers")
    parsed = urllib.parse.urlparse(link["put_url"])
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        fail("the sealed upload link must be an HTTPS URL without embedded credentials")
    if not all(isinstance(name, str) and isinstance(value, str) for name, value in link["put_headers"].items()):
        fail("the sealed link's PUT headers must be strings")
    if any(name.lower() in ("host", "content-length", "transfer-encoding") for name in link["put_headers"]):
        fail("the sealed link may not sign host or framing headers")
    if type(link["expires_at"]) is not int or type(link["not_before"]) is not int or type(link["deadline"]) is not int:
        fail("the sealed link times must be Unix timestamps")
    target = epoch.epoch_id(link["epoch"])
    deadline = int(epoch.epoch_deadline(target).timestamp())
    if link["deadline"] != deadline or link["not_before"] != int(epoch.epoch_start(target).timestamp()):
        fail(f"sealed link does not match epoch {target}")
    if not 0 < link["expires_at"] <= deadline or link["expires_at"] < link["not_before"]:
        fail(f"sealed link expiry exceeds deadline {target}")
    if link.get("object_key") != epoch.submission_object(target, hotkey):
        fail("the sealed link names the wrong private submission object")
    mode = link.get("upload_mode")
    if mode not in ("first_upload_final", "overwrite_until_deadline"):
        fail("the sealed link has no recognized upload mode")
    if bool(link.get("first_upload_final")) != (mode == "first_upload_final"):
        fail("the sealed link's upload mode flags disagree")
    if bool(link.get("overwrite_until_deadline")) != (mode == "overwrite_until_deadline"):
        fail("the sealed link's upload mode flags disagree")
    if mode == "first_upload_final" and link["put_headers"].get("If-None-Match") != "*":
        fail("a first-upload-final grant must sign If-None-Match: *")
    if mode == "first_upload_final":
        try:
            query = urllib.parse.parse_qs(parsed.query)
            signed = ";".join(query.get("X-Amz-SignedHeaders", [])).lower().split(";")
        except (TypeError, ValueError):
            signed = []
        if "if-none-match" not in signed:
            fail("a first-upload-final grant must bind If-None-Match in its signature")
    if now >= link["expires_at"] or (now < link["not_before"] and not peek):
        fail(f"the submission window for {target} is closed or not open; fetch a fresh link")
    if now >= link["expires_at"] - 60:
        fail("the sealed link expires in less than 60 seconds; fetch a fresh link")
    if field == "current":
        if document.get("epoch") != target or document.get("deadline") != deadline or document.get("expires_at") != link["expires_at"]:
            fail("the outer link and sealed current link disagree")
        if document.get("upload_mode") != mode:
            fail("the outer link and sealed current link disagree on upload mode")
    elif document.get("next_epoch") != target:
        fail("the outer link and sealed next link disagree on epoch")
    return link


def link_for(args: argparse.Namespace, now: int | None = None) -> tuple[str, dict[str, Any]]:
    hotkey, key = signing_key(args)
    address = args.link_file or f"{LINKS}/{urllib.parse.quote(hotkey, safe='')}.json"
    document = read_json(address, maximum=1024 * 1024)
    timestamp = int(time.time()) if now is None else now
    requested = getattr(args, "epoch", None)
    field = "current"
    if requested and document.get("next_epoch") == epoch.epoch_id(requested):
        field = "next"
    elif requested and document.get("epoch") != epoch.epoch_id(requested):
        fail(f"the public link has neither current nor next epoch {requested}")
    return hotkey, open_sealed_link(document, hotkey, key, timestamp, field=field, peek=bool(getattr(args, "peek", False)))


def print_link(link: dict[str, Any], *, show_url: bool = False) -> None:
    target = _linked_epoch(link)
    print(f"epoch: {target}")
    print(f"window: [{utc_stamp(epoch.epoch_start(target).timestamp())}, {utc_stamp(epoch.epoch_deadline(target).timestamp())}) UTC")
    print(f"expires_at: {link['expires_at']} ({utc_stamp(link['expires_at'])})")
    print(f"upload_mode: {link['upload_mode']}")
    print("sealed upload link opened; PUT URL is secret and was not saved")
    if show_url:
        print(link["put_url"])


def upload(link: dict[str, Any], raw: bytes, *, replace: bool = False, cap: int | None = None, epoch_name: str | None = None) -> None:
    if cap is not None:
        document = json.loads(raw)
        target = epoch_name or _linked_epoch(link)
        if len(document["task_ids"]) > cap:
            fail(f"submission has {len(document['task_ids'])} task ids; the per-hotkey cap for {target} is {cap}")
    if link.get("upload_mode") == "overwrite_until_deadline" and not replace:
        fail("this grant is overwrite_until_deadline; pass --replace only if you intend to replace your earlier upload before the deadline")
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        fail(f"batch must contain 1..{MAX_UPLOAD_BYTES} bytes")
    request = urllib.request.Request(
        link["put_url"], data=raw, headers=link["put_headers"], method="PUT"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:  # noqa: S310 - freshly opened sealed PUT
            if not 200 <= reply.status < 300:
                raise InfraError(f"upload returned HTTP {reply.status}; no URL was logged")
            reply.read(1024)
    except urllib.error.HTTPError as exc:
        raise InfraError(f"upload returned HTTP {exc.code}; no URL or response body was logged") from exc
    except (OSError, ValueError) as exc:
        # Some urllib errors wrap the complete URL in their text.
        raise InfraError("upload transport failed; fetch the link again and retry (URL redacted)") from exc


# ---- pool, build and offline check ----


def cache_path(args: argparse.Namespace) -> Path:
    return Path(args.cache or os.environ.get(POOL_CACHE_ENV) or "~/.cache/affine/generator").expanduser()


def load_registry(args: argparse.Namespace, *, index: bool = False) -> pool.PriorPool:
    return pool.PriorPool(cache_path(args), build_index=index)


def read_task_ids(path: Path) -> list[str]:
    """One ID per line, or the id column of a search JSONL report."""
    try:
        lines = path.expanduser().read_text().splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read task IDs {path}: {safe_text(exc)}") from exc
    ids = []
    for number, line in enumerate(lines, 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("{"):
            try:
                row = json.loads(text)
                text = row["id"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ContractError(f"{path}:{number} has no string id") from exc
        if not isinstance(text, str):
            fail(f"{path}:{number} has no string id")
        ids.append(text)
    if not ids or len(ids) > MAX_TASKS_PER_SUBMISSION:
        fail(f"need 1..{MAX_TASKS_PER_SUBMISSION} task IDs")
    if len(set(ids)) != len(ids):
        fail("task IDs must be distinct")
    return ids


def batch_document(epoch_name: str, hotkey: str, task_ids: Sequence[str], key: SigningKey) -> dict[str, Any]:
    submission = protocol.sign_submission(
        epoch_name, hotkey, task_ids, lambda message: key.sign(message).signature, require_activation=False
    )
    # Local activation is a chain condition, not a secret and not something the
    # offline builder can prove. The snapshot performs that check independently.
    return submission.to_dict()


def check_document(raw: bytes, registry: pool.PriorPool | None = None) -> protocol.Submission:
    submission = protocol.Submission.from_json(raw)
    if not protocol.verify_submission(submission, known_task_ids=registry.__contains__ if registry else None):
        fail("batch signature or pool membership is invalid")
    return submission


def cmd_pool(args: argparse.Namespace) -> int:
    manifest = pool.load_manifest(args.manifest_url)
    paths = pool.prepare_shards(cache_path(args), manifest, fetch=True)
    registry = pool.PriorPool(cache_path(args), manifest=manifest, build_index=args.index)
    if len(paths) != 8:
        fail(f"the pinned release must contain eight verified shards, got {len(paths)}")
    print(f"pool: {POOL_NAME}-{POOL_VERSION}")
    print(f"manifest sha256: {POOL_MANIFEST_SHA256}")
    print(f"verified: {len(paths)} shards, {manifest.rows:,} rows")
    print(f"cache: {cache_path(args)}")
    if registry._locations:  # read-only status of the validator's cache; not a second index implementation
        print(f"ID index: {len(registry._locations):,} original 20-hex IDs")
    else:
        print("ID index: not built; run pool --index with pyarrow before build/check pool membership")
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    _, link = link_for(args, now=0 if args.peek else None)
    print_link(link, show_url=args.show_url)
    if args.peek:
        print("inspection only: this command does not prove the submission window is open")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    hotkey, key = signing_key(args)
    ids = read_task_ids(args.task_ids)
    registry = load_registry(args)
    registry.require_ids(ids)  # fail closed without a complete, verified index
    target = epoch.epoch_id(args.epoch) if args.epoch else epoch.epoch_id(time.time())
    document = batch_document(target, hotkey, ids, key)
    check_document(canonical_json(document), registry)
    write_json(args.output, document)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        raw = args.batch.expanduser().read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read batch {args.batch}: {safe_text(exc)}") from exc
    registry = None if args.skip_pool else load_registry(args)
    submission = check_document(raw, registry)
    print(f"valid: epoch {submission.epoch}, hotkey {submission.hotkey}, {len(submission.task_ids)} distinct IDs")
    print(f"batch sha256: {sha256_hex(raw)}")
    print("checked offline: exact schema, Ed25519 signature" + ("" if registry else " (pool membership SKIPPED)"))
    if args.skip_pool:
        print("warning: --skip-pool does not prove pool membership; the validator will still check it")
    else:
        print("pool membership: verified against the pinned ID index")
    return 0


def public_cap(args: argparse.Namespace, target: str) -> int | None:
    document = read_json(getattr(args, "state_url", STATE_URL))
    current = document.get("current_epoch") or document.get("epoch")
    field = "submission_cap" if current == target else "next_submission_cap"
    value = document.get(field)
    if not isinstance(value, dict) or value.get("epoch") != target:
        return None
    cap = value.get("cap")
    if type(cap) is not int or not 1 <= cap <= MAX_TASKS_PER_SUBMISSION:
        fail("public submission cap is invalid")
    return cap


def cmd_upload(args: argparse.Namespace) -> int:
    try:
        raw = args.batch.expanduser().read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read batch {args.batch}: {safe_text(exc)}") from exc
    hotkey, link = link_for(args)
    registry = None if args.skip_pool else load_registry(args)
    submission = check_document(raw, registry)
    if submission.hotkey != hotkey:
        fail("the batch hotkey does not own this sealed link")
    if submission.epoch != _linked_epoch(link):
        fail(f"batch epoch {submission.epoch} does not match link epoch {_linked_epoch(link)}")
    cap = link["max_tasks"] if "max_tasks" in link else public_cap(args, _linked_epoch(link))
    upload(link, raw, replace=args.replace, cap=cap, epoch_name=_linked_epoch(link))
    print(f"uploaded epoch {submission.epoch}: {len(submission.task_ids)} tasks; PUT URL not logged")
    return 0


def status_for_hotkey(document: Any, hotkey: str) -> Any:
    """Tolerate either list- or mapping-shaped per-hotkey public result rows."""
    if isinstance(document, dict):
        for key in ("hotkeys", "submissions", "results", "miners", "per_hotkey"):
            value = document.get(key)
            if isinstance(value, dict) and hotkey in value:
                return value[hotkey]
            if isinstance(value, list):
                found = [row for row in value if isinstance(row, dict) and row.get("hotkey") == hotkey]
                if found:
                    return found[0]
        if document.get("hotkey") == hotkey:
            return document
    if isinstance(document, list):
        return [row for row in document if isinstance(row, dict) and row.get("hotkey") == hotkey]
    return None


def cmd_status(args: argparse.Namespace) -> int:
    state = read_json(args.state_url)
    if not isinstance(state, dict):
        fail("Generator state must be a JSON object")
    targets = args.epochs or [state.get("epoch") or state.get("current_epoch") or epoch.epoch_id(time.time())]
    targets = [epoch.epoch_id(target) for target in targets]
    public_state = {key: value for key, value in state.items() if not any(word in key.lower() for word in ("url", "secret", "seed"))}
    print(json.dumps({"state": public_state}, indent=2, sort_keys=True))
    for target in targets:
        results = read_json(args.results_url.format(epoch=target))
        if results is None:
            print(f"results for {target} are not published yet")
        mine = status_for_hotkey(results, args.hotkey)
        print(json.dumps({"epoch": target, "hotkey": args.hotkey, "results": mine}, indent=2, sort_keys=True))
        if mine is None:
            print(f"no public results for this hotkey in {target}; scoring may still be running or the upload was not eligible")
        # Accepted-all is public; this command never reads a private submission.
        if args.accepted:
            accepted = []
            for line in read_bytes(args.accepted_url).decode().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("submitter") == args.hotkey and (not args.epochs or row.get("epoch") in targets):
                    accepted.append(row)
            print(json.dumps({"accepted_all": accepted}, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SN120 Generator miner: pool, sign, check, sealed upload and public status.")
    root.add_argument("--version", action="version", version=protocol.SCHEMA_VERSION)
    commands = root.add_subparsers(dest="command", required=True)

    p = commands.add_parser("pool", help="download and hash-verify the pinned MANIFEST and all eight shards")
    p.add_argument("--cache", type=Path, help=f"pool cache (default ${POOL_CACHE_ENV} or ~/.cache/affine/generator)")
    p.add_argument("--manifest-url", help="mirror URL for exactly the pinned MANIFEST bytes")
    p.add_argument("--index", action="store_true", help="build the 811,609-ID membership index (requires pyarrow)")
    p.set_defaults(run=cmd_pool)

    p = commands.add_parser("open-link", help="open your sealed link; show only epoch and UTC expiry by default")
    add_key_options(p)
    p.add_argument("--link-file", help="read a public link JSON from this file (for testing/offline diagnostics)")
    p.add_argument("--show-url", action="store_true", help="explicitly print the secret presigned PUT URL")
    p.add_argument("--epoch", help="select the sealed next-epoch grant when it is available")
    p.add_argument("--peek", action="store_true", help="inspect a prefetched next-epoch grant before its window opens")
    p.set_defaults(run=cmd_open)

    p = commands.add_parser("build", help="turn a task IDs file into a canonical signed batch")
    add_key_options(p)
    p.add_argument("--task-ids", required=True, type=Path, help="one 20-hex ID per line, or search JSONL with id fields")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epoch", help="deadline YYYYMMDDTHHMMZ; default: current four-hour UTC epoch")
    p.add_argument("--cache", type=Path)
    p.set_defaults(run=cmd_build)

    for name, help_text in (("check", "offline schema/signature/pool validation"), ("upload", "check then PUT through a freshly opened sealed link")):
        p = commands.add_parser(name, help=help_text)
        p.add_argument("--batch", required=True, type=Path)
        p.add_argument("--cache", type=Path)
        p.add_argument("--skip-pool", action="store_true", help="skip local membership check only; validator still enforces it")
        if name == "upload":
            add_key_options(p)
            p.add_argument("--epoch", help="select the sealed next-epoch grant when available")
            p.add_argument("--replace", action="store_true", help="explicitly replace an earlier upload in overwrite mode")
            p.add_argument("--link-file", help=argparse.SUPPRESS)
            p.add_argument("--state-url", default=STATE_URL, help=argparse.SUPPRESS)
        p.set_defaults(run=cmd_check if name == "check" else cmd_upload)

    p = commands.add_parser("status", help="read public Generator state, epoch results and optionally accepted-all")
    p.add_argument("--hotkey", required=True)
    p.add_argument("--epochs", nargs="+", help="epoch deadline IDs (default: the state epoch)")
    p.add_argument("--accepted", action="store_true", help="also read accepted-all.jsonl")
    p.add_argument("--state-url", default=STATE_URL, help=argparse.SUPPRESS)
    p.add_argument("--results-url", default=RESULTS, help=argparse.SUPPRESS)
    p.add_argument("--accepted-url", default=ACCEPTED_ALL, help=argparse.SUPPRESS)
    p.set_defaults(run=cmd_status)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (V2Error, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"{args.command} failed: {safe_text(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
