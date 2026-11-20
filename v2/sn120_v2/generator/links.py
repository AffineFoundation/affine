"""Deadline-bound, sealed Generator upload links.

Every registered non-zero SN120 UID needs the existing v2 Ed25519 activation
before this module will issue a link.  A link names exactly one epoch-scoped
private object.  During the final ten minutes of an epoch, a separately sealed
``next`` link lets a miner fetch the following epoch's capability in advance.

Presigned URLs are secrets: they occur only inside sealed boxes.  In particular,
the public index, command output and errors produced here never contain them.
Conditional first-upload-final PUTs are used only when an explicit support
probe succeeds and the returned grant signature binds ``If-None-Match: *``.
Otherwise the grant is labelled ``overwrite_until_deadline``; it is never
presented as immutable.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlsplit

from ..chain import Reader, SubtensorClient
from ..errors import ConfigError, ContractError, InfraError, V2Error
from ..links import public_key, read_activated, seal, store_token
from . import epoch
from .constants import EPOCH_SECONDS, NETUID

try:
    from ..r2 import R2Store
except ImportError:  # boto3 is an optional extra; pure link construction does not need it
    R2Store = None

LINK_VERSION = "affine-generator-link-v1"
LINKS_DIR = "generator/links/"
INDEX = "generator/links.json"
PUBLIC_LINKS_DIR = "public/" + LINKS_DIR
PUBLIC_INDEX = "public/" + INDEX
CONTENT_TYPE = "application/octet-stream"
FIRST_UPLOAD_HEADERS = {"Content-Type": CONTENT_TYPE, "If-None-Match": "*"}
FIRST_UPLOAD_FINAL = "first_upload_final"
OVERWRITE_UNTIL_DEADLINE = "overwrite_until_deadline"
PREFETCH_S = 10 * 60
MAX_PRESIGN_S = 7 * 24 * 3600

ConditionalSupport = Callable[[Any], bool]


class Presigner(Protocol):
    def presign_get(self, key: str, ttl_s: int) -> str: ...

    def presign_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]: ...


class GeneratorPresigner:
    """Add conditional PUT signing to the existing R2 store without changing it.

    ``conditional_support`` is an explicit operator/integration probe.  It gets
    the wrapped store and must return exactly ``True``.  No probe means support
    is unverified, so issuance uses the explicit overwrite mode.  Probe and
    conditional-signing failures are not allowed to masquerade as final uploads.

    The private R2 ``_presign`` seam is intentionally confined to this adapter;
    the shared store and protocol modules remain unchanged.
    """

    def __init__(self, store: Any, *, conditional_support: ConditionalSupport | None = None) -> None:
        self.store = store
        self._conditional_support = conditional_support

    def presign_get(self, key: str, ttl_s: int) -> str:
        return self.store.presign_get(key, ttl_s)

    def presign_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
        return self.store.presign_put(key, ttl_s)

    def first_upload_supported(self) -> bool:
        if self._conditional_support is None:
            return False
        try:
            return self._conditional_support(self.store) is True
        except Exception:
            return False

    def presign_first_upload_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
        try:
            sign = self.store._presign
        except AttributeError:
            raise ConfigError("this store cannot sign a conditional first-upload PUT") from None
        url = sign(
            "put_object",
            key,
            ttl_s,
            {"ContentType": CONTENT_TYPE, "IfNoneMatch": "*"},
        )
        return url, dict(FIRST_UPLOAD_HEADERS)


def eligible(registrations: Iterable[Any], activated: Iterable[str]) -> list[str]:
    """Registered non-zero hotkeys intersection the mandatory activated set."""
    activated_set = set(activated)
    return sorted(
        {
            registration.hotkey
            for registration in registrations
            if registration.uid != 0 and registration.hotkey in activated_set
        }
    )


def _json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")


def _now(value: dt.datetime | int | float) -> int:
    # epoch_id performs the same aware-UTC and numeric input validation as the
    # rest of Generator.  Convert only after that validation has succeeded.
    epoch.epoch_id(value)
    if isinstance(value, dt.datetime):
        return int(value.timestamp())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("link issue time must be finite")
    return int(value)


def _site_path(hotkey: str) -> str:
    object_key = epoch.link_object(hotkey)
    if not object_key.startswith("public/"):
        raise ContractError("Generator link object is not in the public namespace")
    return object_key.removeprefix("public/")


def _supported(presigner: Presigner, probe: ConditionalSupport | None) -> bool:
    try:
        if probe is not None:
            return probe(presigner) is True
        method = getattr(presigner, "first_upload_supported", None)
        return callable(method) and method() is True
    except Exception:
        return False


def _binds_if_none_match(url: Any) -> bool:
    """Check SigV4's signed-header list without ever putting the URL in an error."""
    try:
        if not isinstance(url, str):
            return False
        query = parse_qs(urlsplit(url).query)
        signed = query.get("X-Amz-SignedHeaders", [])
        return any("if-none-match" in value.lower().split(";") for value in signed)
    except (TypeError, ValueError):
        return False


def _conditional_put(presigner: Presigner, key: str, ttl_s: int) -> tuple[str, dict[str, str]] | None:
    try:
        method = getattr(presigner, "presign_first_upload_put")
        url, headers = method(key, ttl_s)
        headers = dict(headers)
    except Exception:
        return None
    # Extra required headers would make this small-submission grant a different
    # contract.  A header alone is insufficient: it must also be signature-bound.
    if headers != FIRST_UPLOAD_HEADERS or not _binds_if_none_match(url):
        return None
    return url, headers


def _plain_put(presigner: Presigner, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
    try:
        url, headers = presigner.presign_put(key, ttl_s)
        return url, dict(headers)
    except Exception as exc:
        raise InfraError(f"Generator PUT presigning failed for {key!r}: {type(exc).__name__}") from None


def _get(presigner: Presigner, key: str, ttl_s: int) -> str:
    try:
        return presigner.presign_get(key, ttl_s)
    except Exception as exc:
        raise InfraError(f"Generator GET presigning failed for {key!r}: {type(exc).__name__}") from None


def _grant(
    presigner: Presigner,
    hotkey: str,
    epoch_name: str,
    kind: str,
    issued_at: int,
    expires_at: int,
    prefer_first: bool,
    max_tasks: int | None = None,
) -> tuple[dict[str, Any], str]:
    deadline = int(epoch.epoch_deadline(epoch_name).timestamp())
    key = epoch.submission_object(epoch_name, hotkey)
    ttl_s = expires_at - issued_at
    conditional = _conditional_put(presigner, key, ttl_s) if prefer_first else None
    mode = FIRST_UPLOAD_FINAL if conditional is not None else OVERWRITE_UNTIL_DEADLINE
    put_url, put_headers = conditional if conditional is not None else _plain_put(presigner, key, ttl_s)
    link = {
        "version": LINK_VERSION,
        "kind": kind,
        "hotkey": hotkey,
        "epoch": epoch_name,
        "object_key": key,
        "issued_at": issued_at,
        "not_before": int(epoch.epoch_start(epoch_name).timestamp()),
        "deadline": deadline,
        "expires_at": expires_at,
        "put_url": put_url,
        "put_headers": put_headers,
        "get_url": _get(presigner, key, ttl_s),
        "upload_mode": mode,
        "first_upload_final": mode == FIRST_UPLOAD_FINAL,
        "overwrite_until_deadline": mode == OVERWRITE_UNTIL_DEADLINE,
        **({"max_tasks": max_tasks} if max_tasks is not None else {}),
    }
    return link, mode


def link_files(
    hotkeys: Iterable[str],
    presigner: Presigner,
    now: dt.datetime | int | float,
    *,
    ttl_s: int = EPOCH_SECONDS,
    first_upload_final: bool = False,
    conditional_support: ConditionalSupport | None = None,
    caps: Mapping[str, int] | None = None,
) -> dict[str, bytes]:
    """Build sealed site files for an already registered/activated ID set.

    The caller must obtain ``hotkeys`` through :func:`eligible` and
    ``read_activated``.  Invalid or non-Ed25519 addresses are skipped.  The
    returned map is site-relative path -> bytes and always includes the index.

    ``first_upload_final`` requests a conditional grant, not a promise that the
    provider supports one.  Without a successful explicit probe and a correctly
    signed grant, the link says ``overwrite_until_deadline``.  The default is
    therefore deliberately and unambiguously overwrite-until-deadline mode.
    """
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, int) or not 0 < ttl_s <= MAX_PRESIGN_S:
        raise ConfigError(f"Generator link ttl_s must be an integer in 1..{MAX_PRESIGN_S}")
    issued_at = _now(now)
    current = epoch.epoch_id(now)
    deadline = int(epoch.epoch_deadline(current).timestamp())
    current_expiry = min(deadline, issued_at + ttl_s)
    following = epoch.epoch_id(deadline)
    next_deadline = int(epoch.epoch_deadline(following).timestamp())
    # The prefetched grant receives the same requested lifetime from issuance,
    # capped at its own (not the current epoch's) deadline.
    next_expiry = min(next_deadline, issued_at + ttl_s)
    prefetch = 0 < deadline - issued_at <= PREFETCH_S
    caps = dict(caps or {})
    for target, value in caps.items():
        if epoch.epoch_id(target) != target or type(value) is not int or value < 1:
            raise ConfigError("Generator link max_tasks must be a positive integer per epoch")
    prefer_first = bool(first_upload_final and _supported(presigner, conditional_support))

    files: dict[str, bytes] = {}
    modes: set[str] = set()
    candidates = sorted({hotkey for hotkey in hotkeys if isinstance(hotkey, str) and hotkey})
    for hotkey in candidates:
        try:
            sealed_to = public_key(hotkey).hex()
            path = _site_path(hotkey)
            current_link, current_mode = _grant(
                presigner, hotkey, current, "current", issued_at, current_expiry, prefer_first, caps.get(current)
            )
            outer = {
                "version": LINK_VERSION,
                "hotkey": hotkey,
                "issued_at": issued_at,
                "epoch": current,
                "deadline": deadline,
                "expires_at": current_expiry,
                "sealed_to": sealed_to,
                "upload_mode": current_mode,
                **({"max_tasks": caps[current]} if current in caps else {}),
                "current": seal(current_link, hotkey),
            }
            if prefetch:
                next_link, _ = _grant(
                    presigner, hotkey, following, "next", issued_at, next_expiry, prefer_first, caps.get(following)
                )
                outer["next_epoch"] = following
                if following in caps:
                    outer["next_max_tasks"] = caps[following]
                outer["next"] = seal(next_link, hotkey)
        except ContractError:
            # A valid SS58 address can still be a non-Ed25519/non-curve key.
            # Such an address cannot open a sealed box, so it gets no file.
            # Store/configuration errors and unexpected coding faults propagate.
            continue
        modes.add(current_mode)
        files[path] = _json(outer)

    default_mode = FIRST_UPLOAD_FINAL if prefer_first else OVERWRITE_UNTIL_DEADLINE
    index = {
        "version": LINK_VERSION,
        "issued_at": issued_at,
        "epoch": current,
        "deadline": deadline,
        "hotkeys": sorted(Path(name).stem for name in files),
        "link_file": LINKS_DIR + "<hotkey>.json",
        "prefetch_before_deadline_seconds": PREFETCH_S,
        "upload_mode": next(iter(modes)) if len(modes) == 1 else ("mixed" if modes else default_mode),
        **({"max_tasks": caps[current]} if current in caps else {}),
        **({"next_epoch": following, "next_max_tasks": caps[following]} if prefetch and following in caps else {}),
    }
    files[INDEX] = _json(index)
    return files


def _local_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or relative.is_absolute()
        or str(relative) != name
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ContractError("unsafe Generator site-file path")
    return root.joinpath(*relative.parts)


def _atomic_bytes(path: Path, data: bytes) -> bool:
    """Idempotently replace one file; readers see either complete version."""
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise InfraError(f"cannot read existing Generator site file {path}: {type(exc).__name__}") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise InfraError(f"cannot publish Generator site file {path}: {type(exc).__name__}") from None
    return True


def write_files(files: Mapping[str, bytes], site_dir: str | Path) -> int:
    """Atomically publish site files, index last, and remove stale link files."""
    root = Path(site_dir)
    if not root.is_dir():
        raise ConfigError(f"the site directory {root} does not exist; this job never makes it")
    # Validate the complete set before the first mutation: an unsafe entry must
    # not leave a partially published generation or remove an incumbent link.
    for name in files:
        _local_path(root, name)
    try:
        for name in sorted(files, key=lambda item: item == INDEX):
            _atomic_bytes(_local_path(root, name), files[name])
        keep = {name for name in files if name.startswith(LINKS_DIR) and name.endswith(".json")}
        removed = 0
        directory = root / LINKS_DIR
        if directory.is_dir():
            for path in directory.iterdir():
                name = LINKS_DIR + path.name
                if path.is_file() and path.suffix == ".json" and name not in keep:
                    path.unlink()
                    removed += 1
        return removed
    except (ConfigError, ContractError, InfraError):
        raise
    except Exception:
        raise InfraError("local Generator link publication failed") from None


def mirror_files(files: Mapping[str, bytes], store: Any) -> int:
    """Optionally mirror the same documents under public/generator/ in R2."""
    try:
        for name in sorted(files, key=lambda item: item == INDEX):
            _local_path(Path(""), name)  # validate before adding the R2 prefix
            store.put("public/" + name, files[name])
        keep = {"public/" + name for name in files}
        stale = [
            key
            for key in store.list(PUBLIC_LINKS_DIR)
            if key.endswith(".json") and key not in keep
        ]
        for key in stale:
            store.delete(key)
        return len(stale)
    except Exception as exc:
        raise InfraError(f"Generator R2 link mirror failed: {type(exc).__name__}") from None


def _registered_and_activated(network: str, netuid: int) -> tuple[list[str], list[str]]:
    """One read-only registration snapshot plus its mandatory activation read."""
    try:
        client = SubtensorClient.connect(netuid, network)  # no wallet: read only
    except Exception as exc:
        raise InfraError(f"cannot connect for the Generator registration snapshot: {type(exc).__name__}") from None
    try:
        snapshot = Reader(client, netuid).snapshot()
        waiting = sorted({reg.hotkey for reg in snapshot.registrations if reg.uid != 0})
        activated = read_activated(client, waiting, netuid) if waiting else set()
        return waiting, eligible(snapshot.registrations, activated)
    except Exception as exc:
        raise InfraError(f"cannot read Generator registrations and activations: {type(exc).__name__}") from None
    finally:
        with contextlib.suppress(Exception):
            client.close()


def _issue(args: argparse.Namespace) -> int:
    if R2Store is None:
        raise ConfigError("boto3 is not installed: install the 'r2' extra")
    credentials: Mapping[str, str] | None = None
    if args.credentials_file:
        try:
            credentials = store_token(args.credentials_file)
        except V2Error as exc:
            # store_token deliberately names settings without showing values;
            # retain those controlled configuration diagnostics.
            if isinstance(exc, ConfigError):
                raise
            raise InfraError("cannot read the Generator link credentials") from None
        except Exception:
            raise ConfigError("cannot read the Generator link credentials") from None
    try:
        store = R2Store.from_env({**os.environ, **credentials} if credentials is not None else None)
    except V2Error as exc:
        if isinstance(exc, ConfigError):
            raise
        raise InfraError(f"cannot open the Generator link presigner: {type(exc).__name__}") from None
    except Exception:
        raise InfraError("cannot open the Generator link presigner") from None

    waiting, ready = _registered_and_activated(args.network, args.netuid)
    files = link_files(
        ready,
        GeneratorPresigner(store),
        int(time.time()),
        ttl_s=args.ttl_s,
        first_upload_final=args.first_upload_final,
    )
    removed = write_files(files, args.site_dir)
    mirrored = 0
    if args.mirror_r2:
        mirrored = mirror_files(files, store)
    index = json.loads(files[INDEX])
    print(
        f"links: issued {len(files) - 1} of {len(waiting)} registered non-zero hotkeys; "
        f"epoch {index['epoch']}; mode {index['upload_mode']}; "
        f"removed {removed}; mirrored {len(files) if args.mirror_r2 else 0}; stale mirrored {mirrored}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sn120_v2.generator.links",
        description="Issue activation-gated, deadline-bound sealed Generator upload links.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue", help="read the chain and publish one epoch's sealed links")
    issue.add_argument("--site-dir", required=True, help="existing public site root; receives generator/links and links.json")
    issue.add_argument(
        "--credentials-file",
        default="",
        help="optional KEY=VALUE R2 token file; otherwise all SN120_R2_* settings come from the environment",
    )
    issue.add_argument("--network", default="finney", help="read-only chain connection (default finney)")
    issue.add_argument("--netuid", type=int, default=NETUID, help=argparse.SUPPRESS)
    issue.add_argument("--ttl-s", type=int, default=EPOCH_SECONDS, help="requested link lifetime, capped at each epoch deadline")
    issue.add_argument(
        "--first-upload-final",
        action="store_true",
        help="prefer If-None-Match:* after verified support; otherwise explicitly overwrite_until_deadline",
    )
    issue.add_argument("--mirror-r2", action="store_true", help="also publish under public/generator/ in R2")
    issue.set_defaults(run=_issue)
    args = parser.parse_args(argv)
    if args.netuid != NETUID:
        print(f"issue failed: Generator is fixed to SN120 netuid {NETUID}", file=sys.stderr)
        return 1
    try:
        return args.run(args)
    except V2Error as exc:
        print(f"issue failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
