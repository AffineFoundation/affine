"""Mechanism-1 Generator weight setter.

One invocation makes at most one timelocked ``SetWeights`` commit on SN120.
The CLI is a dry run unless ``--execute`` is present.  The two deployment timer
instants are 35:20 and 50:20 past each UTC hour; this module contains no timer
or service installation and never changes mechanism 0.

The decision function is pure: callers inject one chain snapshot, the decoded
local payout document, the clock value and the activity/cooldown observations.
SDK, network and wallet construction live only in the I/O runner.  A successful
commit is remembered in ``runs.jsonl``; failed and dry-run attempts are not.

The historical ``/home/const/sn120-slots-mech1/slots_mech1_weights.py`` was not
available when this implementation was written.  The gates below therefore
implement the explicitly supplied control-plane contract rather than claiming
to be a byte-for-byte copy of that unavailable file.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..canonical import canonical_json, is_sha256
from ..errors import ContractError
from ..payout import check_weights
from .constants import (
    BURN_UID,
    MECH_ID,
    NETUID,
    PAYOUT_MAX_AGE_SECONDS,
    PAYOUT_SCHEMA_VERSION,
    PAYOUT_WINDOW_SECONDS,
)
from .payout import Claim, payout_weights
from .protocol import valid_task_id

VERSION_KEY = 0
TIMER_SECONDS_UTC = frozenset({35 * 60 + 20, 50 * 60 + 20})
MIN_UTC_MINUTE = 25
MECH1_MIN_LAST_UPDATE_BLOCKS = 240
SUCCESS_COOLDOWN_SECONDS = 45 * 60
MAX_PAYOUT_BYTES = 16 * 1024 * 1024
WALLET_NAME = "default"
WALLET_HOTKEY = "default"
ACTIVE_SYSTEMD_STATES = frozenset({"active", "activating", "reloading"})
# Both the current assessment writer and the newer v2 writer are mechanism 0.
# The legacy burn unit is included as a guard only; this module never starts it.
MECH0_WRITER_UNITS = tuple(
    f"{name}.{kind}"
    for name in ("affine-transition-weights", "sn120-v2-weights", "affine-hourly-burn")
    for kind in ("service", "timer")
)


@dataclass(frozen=True, slots=True)
class ChainState:
    """The complete chain observations needed by :func:`decide`."""

    block: int
    hotkeys: Mapping[int, str]
    mech1_last_update: int


@dataclass(frozen=True, slots=True)
class Decision:
    """A side-effect-free instruction for the runner."""

    status: str  # skip, dry_run or submit
    reason: str
    weights: Mapping[int, float] = field(default_factory=dict)
    burn: bool = False
    block: int = 0
    payout_updated_at: float | None = None

    @property
    def uids(self) -> list[int]:
        return sorted(self.weights)

    @property
    def values(self) -> list[float]:
        """Floating-point shares; ``SetWeights`` performs the SDK u16 conversion."""
        return [self.weights[uid] for uid in self.uids]


@dataclass(frozen=True, slots=True)
class _Payout:
    document: Mapping[str, Any] | None
    error: str = ""
    sha256: str = ""


def _timestamp(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(f"{where} must be a finite Unix timestamp")
    try:
        dt.datetime.fromtimestamp(value, dt.timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ContractError(f"{where} is outside the supported UTC range") from exc
    return float(value)


def _utc(now: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(_timestamp(now, "now"), dt.timezone.utc)


def _skip(reason: str, *, block: int = 0) -> Decision:
    return Decision("skip", reason, block=block)


def _gates(
    chain: ChainState | None,
    *,
    now: float,
    mech0_writer_active: bool,
    last_success_at: float | None,
    ignore_window: bool = False,
) -> Decision | None:
    """Allow scheduler jitter without weakening any non-time safety gate."""
    if type(mech0_writer_active) is not bool:
        return _skip("mechanism-0 writer activity could not be established")
    if mech0_writer_active:
        return _skip("mechanism-0 weight writer is active")
    stamp = _utc(now)
    if not ignore_window:
        if stamp.minute < MIN_UTC_MINUTE:
            return _skip("current UTC minute is before 25")
        elapsed = stamp.minute * 60 + stamp.second
        if not any(instant <= elapsed < instant + 240 for instant in TIMER_SECONDS_UTC):
            return _skip("outside the four-minute windows after 35:20 and 50:20 UTC")
    if last_success_at is not None:
        try:
            last = _timestamp(last_success_at, "last successful write")
        except ContractError:
            return _skip("last successful write timestamp is invalid")
        if stamp.timestamp() - last < SUCCESS_COOLDOWN_SECONDS:
            return _skip("own successful mechanism-1 write was less than 45 minutes ago")
    if chain is not None:
        if type(chain.block) is not int or chain.block < 0:
            return _skip("chain head is invalid")
        if type(chain.mech1_last_update) is not int or not 0 <= chain.mech1_last_update <= chain.block:
            return _skip("mechanism-1 LastUpdate is invalid", block=chain.block if type(chain.block) is int else 0)
        if chain.block - chain.mech1_last_update < MECH1_MIN_LAST_UPDATE_BLOCKS:
            return _skip("mechanism-1 LastUpdate is fewer than 240 blocks old", block=chain.block)
        if not isinstance(chain.hotkeys, Mapping) or not isinstance(chain.hotkeys.get(BURN_UID), str) or not chain.hotkeys[BURN_UID]:
            return _skip("UID 0 owner registration is unavailable", block=chain.block)
        for uid, hotkey in chain.hotkeys.items():
            if type(uid) is not int or uid < 0 or not isinstance(hotkey, str) or not hotkey:
                return _skip("chain registration mapping is invalid", block=chain.block)
    return None


def _burn(reason: str, chain: ChainState, updated_at: float | None = None) -> Decision:
    return Decision("submit", reason, {BURN_UID: 1.0}, True, chain.block, updated_at)


def _claims(document: Mapping[str, Any], now: float) -> tuple[Claim, ...]:
    """Validate protocol payout and its optional public informational fields."""
    required = {"version", "updated_at", "window_seconds", "max_age_seconds", "claims"}
    allowed = required | {"weights", "epochs", "window"}
    if not required <= set(document) or set(document) - allowed:
        raise ContractError("payout schema or fields are invalid")
    if document.get("version") != PAYOUT_SCHEMA_VERSION:
        raise ContractError("payout schema or fields are invalid")
    if document.get("window_seconds") != PAYOUT_WINDOW_SECONDS or type(document.get("window_seconds")) is not int:
        raise ContractError("payout window is invalid")
    if document.get("max_age_seconds") != PAYOUT_MAX_AGE_SECONDS or type(document.get("max_age_seconds")) is not int:
        raise ContractError("payout maximum age is invalid")
    updated_at = _timestamp(document.get("updated_at"), "payout updated_at")
    # Future timestamps are checked by the freshness branch in decide().
    rows = document.get("claims")
    if not isinstance(rows, list):
        raise ContractError("payout claims must be a list")
    result: list[Claim] = []
    seen: set[str] = set()
    fields = {"task_id", "uid", "hotkey", "uploaded_at", "file_sha256"}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ContractError("payout claim fields are invalid")
        task_id = row["task_id"]
        if not valid_task_id(task_id) or task_id in seen:
            raise ContractError("payout task IDs are invalid or duplicated")
        seen.add(task_id)
        if type(row["uid"]) is not int or row["uid"] < 0:
            raise ContractError("payout claim UID is invalid")
        if not isinstance(row["hotkey"], str) or not row["hotkey"]:
            raise ContractError("payout claim hotkey is invalid")
        uploaded_at = _timestamp(row["uploaded_at"], "claim uploaded_at")
        if uploaded_at > now:
            raise ContractError("payout claim timestamp is in the future")
        if not is_sha256(row["file_sha256"]):
            raise ContractError("payout claim SHA-256 is invalid")
        result.append(Claim(task_id, row["uid"], row["hotkey"], uploaded_at, row["file_sha256"]))
    return tuple(result)


def decide(
    chain: ChainState,
    payout_document: Mapping[str, Any] | None,
    *,
    now: float,
    mech0_writer_active: bool,
    last_success_at: float | None = None,
    execute: bool = False,
    wallet_hotkey: str | None = None,
    ignore_window: bool = False,
) -> Decision:
    """Return the sole permitted action for this invocation, without any I/O.

    Missing or malformed payout data and stale/future payout freshness select a
    full UID-0 burn. Changed or missing earning hotkeys redirect only their own
    equal claim shares to UID 0; other miners are never silently topped up.
    Chain/identity uncertainty instead skips: it is not evidence for a burn.
    """
    try:
        blocked = _gates(
            chain,
            now=now,
            mech0_writer_active=mech0_writer_active,
            last_success_at=last_success_at,
            ignore_window=ignore_window,
        )
    except ContractError:
        return _skip("current timestamp is invalid")
    if blocked is not None:
        return blocked
    if not isinstance(chain.hotkeys, Mapping) or not isinstance(chain.hotkeys.get(BURN_UID), str) or not chain.hotkeys[BURN_UID]:
        return _skip("UID 0 owner registration is unavailable", block=chain.block)
    if wallet_hotkey is not None and wallet_hotkey != chain.hotkeys[BURN_UID]:
        return _skip("default/default wallet is not the hotkey registered at UID 0", block=chain.block)

    current = _timestamp(now, "now")
    if payout_document is None:
        decision = _burn("payout.json is missing or unreadable; burning UID 0", chain)
    elif not isinstance(payout_document, Mapping):
        decision = _burn("payout.json is invalid; burning UID 0", chain)
    else:
        updated_at: float | None = None
        try:
            claims = _claims(payout_document, current)
            updated_at = _timestamp(payout_document["updated_at"], "payout updated_at")
        except (ContractError, KeyError, TypeError, ValueError):
            decision = _burn("payout.json is invalid; burning UID 0", chain)
        else:
            if updated_at > current or current - updated_at > PAYOUT_MAX_AGE_SECONDS:
                decision = _burn("payout.json is stale or future-dated; burning UID 0", chain, updated_at)
            else:
                try:
                    table = payout_weights(claims, chain.hotkeys, now=current, updated_at=updated_at)
                    table = check_weights(table)
                    if any(uid not in chain.hotkeys for uid in table):
                        raise ContractError("payout selected an unregistered UID")
                except (ContractError, TypeError, ValueError):
                    decision = _burn("payout weights are invalid; burning UID 0", chain, updated_at)
                else:
                    changed = sorted({claim.uid for claim in claims if chain.hotkeys.get(claim.uid) != claim.hotkey})
                    redirected = [uid for uid in changed if table.get(BURN_UID, 0.0) > 0]
                    burn = table == {BURN_UID: 1.0}
                    if burn:
                        reason = "every earning UID hotkey changed or is unregistered (UIDs " + ",".join(map(str, redirected)) + "); burning UID 0"
                    elif redirected:
                        reason = "valid trailing Generator payout; shares of changed/unregistered UIDs " + ",".join(map(str, redirected)) + " sent to UID 0"
                    else:
                        reason = "valid trailing Generator payout"
                    decision = Decision("submit", reason, table, burn, chain.block, updated_at)
    if not execute:
        return Decision(
            "dry_run",
            "dry run; " + decision.reason,
            decision.weights,
            decision.burn,
            decision.block,
            decision.payout_updated_at,
        )
    return decision


def _load_sdk() -> Any:
    """Import bittensor only when the runner actually needs chain access."""
    try:
        return importlib.import_module("bittensor")
    except ImportError as exc:
        raise RuntimeError("bittensor SDK is unavailable") from exc


def storage_index(netuid: int, mechid: int) -> int:
    """Finney's single NetUidStorageIndex for a subnet mechanism."""
    if type(netuid) is not int or type(mechid) is not int or netuid < 0 or mechid < 0:
        raise ValueError("netuid and mechid must be nonnegative integers")
    return mechid * 4096 + netuid


def read_chain(chain: Any, sdk: Any) -> ChainState:
    """Read registrations and mechanism-1 LastUpdate at the current head."""
    block = int(chain.block)
    storage = sdk.storage.SubtensorModule
    rows = chain.query_map(storage.Keys, [NETUID])
    hotkeys = {int(uid): str(hotkey) for uid, hotkey in dict(rows).items()}
    if sorted(hotkeys) != list(range(len(hotkeys))):
        raise RuntimeError("registration snapshot is not a complete UID vector")
    updates = chain.query(storage.LastUpdate, [storage_index(NETUID, MECH_ID)])
    if not isinstance(updates, Sequence) or isinstance(updates, (str, bytes)) or not updates:
        raise RuntimeError("mechanism-1 LastUpdate did not return the UID vector")
    if len(hotkeys) != len(updates):
        raise RuntimeError("registration and mechanism-1 LastUpdate vectors differ in length")
    return ChainState(block, hotkeys, int(updates[BURN_UID]))


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def load_payout(path: str | Path) -> _Payout:
    """Read only the operator-supplied local payout.json; never fetch a URL."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_PAYOUT_BYTES:
            raise ValueError("payout document is too large")
        raw = path.read_bytes()
        document = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError, TypeError):
        return _Payout(None, "missing or invalid")
    if not isinstance(document, Mapping):
        return _Payout(None, "missing or invalid", hashlib.sha256(raw).hexdigest())
    return _Payout(document, sha256=hashlib.sha256(raw).hexdigest())


def last_success_at(path: str | Path) -> float | None:
    """Read only successful execute records, never dry runs or failed writes."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    latest: float | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("runs.jsonl row is not an object")
        if row.get("status") != "submitted":
            continue
        if row.get("execute") is not True or row.get("mechid") != MECH_ID or row.get("netuid") != NETUID:
            continue
        observed = _timestamp(row.get("observed_at"), "runs.jsonl observed_at")
        latest = observed if latest is None else max(latest, observed)
    return latest


def _atomic_append(path: Path, row: Mapping[str, Any]) -> None:
    """Atomically replace runs.jsonl with old bytes plus exactly one JSON row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".append.lock")
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = path.read_bytes() if path.exists() else b""
        if previous and not previous.endswith(b"\n"):
            raise ValueError("runs.jsonl does not end at a JSON row boundary")
        encoded = canonical_json(dict(row)) + b"\n"
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as scratch:
                temporary_name = scratch.name
                scratch.write(previous)
                scratch.write(encoded)
                scratch.flush()
                os.fsync(scratch.fileno())
            os.replace(temporary_name, path)
            temporary_name = ""
        finally:
            if temporary_name:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_name)


@contextlib.contextmanager
def _one_runner(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".writer.lock"), "a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def _mech0_writer_active() -> bool:
    """Check known mech-0 units read-only; an uncertain probe fails closed."""
    for unit in MECH0_WRITER_UNITS:
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", unit],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("cannot establish mechanism-0 writer activity") from exc
        if result.stdout.strip() in ACTIVE_SYSTEMD_STATES:
            return True
    return False


def _base_report(now: float, execute: bool) -> dict[str, Any]:
    stamp = _utc(now)
    return {
        "observed_at": stamp.timestamp(),
        "timestamp": stamp.isoformat().replace("+00:00", "Z"),
        "execute": execute,
        "netuid": NETUID,
        "mechid": MECH_ID,
        "version_key": VERSION_KEY,
    }


def _decision_report(decision: Decision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "block": decision.block,
        "burn": decision.burn,
        "uids": decision.uids,
        "weights": {str(uid): decision.weights[uid] for uid in decision.uids},
        "payout_updated_at": decision.payout_updated_at,
    }


def run_once(
    *,
    payout_path: str | Path,
    runs_path: str | Path,
    execute: bool = False,
    network: str = "finney",
    wallet_name: str = WALLET_NAME,
    wallet_hotkey_name: str = WALLET_HOTKEY,
    wallet_path: str | Path | None = None,
    now: float | None = None,
    sdk: Any = None,
    chain: Any = None,
    wallet: Any = None,
    mech0_active: Callable[[], bool] | bool | None = None,
    successful_at: float | None = None,
    ignore_window: bool = False,
) -> dict[str, Any]:
    """Run one attempt with injectable chain/SDK fakes and lazy production I/O."""
    if type(execute) is not bool:
        raise ValueError("execute must be a boolean")
    observed = time.time() if now is None else _timestamp(now, "now")
    payout_path, runs_path = Path(payout_path), Path(runs_path)
    report = _base_report(observed, execute)

    def record(extra: Mapping[str, Any]) -> dict[str, Any]:
        report.update(extra)
        _atomic_append(runs_path, report)
        return dict(report)

    with _one_runner(runs_path) as alone:
        if not alone:
            return record({"status": "skip", "reason": "another mechanism-1 writer holds the lock"})
        try:
            previous = last_success_at(runs_path) if successful_at is None else successful_at
            probe = _mech0_writer_active if mech0_active is None else mech0_active
            active = probe() if callable(probe) else probe
            early = _gates(None, now=observed, mech0_writer_active=active, last_success_at=previous, ignore_window=ignore_window)
            if early is not None:
                return record(_decision_report(early))

            # No SDK import, wallet read or network construction occurs before
            # the local activity/time/cooldown gates have all passed.
            if sdk is None:
                sdk = _load_sdk()
            if chain is None:
                chain = sdk.Subtensor(network)
            if execute and wallet is None:
                options: dict[str, Any] = {"name": wallet_name, "hotkey": wallet_hotkey_name}
                if wallet_path is not None:
                    options["path"] = str(wallet_path)
                wallet = sdk.Wallet(**options)
            signer = str(wallet.hotkey.ss58_address) if execute else None
            state = read_chain(chain, sdk)
            loaded = load_payout(payout_path)
            decision = decide(
                state,
                loaded.document,
                now=observed,
                mech0_writer_active=active,
                last_success_at=previous,
                execute=execute,
                wallet_hotkey=signer,
                ignore_window=ignore_window,
            )
            if loaded.sha256:
                report["payout_sha256"] = loaded.sha256
            if decision.status != "submit":
                return record(_decision_report(decision))
            check_weights(decision.weights)  # an empty/all-zero table can never reach SetWeights

            # SetWeights owns u16 normalization and timelocked commit creation.
            # These are positional values in sorted UID order, not pre-rounded
            # integers.  There is deliberately no mechid-0 code path.
            intent = sdk.SetWeights(
                netuid=NETUID,
                uids=decision.uids,
                weights=decision.values,
                mechid=MECH_ID,
                version_key=VERSION_KEY,
            )
            plan = chain.plan(intent, wallet)
            if not bool(getattr(plan, "ok", False)):
                return record(_decision_report(decision) | {"status": "failed", "reason": "chain plan refused mechanism-1 commit"})
            if str(getattr(plan, "signer_address", "")) != state.hotkeys[BURN_UID]:
                return record(_decision_report(decision) | {"status": "failed", "reason": "chain plan signer is not UID 0"})
            result = chain.execute(
                intent,
                wallet,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                retries=0,
            )
            if not bool(getattr(result, "success", False)):
                return record(_decision_report(decision) | {"status": "failed", "reason": "mechanism-1 commit was not successful"})
            return record(
                _decision_report(decision)
                | {
                    "status": "submitted",
                    "reason": decision.reason,
                    "block_hash": str(getattr(result, "block_hash", "") or ""),
                }
            )
        except Exception as exc:
            # Error text from wallet/RPC libraries can contain sensitive local
            # material.  Keep only its exception type in stdout and runs.jsonl.
            return record(
                {
                    "status": "failed",
                    "reason": f"mechanism-1 writer failed safely ({type(exc).__name__})",
                    "submission_outcome": "unknown" if execute else "not_attempted",
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sn120_v2.generator.weights",
        description="One SN120 mechanism-1 Generator weight attempt; dry run unless --execute is supplied.",
    )
    parser.add_argument("--payout", type=Path, required=True, help="local Generator payout.json (no URL fetching)")
    parser.add_argument("--runs", type=Path, help="atomic run journal; default: runs.jsonl beside payout.json")
    parser.add_argument("--network", default="finney", help="chain network or node address (default: finney)")
    parser.add_argument("--execute", action="store_true", help="actually plan and submit one mechid=1 timelocked commit")
    parser.add_argument("--ignore-window", action="store_true", help="manual operator run: skip only the minute>=25 and timer-window gates")

    parser.add_argument("--wallet-name", default=WALLET_NAME)
    parser.add_argument("--wallet-hotkey", default=WALLET_HOTKEY)
    parser.add_argument("--wallet-path", type=Path)
    args = parser.parse_args(argv)
    report = run_once(
        payout_path=args.payout,
        runs_path=args.runs or args.payout.with_name("runs.jsonl"),
        execute=args.execute,
        ignore_window=args.ignore_window,
        network=args.network,
        wallet_name=args.wallet_name,
        wallet_hotkey_name=args.wallet_hotkey,
        wallet_path=args.wallet_path,
    )
    print(canonical_json(report).decode("utf-8"))
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
