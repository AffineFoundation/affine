"""Cutover import: v2 starts from v1's state. Read-only, and it fails closed.

One command builds the two records that the controller starts from:

  GenesisSpec  the fresh MiMo genesis head and v1's validator hotkey
  V1Import     v1's last-10 kings, every seat and hotkey that v1 already
               served, the seats v1 moved to the back of the queue, and the
               seats that are owed a turn at the front

It reads v1's public files only (k1.json and mining.json). It never reads a
v1 host, the object store or a credential, and it never writes to the chain.
If v1's data is incomplete or inconsistent, it raises ``CutoverError`` and
writes nothing.

    python -m sn120_v2.cutover --dry-run
    python -m sn120_v2.cutover --master-digest <sha256> --validator-hotkey <ss58> --undecided front --out <dir>

The runbook is v2/CUTOVER.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import is_sha256, sha256_hex
from .errors import ConfigError, InfraError, V2Error, VerifyError
from .params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, GenesisSpec, Params
from .payout import weights
from .queue import BACK, FRONT, NORMAL, plan_sync, waiting, zone
from .types import ChainSnapshot, King, QueueEntry, V1Import

K1_URL = "https://affine.io/k1.json"
MINING_URL = "https://affine.io/mining.json"
K1_VERSION, QUEUE_VERSION, QUEUE_RULE = "sn120-k1-public-v1", "sn120-k1-queue-v2", "hotkey-registration-once"
DECIDED = ("green", "red")  # every other v1 verdict (pending, timeout-promote) is undecided: never a king
V1_PAYEES = 10  # v1 pays its last 10 distinct GREEN UIDs
MAX_EPOCH_GAP = 2  # mining.json may be at most this many epochs after the last epoch in k1.json
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

Seat = tuple[int, int]  # (uid, registration_block)


class CutoverError(VerifyError):
    """v1's public data is incomplete or inconsistent. Nothing is imported."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise CutoverError(message)


def _get(obj: Any, key: str, kind: type, where: str) -> Any:
    """A required field of one JSON type. A missing or mistyped field fails closed."""
    value = obj.get(key) if isinstance(obj, dict) else None
    ok = type(value) in ((int, float) if kind is float else (kind,))
    _need(ok and (kind is not float or math.isfinite(value)), f"{where}: field {key!r} is missing or is not a {kind.__name__}")
    return value


def _seat(row: Any, block_key: str, chain_block: int, where: str) -> Seat:
    """The (uid, registration block) of a v1 row. UID 0 and a block the chain has not reached fail closed."""
    uid, block = _get(row, "uid", int, where), _get(row, block_key, int, where)
    _need(uid >= 1 and 0 < block <= chain_block, f"{where}: seat ({uid}, {block}) is UID 0 or is not on the chain")
    return (uid, block)


def fifo(seat: Seat) -> tuple[int, int]:
    """Queue order: registration block, then UID."""
    return (seat[1], seat[0])


def check_ss58(address: Any) -> str:
    """Return ``address`` if it is a valid 32-byte SS58 address of network 42. One wrong character fails closed."""
    try:
        number = 0
        for char in address:
            number = number * 58 + _B58.index(char)
        raw = number.to_bytes(35, "big")
    except (TypeError, ValueError, OverflowError):
        raw = b""
    if len(raw) != 35 or raw[0] != 42 or hashlib.blake2b(b"SS58PRE" + raw[:33], digest_size=64).digest()[:2] != raw[33:]:
        raise ConfigError(f"{address!r} is not a valid SS58 address")
    return address


def build_genesis(master_digest: Any, validator_hotkey: Any) -> GenesisSpec:
    """The fresh MiMo genesis. The operator gives the fp32 master digest and the validator hotkey; neither has a default."""
    if not is_sha256(master_digest):
        raise ConfigError("the fp32 master digest must be the lowercase 64-hex sha256 digest of the real master files")
    return GenesisSpec(GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, GENESIS_BASE_HASH, master_digest, check_ss58(validator_hotkey))


def build_import(
    k1: Mapping[str, Any],
    mining: Mapping[str, Any],
    *,
    now: float,
    undecided: str | None = None,
    max_age_s: int = 3600,
    source: str = "",
) -> V1Import:
    """Check v1's public state and turn it into a ``V1Import``.

    ``undecided`` is the operator's decision for turns that v1 took but never
    judged: ``"front"`` (not charged, first in the v2 queue) or ``"served"``
    (charged, no new turn). With such turns and no decision, the import stops.
    """
    queue = _get(k1, "queue", dict, "k1.json")
    _need(k1.get("version") == K1_VERSION and queue.get("version") == QUEUE_VERSION, "k1.json has an unknown format version")
    _need(queue.get("enabled") is True and queue.get("queue_rule") == QUEUE_RULE, "the v1 queue is off or uses an unknown rule")
    age = now - _get(k1, "generated_at", float, "k1.json")
    _need(-300 <= age <= max_age_s, f"k1.json is {age:.0f} s old (limit {max_age_s} s): refusing a stale read")
    chain_block = _get(queue, "chain_block", int, "queue")

    # 1. What v1 served. The served lists and the turn history must agree.
    pairs = [{"uid": p[0], "block": p[1]} if type(p) is list and len(p) == 2 else p for p in _get(queue, "served_registrations", list, "queue")]
    served_regs = {_seat(pair, "block", chain_block, "served_registrations") for pair in pairs}
    _need(len(served_regs) == len(pairs) == _get(queue, "registrations_served", int, "queue"), "served registration count mismatch")
    hotkeys = _get(queue, "served_hotkeys", list, "queue")
    _need(all(type(hotkey) is str for hotkey in hotkeys), "served_hotkeys must hold addresses")
    served_hotkeys = set(hotkeys)
    _need(len(served_hotkeys) == len(hotkeys) == _get(queue, "hotkeys_served", int, "queue"), "served hotkey count mismatch")
    history: dict[int, tuple[Any, str, Any]] = {}  # epoch -> (uid, hotkey, registration block)
    last_turn: dict[str, int] = {}  # hotkey -> epoch of its latest turn
    for row in _get(queue, "served", list, "queue"):
        epoch, hotkey = _get(row, "epoch_index", int, "served turn"), _get(row, "hotkey", str, "served turn")
        _need(not history or epoch > max(history), "served turns are not in epoch order")
        if row.get("reg_block") is not None:  # turns from before the registration rule carry no block
            _need(_seat(row, "reg_block", chain_block, "served turn") in served_regs, f"turn e{epoch} is not in served_registrations")
        history[epoch], last_turn[hotkey] = (row.get("uid"), hotkey, row.get("reg_block")), epoch
    _need(set(last_turn) == served_hotkeys, "served_hotkeys does not match the turn history")
    window = _get(k1, "turns", list, "k1.json")
    _need(bool(window), "k1.json has no turns")
    first, last = _get(window[0], "epoch_index", int, "turn"), _get(window[-1], "epoch_index", int, "turn")
    turns = [turn for turn in window if isinstance(turn, dict) and turn.get("status") == "turn"]
    shown = {turn.get("epoch_index"): (turn.get("uid"), turn.get("hotkey"), turn.get("reg_block")) for turn in turns}
    _need(shown == {epoch: row for epoch, row in history.items() if epoch >= first}, "the public turns do not match the served turns")

    # 2. Kings. v1's payees are its last 10 distinct GREEN UIDs; derive them again from the public turns.
    payees = sorted(_get(k1, "payees", list, "k1.json"), key=lambda payee: -_get(payee, "green_epoch", int, "payee"))
    listed = [(p["green_epoch"], _get(p, "uid", int, "payee"), check_ss58(p.get("hotkey"))) for p in payees]
    uids = {uid for _, uid, _ in listed}
    _need(1 <= len(listed) <= V1_PAYEES and len(uids) == len(listed) and 0 not in uids, "payees must be 1 to 10 distinct miner UIDs")
    _need(all(hotkey in served_hotkeys for _, _, hotkey in listed), "a payee never had a turn")
    derived: list[tuple[int, int, str]] = []
    for turn in reversed(turns):
        if turn.get("verdict") == "green" and len(derived) < V1_PAYEES and turn["uid"] not in {uid for _, uid, _ in derived}:
            derived.append((turn["epoch_index"], turn["uid"], turn["hotkey"]))
    _need(
        listed[: len(derived)] == derived and all(epoch < first for epoch, _, _ in listed[len(derived) :]),
        "payees do not match the GREEN turns in k1.json",
    )
    when = {turn["epoch_index"]: turn.get("turn_at") for turn in turns}
    score = {p["green_epoch"]: p.get("z") for p in payees}
    kings = tuple(
        King(uid, hotkey, f"v1-e{epoch}", "", int(when.get(epoch) or 0), float(score[epoch]) if type(score[epoch]) in (int, float) else 0.0)
        for epoch, uid, hotkey in reversed(listed)
    )

    # 3. Seats owed a turn: v1's open retry grants, and turns that v1 took but never decided.
    owed = {_seat(row, "registration_block", chain_block, "grant"): _get(row, "hotkey", str, "grant") for row in _get(queue, "grants", list, "queue")}
    open_turns = [turn for turn in turns if turn.get("verdict") not in DECIDED and last_turn[turn["hotkey"]] == turn["epoch_index"]]
    if open_turns and undecided not in ("front", "served"):
        raise CutoverError(
            f"{len(open_turns)} v1 turn(s) have no verdict (epochs {[turn['epoch_index'] for turn in open_turns]}). Decide with "
            "--undecided front (not charged, first in the v2 queue) or --undecided served (charged, no new turn)"
        )
    if undecided == "front":
        owed |= {_seat(turn, "reg_block", chain_block, "undecided turn"): turn["hotkey"] for turn in open_turns}

    # 4. Seats that v1 lists as waiting must not be served.
    back = _get(queue, "expired_back_of_queue", list, "queue")
    demoted = {_seat(row, "registration_block", chain_block, "back-of-queue seat") for row in back}
    for row in back + [row for row in _get(queue, "next", list, "queue") if not (isinstance(row, dict) and row.get("grant"))]:
        seat, hotkey = _seat(row, "registration_block", chain_block, "waiting seat"), _get(row, "hotkey", str, "waiting seat")
        _need(seat not in served_regs and hotkey not in served_hotkeys, f"waiting seat {seat} is also listed as served")
    _need(len(demoted) == len(back) == _get(queue, "expired_back_of_queue_count", int, "queue"), "back-of-queue count mismatch")
    _need(len(back) <= _get(queue, "registrations_unserved", int, "queue"), "more back-of-queue seats than unserved registrations")

    # 5. mining.json must describe the same network, the same epoch and a head that k1.json knows.
    match = re.search(r"-e(\d+)$", str(mining.get("epoch")))
    _need(mining.get("netuid") == Params().netuid, "mining.json is for another subnet")
    _need(match is not None and 0 <= int(match[1]) - last <= MAX_EPOCH_GAP, "mining.json and k1.json are not from the same epoch")
    head = mining.get("checkpoint")
    _need(is_sha256(head) and any(head in (t.get("head"), t.get("candidate")) for t in window), "k1.json does not know the v1 head in mining.json")

    return V1Import(
        source=f"{source}; v1_last_epoch={last}; v1_head={head}; undecided={undecided or 'none'}".lstrip("; "),
        chain_block=chain_block,
        kings=kings,
        served_regs=tuple(sorted(served_regs - set(owed))),
        served_hotkeys=tuple(sorted(served_hotkeys - set(owed.values()))),
        demoted=tuple(sorted(demoted, key=fifo)),
        front=tuple(sorted(owed, key=fifo)),
    )


def preview_queue(imp: V1Import, snapshot: ChainSnapshot, params: Params) -> list[QueueEntry]:
    """The queue v2 starts with for one chain snapshot, in turn order: front, then normal, then back."""
    _need(snapshot.block >= imp.chain_block, "the chain snapshot is older than v1's last chain read")
    plan = plan_sync({}, imp.served_regs, imp.served_hotkeys, snapshot, params.netuid, params.burn_uid)
    front, demoted = set(imp.front), set(imp.demoted)
    return waiting({reg.key: QueueEntry(reg, front=reg.key in front, demoted=reg.key in demoted) for reg in plan.added})


def chain_report(imp: V1Import, genesis: GenesisSpec | None, snapshot: ChainSnapshot, params: Params) -> list[str]:
    """Starting queue and first weights for one chain snapshot. Refuses a start that would pay no king."""
    seats, registered = preview_queue(imp, snapshot, params), snapshot.hotkeys()
    _need(genesis is None or genesis.validator_hotkey in registered.values(), "the validator hotkey is not registered on the subnet")
    table = weights(imp.kings, registered, params)
    _need(table != {params.burn_uid: 1.0}, "no imported king is still registered: v2 would burn all weight")
    count = [sum(zone(seat) == z for seat in seats) for z in (FRONT, NORMAL, BACK)]
    lines = [f"CHAIN at block {snapshot.block}: {len(seats)} seats wait ({count[0]} front, {count[1]} normal, {count[2]} back). First 20:"]
    lines += [f"    {i}. uid {s.reg.uid} block {s.reg.registration_block} {s.reg.hotkey}" for i, s in enumerate(seats[:20], 1)]
    return lines + [f"  first weights: {len(table)} UIDs, {1 / len(table):.4f} each: {sorted(table)}"]


def describe(imp: V1Import, genesis: GenesisSpec | None, k1: Mapping[str, Any]) -> list[str]:
    """What the import holds, as text for the operator."""
    turns = {(turn.get("uid"), turn.get("reg_block")): turn for turn in k1["turns"]}
    lines = ["GENESIS: not built (needs --master-digest and --validator-hotkey)"]
    if genesis is not None:
        lines = [f"GENESIS {genesis.model_repo} @ {genesis.model_revision}", f"  base hash {genesis.base_hash}"]
        lines += [f"  fp32 master digest {genesis.base_fp32_master_digest}", f"  validator hotkey {genesis.validator_hotkey}"]
    lines += [f"V1 IMPORT at v1 chain block {imp.chain_block}", f"  source: {imp.source}", f"  kings, oldest first ({len(imp.kings)}):"]
    lines += [f"    {king.round_id} uid {king.uid} {king.hotkey} z={king.z:.3f}" for king in imp.kings]
    lines.append(f"  served: {len(imp.served_regs)} registrations and {len(imp.served_hotkeys)} hotkeys get no new turn")
    lines.append(f"  back of the queue: {len(imp.demoted)} seats (v1 found no valid file for too long)")
    lines.append(f"  front of the queue: {len(imp.front)} seats owed a turn, in turn order:")
    for uid, block in imp.front:
        turn = turns.get((uid, block), {})
        batch = f"v1 e{turn.get('epoch_index', '?')} took {turn.get('tasks', '?')} prompts, {turn.get('rollouts', '?')} rollouts"
        lines.append(f"    uid {uid} block {block} {turn.get('hotkey', '?')} ({batch})")
    return lines


def read_source(location: str) -> bytes:
    """Read a public https URL or a local file. A failed read is an ``InfraError``: nothing is imported."""
    try:
        if location.startswith("https://"):
            request = urllib.request.Request(location, headers={"User-Agent": "sn120-v2-cutover"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        return Path(location).read_bytes()
    except OSError as exc:
        raise InfraError(f"cannot read {location}: {exc}") from exc


def run(args: argparse.Namespace) -> list[str]:
    raw = {"k1.json": read_source(args.k1), "mining.json": read_source(args.mining)}
    try:
        k1, mining = (json.loads(data) for data in raw.values())
    except ValueError as exc:
        raise CutoverError(f"a v1 file is not JSON: {exc}") from exc
    source = "; ".join(f"{name} sha256={sha256_hex(data)}" for name, data in raw.items())
    imp = build_import(k1, mining, now=time.time(), undecided=args.undecided, max_age_s=args.max_age_s, source=source)
    genesis = build_genesis(args.master_digest, args.validator_hotkey) if args.out or args.master_digest or args.validator_hotkey else None
    lines = describe(imp, genesis, k1)
    if args.chain:
        lines += chain_report(imp, genesis, ChainSnapshot.from_json(read_source(args.chain)), Params())
    if args.out is None:
        return lines + ["DRY RUN: nothing was written."]
    out = Path(args.out)
    if out.exists():
        raise ConfigError(f"{out} exists already: refusing to replace an earlier import")
    out.mkdir(parents=True)
    files = raw | {"genesis.json": genesis.to_json(), "v1_import.json": imp.to_json()}
    for name, data in files.items():
        (out / name).write_bytes(data)
    return lines + [f"WROTE {out / name} sha256={sha256_hex(data)}" for name, data in files.items()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.cutover", description="Build the v2 genesis and the import of v1's state.")
    parser.add_argument("--k1", default=K1_URL, help="v1 k1.json: https URL or file (default: %(default)s)")
    parser.add_argument("--mining", default=MINING_URL, help="v1 mining.json: https URL or file (default: %(default)s)")
    parser.add_argument("--chain", help="file with a ChainSnapshot as JSON: also check and print the starting queue and weights")
    parser.add_argument("--master-digest", help="sha256 digest of the genesis fp32 master files")
    parser.add_argument("--validator-hotkey", help="SS58 address of the validator hotkey that v1 used")
    parser.add_argument("--undecided", choices=("front", "served"), help="what to do with v1 turns that have no verdict")
    parser.add_argument("--max-age-s", type=int, default=3600, help="oldest k1.json accepted, in seconds (default: %(default)s)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print what would be imported; write nothing")
    mode.add_argument("--out", help="new directory for genesis.json, v1_import.json and the v1 files that were read")
    args = parser.parse_args(argv)
    try:
        print("\n".join(run(args)))
    except V2Error as exc:
        print(f"CUTOVER IMPORT REFUSED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
