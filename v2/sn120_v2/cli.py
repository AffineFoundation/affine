"""Command line: ``python -m sn120_v2 <command>``.

  sim         a whole validator on CPU with fake training and scoring; no GPU, no network
  bootstrap   first start: pin the parameters, write the genesis head, the state carried over
              from v1 and the first queue, then publish llms.txt and mining.json
  run         the round loop
  status      read-only summary; with ``--stack`` also the count of queued files per verdict rule
  publish     read-only for the journal: write llms.txt and mining.json again
  round       print the summary of a closed round as a short text to paste
  set-params  pin new parameters (stop ``run`` first: this command takes over the journal)
  end-round   end the open round now with no decision, when its box is dead: nobody is charged, and its
              seats are first in the next round (this command takes over the journal). With a box result
              of that round in the store it ends nothing and exits with 3: ``run`` judges that result
  release-seat  undo what a fault of the box did to seats: a held seat takes a turn again, and the count of
              failures of a seat is 0 again (stop ``run`` first: this command takes over the journal)
  front-seats  between rounds: send the waiting seats of the named hotkeys to the front and clear ``demoted``;
              nothing is charged or counted. ``--dryrun`` shows the change and writes nothing (stop ``run`` first:
              without ``--dryrun`` this command takes over the journal). Every reader of the journal needs this code
  force-sync  for a chain sync that ``run`` refuses because more than a quarter of the seats would leave:
              show the changes that two chain reads agree on; with ``--confirm-remove N`` apply them once
              (stop ``run`` first: with ``--confirm-remove`` this command takes over the journal)

``--stack module:function`` names the builder of the parts. The real ones are
``sn120_v2.assembly:stack``; the CPU ones are ``sn120_v2.sim:stack``.
Nothing here has a default path. The state directory is always an argument.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from . import layout
from .controller import Controller, Stack, StackContext
from .errors import ConfigError, FencedError, InfraError, V2Error
from .journal import JOURNAL_FILE, Journal, Mirror
from .params import GenesisSpec, Params, load_params
from .payout import payees
from .public import phase, round_text, snapshot
from .queue import BACK, FRONT, FrontMove, SyncPlan, plan_front, plan_sync, waiting, zone
from .sim import SimWorld, sim_params
from .state import SEATS_FRONTED, State, load
from .store import verify_files
from .types import ChainSnapshot, V1Import, Verdict


def now() -> int:
    return int(time.time())


def summary(state: State, params: Params) -> dict[str, Any]:
    return {
        "phase": phase(state),
        "head": state.head.version_id,
        "head_folded": state.head.folded,
        "heads": len(state.heads),
        "rounds_closed": state.round_seq - (1 if state.round else 0),
        "last_outcomes": [record.outcome for record in state.rounds[-10:]],
        "queue": len(state.queue),
        "served": len(state.served_regs),
        "kings": [king.uid for king in payees(state.kings, params.kings_window)],
        "params_hash": params.params_hash,
        "journal_records": state.last_seq + 1,
    }


def load_stack(spec: str, context: StackContext) -> Stack:
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ConfigError(f"--stack must be module:function, got {spec!r}")
    builder: Callable[[StackContext], Stack] = getattr(importlib.import_module(module_name), function_name)
    return builder(context)


def open_stack(args: argparse.Namespace, command: str) -> tuple[StackContext, Stack]:
    given = load_params(args.params) if getattr(args, "params", None) else None
    context = StackContext(Path(args.state_dir), given or Params(), command)
    stack = load_stack(args.stack, context)
    pinned = (context.state_dir / JOURNAL_FILE).exists() and load(context.state_dir / JOURNAL_FILE).params
    if pinned:  # the journal's parameters are the only ones in force
        if given and command != "set-params" and given.params_hash != pinned.params_hash:
            raise ConfigError("--params differs from the parameters pinned in the journal; change them with set-params")
        context.params = pinned
    return context, stack


def open_controller(args: argparse.Namespace, command: str) -> tuple[Controller, Stack]:
    """Take over the journal as its one writer. A controller that still runs elsewhere stops at its next write."""
    context, stack = open_stack(args, command)
    owner = f"{os.uname().nodename}:{os.getpid()}"
    journal = Journal(context.state_dir / JOURNAL_FILE, owner, now(), Mirror(stack.store) if stack.mirror else None)
    controller = Controller(journal, stack.store, stack.admission, stack.sampler, context.params, folder=stack.folder, publisher=stack.publisher, pins=stack.pins)
    return controller, stack


def read_state(args: argparse.Namespace, command: str) -> tuple[State, Params, Stack | None]:
    """The state without taking over the journal. With a stack that mirrors the journal, the store's copy is read."""
    if not getattr(args, "stack", None):
        state = load(Path(args.state_dir) / JOURNAL_FILE)
        return state, state.params or Params(), None
    context, stack = open_stack(args, command)
    state = context.load(stack.store if stack.mirror else None)
    return state, context.params, stack


def verdict_rule(verdict: Verdict | None) -> str:
    """The name under which ``status`` counts a queued file."""
    if verdict is None:
        return "no file"
    if verdict.valid:
        return "valid"
    return verdict.rule or "invalid: " + re.sub(r"\d+", "N", verdict.reason)[:80]


def print_round(stack: Stack, round_id: str) -> None:
    print(round_text(json.loads(stack.store.get(layout.round_file(round_id)))), flush=True)


def cmd_sim(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    world = SimWorld(Path(args.state_dir), sim_params())
    uid = 0
    for _ in range(args.rounds):
        for _ in range(args.miners):
            uid += 1
            quality = rng.choice([0.0, 0.0, -0.1, 0.05, 0.25]) if uid % 5 else None  # every fifth miner has no file
            world.register(uid, quality)
        world.round()
    live = summary(world.controller.state, world.params)
    replayed = summary(load(Path(args.state_dir) / JOURNAL_FILE), world.params)
    print(json.dumps({"summary": live, "replay_matches": live == replayed}, indent=2))
    return 0 if live == replayed and live["phase"] == "IDLE" else 1


def cmd_bootstrap(args: argparse.Namespace) -> int:
    read = lambda path: Path(path).read_text(encoding="utf-8")
    genesis = GenesisSpec.from_json(read(args.genesis))
    files = json.loads(read(args.genesis_files))
    v1 = V1Import.from_json(read(args.v1_import)) if args.v1_import else None
    chain = ChainSnapshot.from_json(read(args.chain)) if args.chain else None
    controller, stack = open_controller(args, "bootstrap")
    if controller.state.heads:
        raise V2Error("this journal already has a genesis; refusing to bootstrap again")
    # Fail closed before the journal gets a genesis: the head's files must be in the store with their hashes.
    print("checking every genesis file in the store against its sha256 (the weights are large: this takes minutes)", file=sys.stderr, flush=True)
    verify_files(stack.store, layout.public_base_prefix(genesis.base_hash), files["base_files"])
    verify_files(stack.store, layout.public_master_prefix(genesis.base_fp32_master_digest), files["master_files"])
    controller.bootstrap(genesis, files["base_files"], files["master_files"], now(), v1)
    if chain is not None:
        controller.sync_registrations(chain, now())
    params = controller.params
    # llms.txt and mining.json go out before the first round, with the first queue and the verdict on each waiting file.
    published = controller.publish(now(), check_files=True)
    print(json.dumps(summary(controller.state, params) | {"published": published}, indent=2))
    if controller.last_error:
        print(f"not finished: {controller.last_error}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    controller, stack = open_controller(args, "run")
    if stack.runner is None:
        raise ConfigError("this stack has no round runner")
    done = idle = 0
    while args.rounds is None or done < args.rounds:
        record = None
        try:
            if stack.chain is not None:
                controller.sync_registrations(stack.chain(), now())
            record = controller.run_round(stack.runner, now, stack.entropy() if stack.entropy else os.urandom(16).hex())
        except FencedError:
            raise
        except InfraError as exc:  # the chain or the store did not answer: nothing was decided, so the next pass continues
            print(f"waiting: {exc}", file=sys.stderr, flush=True)
        if record is None:
            idle += 1
            state = controller.state
            note = f"idle: {len(waiting(state.queue))} seats wait, phase {phase(state)}"
            print(note + (f", last problem: {controller.last_error}" if controller.last_error else ""), file=sys.stderr, flush=True)
            controller.publish(now(), only_changes=True)  # a new or changed miner file shows up while no round runs
            if args.max_idle is not None and idle >= args.max_idle:
                break
            time.sleep(args.idle_sleep)
            continue
        done, idle = done + 1, 0
        print_round(stack, record.round_id)
    print(json.dumps(summary(controller.state, controller.params), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state, params, stack = read_state(args, "status")
    report = summary(state, params)
    # A seat is held after params.max_infra_attempts counted failures (no_decision); release-seat undoes both.
    report["held_seats"] = sorted(seat.reg.uid for seat in state.queue.values() if seat.held)
    report["seats_with_counted_failures"] = sum(1 for seat in state.queue.values() if seat.infra_attempts and not seat.held)
    if stack is not None:
        seats = waiting(state.queue)
        verdicts = stack.admission.verdicts([seat.reg.hotkey for seat in seats])
        report["queued_files"] = dict(sorted(Counter(verdict_rule(verdicts.get(seat.reg.hotkey)) for seat in seats).items()))
    print(json.dumps(report, indent=2))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    state, params, stack = read_state(args, "publish")
    if stack.publisher is None:
        raise ConfigError("this stack has no publisher")
    checked = now()  # this reads the file of every waiting seat
    verdicts = stack.admission.verdicts([seat.reg.hotkey for seat in waiting(state.queue)])
    stack.publisher.publish(snapshot(state, params, now(), verdicts, checked))
    valid = sum(verdict.valid for verdict in verdicts.values())
    print(json.dumps({"published": True, "head": state.head.version_id, "queue": len(state.queue), "valid_files": valid}))
    return 0


def cmd_round(args: argparse.Namespace) -> int:
    state, _, stack = read_state(args, "round")
    if not args.round and not state.rounds:
        raise V2Error("no round has closed yet")
    key = layout.round_file(args.round or state.rounds[-1].round_id)
    if args.json:
        print(stack.store.get(key).decode())
    else:
        print(round_text(json.loads(stack.store.get(key))))
    return 0


def changed_params(before: Params, after: Params) -> dict[str, dict[str, Any]]:
    """Each parameter whose value differs, with both values. A value of the recipe has the name ``recipe.<name>``."""

    def flat(params: Params) -> dict[str, Any]:
        data = params.to_dict()
        return data | {f"recipe.{name}": value for name, value in data.pop("recipe").items()}

    old, new = flat(before), flat(after)
    return {name: {"before": old[name], "after": new[name]} for name in old if old[name] != new[name]}


def cmd_set_params(args: argparse.Namespace) -> int:
    controller, _ = open_controller(args, "set-params")
    before = controller.params
    controller.set_params(load_params(args.params), now())
    after = controller.params
    # A key that the file does not hold is back at its default, so the operator must see every value that moved.
    print(json.dumps({"params_hash_before": before.params_hash, "params_hash": after.params_hash, "changed": changed_params(before, after)}))
    return 0


RESULT_STORED = 3  # exit code of end-round when it ends nothing because a box result waits for run


def cmd_end_round(args: argparse.Namespace) -> int:
    controller, stack = open_controller(args, "end-round")
    record = controller.state.round
    if record is None or record.decision is not None:
        raise V2Error("no round is open and undecided; a decided round is finished by run")
    stored = [lease.lease_id for lease in record.leases if lease.status == "open" and stack.store.exists(layout.sealed_result(record.round_id, lease.lease_id))]
    if stored and not args.discard_result:
        note = f"kept: round {record.round_id} has a box result in the store ({', '.join(stored)}); run judges it."
        print(note + " Nothing was ended. --discard-result ends the round without it.", file=sys.stderr)
        return RESULT_STORED
    while (record := controller.state.round) is not None and record.decision is None:
        lease = next(lease for lease in record.leases if lease.status == "open")
        controller.fail_lease(lease.lease_id, "ended by the operator", now())
    print(json.dumps(summary(controller.state, controller.params), indent=2))
    return 0


def cmd_release_seat(args: argparse.Namespace) -> int:
    controller, _ = open_controller(args, "release-seat")
    marked = sorted((seat for seat in controller.state.queue.values() if seat.held or seat.infra_attempts), key=lambda seat: seat.reg.key)
    if not args.all:
        clean = sorted(set(args.uid) - {seat.reg.uid for seat in marked})
        if clean:
            raise V2Error(f"no seat of UID {', '.join(map(str, clean))} is held or has a counted failure; nothing was released")
        marked = [seat for seat in marked if seat.reg.uid in args.uid]
    if not marked:
        raise V2Error("no seat is held or has a counted failure; nothing was released")
    for seat in marked:
        controller.release_seat(seat.reg.key, now())
    released = [
        {"uid": seat.reg.uid, "registration_block": seat.reg.registration_block, "hotkey": seat.reg.hotkey, "was_held": seat.held, "counted_failures": seat.infra_attempts}
        for seat in marked
    ]
    print(json.dumps({"released": released}, indent=2))
    return 0


def front_report(state: State, hotkeys: list[str]) -> tuple[list[FrontMove], dict[str, Any]]:
    """What ``front-seats`` changes in this state. Refused when a round is open or a hotkey has no waiting seat."""
    if state.round is not None:
        raise V2Error(f"nothing was changed: round {state.round.round_id} is open (phase {phase(state)}). The command runs only between rounds, in phase IDLE")
    moves, order = plan_front(state.queue, state.served_hotkeys, hotkeys)
    named = {move.seat.reg.key for move in moves}
    before = {seat.reg.key: place for place, seat in enumerate(waiting(state.queue))}
    seats = [
        {
            "uid": move.seat.reg.uid,
            "hotkey": move.seat.reg.hotkey,
            "registration_block": move.seat.reg.registration_block,
            "changed": move.changed,
            "position": {"before": move.position_before, "after": move.position_after},
            "front": {"before": move.seat.front, "after": True},
            "demoted": {"before": move.seat.demoted, "after": False},
            "held": move.seat.held,
            "counted_failures": move.seat.infra_attempts,
        }
        for move in moves
    ]
    report = {
        "seats": seats,
        "seats_to_change": sum(move.changed for move in moves),
        "other_seats_whose_position_shifts": sum(1 for place, seat in enumerate(order) if seat.reg.key not in named and before[seat.reg.key] != place),
        "front_seats_after": [seat.reg.uid for seat in order if seat.front],
        "charged": 0,
    }
    return moves, report


def cmd_front_seats(args: argparse.Namespace) -> int:
    """Send waiting seats to the front, for turns outside v2 that were never judged. All or nothing, and only between rounds."""
    reason = args.reason.strip()
    if not reason:
        raise V2Error("nothing was changed: --reason is empty")
    state, _, _ = read_state(args, "front-seats")  # the journal copy in the store; this takes over nothing
    moves, report = front_report(state, args.hotkey)
    report = {"dry_run": args.dry_run, "written": False, "reason": reason} | report
    if not report["seats_to_change"]:
        print(json.dumps(report, indent=2))
        print("nothing to change: each of these seats is at the front already and is not demoted. Nothing was written.", file=sys.stderr)
        return 0
    if args.dry_run:
        print(json.dumps(report, indent=2))
        print("dry run: nothing was written, to the journal or to the store. The same command without --dryrun writes exactly this.", file=sys.stderr)
        print(
            f"BEFORE the real run: every host that reads the journal must be on this commit. On AffineValidator the weights writer and the links job read it. "
            f"A checkout that does not know the record kind {SEATS_FRONTED!r} refuses the whole journal. AFTER the real run no host can go back to an older commit.",
            file=sys.stderr,
        )
        return 0
    controller, _ = open_controller(args, "front-seats")
    again, check = front_report(controller.state, args.hotkey)
    if check != {key: report[key] for key in check}:
        raise V2Error("nothing was changed: the journal moved between the read and the takeover. Run the command again")
    controller.front_seats([move.seat.reg.key for move in again if move.changed], reason, now())
    replayed = load(Path(args.state_dir) / JOURNAL_FILE)
    report |= {"written": True, "journal_record": controller.state.last_seq, "replay_matches": replayed.queue == controller.state.queue}
    print(json.dumps(report, indent=2))
    print(f"written: journal record {controller.state.last_seq} ({SEATS_FRONTED}). No host can go back to a commit that does not know this record kind.", file=sys.stderr)
    return 0 if report["replay_matches"] else 1


CHAIN_READ_WAIT_S = 6.0  # between two tries for a chain read at a later block; a block takes 12 seconds
CHAIN_READ_TRIES = 10


def two_chain_reads(read: Callable[[], ChainSnapshot]) -> tuple[ChainSnapshot, ChainSnapshot]:
    """Two complete reads of the registrations, the second at a later block than the first."""
    first = second = read()
    for _ in range(CHAIN_READ_TRIES):
        if second.block > first.block:
            return first, second
        time.sleep(CHAIN_READ_WAIT_S)
        second = read()
    if second.block > first.block:
        return first, second
    raise InfraError(f"the chain stayed at block {first.block} for {CHAIN_READ_TRIES * CHAIN_READ_WAIT_S:g} s: there is no second read to compare with")


def _changes(plan: SyncPlan) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int, str]]]:
    """What a sync plan changes, without the clock times of a read: seats that leave, seats that enter, new hotkeys."""
    return list(plan.removed), [reg.key for reg in plan.added], [(*reg.key, reg.hotkey) for reg in plan.rehotkeyed]


def cmd_force_sync(args: argparse.Namespace) -> int:
    """Apply once a chain sync that ``run`` refuses. ``run`` itself never forces a sync.

    Two guards stand in for the refusal. The changes must be the same in two
    chain reads at different blocks, so a read that is wrong for one moment
    does not pass. And the operator must give the number of seats that leave,
    as the list without ``--confirm-remove`` showed it.
    """
    controller = None
    if args.confirm_remove is None:
        state, params, stack = read_state(args, "force-sync")
    else:
        controller, stack = open_controller(args, "force-sync")
        state, params = controller.state, controller.params
    if stack.chain is None:
        raise ConfigError("this stack reads no chain, so there is no sync to apply")
    first, second = two_chain_reads(stack.chain)
    plans = [plan_sync(state.queue, state.served_regs, state.served_hotkeys, chain, params.netuid, params.burn_uid, force=True) for chain in (first, second)]
    plan = plans[1]
    if _changes(plans[0]) != _changes(plan):
        counts = " and ".join(f"remove {len(each.removed)}, add {len(each.added)} at block {chain.block}" for each, chain in zip(plans, (first, second)))
        raise InfraError(f"two chain reads give different changes of the queue ({counts}); nothing was changed. Run the command again")
    seats, leave = len(state.queue), len(plan.removed)
    print(f"chain blocks {first.block} and {second.block} ({len(second.registrations)} registrations) give the same changes to the queue of {seats} seats:")
    print(f"  remove {leave} seats, add {len(plan.added)} seats, change the hotkey of {len(plan.rehotkeyed)} seats")
    names = {FRONT: "front", BACK: "back"}
    for key in plan.removed:
        seat = state.queue[key]
        flags = names.get(zone(seat), "normal") + (", held" if seat.held else "")
        print(f"  remove uid {key[0]:>4} registration block {key[1]:>9} {seat.reg.hotkey} ({flags})")
    if seats and leave == seats:
        print(f"WARNING: not one of the {seats} seats is on this chain. A read of another network or subnet looks like this")
    if controller is None:
        print(f"nothing was changed. To apply exactly this, stop run and give the same command with --confirm-remove {leave}")
        return 0
    if args.confirm_remove != leave:
        raise V2Error(f"the chain now removes {leave} seats, not the confirmed {args.confirm_remove}; nothing was changed. Read the list again")
    applied = controller.sync_registrations(second, now(), force=True)
    if _changes(applied) != _changes(plan):
        raise V2Error("the applied changes differ from the listed ones")  # one state and one read: this cannot happen
    print(f"applied at chain block {second.block}: the queue has {len(controller.state.queue)} seats", flush=True)
    print(json.dumps(summary(controller.state, controller.params), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sn120_v2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    stack_help = "module:function that builds the parts: sn120_v2.assembly:stack (real) or sn120_v2.sim:stack (CPU)"

    def command(name: str, handler: Callable[[argparse.Namespace], int], stack: bool | None = None) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--state-dir", required=True, help="directory of the journal (and of the local store, if used)")
        if stack is not None:
            sub.add_argument("--stack", required=stack, help=stack_help)
            if name not in ("bootstrap", "set-params"):
                sub.add_argument("--params", help="optional check: refused when it differs from the parameters pinned in the journal")
        return sub

    sim = command("sim", cmd_sim)
    sim.add_argument("--rounds", type=int, default=5)
    sim.add_argument("--miners", type=int, default=4, help="new registrations per round")
    sim.add_argument("--seed", type=int, default=0)
    boot = command("bootstrap", cmd_bootstrap, stack=True)
    boot.add_argument("--params", help="JSON file of parameter overrides; the result is pinned in the journal")
    boot.add_argument("--genesis", required=True, help="genesis.json (a GenesisSpec)")
    boot.add_argument("--genesis-files", required=True, help='JSON with "base_files" and "master_files": {file name: sha256}')
    boot.add_argument("--v1-import", help="v1_import.json (a V1Import)")
    boot.add_argument("--chain", help="chain.json (a ChainSnapshot) to fill the queue")
    run = command("run", cmd_run, stack=True)
    run.add_argument("--rounds", type=int, help="stop after this many rounds (default: run until stopped)")
    run.add_argument("--max-idle", type=int, help="stop after this many polls with nothing to run")
    run.add_argument("--idle-sleep", type=float, default=30.0, help="seconds between polls when nothing is ready")
    command("status", cmd_status, stack=False)
    command("publish", cmd_publish, stack=True)
    one = command("round", cmd_round, stack=True)
    one.add_argument("--round", help="round id, such as r000001 (default: the last closed round)")
    one.add_argument("--json", action="store_true", help="print the stored JSON record instead of the text")
    setp = command("set-params", cmd_set_params, stack=True)
    setp.add_argument("--params", required=True, help="JSON file of parameter overrides on the defaults; the result replaces the pinned set")
    end = command("end-round", cmd_end_round, stack=True)
    end.add_argument("--discard-result", action="store_true", help="end the round also when a box result of it is in the store; that result is never judged")
    which = command("release-seat", cmd_release_seat, stack=True).add_mutually_exclusive_group(required=True)
    which.add_argument("--uid", type=int, action="append", help="release the seat of this UID; give the flag again for more UIDs")
    which.add_argument("--all", action="store_true", help="release every seat that is held or has a counted failure")
    front = command("front-seats", cmd_front_seats, stack=True)
    front.add_argument("--hotkey", action="append", required=True, help="the hotkey (SS58 address) of a waiting seat; give the flag again for more hotkeys")
    front.add_argument("--reason", required=True, help="why these seats go to the front; the journal record keeps this text")
    front.add_argument("--dryrun", "--dry-run", dest="dry_run", action="store_true", help="print what would change; write nothing")
    force = command("force-sync", cmd_force_sync, stack=True)
    force.add_argument("--confirm-remove", type=int, metavar="N", help="apply the changes, when exactly N seats leave; without it the changes are only shown")
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except V2Error as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
