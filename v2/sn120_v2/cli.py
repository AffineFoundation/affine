"""Command line: ``python -m sn120_v2 <command>``.

  sim         a whole validator on CPU with fake training and scoring; no GPU, no network
  bootstrap   first start: pin the parameters, write the genesis head, the state carried over
              from v1 and the first queue, then publish llms.txt and mining.json
  run         the round loop
  status      read-only summary; with ``--stack`` also the count of queued files per verdict rule
  publish     read-only for the journal: write llms.txt and mining.json again
  round       print the summary of a closed round as a short text to paste
  set-params  pin new parameters (stop ``run`` first: this command takes over the journal)

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
from .queue import waiting
from .sim import SimWorld, sim_params
from .state import State, load
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
    controller = Controller(journal, stack.store, stack.admission, stack.sampler, context.params, folder=stack.folder, publisher=stack.publisher)
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


def cmd_set_params(args: argparse.Namespace) -> int:
    controller, _ = open_controller(args, "set-params")
    before = controller.params.params_hash
    controller.set_params(load_params(args.params), now())
    print(json.dumps({"params_hash_before": before, "params_hash": controller.params.params_hash}))
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
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except V2Error as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
