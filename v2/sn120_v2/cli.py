"""Command line: ``python -m sn120_v2 <command>``.

  sim        a whole validator on CPU with fake training and scoring; no GPU, no network
  bootstrap  first start: write the genesis head, the state carried over from v1, and the first queue
  run        the round loop, with the parts named by ``--stack module:function``
  status     read-only summary of a state directory

Nothing here has a default path. The state directory is always an argument.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .controller import Controller, Stack, StackContext
from .errors import ConfigError, InfraError, V2Error
from .journal import Journal
from .params import GenesisSpec, Params, load_params
from .payout import payees
from .public import phase
from .sim import JOURNAL_FILE, SimWorld, sim_params
from .state import State, load, replay
from .types import ChainSnapshot, V1Import


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
    }


def load_stack(spec: str, context: StackContext) -> Stack:
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ConfigError(f"--stack must be module:function, got {spec!r}")
    builder: Callable[[StackContext], Stack] = getattr(importlib.import_module(module_name), function_name)
    return builder(context)


def open_controller(args: argparse.Namespace) -> tuple[Controller, Stack, Params]:
    params = load_params(args.params) if args.params else Params()
    state_dir = Path(args.state_dir)
    journal = Journal(state_dir / JOURNAL_FILE, f"{os.uname().nodename}:{os.getpid()}", now())
    stack = load_stack(args.stack, StackContext(state_dir, params, replay(journal.records)))
    controller = Controller(journal, stack.store, stack.admission, stack.sampler, params, folder=stack.folder, publisher=stack.publisher)
    return controller, stack, params


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
    controller, _, params = open_controller(args)
    controller.bootstrap(genesis, files["base_files"], files["master_files"], now(), v1)
    if args.chain:
        controller.sync_registrations(ChainSnapshot.from_json(read(args.chain)), now())
    print(json.dumps(summary(controller.state, params), indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    controller, stack, params = open_controller(args)
    done = idle = 0
    while args.rounds is None or done < args.rounds:
        if stack.chain is not None:
            try:
                controller.sync_registrations(stack.chain(), now())
            except InfraError as exc:
                print(f"chain sync skipped: {exc}", file=sys.stderr)
        record = controller.run_round(stack.runner, now, stack.entropy() if stack.entropy else os.urandom(16).hex())
        if record is None:
            idle += 1
            if args.max_idle is not None and idle >= args.max_idle:
                break
            time.sleep(args.idle_sleep)
            continue
        done, idle = done + 1, 0
        winner = record.decision.winner if record.decision else None
        print(json.dumps({"round": record.round_id, "outcome": record.outcome, "winner_uid": winner.uid if winner else None}))
    print(json.dumps(summary(controller.state, params), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    params = load_params(args.params) if args.params else Params()
    print(json.dumps(summary(load(Path(args.state_dir) / JOURNAL_FILE), params), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sn120_v2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Callable[[argparse.Namespace], int], stack: bool = False) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--state-dir", required=True, help="directory of the journal (and of the local store, if used)")
        if name != "sim":
            sub.add_argument("--params", help="JSON file of parameter overrides")
        if stack:
            sub.add_argument("--stack", required=True, help="module:function that builds the parts, e.g. sn120_v2.sim:stack")
        return sub

    sim = command("sim", cmd_sim)
    sim.add_argument("--rounds", type=int, default=5)
    sim.add_argument("--miners", type=int, default=4, help="new registrations per round")
    sim.add_argument("--seed", type=int, default=0)
    boot = command("bootstrap", cmd_bootstrap, stack=True)
    boot.add_argument("--genesis", required=True, help="genesis.json (a GenesisSpec)")
    boot.add_argument("--genesis-files", required=True, help='JSON with "base_files" and "master_files": {file name: sha256}')
    boot.add_argument("--v1-import", help="v1_import.json (a V1Import)")
    boot.add_argument("--chain", help="chain.json (a ChainSnapshot) to fill the queue")
    run = command("run", cmd_run, stack=True)
    run.add_argument("--rounds", type=int, help="stop after this many rounds (default: run until stopped)")
    run.add_argument("--max-idle", type=int, help="stop after this many polls with nothing to run")
    run.add_argument("--idle-sleep", type=float, default=30.0, help="seconds between polls when nothing is ready")
    command("status", cmd_status)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except V2Error as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
