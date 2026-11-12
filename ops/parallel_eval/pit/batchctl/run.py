"""Run the batch controller with every part, in shadow mode or in live mode.

    python -m ops.parallel_eval.pit.batchctl.run check --shadow <options>
    python -m ops.parallel_eval.pit.batchctl.run seed  --shadow <options>
    python -m ops.parallel_eval.pit.batchctl.run run   --shadow <options>
    python -m ops.parallel_eval.pit.batchctl.run run   --live --arm-live <run id> <options>

check   Test the settings and every part that ``run`` needs, and print what
        ``run`` would do. It writes nothing and rents nothing.
seed    Copy the live HEAD into the R2 prefix of the run as its first state
        ("genesis"). The live side is only read. It prints the state id and
        keeps it in the directory of the run for ``run``.
run     The controller loop, until SIGTERM or SIGINT. A fatal error ends the
        process; a supervisor starts it again and it recovers from its
        journal and R2.

The parts
---------
batch source  the live training queue, read-only (``feed.LiveQueue``). A
              batch is taken when the live gate has drawn its round.
box agent     ``agent.LiveBoxAgent``: one job for each lease on its rented
              box (train with profile rtx5090-u6, probe with the TP2 layout,
              HEAD blocks of the baseline).
judge         ``judge.GateJudge``: the decision function of the live gate
              against a HEAD baseline of the same hardware and layout.
pay hook      ``settle.Settler``, live mode only: the gate decision of each
              final outcome, for the K1 service.

The two modes
-------------
``--shadow``  The process enters the shadow guard of the PIT package first
              (``shadow_guard.enter_shadow``): it can write files only under
              ``PIT_SHADOW_STATE`` and R2 keys only under ``pit-shadow/``.
              The run writes R2 under ``pit-shadow/batchctl/<run id>/`` and
              rents boxes with a shadow name. There is no pay hook: nothing
              is written for the gate, the queue or the website.
``--live``    Needs ``--arm-live`` with the run id a second time. Without
              it nothing starts, for each of the three commands. The run
              writes R2 under ``pit/batchctl/<run id>/``, rents boxes with a
              live name, and the pay hook writes gate decisions.

In both modes the process submits no weights and writes no score file.

The directory of the run (``--state-dir``) holds ``controller/`` (journal,
lock, status) and ``private/`` (baseline blocks, records of the judge,
marks of the pay hook, short-lived files). In shadow mode it must be under
``PIT_SHADOW_STATE``; the default is ``$PIT_SHADOW_STATE/batchctl/<run id>``.
In both modes it must be outside every live root.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .. import PRICE_CEILING_USD_H, SHARDED_GPUS
from .. import shadow_guard
from ..gate_writer import GateWriteError, HostGateWriter, build_host_rounds
from ..shadow import ShadowWriteError
from . import MODE_LIVE, MODE_SHADOW
from . import core as C
from . import genesis, hostlink, recipe
from .agent import RUN_CLASSES, AgentConfig, LiveBoxAgent, config_for, host_tools
from .baseline import Baselines
from .controller import Config, Controller, ladder_for
from .feed import LiveQueue, epoch_of
from .guard import MARKERS, GuardedStore, StateRoot, check_outside_live, check_run_id, run_prefix
from .interfaces import BatchRef, NotFound, StoreError
from .judge import Fleet, GateJudge, Records
from .layout import MAX_COMPLETE_BYTES, Layout, is_hex64, sha256_hex
from .lium import NEVER_NAMES_DEFAULT, PROVIDER_NAME, LiumConfig, LiumProvider
from .r2 import CONTENT_TYPE, R2Presigner, R2Store, client_from_bucket_config
from .rounds import HostRounds, RoundError, RoundNotReady
from .settle import SettleError, Settler
from .timers import SystemClock

MAX_BOXES = 8
MIN_RAM_GB = 300.0
DEFAULT_GPUS = 'RTX5090'
DEFAULT_MAX_TOKENS = 65536
GENESIS_FILE = 'GENESIS.json'
# The size of one uploaded part of a new state.
CHUNK_BYTES = 256 << 20


class RunnerError(RuntimeError):
    """The run cannot start with these settings. Nothing was started."""


# ----------------------------------------------------------------------------
# Options and settings
# ----------------------------------------------------------------------------

def options(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='batch controller with every part: shadow mode, or live mode behind a flag')
    parser.add_argument('command', choices=('check', 'seed', 'run'))
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument('--shadow', action='store_true', help='write only the shadow state root and R2 under pit-shadow/')
    which.add_argument('--live', action='store_true', help='the live run; needs --arm-live')
    parser.add_argument('--arm-live', default=None, metavar='RUN_ID',
                        help='live mode starts only when this repeats the run id')
    parser.add_argument('--run-id', required=True, help='4 to 32 characters, a-z and 0-9; names the R2 prefix and the boxes')
    parser.add_argument('--state-dir', default=None, help='directory of this run (controller/ and private/)')
    parser.add_argument('--bucket-config', required=True, help='file with the R2 endpoint, bucket and keys')
    parser.add_argument('--life-root', required=True, help='live queue, read-only: the directory with CURRENT.json and epochs/')
    parser.add_argument('--gate-state', required=True, help='state directory of the live gate: the parent of epochs/ and heldout/')
    parser.add_argument('--inbox', action='append', default=[], help='one inbox directory of the live queue (read-only)')
    parser.add_argument('--turns-log', default=None, help='turns log of the live queue (read-only)')
    parser.add_argument('--pending', default=None, help='pending directory of the live queue (read-only)')
    parser.add_argument('--genesis-state', default=None, help='state id of the first HEAD; the default is what seed kept')
    parser.add_argument('--sharded-host', default=None, help='directory of the ROOT-admitted sharded trainer host package')
    parser.add_argument('--gpus', default=DEFAULT_GPUS, help='the GPU class of this run')
    parser.add_argument('--max-boxes', type=int, default=MAX_BOXES)
    parser.add_argument('--max-infra-failures', type=int, default=3, help='lost boxes of one batch before it is given up')
    parser.add_argument('--max-cost-usd', type=float, default=None, help='stop new rentals above this cost')
    parser.add_argument('--price-ceiling-usd-h', type=float, default=PRICE_CEILING_USD_H,
                        help='highest price of one box per hour; never above the ceiling of the PIT policy')
    parser.add_argument('--never-name', action='append', default=[],
                        help='one more word of a node or pod that is never rented and never removed')
    parser.add_argument('--lease-hours', type=float, default=6.0, help='longest time of one job on a box')
    parser.add_argument('--box-hours', type=float, default=8.0, help='lifetime that a rental gets')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help='token cap of a round whose settings name none')
    parser.add_argument('--max-head-blocks', type=int, default=None, help='most HEAD blocks that one box runs')
    parser.add_argument('--settle-every-s', type=float, default=30.0, help='live: time between two looks of the pay hook')
    parser.add_argument('--survey', action='store_true', help='check: also list the offers of each rung')
    seed = parser.add_argument_group('seed')
    seed.add_argument('--master-index-dir', default=None, help='directory with <checkpoint id>.json of the fp32 master')
    seed.add_argument('--checkpoint', default=None, help='checkpoint id; the default is the id of CURRENT.json')
    seed.add_argument('--ckpt-prefix', default=genesis.DEFAULT_CKPT_PREFIX, help='R2 prefix of the live checkpoints')
    seed.add_argument('--source-bucket', default=None, help='bucket of the live keys, if it is another bucket')
    return parser.parse_args(argv)


def mode_of(args: argparse.Namespace) -> str:
    return MODE_LIVE if args.live else MODE_SHADOW


def armed(args: argparse.Namespace) -> bool:
    """Live mode is armed only by the flag on the command line, with the run id a second time."""
    return bool(args.live) and args.arm_live is not None and args.arm_live == args.run_id


@dataclass(frozen=True)
class Folders:
    state_dir: Path
    controller: Path
    private: Path

    @property
    def scratch(self) -> Path:
        return self.private / 'scratch'

    @property
    def genesis(self) -> Path:
        return self.private / GENESIS_FILE


def folders(args: argparse.Namespace, env: Mapping[str, str]) -> Folders:
    """The directory of the run. Nothing is made. A directory in or around a live root is refused."""
    mode = mode_of(args)
    check_run_id(args.run_id)
    shadow = env.get('PIT_SHADOW_STATE')
    if mode == MODE_SHADOW and not shadow:
        raise RunnerError('PIT_SHADOW_STATE is not set: shadow mode needs its own state root')
    if args.state_dir is not None:
        state_dir = Path(args.state_dir)
    elif mode == MODE_SHADOW:
        state_dir = Path(shadow) / 'batchctl' / args.run_id
    else:
        raise RunnerError('--state-dir is needed')
    state_dir = Path(os.path.realpath(str(state_dir)))
    if mode == MODE_SHADOW and not str(state_dir).startswith(os.path.realpath(shadow).rstrip('/') + '/'):
        raise RunnerError('the directory of a shadow run must be under PIT_SHADOW_STATE')
    check_outside_live(state_dir, env)
    return Folders(state_dir, state_dir / 'controller', state_dir / 'private')


def genesis_state(args: argparse.Namespace, where: Folders) -> str:
    if args.genesis_state is not None:
        named = args.genesis_state
    else:
        try:
            doc = json.loads(where.genesis.read_text())
        except (OSError, ValueError):
            raise RunnerError('no first state: run the seed command, or give --genesis-state') from None
        if not isinstance(doc, dict) or doc.get('run_id') != args.run_id or doc.get('mode') != mode_of(args):
            raise RunnerError('the first state on file is of another run or mode: run the seed command')
        named = doc.get('genesis_state')
    if not is_hex64(named):
        raise RunnerError('the first state is not a state id (sha256)')
    return named


def settings(args: argparse.Namespace, first_state: str, more_never: Sequence[str] = ()) -> Tuple[Config, LiumConfig]:
    """``more_never``: the names that the live lane never rents and never removes."""
    mode = mode_of(args)
    gpus = tuple(gpu.strip() for gpu in args.gpus.split(',') if gpu.strip())
    if not 1 <= args.max_boxes <= MAX_BOXES:
        raise RunnerError('max-boxes must be 1 to %d' % MAX_BOXES)
    if len(gpus) != 1:
        # One class: every probe of the run then has one hardware fingerprint.
        raise RunnerError('a run names one GPU class')
    if gpus[0] not in RUN_CLASSES:
        raise RunnerError('there is no box job for the GPU class %s' % gpus[0])
    never = list(NEVER_NAMES_DEFAULT)
    for word in (*more_never, *args.never_name):
        if not isinstance(word, str) or not word.strip():
            raise RunnerError('a never-name is a word')
        if word not in never:
            never.append(word)
    try:
        core = C.CoreConfig(
            max_boxes=args.max_boxes, max_infra_failures=args.max_infra_failures, ladder=ladder_for(PROVIDER_NAME, gpus),
        )
        config = Config(
            run_id=args.run_id, genesis_state=first_state, core=core, mode=mode, live_armed=armed(args),
            gpu_allow_list=gpus,
            # A run that names a sharded class (RTX 5090) opens the gate for itself. The default of
            # the PIT package, which the serial lane reads, is not changed.
            allow_sharded_gpus=any(gpu in SHARDED_GPUS for gpu in gpus),
            price_ceiling_usd_h=args.price_ceiling_usd_h,
            lease_ttl_s=args.lease_hours * 3600.0, box_ttl_s=args.box_hours * 3600.0, max_cost_usd=args.max_cost_usd,
        )
        lium = LiumConfig(run_id=args.run_id, gpus=gpus, max_boxes=args.max_boxes, gpu_count=config.gpu_count,
                          min_ram_gb=MIN_RAM_GB, never_names=tuple(never), mode=mode)
    except ValueError as exc:
        raise RunnerError(str(exc)) from None
    return config, lium


def host_never_names() -> Tuple[str, ...]:
    """The never-list of the live lane. This run obeys it too."""
    names = hostlink.value('never_names')
    if not isinstance(names, (tuple, list)) or not all(isinstance(name, str) and name for name in names):
        raise RunnerError('the never-list of the live lane is not a list of names')
    return tuple(names)


# ----------------------------------------------------------------------------
# The parts
# ----------------------------------------------------------------------------

class ReadySource:
    """``BatchSource``: the batches of the live queue whose round the live gate has drawn.

    A batch without its round cannot be probed, so it gets no box yet. It is listed when the round is there.
    """

    def __init__(self, queue: LiveQueue, rounds: Any, log: Callable[[str], None] = lambda text: None):
        self.queue = queue
        self.rounds = rounds
        self.log = log
        self.waiting: Dict[str, str] = {}

    def list(self) -> Sequence[BatchRef]:
        ready, waiting = [], {}
        for ref in self.queue.list():
            try:
                self.rounds.of(epoch_of(ref.batch_id))
            except (RoundNotReady, RoundError) as exc:
                waiting[ref.batch_id] = str(exc)[:200]
                continue
            ready.append(ref)
        for batch_id in sorted(set(waiting) - set(self.waiting)):
            self.log('batch %s waits for its round: %s' % (batch_id, waiting[batch_id]))
        self.waiting = waiting
        return ready


class ControllerFleet(Fleet):
    """``Fleet`` on the journal of the controller. It knows nothing before ``bind``."""

    def __init__(self) -> None:
        self.controller: Optional[Controller] = None

    def bind(self, controller: Controller) -> None:
        self.controller = controller

    def live(self) -> Collection[int]:
        return self.controller.fleet().live if self.controller is not None else ()

    def others(self, fence: int, base_state: str) -> int:
        if self.controller is None:
            return 0
        view = self.controller.fleet()
        count = sum(1 for other, base in view.starting.items() if other != fence and base == base_state)
        # A batch without a box starts from the newest HEAD.
        return count + (view.waiting if view.head_state == base_state else 0)

    def head_state(self) -> Optional[str]:
        return self.controller.fleet().head_state if self.controller is not None else None


def say(doc: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(doc, sort_keys=True, default=str) + '\n')
    sys.stdout.flush()


def logger(part: str) -> Callable[[str], None]:
    return lambda text: say({'event': part, 'text': str(text)[:1000], 'at': round(time.time(), 1)})


def raw_store(args: argparse.Namespace) -> Tuple[R2Store, R2Presigner]:
    try:
        client, bucket = client_from_bucket_config(Path(args.bucket_config))
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
        # Only the type of the error: the text of an error must never show a part of the config.
        raise RunnerError('cannot use the bucket config: %s' % type(exc).__name__) from None
    return R2Store(client, bucket), R2Presigner(client, bucket)


def live_queue(args: argparse.Namespace) -> LiveQueue:
    return LiveQueue(Path(args.life_root), inboxes=[Path(inbox) for inbox in args.inbox],
                     turns_log=Path(args.turns_log) if args.turns_log else None,
                     pending=Path(args.pending) if args.pending else None)


def scorer(gate_state: Path) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """The decision function of the live gate and its baseline builder. They must read the rounds of this gate state."""
    score, blocks = hostlink.load('score'), hostlink.load('blocks_baseline')
    reads = Path(os.path.realpath(str(hostlink.value('heldout'))))
    if reads != Path(os.path.realpath(str(gate_state / 'heldout'))):
        raise RunnerError('the live scorer reads its rounds from another directory than --gate-state names')
    return score, blocks


def agent_config(args: argparse.Namespace, config: Config, scratch: Path) -> AgentConfig:
    harness = hostlink.load('public_harness')()
    return config_for(
        config.gpu_allow_list[0], harness=harness if isinstance(harness, Mapping) else {}, scratch=scratch,
        transfer=recipe.Transfer(content_type=CONTENT_TYPE, chunk_bytes=CHUNK_BYTES),
        sharded_host=Path(args.sharded_host) if args.sharded_host else None, max_head_blocks=args.max_head_blocks,
    )


def gate_writer(gate_state: Path) -> HostGateWriter:
    """The one writer of the live gate, with the round store of the gate. Live mode only.

    It is built the way the serial lane builds it (``host_wiring``). A host part that a later write needs
    (the rounds, the K1 grant function) must be there now.
    """
    try:
        rounds = build_host_rounds(gate_state=str(gate_state), heldout=str(gate_state / 'heldout'))
        writer = HostGateWriter(epochs_root=gate_state / 'epochs', shadow=False, rounds=rounds)
        writer.preflight()
    except (GateWriteError, ImportError) as exc:
        raise RunnerError('the gate writer cannot be built: %s: %s' % (type(exc).__name__, str(exc)[:300])) from None
    return writer


@dataclass
class Parts:
    mode: str
    config: Config
    where: Folders
    controller: Controller
    source: ReadySource
    settler: Optional[Settler]


def make_private(where: Folders) -> None:
    for folder in (where.state_dir, where.private, where.scratch):
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)


def state_root(args: argparse.Namespace, where: Folders, env: Mapping[str, str]) -> StateRoot:
    mode = mode_of(args)
    if (where.controller / MARKERS[mode]).exists():
        return StateRoot.open(where.controller, args.run_id, env, mode)
    return StateRoot.create(where.controller, args.run_id, env, mode)


def build(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None) -> Parts:
    """Every part of a run, wired. Makes the directory of the run. Rents nothing and starts nothing."""
    env = os.environ if env is None else env
    mode = mode_of(args)
    where = folders(args, env)
    gate_state = Path(args.gate_state)
    # Each part of the host is loaded before a file of the run is made: a part that is missing stops the start.
    score, blocks = scorer(gate_state)
    tools = host_tools()
    cli = hostlink.load('lium')
    config, lium = settings(args, genesis_state(args, where), host_never_names())
    writer = gate_writer(gate_state) if mode == MODE_LIVE else None
    make_private(where)
    root = state_root(args, where, env)
    raw, presigner = raw_store(args)
    guarded = GuardedStore(raw, args.run_id, mode)
    rounds = HostRounds(gate_state, max_tokens=args.max_tokens)
    queue = live_queue(args)
    baselines = Baselines(where.private / 'baseline')
    records = Records(where.private / 'judged')
    fleet = ControllerFleet()
    provider = LiumProvider(cli, lium)
    judge = GateJudge(run_id=args.run_id, rounds=rounds, baselines=baselines, records=records, score=score,
                      blocks_baseline=blocks, fleet=fleet, log=logger('judge'))
    try:
        agent = LiveBoxAgent(
            run_id=args.run_id, store=guarded, presigner=presigner, providers={PROVIDER_NAME: provider}, queue=queue,
            rounds=rounds, baselines=baselines, tools=tools, config=agent_config(args, config, where.scratch),
            fleet=fleet, log=logger('agent'),
        )
    except ValueError as exc:
        raise RunnerError(str(exc)) from None
    source = ReadySource(queue, rounds, logger('source'))
    controller = Controller(
        root=root, config=config, store=raw, providers={PROVIDER_NAME: provider}, agent=agent, source=source,
        judge=judge, clock=SystemClock(),
    )
    fleet.bind(controller)
    settler = None
    if writer is not None:
        settler = Settler(
            run_id=args.run_id, mode=mode, store=guarded, records=records, writer=writer,
            marks=where.private / 'settled', life_epochs=Path(args.life_root) / 'epochs', alert=controller.alert,
            log=logger('settle'),
        )
    return Parts(mode, config, where, controller, source, settler)


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def plan(args: argparse.Namespace, config: Config, lium: LiumConfig, where: Folders) -> Dict[str, Any]:
    live = config.mode == MODE_LIVE
    return {
        'mode': config.mode, 'run_id': config.run_id, 'r2_prefix': run_prefix(config.run_id, config.mode),
        'state_dir': str(where.state_dir), 'genesis_state': config.genesis_state,
        'ladder': [[rung.provider, rung.gpu, rung.tier] for rung in config.core.ladder],
        'allow_sharded_gpus': config.allow_sharded_gpus, 'gpu_count': config.gpu_count,
        'never_names': list(lium.never_names), 'max_boxes': config.core.max_boxes, 'min_ram_gb': lium.min_ram_gb,
        'price_ceiling_usd_h': config.price_ceiling_usd_h, 'max_cost_usd': config.max_cost_usd,
        'max_infra_failures': config.core.max_infra_failures, 'lease_ttl_s': config.lease_ttl_s,
        'box_ttl_s': config.box_ttl_s,
        # Every place that ``run`` writes. None: this process never writes it, in each mode.
        'writes': {
            'r2': run_prefix(config.run_id, config.mode), 'files': str(where.state_dir),
            'gate_decisions': str(Path(args.gate_state) / 'epochs') if live else None,
            'weights': None, 'score_file': None, 'turn_file': None,
        },
    }


def check(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """-> (what ``run`` would do, what is wrong). Writes nothing, rents nothing."""
    env = os.environ if env is None else env
    wrong: List[str] = []
    where = folders(args, env)
    first_state = genesis_state(args, where)
    gate_state = Path(args.gate_state)
    found: Dict[str, Any] = {}

    def part(name: str, test: Callable[[], Any], *, quiet: bool = False) -> None:
        """``quiet``: report only the type of an error. The text of a bucket error can show a part of the config."""
        try:
            said = test()
        except Exception as exc:
            found[name] = 'failed'
            text = '' if quiet and not isinstance(exc, RunnerError) else ': ' + str(exc)[:300]
            wrong.append('%s: %s%s' % (name, type(exc).__name__, text))
            return
        found[name] = 'ok' if said is None else said

    host_never: List[str] = []

    def never_names() -> None:
        host_never.extend(host_never_names())

    part('never_names_of_the_live_lane', never_names)
    config, lium = settings(args, first_state, host_never)
    report = plan(args, config, lium, where)
    marker = where.controller / MARKERS[config.mode]
    found['state_dir'] = 'exists' if marker.exists() else 'will be created'

    def bucket() -> None:
        raw_store(args)[0].check_bucket()

    part('bucket', bucket, quiet=True)

    def first() -> None:
        guarded = GuardedStore(raw_store(args)[0], args.run_id, config.mode)
        try:
            body = guarded.get(Layout(guarded.prefix).state_manifest(first_state), max_bytes=MAX_COMPLETE_BYTES).data
        except NotFound:
            raise RunnerError('the first state is not in R2: run the seed command') from None
        if sha256_hex(body) != first_state:
            raise RunnerError('the manifest of the first state has other bytes')

    part('first_state', first, quiet=True)

    def boxes() -> Any:
        provider = LiumProvider(hostlink.load('lium'), lium)
        if not args.survey:
            return None
        return [provider.survey(gpu=rung.gpu, tier=rung.tier, max_price_usd_h=config.price_ceiling_usd_h)
                for rung in config.core.ladder]

    part('lium', boxes)

    def score_function() -> None:
        scorer(gate_state)

    part('live_scorer', score_function)

    def box_helpers() -> None:
        host_tools()

    part('box_helpers', box_helpers)

    def box_agent() -> None:
        problems = agent_config(args, config, Path(tempfile.gettempdir())).problems()
        if problems:
            raise RunnerError('; '.join(problems))

    part('box_agent', box_agent)

    def queue() -> Dict[str, Any]:
        live = live_queue(args)
        source = ReadySource(live, HostRounds(gate_state, max_tokens=args.max_tokens))
        ready = source.list()
        return {'batches_ready': [ref.batch_id for ref in ready], 'wait_for_round': dict(source.waiting),
                'skipped': dict(live.skipped)}

    part('live_queue', queue)
    if config.mode == MODE_LIVE:
        # After the scorer: it puts the code directory of the live gate on the module path.
        def writer() -> None:
            gate_writer(gate_state)

        part('gate_writer', writer)
    report.update(found)
    return report, wrong


def seed(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    mode = mode_of(args)
    if mode == MODE_LIVE and not armed(args):
        raise RunnerError('live mode needs --arm-live with the run id')
    if not args.master_index_dir:
        raise RunnerError('seed needs --master-index-dir')
    where = folders(args, env)
    raw, _ = raw_store(args)
    raw.check_bucket()
    head = genesis.read_live_head(Path(args.life_root), Path(args.master_index_dir), checkpoint=args.checkpoint)
    state_id = genesis.seed(raw, head, prefix=run_prefix(args.run_id, mode), ckpt_prefix=args.ckpt_prefix,
                            src_bucket=args.source_bucket, log=logger('seed'))
    make_private(where)
    doc = {'run_id': args.run_id, 'mode': mode, 'genesis_state': state_id, 'checkpoint': head.checkpoint_id,
           'fp32_master_digest': head.master_digest}
    tmp = where.genesis.with_name(GENESIS_FILE + '.tmp')
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as handle:
        handle.write(json.dumps(doc, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, where.genesis)
    say(dict(doc, event='seeded'))
    return 0


def settle_once(settler: Settler, alert: Callable[[str, Any], None]) -> None:
    try:
        settler.run_once()
    except ShadowWriteError:
        raise
    except Exception as exc:
        alert('settle_failed', '%s: %s' % (type(exc).__name__, str(exc)[:200]))


def settle_loop(settler: Settler, stop: threading.Event, every_s: float, alert: Callable[[str, Any], None]) -> None:
    while not stop.wait(every_s):
        settle_once(settler, alert)


def run(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None, *,
        stop: Optional[threading.Event] = None) -> int:
    """``stop``: a caller that gives it ends the loop itself; no signal handler is set and a fatal error is raised."""
    parts = build(args, env)
    controller = parts.controller.open()
    own_stop = stop is None
    stop = threading.Event() if stop is None else stop
    if own_stop:
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda *_: stop.set())
    say({'event': 'started', 'mode': parts.mode, 'run_id': args.run_id,
         'r2_prefix': run_prefix(args.run_id, parts.mode), 'pay_hook': parts.settler is not None})
    worker = None
    if parts.settler is not None:
        worker = threading.Thread(target=settle_loop, name='settle', daemon=True,
                                  args=(parts.settler, stop, args.settle_every_s, controller.alert))
        worker.start()
    try:
        controller.run(stop)
    except BaseException as exc:
        # A worker thread may hang for ever: the process ends here, and the next one recovers.
        say({'event': 'fatal', 'error': '%s: %s' % (type(exc).__name__, str(exc)[:500])})
        if own_stop:
            os._exit(1)
        stop.set()
        raise
    stop.set()
    if worker is not None:
        worker.join(timeout=120.0)
        # The outcomes of the last loop. What is not settled now is settled by the next process.
        settle_once(parts.settler, controller.alert)
    controller.close()
    say({'event': 'stopped', 'run_id': args.run_id})
    return 0


def enter_mode(args: argparse.Namespace, env: Optional[MutableMapping[str, str]] = None) -> None:
    """Shadow: start the hard deny of the PIT package before any work. Live: refuse without the flag."""
    if args.live:
        if not armed(args):
            raise RunnerError('live mode needs --arm-live with the run id: nothing was started')
        return
    # The ssh helper of the live lane keeps the host keys of the rented boxes there. Not a live root.
    known_hosts = Path.home() / '.lium' / 'known_hosts'
    shadow_guard.enter_shadow(env, scratch=(tempfile.gettempdir(), str(known_hosts)))


# What stops a command before it starts work. Each one is reported in one line, with exit code 2.
REFUSALS = (RunnerError, ShadowWriteError, shadow_guard.ShadowConfigError, genesis.GenesisError,
            hostlink.HostLinkError, SettleError, StoreError)


def main(argv: Optional[Sequence[str]] = None, env: Optional[MutableMapping[str, str]] = None) -> int:
    args = options(argv)
    try:
        enter_mode(args, env)
        if args.command == 'seed':
            return seed(args, env)
        if args.command == 'check':
            report, wrong = check(args, env)
            report['wrong'] = wrong
            say(report)
            return 1 if wrong else 0
        return run(args, env)
    except REFUSALS as exc:
        say({'error': '%s: %s' % (type(exc).__name__, str(exc)[:500])})
        return 2


if __name__ == '__main__':
    sys.exit(main())
