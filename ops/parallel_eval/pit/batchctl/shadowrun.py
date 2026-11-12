"""Start the batch controller in shadow mode on the validator host.

    python -m ops.parallel_eval.pit.batchctl.shadowrun check <options>
    python -m ops.parallel_eval.pit.batchctl.shadowrun run   <options>

check   Build every part, test the settings, the state root, the host helper
        and the bucket, and print what ``run`` would do. It writes nothing
        and rents nothing. ``--survey`` also lists the offers of each rung.
run     The controller loop, until SIGTERM or SIGINT. A fatal error ends the
        process; a supervisor starts it again and it recovers from its
        journal and R2.

What a shadow run can write, and nothing else:

* its own state root (journal, lock, status file): a directory that is not
  in or around a live root (``guard.StateRoot``);
* R2 keys under ``pit-shadow/batchctl/<run id>/`` (``guard.GuardedStore``);
* boxes with a shadow name of this run (``lium.LiumProvider``).

There is no other mode in this file, and no code for the live state, the
live HEAD or the weights of the subnet.

Parts that are not wired yet are in ``STUBS``. Each one fails closed: with
the batch feed not wired no batch comes in, so no box is rented.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import MODE_SHADOW
from . import core as C
from . import hostlink
from .controller import Config, Controller, ladder_for
from .guard import MARKER, StateRoot, check_outside_live, shadow_prefix
from .interfaces import BatchRef, BoxError, JobSpec, JudgeNotReady
from .lium import PROVIDER_NAME, LiumConfig, LiumProvider
from .r2 import R2Store, client_from_bucket_config
from .timers import SystemClock

# At most this many boxes at one time, whatever the options say.
MAX_SHADOW_BOXES = 8
MIN_RAM_GB = 300.0
DEFAULT_GPUS = 'RTX5090'


class NoBatches:
    """The batch feed is not wired to the live queue yet. No batch comes in."""

    def list(self) -> Sequence[BatchRef]:
        return ()


class NoLaunch:
    """The box agent is not wired yet. No job is started on a box."""

    def launch(self, box_id: str, provider: str, spec: JobSpec) -> None:
        raise BoxError('the box agent is not wired yet')


class NoVerdict:
    """The judge is not wired yet. No batch gets a verdict."""

    def verdict(self, result: bytes, *, batch_id: str, base_state: str) -> str:
        raise JudgeNotReady('the judge is not wired yet')


# What is not wired yet, for the operator. ``check`` prints it.
STUBS: Dict[str, str] = {
    'batch_feed': 'not wired to the live queue: no batch comes in, so nothing is rented',
    'box_agent': 'not wired: no job is started on a box (box/job.py, box/steps.py and recipe.py are ready for it)',
    'judge': 'not wired: no batch gets a verdict',
    'genesis_import': 'not written: the first state must be put under the shadow prefix by other means',
}


def options(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='batch controller, shadow mode only')
    parser.add_argument('command', choices=('check', 'run'))
    parser.add_argument('--run-id', required=True, help='4 to 32 characters, a-z and 0-9; names the R2 prefix and the boxes')
    parser.add_argument('--state-root', required=True, help='directory of this run: journal, lock, status')
    parser.add_argument('--bucket-config', required=True, help='file with the R2 endpoint, bucket and keys')
    parser.add_argument('--genesis-state', required=True, help='state id (sha256) of the first HEAD of this run')
    parser.add_argument('--gpus', default=DEFAULT_GPUS, help='GPU classes of this run, comma-separated, first is tried first')
    parser.add_argument('--max-boxes', type=int, default=MAX_SHADOW_BOXES)
    parser.add_argument('--max-infra-failures', type=int, default=3, help='lost boxes of one batch before it is given up')
    parser.add_argument('--max-cost-usd', type=float, default=None, help='stop new rentals above this cost')
    parser.add_argument('--lease-hours', type=float, default=8.0, help='longest time of one job on a box')
    parser.add_argument('--box-hours', type=float, default=10.0, help='lifetime that a rental gets')
    parser.add_argument('--survey', action='store_true', help='check: also list the offers of each rung')
    return parser.parse_args(argv)


def settings(args: argparse.Namespace) -> Tuple[Config, LiumConfig]:
    gpus = tuple(gpu.strip() for gpu in args.gpus.split(',') if gpu.strip())
    if not 1 <= args.max_boxes <= MAX_SHADOW_BOXES:
        raise ValueError('max-boxes must be 1 to %d' % MAX_SHADOW_BOXES)
    core = C.CoreConfig(
        max_boxes=args.max_boxes, max_infra_failures=args.max_infra_failures, ladder=ladder_for(PROVIDER_NAME, gpus),
    )
    config = Config(
        run_id=args.run_id, genesis_state=args.genesis_state, core=core, mode=MODE_SHADOW, gpu_allow_list=gpus,
        lease_ttl_s=args.lease_hours * 3600.0, box_ttl_s=args.box_hours * 3600.0, max_cost_usd=args.max_cost_usd,
    )
    lium = LiumConfig(run_id=args.run_id, gpus=gpus, max_boxes=args.max_boxes, gpu_count=config.gpu_count,
                      min_ram_gb=MIN_RAM_GB)
    return config, lium


def state_root(args: argparse.Namespace, *, create: bool, env: Optional[Mapping[str, str]] = None) -> Optional[StateRoot]:
    """The state root of the run. ``create`` False: an existing root is
    opened, a missing one is only checked (it is outside every live root)."""
    path = Path(args.state_root)
    if (path / MARKER).exists():
        return StateRoot.open(path, args.run_id, env)
    if create:
        return StateRoot.create(path, args.run_id, env)
    check_outside_live(path, env)
    return None


def provider(lium: LiumConfig) -> LiumProvider:
    return LiumProvider(hostlink.load('lium'), lium)


def store(args: argparse.Namespace) -> R2Store:
    client, bucket = client_from_bucket_config(Path(args.bucket_config))
    return R2Store(client, bucket)


def plan(config: Config, lium: LiumConfig) -> Dict[str, Any]:
    return {
        'mode': config.mode, 'run_id': config.run_id, 'r2_prefix': shadow_prefix(config.run_id),
        'genesis_state': config.genesis_state,
        'ladder': [[rung.provider, rung.gpu, rung.tier] for rung in config.core.ladder],
        'gpu_allow_list': list(config.gpu_allow_list or ()), 'gpu_count': config.gpu_count,
        'max_boxes': config.core.max_boxes, 'min_ram_gb': lium.min_ram_gb,
        'price_ceiling_usd_h': config.price_ceiling_usd_h, 'max_cost_usd': config.max_cost_usd,
        'max_infra_failures': config.core.max_infra_failures,
        'lease_ttl_s': config.lease_ttl_s, 'box_ttl_s': config.box_ttl_s,
        'stubs': dict(STUBS),
    }


def check(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """-> (what ``run`` would do, what is wrong). Writes nothing, rents nothing."""
    config, lium = settings(args)
    report, wrong = plan(config, lium), []
    root = state_root(args, create=False, env=env)
    report['state_root'] = 'exists' if root is not None else 'will be created'
    try:
        store(args).check_bucket()
        report['bucket'] = 'ok'
    except Exception as exc:
        report['bucket'] = 'failed'
        wrong.append('bucket: %s' % type(exc).__name__)
    try:
        boxes = provider(lium)
        report['host_helper_lium'] = 'ok'
        if args.survey:
            report['offers'] = [
                boxes.survey(gpu=rung.gpu, tier=rung.tier, max_price_usd_h=config.price_ceiling_usd_h)
                for rung in config.core.ladder
            ]
    except Exception as exc:
        report['host_helper_lium'] = 'failed'
        wrong.append('lium: %s: %s' % (type(exc).__name__, str(exc)[:200]))
    return report, wrong


def build(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None) -> Controller:
    config, lium = settings(args)
    return Controller(
        root=state_root(args, create=True, env=env), config=config, store=store(args),
        providers={PROVIDER_NAME: provider(lium)}, agent=NoLaunch(), source=NoBatches(), judge=NoVerdict(),
        clock=SystemClock(),
    )


def say(doc: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(doc, sort_keys=True) + '\n')
    sys.stdout.flush()


def run(args: argparse.Namespace) -> int:
    controller = build(args).open()
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    say({'event': 'shadow_started', 'run_id': args.run_id, 'r2_prefix': shadow_prefix(args.run_id), 'stubs': STUBS})
    try:
        controller.run(stop)
    except BaseException as exc:
        # A worker thread may hang for ever: the process ends here, and the next one recovers.
        say({'event': 'shadow_fatal', 'error': '%s: %s' % (type(exc).__name__, str(exc)[:500])})
        os._exit(1)
    controller.close()
    say({'event': 'shadow_stopped', 'run_id': args.run_id})
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = options(argv)
    if args.command == 'run':
        return run(args)
    report, wrong = check(args)
    report['wrong'] = wrong
    say(report)
    return 1 if wrong else 0


if __name__ == '__main__':
    sys.exit(main())
