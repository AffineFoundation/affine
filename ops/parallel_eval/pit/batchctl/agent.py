"""The box agent: it starts the one job of a lease on its rented box.

``launch`` does this for one lease, in this order:

1. It reads what the lease needs on the validator host: the batch in the live
   queue (read-only), the round of its epoch, the base state in R2 and the
   HEAD blocks that this box takes. A problem here stops the launch before
   the box is touched.
2. It asks the box if the job of this lease started there before. The job
   keeps a claim directory for that. If it did, nothing more is done.
3. It prepares the box with the helpers of the live lane, unchanged and in
   the order of the serial lane: boot, probe bundle, trainer bundle, the
   signed inputs of the epoch, the trainer script. A box class that trains
   sharded also gets the sharded trainer host package.
4. It writes the documents of the lease (``recipe.py``), the code of the job
   and the probe tool to the box.
5. It starts ``box/job.py`` detached.

One box job and one staging layout: the job is ``box/job.py``. It downloads the
base state, trains, probes and uploads through signed URLs, and what it
uploads goes under the staging prefix of the lease only.

While steps 3 and 4 run, the agent writes the heartbeat of the lease, so a
slow preparation is not taken for a dead box. It does that for a bounded time
(``prepare_budget_s``). A box that is not ready after that loses its lease,
and the batch gets a new box.

Host-private data (the tasks of the round, the seeds, the mask key) goes to
the box over ssh only, in one file that the job removes. It is in no object
store and in no journal.

The trainer checks its ROOT admission and the fp32 master digest of the lease
on the box, as on the serial lane. Nothing here can turn those checks off: the
agent refuses a train command that names another digest or sets the bootstrap
override (``recipe.check_train_cmd``), and it ships a sharded trainer package
only if that package holds its ROOT admission file.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import shlex
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from .. import probe_layout as PL
from .. import probe_serve
from ..shadow import ShadowWriteError
from . import hostlink, recipe
from .baseline import Baselines
from .files import INDEX_NAME, MAX_INDEX_BYTES, Index, parse_index, slots_for
from .guard import check_key
from .interfaces import BoxError, JobSpec, NotFound, ObjectStore, StoreError
from .judge import Fleet
from .layout import (
    MAX_COMPLETE_BYTES, MAX_HEARTBEAT_BYTES, Layout, StateManifest, heartbeat_body, load_json, parse_heartbeat,
    parse_manifest, sha256_hex,
)
from .masking import mask_key
from .rounds import Round, Rounds

PACKAGE = Path(__file__).resolve().parent
JOB_CODE = PACKAGE / 'box' / 'job.py'
STEPS_CODE = PACKAGE / 'box' / 'steps.py'
PLAN_BUILDER = PACKAGE.parents[1] / 'live' / 'pe_probe_plan.py'
# The trainer host package that the trainer bundle of the live lane ships.
BUNDLED_TRAINER_HOST = 'host-v834e'
# What a trainer host package must hold. The trainer checks the admission on the box.
TRAINER_LAUNCHER = 'mimo_host_trainer_v83.py'
ROOT_ADMISSION = 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json'
# The heartbeat numbers of the job on the box start above this (``box/job.py``). The host stays below.
BOX_HEARTBEAT_BASE = 1000000
URL_MAX_S = 7 * 86400 - 60
MIN_JOB_S = 3600.0


@dataclass(frozen=True)
class Tools:
    """The helpers of the live lane that prepare a box. ``host_tools`` gives the real ones."""

    pod_of: Callable[[Mapping[str, Any]], Any]
    boot: Callable[[Any], Any]
    probe_bundle: Callable[[Any], Any]
    trainer_bundle: Callable[[Any], Any]
    ship_inputs: Callable[..., Any]
    ship_trainer: Callable[[Any], Any]
    train_cmd: Callable[..., str]
    put: Callable[..., Any]
    put_tar: Callable[..., Any]


def host_tools() -> Tools:
    load = hostlink.load
    return Tools(
        pod_of=load('pod_of'), boot=load('boot_helpers'), probe_bundle=load('push_probe_bundle'),
        trainer_bundle=load('bundle_trainer'), ship_inputs=load('ship_inputs'), ship_trainer=load('ship_train_pit'),
        train_cmd=load('train_cmd'), put=load('put'), put_tar=load('put_tar'),
    )


@dataclass(frozen=True)
class AgentConfig:
    klass: recipe.BoxClass
    paths: recipe.BoxPaths
    transfer: recipe.Transfer
    # The serving layouts of the probe. One layout only: every probe of a run then has one fingerprint.
    plan: PL.ProbePlan
    # The public harness of the live lane. It names the sampling of the probe.
    harness: Mapping[str, Any]
    # A private directory on the validator host for short-lived files.
    scratch: Path
    # The signed package of the sharded trainer host, for a box class that needs it.
    sharded_host: Optional[Path] = None
    # The longest time from the first launch of a lease in which the agent keeps the lease alive.
    prepare_budget_s: float = 2700.0
    # The time that the end of a lease keeps for the outcome and the rest of the upload.
    head_reserve_s: int = 1500
    # The largest number of HEAD blocks that one box takes. None: no limit.
    max_head_blocks: Optional[int] = None

    def problems(self) -> List[str]:
        """What stops a launch for every lease. Empty: the agent can start."""
        wrong = []
        if not self.harness.get('harness') or self.harness.get('sampling_contract') is None:
            wrong.append('the public harness names no harness or no sampling contract: a probe would have no '
                         'fingerprint and no batch could get a verdict')
        if len(self.plan.layouts) != 1:
            wrong.append('the probe plan has more than one layout: the probes of one run must share one fingerprint')
        for path in (JOB_CODE, STEPS_CODE, PLAN_BUILDER):
            if not path.is_file():
                wrong.append('%s is missing' % path.name)
        if not Path(self.scratch).is_dir():
            wrong.append('the scratch directory is missing')
        if self.klass.trainer_host != BUNDLED_TRAINER_HOST:
            why = self.sharded_host_problem()
            if why:
                wrong.append(why)
        return wrong

    def sharded_host_problem(self) -> Optional[str]:
        name = self.klass.trainer_host
        if self.sharded_host is None:
            return 'the box class trains with %s and no package of it is given' % name
        folder = Path(self.sharded_host)
        if not (folder / TRAINER_LAUNCHER).is_file():
            return 'the package of %s has no trainer launcher' % name
        if not (folder / ROOT_ADMISSION).is_file():
            return 'the package of %s has no ROOT admission file: it is not shipped' % name
        return None


@dataclass(frozen=True)
class Lease:
    """What one lease needs, read on the validator host before the box is touched."""

    files: recipe.LeaseFiles
    epoch: int
    epoch_dir: Path
    rnd: Round
    manifest: StateManifest
    base: Index
    binding: Mapping[str, Any]
    # The batch was made for the base checkpoint of the lease. If not, the box trains nothing.
    bound: bool


class LiveBoxAgent:
    """``BoxAgent`` for a rented box. Safe for many threads; one launch for each lease at a time."""

    def __init__(self, *, run_id: str, store: ObjectStore, presigner: Any, providers: Mapping[str, Any], queue: Any,
                 rounds: Rounds, baselines: Baselines, tools: Tools, config: AgentConfig,
                 fleet: Optional[Fleet] = None, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] = lambda text: None):
        wrong = config.problems()
        if wrong:
            raise ValueError('the box agent cannot start: ' + '; '.join(wrong))
        self.run_id = run_id
        self.store = store
        self.layout = Layout(store.prefix)
        self.presigner = presigner
        self.providers = dict(providers)
        self.queue = queue
        self.rounds = rounds
        self.baselines = baselines
        self.tools = tools
        self.cfg = config
        self.fleet = fleet or Fleet()
        self.clock = clock
        self.log = log
        self._guard = threading.Lock()
        self._busy: Dict[int, threading.Lock] = {}
        self._first: Dict[int, float] = {}

    # -- the interface -------------------------------------------------------

    def launch(self, box_id: str, provider: str, spec: JobSpec) -> None:
        with self._guard:
            busy = self._busy.setdefault(spec.fence, threading.Lock())
            first = self._first.setdefault(spec.fence, self.clock())
        if not busy.acquire(blocking=False):
            raise BoxError('the box of this lease is being prepared')
        try:
            self._launch(box_id, provider, spec, first)
        except (BoxError, ShadowWriteError) as exc:
            self.log('launch %s f%d: %s' % (spec.batch_id, spec.fence, str(exc)[:300]))
            raise
        except Exception as exc:
            # A box that cannot be prepared must cost one lease, not stop the controller.
            self.log('launch %s f%d: %s: %s' % (spec.batch_id, spec.fence, type(exc).__name__, str(exc)[:300]))
            raise BoxError('%s: %s' % (type(exc).__name__, str(exc)[:300])) from None
        finally:
            busy.release()

    def _launch(self, box_id: str, provider_name: str, spec: JobSpec, first: float) -> None:
        if self.clock() - first > self.cfg.prepare_budget_s:
            raise BoxError('the time to prepare the box of this lease is spent')
        lease = self._lease(spec)
        provider = self.providers.get(provider_name)
        target = provider.ssh_target(box_id) if provider is not None else None
        if target is None:
            raise BoxError('the box has no address yet')
        pod = self.tools.pod_of(target)
        claim = shlex.quote(lease.files.at('claimed'))
        said = str(pod.run('test -d %s && echo CLAIMED || echo NEW' % claim, timeout=60, check=False))
        if 'CLAIMED' in said:
            return
        if 'NEW' not in said:
            raise BoxError('the box does not answer')
        with self._beating(spec, first):
            self._prepare(pod, lease)
            blocks = self._write(pod, spec, lease, first)
        self._start(pod, lease.files)
        self.log('launch %s f%d: the job runs (HEAD blocks: %s)' % (
            spec.batch_id, spec.fence, ', '.join(str(n) for n in blocks) or 'none'))

    # -- what the lease needs ------------------------------------------------

    def _base(self, spec: JobSpec) -> Tuple[StateManifest, Index]:
        lay = self.layout
        if spec.base_prefix != lay.state_prefix(spec.base_state) or spec.staging_prefix != lay.staging_prefix(
                spec.batch_id, spec.fence):
            raise BoxError('the lease names a prefix of another run')
        blob = self.store.get(lay.state_manifest(spec.base_state), max_bytes=MAX_COMPLETE_BYTES)
        if sha256_hex(blob.data) != spec.base_state:
            raise BoxError('the manifest of the base state has other bytes')
        doc = load_json(blob.data)
        manifest = parse_manifest(doc, batch_id=doc.get('batch_id'), parent=doc.get('parent'))
        objects = {ref.name: ref for ref in manifest.objects}
        if INDEX_NAME not in objects:
            raise BoxError('the base state has no file index')
        raw = self.store.get(lay.state_object(spec.base_state, INDEX_NAME), max_bytes=MAX_INDEX_BYTES).data
        if sha256_hex(raw) != objects[INDEX_NAME].sha256:
            raise BoxError('the file index of the base state has other bytes')
        return manifest, parse_index(raw, objects)

    def _lease(self, spec: JobSpec) -> Lease:
        live = self.queue.batch(spec.batch_id)
        if live is None:
            raise BoxError('the batch is not in the live queue')
        if live.ref != spec.batch_ref:
            # Never train other documents than the ones that were admitted.
            raise BoxError('the batch changed after it was admitted')
        rnd = self.rounds.of(live.epoch)
        manifest, base = self._base(spec)
        return Lease(
            files=recipe.lease_files(self.cfg.paths, spec), epoch=live.epoch, epoch_dir=Path(live.epoch_dir),
            rnd=rnd, manifest=manifest, base=base, binding={'checkpoint_id': live.checkpoint_id},
            bound=live.checkpoint_id == base.checkpoint_id,
        )

    def _blocks(self, spec: JobSpec, lease: Lease, budget_s: int) -> Tuple[int, ...]:
        """The HEAD blocks that the box of this lease runs after the probe of its batch."""
        if not lease.bound or budget_s < recipe.HEAD_MIN_S:
            # A box that trains nothing runs no step, and a lease with no time for a block starts none.
            return ()
        # Boxes that start on this base soon take blocks too: the blocks are split across the boxes.
        return self.baselines.claim(
            lease.rnd, lease.base.checkpoint_id, spec.fence, live=set(self.fleet.live()) | {spec.fence},
            others=self.fleet.others(spec.fence, spec.base_state), limit=self.cfg.max_head_blocks,
        )

    def _urls(self, spec: JobSpec, manifest: StateManifest) -> Dict[str, Any]:
        lay = self.layout
        expires = min(URL_MAX_S, spec.job_ttl_s + 3600.0)

        def get(key: str) -> str:
            return self.presigner.get_url(check_key(lay.prefix, key), expires)

        def put(key: str) -> str:
            return self.presigner.put_url(check_key(lay.prefix, key), expires)

        slots = slots_for(manifest, self.cfg.transfer.chunk_bytes) + [INDEX_NAME]
        return {
            'base': {
                'manifest': get(lay.state_manifest(spec.base_state)),
                'objects': {ref.name: get(lay.state_object(spec.base_state, ref.name)) for ref in manifest.objects},
            },
            'put': {
                'objects': {name: put(lay.staging_object(spec.batch_id, spec.fence, name)) for name in slots},
                'heartbeat': put(lay.staging_heartbeat(spec.batch_id, spec.fence)),
                'result': put(lay.staging_result(spec.batch_id, spec.fence)),
                'complete': put(lay.staging_complete(spec.batch_id, spec.fence)),
            },
        }

    # -- the box -------------------------------------------------------------

    def _prepare(self, pod: Any, lease: Lease) -> None:
        tools, klass = self.tools, self.cfg.klass
        tools.boot(pod)
        tools.probe_bundle(pod)
        tools.trainer_bundle(pod)
        if klass.trainer_host != BUNDLED_TRAINER_HOST:
            why = self.cfg.sharded_host_problem()
            if why:
                raise BoxError(why)
            quote = shlex.quote
            dest = '%s/%s' % (self.cfg.paths.fp32.rsplit('/', 1)[0], klass.trainer_host)
            pod.run('rm -rf %s' % quote(dest), timeout=120)
            tools.put_tar(pod, Path(self.cfg.sharded_host), dest)
            said = str(pod.run('test -f %s && test -f %s && echo HOST_OK' % (
                quote(dest + '/' + TRAINER_LAUNCHER), quote(dest + '/' + ROOT_ADMISSION)), timeout=60, check=False))
            if 'HOST_OK' not in said:
                raise BoxError('the trainer host package is not complete on the box')
        with tempfile.TemporaryDirectory(prefix='batchctl-in-', dir=str(self.cfg.scratch)) as tmp:
            tools.ship_inputs(pod, epoch_dir=lease.epoch_dir, epoch=lease.epoch, work_jd=Path(tmp))
        tools.ship_trainer(pod)

    def _write(self, pod: Any, spec: JobSpec, lease: Lease, first: float) -> Tuple[int, ...]:
        """Write the documents of the lease to the box. -> the HEAD blocks that the job runs."""
        cfg, files, rnd = self.cfg, lease.files, lease.rnd
        quote = shlex.quote

        def put(path: str, data: bytes, mode: str = '644') -> None:
            self.tools.put(pod, path, data, mode)

        def document(doc: Mapping[str, Any]) -> bytes:
            return json.dumps(doc, sort_keys=True, allow_nan=False).encode()

        pod.run('mkdir -p %s %s && chmod 700 %s' % (quote(files.private), quote(files.probe_dir),
                                                  quote(files.private)), timeout=60)
        put(files.job_code, JOB_CODE.read_bytes(), '700')
        put(files.steps_code, STEPS_CODE.read_bytes(), '700')
        put(files.plan_builder, PLAN_BUILDER.read_bytes())
        tool = probe_serve.ship(lambda path, text: put(path, text.encode()), files.probe_dir, cfg.plan)

        # The job ends before the controller calls the lease expired: the time that the preparation took is taken off.
        left = spec.job_ttl_s - (self.clock() - first) - 120.0
        if left < MIN_JOB_S:
            raise BoxError('the lease has too little time left for a job')
        timeout = cfg.plan.eval_timeout_s()
        budget = int(left - cfg.head_reserve_s)
        head = [
            recipe.HeadBlock(block=number, timeout_s=timeout, probe_cmd=probe_serve.detached_command(
                preamble='', job_dir=files.head_dir(number), candidate_id=lease.base.checkpoint_id,
                plan_path=files.head_plan(number), root=cfg.paths.probe_root))
            for number in self._blocks(spec, lease, budget)
        ]
        args = recipe.probe_args(
            indices=list(rnd.indices), plan_seed=rnd.plan_seed,
            mask_key=mask_key(rnd.plan_seed, self.run_id, spec.batch_id), env_id=rnd.env_id, k=rnd.k_cand,
            max_tokens=rnd.max_tokens, max_num_seqs=cfg.plan.primary.concurrency(), harness=cfg.harness,
            head=[{'block': one.block, 'plan_seed': rnd.block_seeds[one.block], 'k': rnd.k_block} for one in head],
        )
        job = recipe.job_document(
            dataclasses.replace(spec, job_ttl_s=left), paths=cfg.paths, klass=cfg.klass, base=lease.base,
            binding=lease.binding, train_cmd=self.tools.train_cmd(expected_master=spec.expected_fp32_master_digest),
            probe_cmd=probe_serve.detached_command(
                preamble='', job_dir=files.probe_dir, candidate_id=recipe.CANDIDATE_TOKEN, plan_path=files.plan,
                root=cfg.paths.probe_root),
            probe_timeout_s=timeout, urls=self._urls(spec, lease.manifest), transfer=cfg.transfer, head=head,
            tool_files=sorted(tool.values()), head_budget_s=max(budget, 0),
        )
        put(files.steps_config, document(recipe.steps_config(
            spec, paths=cfg.paths, klass=cfg.klass, base=lease.base, head=head)))
        put(files.probe_args, document(args), '600')
        put(files.job, document(job), '600')
        return tuple(one.block for one in head)

    def _start(self, pod: Any, files: recipe.LeaseFiles) -> None:
        quote = shlex.quote
        claim = quote(files.at('claimed'))
        said = str(pod.run(
            'cd %s && (nohup setsid python3 %s %s > %s 2>&1 < /dev/null &); '
            'for n in 1 2 3 4 5 6 7 8 9 10; do test -d %s && break; sleep 1; done; '
            'test -d %s && echo JOB_STARTED || echo JOB_ABSENT' % (
                quote(files.work), quote(files.job_code), quote(files.job), quote(files.job_log), claim, claim),
            timeout=120, check=False))
        if 'JOB_STARTED' not in said:
            raise BoxError('the job did not start on the box')

    # -- the heartbeat while the box is prepared -----------------------------

    def _beat(self, key: str, fence: int) -> None:
        try:
            try:
                seq = parse_heartbeat(self.store.get(key, max_bytes=MAX_HEARTBEAT_BYTES).data, fence=fence)
            except (NotFound, ValueError):
                seq = 0
            if seq + 1 < BOX_HEARTBEAT_BASE:
                self.store.put(key, heartbeat_body(fence=fence, seq=seq + 1, stage='prepare'))
        except StoreError:
            pass

    @contextlib.contextmanager
    def _beating(self, spec: JobSpec, first: float) -> Iterator[None]:
        stop = threading.Event()
        key = self.layout.staging_heartbeat(spec.batch_id, spec.fence)

        def beat() -> None:
            while not stop.wait(spec.heartbeat_every_s) and self.clock() - first <= self.cfg.prepare_budget_s:
                self._beat(key, spec.fence)

        self._beat(key, spec.fence)
        thread = threading.Thread(target=beat, name='prepare-f%d' % spec.fence, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=60.0)


__all__ = ['AgentConfig', 'LiveBoxAgent', 'Tools', 'host_tools']
