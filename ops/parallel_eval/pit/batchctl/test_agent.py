"""The box agent: what it does to a rented box for one lease.

The box is a fake pod that keeps what is written to it. The helpers of the
live lane are fakes that record their calls. The documents that the agent
writes are checked with the real code of the box side (``box/job.py``,
``box/steps.py``), and the round is the fixture of the live scorer.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from ops.parallel_eval.live import pe_box
from ops.parallel_eval.pit import probe_layout as PL
from ops.parallel_eval.pit import probe_serve
from ops.parallel_eval.pit.batchctl import agent as A
from ops.parallel_eval.pit.batchctl import recipe
from ops.parallel_eval.pit.batchctl.box import job as box_job
from ops.parallel_eval.pit.batchctl.box import steps as box_steps
from ops.parallel_eval.pit.batchctl.feed import LiveBatch
from ops.parallel_eval.pit.batchctl.guard import GuardedStore
from ops.parallel_eval.pit.batchctl.interfaces import BoxError, JobSpec
from ops.parallel_eval.pit.batchctl.layout import Layout, parse_heartbeat
from ops.parallel_eval.pit.batchctl.masking import mask_key
from ops.parallel_eval.pit.batchctl.r2 import CONTENT_TYPE
from ops.parallel_eval.pit.batchctl.sim import RawStore
from ops.parallel_eval.pit.batchctl.test_fakes import model_ids, put_model
from ops.parallel_eval.pit.batchctl.test_judge import EPOCH, RUN, FakeFleet, World
from ops.parallel_eval.pit.shadow import ShadowWriteError

BATCH = 'e%d' % EPOCH
FENCE = 7
HARNESS = {'harness': 'live-math-v3', 'mimo_answer_projection': 'boxed', 'sampling_contract': {'temperature': 1.0}}
TP2_ONLY = PL.ProbePlan((PL.RTX5090_TP2,))


class FakePod:
    def __init__(self) -> None:
        self.commands: List[str] = []
        self.files: Dict[str, bytes] = {}
        self.modes: Dict[str, str] = {}
        self.claimed = False
        self.answers = True
        self.starts = True

    def run(self, cmd: str, inp: Optional[bytes] = None, timeout: int = 600, check: bool = True, retries: int = 0) -> str:
        self.commands.append(cmd)
        if not self.answers:
            return ''
        if cmd.startswith('test -d ') and 'CLAIMED' in cmd:
            return 'CLAIMED\n' if self.claimed else 'NEW\n'
        if 'nohup setsid python3' in cmd:
            self.claimed = self.starts
            return 'JOB_STARTED\n' if self.starts else 'JOB_ABSENT\n'
        if 'HOST_OK' in cmd:
            return 'HOST_OK\n'
        return ''


class Calls:
    """The helpers of the live lane, as fakes that record what they are asked."""

    def __init__(self, pod: FakePod) -> None:
        self.pod = pod
        self.seen: List[Any] = []
        self.fail: Optional[str] = None

    def note(self, name: str, *more: Any) -> None:
        self.seen.append((name,) + more)
        if self.fail == name:
            raise RuntimeError('%s broke' % name)

    def tools(self) -> A.Tools:
        def put(pod: FakePod, dest: str, data: bytes, mode: str = '644') -> None:
            pod.files[dest], pod.modes[dest] = bytes(data), mode

        return A.Tools(
            pod_of=lambda target: self.note('pod_of', target['id']) or self.pod,
            boot=lambda pod: self.note('boot'), probe_bundle=lambda pod: self.note('probe_bundle'),
            trainer_bundle=lambda pod: self.note('trainer_bundle'),
            ship_inputs=lambda pod, **kw: self.note('ship_inputs', kw['epoch'], str(kw['epoch_dir'])),
            ship_trainer=lambda pod: self.note('ship_trainer'),
            train_cmd=pe_box._train_pit_inner_cmd, put=put,
            put_tar=lambda pod, local, remote: self.note('put_tar', str(local), remote),
        )

    def names(self) -> List[str]:
        return [call[0] for call in self.seen]


class FakeProvider:
    def __init__(self) -> None:
        self.target: Optional[Dict[str, Any]] = {'id': 'box-1', 'ip': '192.0.2.1', 'port': 22, 'huid': 'h', 'name': 'n'}

    def ssh_target(self, box_id: str) -> Optional[Dict[str, Any]]:
        return self.target


class FakeSigner:
    def get_url(self, key: str, expires_s: float) -> str:
        return 'https://signed.invalid/get/%s?e=%d' % (key, expires_s)

    def put_url(self, key: str, expires_s: float) -> str:
        return 'https://signed.invalid/put/%s?e=%d' % (key, expires_s)


class FakeQueue:
    def __init__(self, batch: Optional[LiveBatch]) -> None:
        self.found = batch

    def batch(self, batch_id: str) -> Optional[LiveBatch]:
        return self.found if self.found is not None and self.found.batch_id == batch_id else None


class Bench:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.world = World(tmp / 'round')
        self.raw = RawStore()
        self.store = GuardedStore(self.raw, RUN)
        self.layout = Layout(self.store.prefix)
        self.manifest, state_files = put_model(self.store, self.layout, 'head')
        self.checkpoint, self.entry, self.master = model_ids(state_files)
        self.batch = LiveBatch(epoch=EPOCH, uid=11, hotkey='5F' + 'a' * 46, epoch_dir=tmp / 'life' / 'epochs' / str(EPOCH),
                               checkpoint_id=self.checkpoint, documents=('ab' * 32,), source='k1-turn')
        self.queue = FakeQueue(self.batch)
        self.pod = FakePod()
        self.calls = Calls(self.pod)
        self.provider = FakeProvider()
        self.fleet = FakeFleet()
        self.logs: List[str] = []
        self.now = 1000.0
        (tmp / 'scratch').mkdir()
        host = tmp / 'trainer-host-v834f'
        host.mkdir()
        (host / A.TRAINER_LAUNCHER).write_text('# launcher\n')
        (host / A.ROOT_ADMISSION).write_text('{}')
        self.config = A.AgentConfig(
            klass=recipe.rtx5090x8(), paths=recipe.BoxPaths(),
            transfer=recipe.Transfer(content_type=CONTENT_TYPE, chunk_bytes=1 << 20), plan=TP2_ONLY, harness=HARNESS,
            scratch=tmp / 'scratch', sharded_host=host,
        )
        self.agent = self.make()

    def make(self, **changes: Any) -> A.LiveBoxAgent:
        return A.LiveBoxAgent(
            run_id=RUN, store=self.store, presigner=FakeSigner(), providers={'lium': self.provider}, queue=self.queue,
            rounds=self.world.rounds, baselines=self.world.baselines, tools=self.calls.tools(),
            config=dataclasses.replace(self.config, **changes), fleet=self.fleet, clock=lambda: self.now,
            log=self.logs.append,
        )

    def spec(self, fence: int = FENCE, **changes: Any) -> JobSpec:
        fields = dict(
            run_id=RUN, batch_id=BATCH, batch_ref=self.batch.ref, fence=fence, attempt=1,
            base_state=self.manifest.state_id, base_seq=0, base_prefix=self.layout.state_prefix(self.manifest.state_id),
            expected_fp32_master_digest=self.master, staging_prefix=self.layout.staging_prefix(BATCH, fence),
            heartbeat_every_s=60.0, job_ttl_s=6 * 3600.0,
        )
        fields.update(changes)
        return JobSpec(**fields)

    def launch(self, agent: Optional[A.LiveBoxAgent] = None, **changes: Any) -> recipe.LeaseFiles:
        spec = self.spec(**changes)
        (agent or self.agent).launch('box-1', 'lium', spec)
        return recipe.lease_files(self.config.paths, spec)

    def doc(self, path: str) -> Dict[str, Any]:
        return json.loads(self.pod.files[path])


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    return Bench(tmp_path / 'own')


def test_launch_prepares_the_box_as_the_serial_lane_does_and_starts_one_job(bench: Bench) -> None:
    files = bench.launch()
    assert bench.calls.names() == ['pod_of', 'boot', 'probe_bundle', 'trainer_bundle', 'put_tar', 'ship_inputs',
                                   'ship_trainer']
    assert bench.calls.seen[4] == ('put_tar', str(bench.config.sharded_host),
                                   '/workspace/affine-mimo-trainer-prod/host-v834f')
    assert bench.calls.seen[5] == ('ship_inputs', EPOCH, str(bench.batch.epoch_dir))
    starts = [cmd for cmd in bench.pod.commands if 'nohup setsid python3' in cmd]
    assert len(starts) == 1 and files.job_code in starts[0] and files.job in starts[0] and files.job_log in starts[0]
    # The code of the job is the code of this package, and the probe tool is the tool of the PIT package.
    assert bench.pod.files[files.job_code] == A.JOB_CODE.read_bytes()
    assert bench.pod.files[files.steps_code] == A.STEPS_CODE.read_bytes()
    assert bench.pod.files[files.plan_builder] == A.PLAN_BUILDER.read_bytes()
    assert bench.pod.files[files.probe_tool] == (Path(probe_serve.__file__).parent / 'probe_serve.py').read_bytes()
    layouts = bench.doc(files.probe_dir + '/' + probe_serve.LAYOUTS_NAME)['layouts']
    assert [(one['name'], one['gpus_per_replica']) for one in layouts] == [('rtx5090-tp2', 2)]
    assert bench.pod.modes[files.job] == '600' and bench.pod.modes[files.probe_args] == '600'
    assert any('the job runs' in text for text in bench.logs)


def test_the_job_is_a_job_that_the_box_accepts_with_the_train_and_probe_of_the_5090(bench: Bench) -> None:
    files = bench.launch()
    job = bench.doc(files.job)
    box_job.check_job(job)
    assert (job['run_id'], job['batch_id'], job['fence'], job['base_state']) == (
        RUN, BATCH, FENCE, bench.manifest.state_id)
    assert job['expected_fp32_master_digest'] == bench.master and job['binding'] == {'checkpoint_id': bench.checkpoint}
    assert job['require']['gpus'] == 8 and job['require']['gpu_name'] == 'RTX 5090'
    steps = {step['name']: step for step in job['steps']}
    train = steps['train']
    assert train['script'] == pe_box._train_pit_inner_cmd(expected_master=bench.master)
    assert train['env'] == {'PIT_ZERO3': '1', 'PIT_ALLOW_SHARDED_GPUS': '1', 'PIT_ZERO3_PROFILE': 'rtx5090-u6'}
    assert 'BOOTSTRAP_OVERRIDE=' not in train['script']
    probe = steps['probe']
    assert files.probe_tool in probe['script'] and recipe.CANDIDATE_TOKEN in probe['script']
    assert '--layouts %s/%s' % (files.probe_dir, probe_serve.LAYOUTS_NAME) in probe['script']
    assert probe['timeout_s'] == TP2_ONLY.eval_timeout_s()
    # The job ends before the controller calls the lease expired.
    assert job['job_ttl_s'] < 6 * 3600.0
    # The base goes where the trainer of the live lane reads it.
    assert job['place'][0] == {'prefix': 'bf16/', 'dir': '/workspace/pst/in/actor/' + bench.checkpoint}
    assert job['place'][1] == {'prefix': 'fp32/', 'dir': '/workspace/affine-mimo-trainer-prod/fp32-master/' + bench.entry}
    config = bench.doc(files.steps_config)
    assert (config['trainer_host'], config['sharded'], config['zero3_profile']) == ('host-v834f', True, 'rtx5090-u6')
    assert config['expected_master'] == bench.master and config['base_checkpoint'] == bench.checkpoint


def test_the_probe_arguments_are_the_round_of_the_live_gate(bench: Bench) -> None:
    files = bench.launch()
    args = bench.doc(files.probe_args)
    rnd = bench.world.rnd
    assert args['indices'] == list(rnd.indices) and args['plan_seed'] == rnd.plan_seed
    assert (args['k'], args['max_tokens'], args['env_id']) == (rnd.k_cand, rnd.max_tokens, rnd.env_id)
    assert args['max_num_seqs'] == PL.RTX5090_TP2.max_num_seqs and args['harness'] == HARNESS
    assert args['mask_key'] == mask_key(rnd.plan_seed, RUN, BATCH)
    local = bench.tmp / 'probe-args.json'
    local.write_bytes(bench.pod.files[files.probe_args])
    assert box_steps.load_probe_args({'probe_args': str(local)})['indices'] == list(rnd.indices)


def test_a_box_alone_takes_each_head_block_and_boxes_that_come_split_them(bench: Bench) -> None:
    files = bench.launch()
    job, args = bench.doc(files.job), bench.doc(files.probe_args)
    rnd = bench.world.rnd
    names = [step['name'] for step in job['steps']]
    assert names == ['clock', 'train', 'collect', 'plan', 'probe', 'headplan', 'head-0', 'head-1', 'head-2', 'outcome']
    assert args['head'] == [{'block': n, 'plan_seed': rnd.block_seeds[n], 'k': rnd.k_block} for n in range(3)]
    config = bench.doc(files.steps_config)
    assert sorted(config['head']['blocks']) == ['0', '1', '2']
    head = {step['name']: step for step in job['steps']}['head-1']
    assert files.head_dir(1) + '/probe_serve.py' in head['script'] and bench.checkpoint in head['script']
    assert head['script'].rstrip().endswith('exit 0') and recipe.CANDIDATE_TOKEN not in head['script']
    assert files.head_dir(1) in job['wipe']

    # A second run: other batches wait for a box, so each box takes one block.
    other = Bench(bench.tmp / 'second')
    other.fleet.fences, other.fleet.head = {1, 2}, other.manifest.state_id
    asked = []
    other.fleet.others = lambda fence, base_state: asked.append((fence, base_state)) or 5
    first = other.launch(fence=1)
    other.pod.claimed = False
    second = other.launch(fence=2)
    assert [block['block'] for block in other.doc(first.probe_args)['head']] == [0]
    assert [block['block'] for block in other.doc(second.probe_args)['head']] == [1]
    assert asked == [(1, other.manifest.state_id), (2, other.manifest.state_id)]
    # A launch that is sent again names the same blocks.
    other.pod.claimed = False
    assert [block['block'] for block in other.doc(other.launch(fence=1).probe_args)['head']] == [0]


def test_a_batch_that_was_made_for_another_checkpoint_takes_no_head_block(bench: Bench) -> None:
    bench.queue.found = dataclasses.replace(bench.batch, checkpoint_id='9' * 64)
    files = bench.launch(batch_ref=bench.queue.found.ref)
    job = bench.doc(files.job)
    assert job['binding'] == {'checkpoint_id': '9' * 64}
    assert [step['name'] for step in job['steps']] == ['train', 'collect', 'plan', 'probe', 'outcome']
    assert bench.world.baselines.free(bench.world.rnd, bench.checkpoint, {FENCE}) == [0, 1, 2]


def test_a_lease_that_runs_on_the_box_is_not_started_again(bench: Bench) -> None:
    bench.launch()
    before = (list(bench.calls.seen), dict(bench.pod.files))
    bench.launch()
    assert bench.calls.names()[len(before[0]):] == ['pod_of']
    assert bench.pod.files == before[1]
    assert len([cmd for cmd in bench.pod.commands if 'nohup setsid python3' in cmd]) == 1


def test_every_signed_url_is_inside_the_lease_or_the_base_state(bench: Bench) -> None:
    files = bench.launch()
    job = bench.doc(files.job)
    base, staging = bench.layout.state_prefix(bench.manifest.state_id), bench.layout.staging_prefix(BATCH, FENCE)
    gets = [job['base']['manifest']] + list(job['base']['objects'].values())
    puts = list(job['put']['objects'].values()) + [job['put'][name] for name in ('heartbeat', 'result', 'complete')]
    assert all(url.startswith('https://signed.invalid/get/' + base) for url in gets)
    assert all(url.startswith('https://signed.invalid/put/' + staging) for url in puts)
    assert set(job['base']['objects']) == {ref.name for ref in bench.manifest.objects}
    assert 'index.json' in job['put']['objects'] and len(job['put']['objects']) > len(bench.manifest.objects)


def test_host_private_data_goes_to_the_box_only(bench: Bench) -> None:
    files = bench.launch()
    rnd = bench.world.rnd
    private = [rnd.plan_seed, mask_key(rnd.plan_seed, RUN, BATCH)] + list(rnd.block_seeds)
    stored = b'\n'.join(bench.raw.objs.values()).decode('utf-8', 'replace')
    for text in private:
        assert text not in stored
        assert text in bench.pod.files[files.probe_args].decode()
        for path, data in bench.pod.files.items():
            assert path == files.probe_args or text.encode() not in data, path
    assert files.private in bench.doc(files.job)['wipe']


def test_the_host_keeps_the_lease_alive_while_it_prepares_the_box(bench: Bench) -> None:
    key = bench.layout.staging_heartbeat(BATCH, FENCE)
    seen = []
    agent = bench.make()
    real = agent.tools.trainer_bundle

    def slow(pod: Any) -> None:
        seen.append(parse_heartbeat(bench.store.get(key, max_bytes=4096).data, fence=FENCE))
        real(pod)

    agent.tools = A.Tools(**dict(agent.tools.__dict__, trainer_bundle=slow))
    bench.launch(agent)
    assert seen == [1] and seen[0] < A.BOX_HEARTBEAT_BASE == box_job.HEARTBEAT_BASE
    # A heartbeat of the job on the box is never replaced by the host.
    bench.store.put(key, json.dumps({'schema': 'affine-batchctl-heartbeat-v1', 'fence': FENCE, 'seq': 1000003,
                                     'stage': 'train'}).encode())
    agent._beat(key, FENCE)
    assert parse_heartbeat(bench.store.get(key, max_bytes=4096).data, fence=FENCE) == 1000003


@pytest.mark.parametrize('case', ['no_address', 'batch_changed', 'batch_gone', 'helper_fails', 'no_answer',
                                  'not_started', 'budget_spent', 'other_run', 'round_missing'])
def test_what_stops_a_launch_costs_one_try_and_never_stops_the_controller(bench: Bench, case: str) -> None:
    changes: Dict[str, Any] = {}
    if case == 'no_address':
        bench.provider.target = None
    elif case == 'batch_changed':
        changes['batch_ref'] = 'f' * 64
    elif case == 'batch_gone':
        bench.queue.found = None
    elif case == 'helper_fails':
        bench.calls.fail = 'trainer_bundle'
    elif case == 'no_answer':
        bench.pod.answers = False
    elif case == 'not_started':
        bench.pod.starts = False
    elif case == 'other_run':
        changes['staging_prefix'] = Layout('pit-shadow/batchctl/otherrun/').staging_prefix(BATCH, FENCE)
    elif case == 'round_missing':
        (bench.world.gate / 'epochs' / str(EPOCH) / 'SAMPLE.private.json').unlink()
    if case == 'budget_spent':
        with pytest.raises(BoxError):
            bench.provider.target = None
            bench.launch()
        bench.provider.target = {'id': 'box-1', 'ip': '192.0.2.1', 'port': 22}
        bench.now += bench.config.prepare_budget_s + 1
    with pytest.raises(BoxError):
        bench.launch(**changes)
    assert not [cmd for cmd in bench.pod.commands if 'nohup setsid python3' in cmd] or case == 'not_started'
    assert bench.logs and 'f%d' % FENCE in bench.logs[-1]
    if case in ('no_address', 'batch_changed', 'batch_gone', 'budget_spent', 'other_run', 'round_missing'):
        assert not bench.pod.commands and not bench.pod.files


def test_one_launch_for_a_lease_at_a_time_and_a_guard_error_is_never_hidden(bench: Bench) -> None:
    gate, inside = threading.Event(), threading.Event()
    agent = bench.make()

    def wait(pod: Any) -> None:
        inside.set()
        assert gate.wait(10)

    agent.tools = A.Tools(**dict(agent.tools.__dict__, boot=wait))
    first = threading.Thread(target=lambda: bench.launch(agent))
    first.start()
    assert inside.wait(10)
    with pytest.raises(BoxError, match='is being prepared'):
        bench.launch(agent)
    gate.set()
    first.join(10)
    assert bench.pod.claimed

    def refused(pod: Any) -> None:
        raise ShadowWriteError('a key outside the run')

    other = Bench(bench.tmp / 'second')
    agent = other.make()
    agent.tools = A.Tools(**dict(agent.tools.__dict__, boot=refused))
    with pytest.raises(ShadowWriteError):
        other.launch(agent)


@pytest.mark.parametrize('change, why', [
    (dict(harness={}), 'no harness or no sampling contract'),
    (dict(harness={'harness': 'x'}), 'no harness or no sampling contract'),
    (dict(plan=PL.shadow_plan('RTX5090')), 'more than one layout'),
    (dict(sharded_host=None), 'no package of it is given'),
    (dict(scratch=Path('/nonexistent/scratch')), 'scratch directory is missing'),
])
def test_the_agent_does_not_start_with_settings_that_fail_every_lease(bench: Bench, change: Dict[str, Any],
                                                                      why: str) -> None:
    with pytest.raises(ValueError, match=why):
        bench.make(**change)


def test_a_sharded_trainer_package_without_its_root_admission_is_never_shipped(bench: Bench) -> None:
    (Path(bench.config.sharded_host) / A.ROOT_ADMISSION).unlink()
    with pytest.raises(ValueError, match='no ROOT admission file'):
        bench.make()
    # The file can go away after the start: the launch then stops before the package goes to the box.
    (Path(bench.config.sharded_host) / A.ROOT_ADMISSION).write_text('{}')
    agent = bench.make()
    (Path(bench.config.sharded_host) / A.ROOT_ADMISSION).unlink()
    with pytest.raises(BoxError, match='no ROOT admission file'):
        bench.launch(agent)
    assert 'put_tar' not in bench.calls.names() and not bench.pod.claimed


def head_step(tmp: Path, *, budget_s: int, age_s: Optional[int], probe_cmd: str = 'echo PROBED') -> str:
    """Run the step of one HEAD block as the box does. -> what it prints."""
    work = tmp / 'lease'
    work.mkdir(parents=True, exist_ok=True)
    tool = work / 'tool.py'
    tool.write_text('# tool\n')
    files = recipe.LeaseFiles(str(work))
    if age_s is not None:
        Path(files.clock).write_text('%d\n' % (int(time.time()) - age_s))
    steps = recipe._head_steps(
        files=files, step='true %s\n', stop='true\n', tool_files=[str(tool)], budget_s=budget_s,
        head=[recipe.HeadBlock(block=0, probe_cmd=probe_cmd, timeout_s=7200)],
    )
    assert [step['name'] for step in steps] == ['headplan', 'head-0']
    assert steps[1]['timeout_s'] == 7200 + recipe.HEAD_STEP_MARGIN_S
    done = subprocess.run(['bash', '-c', steps[1]['script']], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_a_head_block_starts_only_while_the_lease_has_time_for_it(tmp_path: Path) -> None:
    own = tmp_path / 'own'
    # 3 h of budget and a job that is 1 h old: the block runs, in its own directory with the tool.
    assert head_step(own / 'a', budget_s=3 * 3600, age_s=3600) == 'PROBED\n'
    assert (own / 'a' / 'lease' / 'head-0' / 'tool.py').is_file()
    # Less than HEAD_MIN_S left: the block does not start, and the step does not fail.
    said = head_step(own / 'b', budget_s=3 * 3600, age_s=3 * 3600 - recipe.HEAD_MIN_S + 60)
    assert 'no time left' in said and 'PROBED' not in said
    assert not (own / 'b' / 'lease' / 'head-0').exists()
    # No clock on the box: no block.
    assert 'no time left' in head_step(own / 'c', budget_s=3 * 3600, age_s=None)
    # A block that starts is ended at the budget, not at its own longest time.
    said = head_step(own / 'd', budget_s=recipe.HEAD_MIN_S + 600, age_s=0, probe_cmd='echo left=$0')
    assert said == 'left=bash\n'
    script = recipe._head_steps(
        files=recipe.LeaseFiles('/workspace/x'), step='true %s\n', stop='true\n', tool_files=['/workspace/t.py'],
        budget_s=100, head=[recipe.HeadBlock(block=2, probe_cmd='echo x', timeout_s=7200)])[1]['script']
    assert 'if [ "$left" -gt 7200 ]; then left=7200; fi' in script and 'timeout -k 60 "$left" bash -c' in script
    with pytest.raises(recipe.RecipeError, match='no time'):
        recipe._head_steps(files=recipe.LeaseFiles('/workspace/x'), step='true %s\n', stop='true\n',
                           tool_files=['/workspace/t.py'], budget_s=0,
                           head=[recipe.HeadBlock(block=2, probe_cmd='echo x', timeout_s=7200)])


def test_a_lease_with_little_time_takes_no_head_block_and_a_limit_caps_the_claim(bench: Bench) -> None:
    # 70 minutes of lease: enough for a job, not enough for a HEAD block after the reserve.
    files = bench.launch(bench.make(head_reserve_s=3000), job_ttl_s=4200.0)
    job = bench.doc(files.job)
    assert [step['name'] for step in job['steps']] == ['train', 'collect', 'plan', 'probe', 'outcome']
    assert not bench.doc(files.probe_args).get('head')
    assert bench.world.baselines.free(bench.world.rnd, bench.checkpoint, {FENCE}) == [0, 1, 2]
    bench.pod.claimed = False
    with pytest.raises(BoxError, match='too little time'):
        bench.launch(fence=FENCE + 1, job_ttl_s=1800.0)
    assert not bench.pod.claimed

    other = Bench(bench.tmp / 'second')
    files = other.launch(other.make(max_head_blocks=1))
    assert [block['block'] for block in other.doc(files.probe_args)['head']] == [0]
    rnd = other.world.rnd
    assert other.world.baselines.claim(rnd, other.checkpoint, 9, live={FENCE, 9}, others=0, limit=0) == ()
    assert other.world.baselines.claim(rnd, other.checkpoint, 10, live={FENCE, 9, 10}, others=0, limit=1) == (1,)


def test_the_paths_of_a_box_are_the_paths_of_the_live_lane() -> None:
    paths = recipe.BoxPaths()
    assert (paths.pst, paths.fp32, paths.ckpt, paths.probe_root) == (pe_box.PST, pe_box.FP32, pe_box.KDIR, pe_box.BOX_ROOT)
    assert A.BUNDLED_TRAINER_HOST == pe_box.BOX_HOST.rsplit('/', 1)[1]
    assert paths.probe_root == PL.BOX_ROOT
