"""Trusted, versioned environment boundary for v1 Prime tasks and legacy artifacts.

Only operator-signed specs select code. Uploaded rollout fields never select an
import, source root, reward hook, runtime, or task data. Dataset snapshots should
be pinned with ``snapshot_spec`` before publishing a production challenge.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = PACKAGE_ROOT/'vendor/legacy'
RESEARCH_ROOT = PACKAGE_ROOT/'vendor/research'
DEFAULT_MASTER = dict(num_train_examples=4, num_eval_examples=0, code_length=1,
                      num_symbols=2, max_turns=2, use_think=True, seed=42,
                      use_candidate_reduction_reward=False)

# Exact positive-share legacy sources, not an inventory of package directories.
# Taskset id is distinct from module for the swerebench-v2-full alias.
_UPSTREAM = {
    'scaleswe': ('scaleswe_v1', 'ScaleSWETaskset'),
    'swerebench_v2': ('swerebench_v2_v1', 'SWERebenchV2Taskset'),
    'r2e_gym': ('r2e_gym_v1', 'R2EGymTaskset'),
    'multiswe': ('multiswe_v1', 'MultiSWETaskset'),
    'swesmith': ('swesmith_v1', 'SWESmithTaskset'),
    'swelego': ('swelego_v1', 'SWELegoTaskset'),
    'terminal_lego': ('terminal_lego_v1', 'TerminalLegoTaskset'),
    'terminal_bench_2': ('terminal_bench_2_v1', 'TerminalBench2Taskset'),
    'nl2repobench': ('nl2repobench_v1', 'NL2RepoTaskset'),
}
_WRAPPERS = {
    'affine_nl2lib': 'NL2LibTaskset', 'affine_math': 'MathTaskset',
    'affine_wiki': 'WikiTaskset', 'affine_agent': 'AffineAgentTaskset',
    'affine_when2call': 'When2CallTaskset', 'affine_tau2': 'AffineTau2Taskset',
    'affine_tau2_synth': 'AffineTau2SynthTaskset', 'affine_tau2_gen': 'AffineTau2GenTaskset',
    'affine_kb_synth': 'AffineKBSynthTaskset', 'affine_logic': 'LogicTaskset',
    'affine_trivia': 'TriviaTaskset', 'affine_trivia_abstain': 'TriviaAbstainTaskset',
    'affine_popqa_abstain': 'PopQAAbstainTaskset', 'affine_ifeval': 'IFEvalTaskset',
    'affine_science': 'ScienceTaskset', 'affine_scitext': 'SciTextTaskset',
    'affine_unscramble': 'UnscrambleTaskset', 'affine_prolog': 'PrologTaskset',
    'affine_wikispeedia': 'WikispeediaTaskset', 'affine_tmax': 'TMaxTaskset',
    'affine_eog': 'AffineEnterpriseOpsTaskset', 'affine_numina': 'AffineNuminaTaskset',
    'affine_sql': 'SqlTaskset', 'affine_autobench': 'AffineAutomationBenchTaskset',
    'affine_uuidctf': 'AffineUUIDCTFTaskset', 'affine_i3code': 'I3CodeTaskset',
    'affine_scicomp': 'SciCompTaskset', 'affine_i3math': 'I3MathTaskset',
    'affine_deshuffle': 'DeshuffleTaskset', 'affine_rgym': 'RGymTaskset',
    'affine_rcore': 'RCoreTaskset', 'affine_pydantic': 'PydanticTaskset',
    'affine_verbatim': 'VerbatimTaskset', 'affine_oolong': 'OolongTaskset',
    'affine_mrcr': 'MRCRTaskset', 'affine_docqa': 'DocQATaskset',
}
REGISTRY = dict(_UPSTREAM, **{n: (n+'_v1', cls) for n, cls in _WRAPPERS.items()})
# Operator snapshots may contain only these trusted, source-pinned subclasses.
# The legacy IFEval taskset deliberately mixes RLVR and IFEval row task types.
SNAPSHOT_TASK_CLASSES = {'affine_ifeval': {
    'RLVRTask': 'IFEvalTaskConfig', 'IFEvalRowTask': 'IFEvalTaskConfig'}}
TOOL_SOURCES = {'affine_wiki', 'affine_agent', 'affine_when2call', 'affine_wikispeedia',
                'affine_eog', 'affine_autobench', 'affine_tau2', 'affine_tau2_synth',
                'affine_tau2_gen', 'affine_kb_synth'}
SIMULATOR_SOURCES = {'affine_tau2', 'affine_tau2_synth', 'affine_tau2_gen', 'affine_kb_synth'}
SINGLE_CODE = {'affine_i3code', 'affine_scicomp'}
SINGLE_TEXT = {'affine_math', 'affine_logic', 'affine_trivia', 'affine_trivia_abstain',
               'affine_popqa_abstain', 'affine_ifeval', 'affine_science', 'affine_scitext',
               'affine_unscramble', 'affine_i3math', 'affine_rgym', 'affine_rcore',
               'affine_pydantic', 'affine_verbatim', 'affine_docqa'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _roots(config):
    legacy = config.get('legacy_root','bundled')
    research = config.get('research_root','bundled')
    return (LEGACY_ROOT if legacy=='bundled' else Path(legacy),
            RESEARCH_ROOT if research=='bundled' else Path(research))


def _snapshot_path(config):
    if config.get('math_corpus_asset') is not None:
        from .math_corpus_provider import asset_path
        return asset_path(config['math_corpus_asset'],PACKAGE_ROOT.parent)
    value = Path(config['task_snapshot'])
    return value if value.is_absolute() else PACKAGE_ROOT.parent/value


def _dependency_versions(source_id, tool_error_policy=None):
    from importlib.metadata import version,PackageNotFoundError
    names = ['verifiers']
    if tool_error_policy=='native-mcp-toolerror-observation-v1':names.append('mcp')
    elif tool_error_policy is not None:raise ValueError('unapproved tool error policy')
    if source_id=='affine_rgym':names+=['reasoning-gym','arckit','bfi','cellpylib','magiccube','pycosat','pyfiglet','pytz','zss']
    if source_id=='affine_verbatim':names+=['faker']
    versions={}
    for name in names:
        try:versions[name]=version(name)
        except PackageNotFoundError:raise RuntimeError(f'required environment dependency missing: {name}')
    return versions


def source_inventory(legacy_root=LEGACY_ROOT):
    import tomllib
    root = Path(legacy_root)
    source = root/'rollouts/rollouts/sources.toml'
    raw = tomllib.loads(source.read_text())
    entries = {}
    for name, record in raw['source'].items():
        active = record.get('share', 1)>0 and raw['mix'].get(record['group'],0)>0
        category = ('singleturn_text' if name in SINGLE_TEXT else 'singleturn_code_sandbox'
                    if name in SINGLE_CODE else 'multiturn_tool_simulator' if name in SIMULATOR_SOURCES
                    else 'tool' if name in TOOL_SOURCES else 'multiturn_sandbox')
        entries[name] = dict(active=active, module=REGISTRY.get(name, (None,None))[0],
            taskset_class=REGISTRY.get(name, (None,None))[1], classification=category,
            source=record, source_manifest_hash=_hash(source))
    return entries


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    version: str = 'prime-v1-1'
    adapter: str = 'prime_v1'
    config: dict = field(default_factory=dict)
    max_turns: int = 8
    max_output_tokens: int = 128
    num_samples: int = 4
    success_reward: float = 1.0
    source_hash: str = ''

    @classmethod
    def from_dict(cls, value):
        result = cls(**value)
        if result.adapter not in ('prime_v1', 'legacy_mastermind', 'resource_prime_v1', 'resource_prime_v1_controlled'):
            raise ValueError('unknown trusted environment adapter')
        from .math_corpus_provider import is_corpus_id
        if result.adapter == 'prime_v1' and result.id not in REGISTRY and not is_corpus_id(result.id):
            raise ValueError('unknown environment source')
        if not 1<=result.max_turns<=128 or not 1<=result.num_samples<=100000 or not 1<=result.max_output_tokens<=32768:
            raise ValueError('environment budget')
        if not math.isfinite(result.success_reward) or not result.source_hash:
            raise ValueError('unversioned environment or invalid threshold')
        if result.adapter in ('resource_prime_v1','resource_prime_v1_controlled'):
            from .resource_session import validate_outer_spec
            validate_outer_spec(result.to_dict())
        if is_corpus_id(result.id):
            from .math_corpus_provider import validate
            validate(result)
        return result

    def to_dict(self):
        return asdict(self)


def _source_hash(spec):
    if spec.adapter == 'legacy_mastermind':
        roots = [Path(__file__).resolve().parents[1]/'prototype/vendor/mastermind']
    else:
        legacy, research = _roots(spec.config)
        roots = [legacy/'rollouts/envs']
        if research.exists():
            roots.append(research/'environments')
    from .math_corpus_provider import is_corpus_id
    if is_corpus_id(spec.id):
        from .math_corpus_provider import validate
        validate(spec,PACKAGE_ROOT.parent)
    files = []
    for i, root in enumerate(roots):
        if not root.exists():
            raise FileNotFoundError(f'trusted environment source missing: {root}')
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix not in ('.pyc',) and '__pycache__' not in p.parts:
                files.append((i, str(p.relative_to(root)), _hash(p)))
    snapshot = spec.config.get('task_snapshot')
    if snapshot:
        files.append(('tasks', _hash(_snapshot_path(spec.config))))
    files.append(('adapter',_hash(__file__)))
    if is_corpus_id(spec.id):
        files.extend((name,_hash(PACKAGE_ROOT/name)) for name in ('math_corpus_provider.py','math_corpus_assets.py','math_corpus.py'))
    if spec.config.get('prolog_session_revision') is not None:
        from .native_common_dispatch import validate_prolog_binding
        validate_prolog_binding(spec)
        files.append(('native_common_dispatch',_hash(PACKAGE_ROOT/'native_common_dispatch.py')))
        for name,digest in sorted(spec.config['prolog_source_files'].items()):files.append(('native_prolog',name,digest))
    if spec.config.get('rcore_terminal_revision') is not None:
        from .native_rcore_common import validate_binding
        validate_binding(spec)
        files.append(('rcore_terminal_adapter',_hash(PACKAGE_ROOT/'native_rcore_common.py')))
        files.append(('rcore_terminal_public',_hash(PACKAGE_ROOT/'native_rcore_boundary.py')))
    policy=spec.config.get('tool_error_policy')
    if policy=='native-mcp-toolerror-observation-v1':
        files.append(('tool_error_adapter',_hash(PACKAGE_ROOT/'native_tool_errors.py')))
    elif policy is not None:raise ValueError('unapproved tool error policy')
    # Config is signed alongside source hash: changes to seed/index mapping matter.
    return hashlib.sha256(_canonical({'files':files,'config':spec.config,'id':spec.id,
                                      'version':spec.version,'adapter':spec.adapter})).hexdigest()


def build_spec(source_id, config=None, legacy_root=LEGACY_ROOT, research_root=None,
               num_samples=4, max_turns=8, max_output_tokens=128, success_reward=1.0):
    config = dict(config or {})
    config.setdefault('legacy_root', 'bundled' if Path(legacy_root)==LEGACY_ROOT else str(legacy_root))
    config.setdefault('research_root','bundled')
    config.setdefault('dependency_versions',_dependency_versions(source_id,config.get('tool_error_policy')))
    if research_root:
        config['research_root'] = str(research_root)
    environment_version='prime-v1-2-native-mcp-errors' if config.get('tool_error_policy')=='native-mcp-toolerror-observation-v1' else 'prime-v1-1'
    from .math_corpus_provider import is_corpus_id,VERSION as corpus_version
    if is_corpus_id(source_id):environment_version=corpus_version
    spec = EnvironmentSpec(source_id, version=environment_version, config=config, num_samples=num_samples,
                           max_turns=max_turns, max_output_tokens=max_output_tokens,
                           success_reward=success_reward)
    return EnvironmentSpec.from_dict(dict(spec.to_dict(), source_hash=_source_hash(spec)))


def legacy_spec(config=None):
    config = dict(config or DEFAULT_MASTER)
    spec = EnvironmentSpec('mastermind', version='legacy-mastermind-v1',
        adapter='legacy_mastermind', config=config, max_turns=config['max_turns'],
        num_samples=config['num_train_examples'], max_output_tokens=96)
    return EnvironmentSpec.from_dict(dict(spec.to_dict(), source_hash=_source_hash(spec)))


def legacy_harness(config=None):
    config = config or DEFAULT_MASTER
    prefix = '<think>Test a candidate.</think><guess>'
    return dict(version='text-tools-v1', policy='candidates', temperature=0.7, top_p=1.0,
        max_output_tokens=96, candidates=[prefix+str(n)+'</guess>' for n in range(config['num_symbols'])])


def _taskset(spec):
    from verifiers.v1 import TasksetConfig
    from verifiers.v1.utils.generic import concrete_type
    legacy, research = _roots(spec.config)
    for p in (legacy/'rollouts/envs').iterdir():
        if p.is_dir():
            sys.path.insert(0,str(p))
    if research.exists():
        for p in (research/'environments').glob('*/*'):
            if p.is_dir():
                sys.path.insert(0,str(p))
    from .math_corpus_provider import is_corpus_id,taskset_source
    module,name=taskset_source(spec) if is_corpus_id(spec.id) else REGISTRY[spec.id]
    cls = getattr(importlib.import_module(module+'.taskset'), name)
    config_cls = concrete_type(cls, TasksetConfig)
    if config_cls is None:
        raise RuntimeError('taskset config type unavailable')
    return cls(config_cls(**spec.config.get('taskset',{})))


def snapshot_spec(spec, path):
    """Operator-only materialization: pin exact trusted task data before publication."""
    if _source_hash(spec) != spec.source_hash:
        raise ValueError('source changed')
    rows = [dict(task_class=type(t).__name__, data=t.data.model_dump(mode='json'),
                 task_config=t.config.model_dump(mode='json'))
            for t in itertools.islice(iter(_taskset(spec)),spec.num_samples)]
    if len(rows)!=spec.num_samples:
        raise ValueError('insufficient trusted task data')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_bytes(_canonical(rows))
    try:reference=str(Path(path).resolve().relative_to(PACKAGE_ROOT.parent))
    except ValueError:reference=str(Path(path).resolve())
    config = dict(spec.config,task_snapshot=reference)
    modified = EnvironmentSpec.from_dict(dict(spec.to_dict(),config=config))
    return EnvironmentSpec.from_dict(dict(modified.to_dict(),source_hash=_source_hash(modified)))


class EnvironmentSession:
    def __init__(self,spec):
        self.spec=spec
        if _source_hash(spec)!=spec.source_hash:
            raise ValueError('trusted environment code or data hash mismatch')
        if spec.adapter=='prime_v1' and _dependency_versions(spec.id,spec.config.get('tool_error_policy'))!=spec.config.get('dependency_versions'):
            raise ValueError('environment dependency version mismatch')
        self.loop=asyncio.new_event_loop()
        self.runtime=None
        self.toolsets=[]
        self.done=False
        self.turns=0

    def _run(self,coro):
        return self.loop.run_until_complete(coro)

    def reset(self,index,seed):
        if type(index) is not int or not 0<=index<self.spec.num_samples:
            raise ValueError('environment index')
        self.close_runtime()
        self.turns=0
        self.done=False
        if self.spec.adapter=='legacy_mastermind':
            path=Path(__file__).resolve().parents[1]/'prototype/vendor/mastermind'
            sys.path.insert(0,str(path))
            self.master=importlib.import_module('mastermind')
            self.env=self.master.load_environment(**self.spec.config)
            row=self.env.dataset[index]
            self.state=dict(answer=row['answer'],trajectory=[],prompt_too_long=False)
            self._run(self.env.setup_state(self.state))
            self.messages=[dict(role='system',content=self.env.system_prompt),
                           dict(role='user',content='Start: make your first guess.')]
            return dict(messages=self.messages.copy(),tools=[],task_hash=hashlib.sha256(_canonical(row)).hexdigest(),task_name=str(index))
        self.taskset=_taskset(self.spec)
        snapshot=self.spec.config.get('task_snapshot')
        if snapshot:
            rows=json.loads(_snapshot_path(self.spec.config).read_text())
            if len(rows)!=self.spec.num_samples:
                raise ValueError('snapshot task count')
            row=rows[index]
            task_cls=self.taskset.task_type()
            config_cls=task_cls.config_type()
            if task_cls.__name__!=row['task_class']:
                allowed=SNAPSHOT_TASK_CLASSES.get(self.spec.id,{})
                if row['task_class'] not in allowed:
                    raise RuntimeError('heterogeneous task snapshot requires task-class adapter')
                import verifiers.v1 as vf
                module=importlib.import_module(REGISTRY[self.spec.id][0]+'.taskset')
                task_cls=getattr(module,row['task_class'])
                if not isinstance(task_cls,type) or not issubclass(task_cls,vf.Task):
                    raise ValueError('trusted snapshot task class is invalid')
                config_cls=getattr(module,allowed[row['task_class']])
                if not isinstance(config_cls,type) or not issubclass(config_cls,vf.TaskConfig):
                    raise ValueError('trusted snapshot task config is invalid')
            self.task=task_cls(task_cls.data_type()(**row['data']),config_cls(**row['task_config']))
        else:
            self.task=next(itertools.islice(iter(self.taskset),index,index+1),None)
            if self.task is None:
                raise ValueError('index absent from taskset')
        if self.spec.id in SIMULATOR_SOURCES:
            raise RuntimeError('tau2 requires its orchestrator/user-simulator harness; direct tool replay is not equivalent')
        self._run(self._prepare())
        return dict(messages=self.messages.copy(),tools=self.tools,task_hash=self.task.hash,
                    task_name=self.task.data.name)

    async def _prepare(self):
        import verifiers.v1 as vf
        from verifiers.v1.trace import TraceTask,AgentInfo
        from verifiers.v1.configs.agent import AgentConfig
        from verifiers.v1.state import state_cls
        from verifiers.v1.runtimes import DockerConfig,DockerRuntime,SubprocessConfig,SubprocessRuntime
        from mcp.server.fastmcp import FastMCP
        from .math_corpus_provider import is_corpus_id
        sandbox=self.task.NEEDS_CONTAINER or (self.spec.id not in SINGLE_TEXT|TOOL_SOURCES and not is_corpus_id(self.spec.id))
        if sandbox:
            self.runtime=DockerRuntime(DockerConfig(image=self.task.data.image or 'python:3.12-slim',
                workdir=self.task.data.workdir or '/app',cpu=1,memory=2))
        else:
            self.runtime=SubprocessRuntime(SubprocessConfig())
        await self.runtime.start()
        self.trace=vf.Trace(task=TraceTask(type=type(self.task).__name__,data=self.task.data,
            hash=self.task.hash,key=self.task.key), agent=AgentInfo(config=AgentConfig()),
            state=state_cls(type(self.task))())
        self.messages=[]
        if self.task.data.system_prompt:
            self.messages.append(dict(role='system',content=self.task.data.system_prompt))
        prompt=self.task.data.prompt
        if isinstance(prompt,str):self.messages.append(dict(role='user',content=prompt))
        elif prompt:self.messages.extend(m.model_dump(mode='json',exclude_none=True) for m in prompt)
        for message in self.messages:self._node(message)
        from verifiers.v1.utils.decorators import invoke
        await invoke(self.task.setup,dict(trace=self.trace,runtime=self.runtime))
        self.mcp=FastMCP('epoch-task')
        self.toolsets=self.task.toolsets(self.task.config)+self.taskset.toolsets(self.taskset.config)
        for tools in self.toolsets:
            await tools.setup()
            await tools.setup_task(self.task.data)
            async def state_pull():return self.trace.state
            async def state_push(before):return None
            tools._pull_state=state_pull
            tools._push_state=state_push
            tools.register(self.mcp)
        self.tools=[{'type':'function','function':dict(name=t.name,description=t.description or '',parameters=t.inputSchema)}
                    for t in await self.mcp.list_tools()]
        if sandbox:
            self.tools.append(dict(type='function',function=dict(name='bash',description='Run a command in the isolated task container.',
                parameters={'type':'object','properties':{'command':{'type':'string'}},'required':['command']})))
        if self.spec.id in SINGLE_CODE:
            self.tools=[]

    def _node(self,message,sampled=False):
        from pydantic import TypeAdapter
        from verifiers.v1.types import Message
        from verifiers.v1.graph import MessageNode
        self.trace.nodes.append(MessageNode(parent=len(self.trace.nodes)-1 if self.trace.nodes else None,
            message=TypeAdapter(Message).validate_python(message),sampled=sampled))

    def step(self,action):
        if self.done:raise ValueError('environment already completed')
        if isinstance(action,str):action={'text':action}
        self.turns+=1
        text=action.get('text','')
        if not isinstance(text,str):raise ValueError('action text')
        if self.spec.adapter=='legacy_mastermind':
            import verifiers as vf
            self.state['trajectory'].append({'completion':[vf.AssistantMessage(content=text)]})
            self.done=self._run(self.env.check_done(self.state))
            feedback=self.state['next_turn_response'][0].content
            reward=float(self.master.solved_reward(self.state))
            observations=[dict(role='user',content=feedback)]
            self.messages.extend([dict(role='assistant',content=text),*observations])
            return dict(observations=observations,done=self.done,reward=reward,
                        classification='positive' if self.done and reward>=self.spec.success_reward else 'negative' if self.done else 'neutral')
        return self._run(self._step(action))

    async def _step(self,action):
        from verifiers.v1.utils.decorators import invoke
        calls=action.get('tool_calls') or []
        if len(calls)>8:raise ValueError('tool-call budget')
        message={'role':'assistant','content':action.get('text','')}
        if calls:
            message['tool_calls']=[dict(id=c.get('id',f'call-{self.turns}-{i}'),name=c['name'],
                arguments=json.dumps(c.get('arguments',{})) if isinstance(c.get('arguments',{}),dict) else c['arguments']) for i,c in enumerate(calls)]
        self._node(message,sampled=True)
        observations=[]
        for call in message.get('tool_calls',[]):
            arguments=json.loads(call['arguments'])
            if call['name']=='bash' and any(t['function']['name']=='bash' for t in self.tools):
                command=arguments.get('command','')
                if not isinstance(command,str) or len(command)>16384:raise ValueError('command budget')
                result=await asyncio.wait_for(self.runtime.run(['bash','-lc',command],{}),60)
                content=json.dumps({'exit_code':result.exit_code,'stdout':result.stdout[:32768],'stderr':result.stderr[:32768]})
            else:
                if self.spec.config.get('tool_error_policy')=='native-mcp-toolerror-observation-v1':
                    from .native_tool_errors import call_tool
                    native=await asyncio.wait_for(call_tool(self.mcp._tool_manager,call['name'],arguments),60)
                    content=native['result']
                else:
                    content=await asyncio.wait_for(self.mcp._tool_manager.call_tool(call['name'],arguments),60)
                if not isinstance(content,str):content=json.dumps(content,default=str)
            observation=dict(role='tool',tool_call_id=call['id'],name=call['name'],content=content)
            observations.append(observation)
            self._node(observation)
        self.done=not calls or self.turns>=self.spec.max_turns
        for hook in self.task.hooks('stop'):
            result=invoke(hook,dict(trace=self.trace,task=self.task.data,runtime=self.runtime,state=self.trace.state))
            if inspect.isawaitable(result):result=await result
            if result:self.done=True
        reward=0.0
        if self.done:
            await invoke(self.task.finalize,dict(trace=self.trace,runtime=self.runtime))
            await self.task.score(self.trace,self.runtime)
            if any(v is None for v in self.trace.rewards.values()):raise RuntimeError('unscored task reward')
            reward=float(self.trace.reward)
            if not math.isfinite(reward):raise ValueError('nonfinite task reward')
        self.messages.extend([message,*observations])
        return dict(observations=observations,done=self.done,reward=reward,
                    classification='positive' if self.done and reward>=self.spec.success_reward else 'negative' if self.done else 'neutral')

    def close_runtime(self):
        for tools in self.toolsets:
            self._run(tools._exit_stack.aclose())
        self.toolsets=[]
        if self.runtime:
            self._run(self.runtime.teardown())
            self.runtime=None

    def close(self):
        self.close_runtime()
        self.loop.close()


def create_session(spec):
    checked=EnvironmentSpec.from_dict(spec) if isinstance(spec,dict) else spec
    if checked.config.get('rcore_terminal_revision') is not None and checked.adapter!='prime_v1':raise ValueError('native RCore marker cannot select alternate adapter')
    if checked.adapter in ('resource_prime_v1','resource_prime_v1_controlled'):
        from .resource_session import create_resource_session
        return create_resource_session(checked.to_dict())
    if checked.config.get('rcore_terminal_revision') is not None:
        from .native_rcore_common import CommonRCoreSession
        return CommonRCoreSession(checked)
    if checked.config.get('prolog_session_revision') is not None:
        from .native_common_dispatch import prolog_session
        return prolog_session(checked)
    return EnvironmentSession(checked)
