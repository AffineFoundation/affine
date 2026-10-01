"""Original EOG Calendar service behind a bounded, tool-only actor boundary.

No miner-selected image, endpoint, SQL, seed or verifier is accepted. SQL
grading stays in the operator process; the network-none service contains only
the original seed and image. This is a controlled native fixture experiment,
not a registration in the shared environment registry.
"""
import asyncio
import hashlib
import json
import re
import subprocess
import time
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

IMAGE = ('shivakrishnareddyma225/enterpriseops-gym-mcp-calendar@sha256:'
         '994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02')
REVISION = 'controlled-original-eog-calendar-v1'
LIMIT = 2_000_000
RPC = '''import sys,json,urllib.request,urllib.error
x=json.load(sys.stdin)
r=urllib.request.Request('http://127.0.0.1:8003'+x['path'],
 data=None if x['method']=='GET' else json.dumps(x['payload']).encode(),
 headers={'Content-Type':'application/json',**x['headers']},method=x['method'])
try:
 with urllib.request.urlopen(r,timeout=60) as f: print(json.dumps({'status':f.status,'body':f.read().decode()}))
except urllib.error.HTTPError as e: print(json.dumps({'status':e.code,'body':e.read().decode()}))
'''

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()

def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def logical_state(value):
    return {k:value[k] for k in ('service','database_id','table_counts','table_data')}

def replay(private_task,runtime,events,reward,expected_state_sha256):
    """Fresh exact original tool replay; never trusts a submitted state/score."""
    if not isinstance(events,list) or len(events)>32:raise ValueError('replay budget')
    session=NativeEOGSession(private_task,runtime=runtime)
    try:
        session.start()
        for event in events:
            if set(event)!={'name','arguments','observation'}:raise ValueError('replay event fields')
            actual=session.call(event['name'],event['arguments'])
            if actual!=event['observation']:raise ValueError('original native observation mismatch')
        actual_grade=session.grade()
        if type(reward) not in (int,float) or actual_grade['reward']!=reward:
            raise ValueError('original native reward mismatch')
        digest=sha(logical_state(session.state()))
        if digest!=expected_state_sha256:raise ValueError('original logical database state mismatch')
        return {'replayed_tools':len(events),'reward':actual_grade['reward'],
                'logical_database_sha256':digest,'exact_observations':True}
    finally:session.close()

def original_modules():
    # The existing source loader establishes only vendored import roots.
    from subnet.environments import build_spec, _taskset
    taskset = _taskset(build_spec('affine_eog', config={'taskset':{'domains':['calendar']}},num_samples=4))
    import enterprise_ops_gym_v1.toolset as toolset
    import affine_eog_v1.taskset as wrapper
    return taskset, toolset, wrapper

def validate_runtime(runtime):
    if set(runtime)!={'revision','image','base_image','shim_sha256','seed','clock'}:
        raise ValueError('runtime descriptor fields')
    from subnet.native_eog_clock import REVISION as clock_revision
    if runtime['revision']!=clock_revision or runtime['base_image']!=IMAGE:
        raise ValueError('runtime profile/base')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',runtime['image']):raise ValueError('immutable runtime image')
    if not re.fullmatch(r'[0-9a-f]{64}',runtime['seed']):raise ValueError('runtime seed')
    import datetime
    if datetime.datetime.fromisoformat(runtime['clock']).tzinfo!=datetime.timezone.utc:raise ValueError('runtime clock')
    if runtime['shim_sha256']!=hashlib.sha256(Path(__file__).with_name('native_eog_clock.py').read_bytes()).hexdigest():
        raise ValueError('approved runtime shim identity')
    return runtime

def container_command(name,runtime=None):
    if not re.fullmatch(r'affine-eog-controlled-[0-9a-f]{16}',name):
        raise ValueError('controlled container identity')
    cmd=['docker','run','-d','--name',name,'--label','affine.native-eog-controlled='+REVISION,
         '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
         '--pids-limit','64','--cpus','1','--memory','512m']
    for target,budget in [('/tmp','64m'),('/app/databases','64m'),('/app/mcp_databases','64m'),('/app/logs','16m')]:
        cmd += ['--tmpfs',target+':rw,noexec,nosuid,size='+budget]
    cmd+=['--env','API_BASE_URL=http://127.0.0.1:8003',
          '--env','FASTAPI_BASE_URL=http://127.0.0.1:8003']
    if runtime:
        validate_runtime(runtime)
        cmd+=['--env','AFFINE_EOG_SEED='+runtime['seed'],'--env','AFFINE_EOG_CLOCK='+runtime['clock']]
    return cmd+[runtime['image'] if runtime else IMAGE]

class NativeEOGSession:
    def __init__(self, private_task, timeout=90,runtime=None):
        self.private=private_task; self.data=private_task['data']; self.timeout=timeout
        services=self.data['services']
        if len(services)!=1 or services[0]['name']!='gym-calendar' or services[0]['image']!=IMAGE:
            raise ValueError('only approved original Calendar fixture supported')
        self.service=services[0]; self.name='affine-eog-controlled-'+uuid.uuid4().hex[:16]
        self.runtime=validate_runtime(runtime) if runtime else None
        self.database_id='vf_'+(sha(private_task)[:32] if runtime else uuid.uuid4().hex)
        self.started=False; self.events=[]
        _,self.original,self.wrapper=original_modules()

    def _rpc(self,path,payload=None,method='POST',headers=None):
        if not self.started or path not in ('/health','/mcp','/api/seed-database','/api/sql-runner','/api/database-state'):
            raise ValueError('operator RPC route/session')
        data=canonical(dict(path=path,payload=payload,method=method,headers=headers or {}))
        if len(data)>LIMIT: raise ValueError('RPC request budget')
        result=subprocess.run(['docker','exec','-i',self.name,'python','-B','-c',RPC],input=data,
            capture_output=True,timeout=self.timeout)
        if result.returncode: raise RuntimeError('native RPC failed: '+result.stderr.decode()[-1800:])
        if len(result.stdout)>LIMIT: raise ValueError('RPC response budget')
        envelope=json.loads(result.stdout)
        if envelope['status']>=400:
            import httpx
            request=httpx.Request(method,'http://isolated'+path)
            response=httpx.Response(envelope['status'],text=envelope['body'],request=request)
            response.raise_for_status()
        return json.loads(envelope['body'])

    def start(self):
        subprocess.run(container_command(self.name,self.runtime),check=True,capture_output=True,timeout=self.timeout)
        self.started=True
        for _ in range(60):
            try:self._rpc('/health',method='GET');break
            except RuntimeError:time.sleep(.5)
        else:raise TimeoutError('original Calendar service startup')
        seed=Path(self.service['seed_file']).read_text()
        value=self._rpc('/api/seed-database',dict(database_id=self.database_id,
             name='Controlled original EOG fixture',description='Original pinned seed',sql_content=seed))
        if not value.get('success',True):raise ValueError('original seed failed')
        self.headers={'Accept':'application/json, text/event-stream',
            **self.original.normalize_headers(self.service['context'],self.database_id)}
        listed=self._rpc('/mcp',dict(jsonrpc='2.0',id=1,method='tools/list',params={}),headers=self.headers)
        available={v['name']:v for v in listed['result']['tools']}
        allowed=self.original.partition_tools(tuple(self.data['selected_tools']),
                 tuple(self.data['restricted_tools']),[set(available)])[0]
        self.tools={name:available[name] for name in allowed}
        self.initial_state=self.state()
        return {'messages':[{'role':'system','content':self.data['system_prompt']},
                            {'role':'user','content':self.data['prompt']}],
                'tools':list(self.tools.values()),'task_id':self.data['name']}

    def call(self,name,arguments):
        if name not in self.tools or not isinstance(arguments,dict):
            raise ValueError('unavailable/restricted native tool')
        response=self._rpc('/mcp',dict(jsonrpc='2.0',id=uuid.uuid4().hex,method='tools/call',
            params={'name':name,'arguments':arguments}),headers=self.headers)
        result=response.get('result')
        observation=json.dumps(result.get('content',result) if result is not None else response['error'])
        self.events.append({'name':name,'arguments':arguments,'observation':observation})
        return observation

    def state(self):
        return self._rpc('/api/database-state',method='GET',headers=self.headers)

    def grade(self):
        # Execute the original function body with only its transport replaced by
        # inside-container HTTP. Comparisons, scalar conversion and failures are
        # therefore precisely the vendored original behavior.
        async def inside(url,payload,headers=None,timeout=60):
            if url!='http://isolated/api/sql-runner':raise ValueError('private grader route')
            return self._rpc('/api/sql-runner',payload,headers=headers)
        fn=self.original.run_verifier
        run=types.FunctionType(fn.__code__,{**fn.__globals__,'post_json':inside},fn.__name__,fn.__defaults__,fn.__closure__)
        from enterprise_ops_gym_v1.models import VerifierSpec
        service=SimpleNamespace(base_url='http://isolated',database_id=self.database_id,
              spec=SimpleNamespace(context=self.service['context']))
        async def evaluate():
            results=await asyncio.gather(*(run(VerifierSpec.model_validate(v),service,i)
                         for i,v in enumerate(self.data['verifiers'])))
            reward=await self.wrapper.AffineEnterpriseOpsTask.solved(None,
                         SimpleNamespace(state=SimpleNamespace(verifier_results=results)))
            return {'reward':reward,'results':results}
        return asyncio.run(evaluate())

    def isolation(self):
        value=json.loads(subprocess.check_output(['docker','inspect',self.name]))[0]
        host=value['HostConfig']
        return {'network_mode':host['NetworkMode'],'read_only':host['ReadonlyRootfs'],
                'binds':host['Binds'],'port_bindings':host['PortBindings'],
                'nonroot_user':value['Config']['User'],'cap_drop':host['CapDrop'],
                'image':value['Image'],'tmpfs':sorted(host['Tmpfs'])}

    def close(self):
        if self.started:
            subprocess.run(['docker','rm','-f',self.name],check=True,capture_output=True,timeout=self.timeout)
            self.started=False
