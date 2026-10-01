"""Authenticated isolated session bridge for the next adapter family.

No live environment module imports this bridge. Worker owns one session and
emits JSON-line responses; no dataset tool/grader code may select host imports.
"""
import base64,json,sys
from contextlib import redirect_stdout
from pathlib import Path
from .adapter_resources import validate_contract,prepare_environment,verify_provider_before_import,prepare_task,verify_docker_before_start
from .environment_resources import canonical,digest

SCHEMA='resource-environment-spec-v1'

def bridge_code_hash():
    root=Path(__file__).parent
    return digest(canonical({name:digest((root/name).read_bytes()) for name in ('resource_session.py','adapter_resources.py','environment_resources.py')}))

def normalize_spec(value):
    if set(value)!={'schema','environment','execution_resources','bridge_code_hash','source_hash'} or value['schema']!=SCHEMA:
        raise ValueError('unsupported resource environment spec')
    validate_contract(value['execution_resources'])
    if value['bridge_code_hash']!=bridge_code_hash():raise ValueError('resource bridge code mismatch')
    body={k:v for k,v in value.items() if k!='source_hash'}
    if digest(canonical(body))!=value['source_hash']:raise ValueError('resource environment source hash mismatch')
    return json.loads(canonical(value))

def build_spec(environment,execution_resources):
    body={'schema':SCHEMA,'environment':environment,'execution_resources':execution_resources,'bridge_code_hash':bridge_code_hash()}
    return normalize_spec({**body,'source_hash':digest(canonical(body))})

def authenticate_spec(envelope,authority):
    from nacl.signing import VerifyKey
    if set(envelope)!={'payload','signer','signature'} or envelope['signer']!=authority:raise ValueError('resource spec authority mismatch')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
    return normalize_spec(envelope['payload'])

def validate_outer_spec(value):
    if value.get('adapter') not in ('resource_prime_v1','resource_prime_v1_controlled'):raise ValueError('resource adapter expected')
    config=value.get('config',{})
    if set(config)!={'signed_resource_spec','authority'}:raise ValueError('unexpected outer resource adapter config')
    inner=authenticate_spec(config['signed_resource_spec'],config['authority'])
    base=inner['environment'];contract=inner['execution_resources']
    expected='resource_prime_v1_controlled' if contract['dependency_scope']=='provider-namespace-controlled' else 'resource_prime_v1'
    if value['adapter']!=expected:raise ValueError('resource adapter dependency scope mismatch')
    if value['id']!=base['id'] or value['version']!=contract['environment_version'] or value['source_hash']!=inner['source_hash']:
        raise ValueError('outer resource adapter identity mismatch')
    for name in ('max_turns','max_output_tokens','num_samples','success_reward'):
        if value[name]!=base[name]:raise ValueError('outer resource budget mismatch')
    return inner

class ResourceSessionProxy:
    """Local Runtime-facing session API; all provider code executes fresh worker.

    Role-local bindings never enter the public spec. They contain only exact
    descriptor/archive/cache paths and role, supplied by authenticated transport.
    """
    def __init__(self,spec,bindings):
        import subprocess,tempfile
        outer=spec.to_dict() if hasattr(spec,'to_dict') else spec
        validate_outer_spec(outer)
        if set(bindings)!={'descriptors','archives','cache','audience'}:raise ValueError('unexpected local resource binding fields')
        self.directory=tempfile.TemporaryDirectory(prefix='affine-resource-worker-')
        root=Path(self.directory.name);request={**bindings,'signed_spec':outer['config']['signed_resource_spec']}
        requestfile=root/'request.json';requestfile.write_bytes(canonical(request));requestfile.chmod(0o600)
        self.logfile=root/'stderr.log';self.logstream=self.logfile.open('w')
        repo=Path(__file__).resolve().parents[1]
        bootstrap='import sys,runpy;sys.path.insert(0,'+repr(str(repo))+');runpy.run_module("subnet.resource_session",run_name="__main__")'
        self.process=subprocess.Popen([sys.executable,'-I','-B','-c',bootstrap,'--request',str(requestfile),'--authority',outer['config']['authority']],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.logstream,text=True,cwd=repo)
    def _call(self,command):
        import os,selectors,time
        if self.process.poll() is not None:raise ValueError('resource worker stopped: '+self.logfile.read_text()[-4000:])
        self.process.stdin.write(json.dumps(command)+'\n');self.process.stdin.flush()
        line=read_protocol_line(self.process.stdout,timeout=600)
        if not line:raise ValueError('resource worker rejected request: '+self.logfile.read_text()[-4000:])
        result=json.loads(line)
        if result.get('ok') is not True:raise ValueError('resource worker response')
        return result['result']
    def reset(self,index,seed):return self._call({'op':'reset','index':index,'seed':seed})
    def step(self,action):return self._call({'op':'step','action':action})
    def close(self):
        import subprocess,signal
        if getattr(self,'process',None) is not None:
            if self.process.poll() is None:
                try:self.process.stdin.write('{"op":"close"}\n');self.process.stdin.flush();self.process.wait(timeout=10)
                except (BrokenPipeError,subprocess.TimeoutExpired):
                    self.process.send_signal(signal.SIGINT)
                    try:self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process.stdin.close();self.process.stdout.close();self.process=None
        if getattr(self,'logstream',None) is not None:self.logstream.close();self.logstream=None
        if getattr(self,'directory',None) is not None:self.directory.cleanup();self.directory=None

def create_resource_session(spec,bindings=None):
    if bindings is None:
        import os
        path=os.environ.get('AFFINE_RESOURCE_BINDINGS_FILE')
        if not path:raise ValueError('resource role-local transport bindings missing')
        bindings=json.loads(Path(path).read_text())
    return ResourceSessionProxy(spec,bindings)

def create_trusted_session(spec,descriptors,prepared):
    """Call only AFTER worker authenticates/guards resources; providers load now."""
    from .environments import EnvironmentSpec,EnvironmentSession
    resource_contract=spec['execution_resources'];audience=prepared['audience'];roots=prepared['resource_roots']
    class ResourceSession(EnvironmentSession):
        async def _prepare(self):
            bound=prepare_task(resource_contract,descriptors,roots,self.task.data.model_dump(mode='json'),
                self.task.config.model_dump(mode='json'),audience=audience)
            if bound['runtime_data'] is None:
                raise ValueError('private grader unavailable to miner; source needs a miner-setup factory')
            self.canonical_task_hash=bound['task_hash']
            self.task=type(self.task)(self.task.data.model_copy(update=bound['runtime_data']),self.task.config)
            # A sandbox must have a signed immutable image; uploaded TaskData
            # and mutable source defaults cannot substitute another image.
            from .environments import SINGLE_TEXT,TOOL_SOURCES
            sandbox=self.task.NEEDS_CONTAINER or self.spec.id not in SINGLE_TEXT|TOOL_SOURCES
            if sandbox:
                image=verify_docker_before_start(resource_contract)
                if image is None:raise ValueError('sandbox missing authenticated Docker content ID')
                self.task=type(self.task)(self.task.data.model_copy(update={'image':image}),self.task.config)
            await super()._prepare()
            self.trace.task.hash=self.canonical_task_hash
        def reset(self,index,seed):
            result=super().reset(index,seed);result['task_hash']=self.canonical_task_hash
            result['environment_version']=resource_contract['environment_version']
            return result
    return ResourceSession(EnvironmentSpec.from_dict(spec['environment']))

def read_protocol_line(stream,*,timeout,max_bytes=32*1024*1024):
    """Partial native writes cannot turn a deadline into blocking readline."""
    import os,selectors,time
    deadline=time.monotonic()+timeout;buffer=bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(stream,selectors.EVENT_READ)
        while not buffer.endswith(b'\n'):
            remaining=deadline-time.monotonic()
            if remaining<=0 or not selector.select(remaining):raise TimeoutError('resource protocol response deadline')
            block=os.read(stream.fileno(),min(65536,max_bytes-len(buffer)+1))
            if not block:
                if buffer:raise ValueError('incomplete resource protocol frame')
                return ''
            buffer.extend(block)
            if len(buffer)>max_bytes:raise ValueError('oversized resource protocol frame')
            if b'\n' in buffer[:-1]:raise ValueError('multiple resource protocol frames without request')
    return buffer.decode('utf-8')

def worker(request,authority,protocol=None):
    # Signature check occurs before reading descriptor identities, archives or
    # constructing an environment. Signing/bootstrap dependencies belong to
    # the separately approved interpreter platform, not untrusted provider code.
    spec=authenticate_spec(request['signed_spec'],authority)
    prepared=prepare_environment(spec['execution_resources'],request['descriptors'],request['archives'],request['cache'],audience=request['audience'])
    sys.path.insert(0,prepared['dependency_root'])
    verify_provider_before_import(spec['execution_resources'],request['descriptors'],prepared['dependency_root'])
    with redirect_stdout(sys.stderr):session=create_trusted_session(spec,request['descriptors'],prepared)
    try:
        for line in sys.stdin:
            command=json.loads(line)
            with redirect_stdout(sys.stderr):
                if command.get('op')=='reset':result=session.reset(command['index'],command['seed'])
                elif command.get('op')=='step':result=session.step(command['action'])
                elif command.get('op')=='close':break
                else:raise ValueError('unsupported resource worker operation')
            print(json.dumps({'ok':True,'result':result}),flush=True,file=protocol)
    finally:session.close()

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--authority',required=True);args=parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('resource worker must start with Python -I -B')
    import os
    # Retain the original output pipe solely for protocol. Provider Python and
    # native os.write(1,...)/subprocess stdout now target the diagnostic stream.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno());sys.stdout=sys.stderr
    try:worker(json.loads(Path(args.request).read_text()),args.authority,protocol)
    finally:protocol.close()
if __name__=='__main__':main()
