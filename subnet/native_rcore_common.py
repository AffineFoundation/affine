"""Prospective common-session bridge to a separately admitted private CPU grader."""
import base64,copy,hashlib,json,math,os,subprocess,sys
from pathlib import Path
from .native_rcore_boundary import PublicActor,digest,canonical,sha
from .resource_session import read_protocol_line
REVISION='original-rcore-common-terminal-v4'
VERSION='prime-native-rcore-terminal-v4'
BINDINGS='AFFINE_RCORE_TERMINAL_BINDINGS_FILE'

def validate_binding(spec):
    if spec.id!='affine_rcore' or spec.adapter!='prime_v1' or spec.version!=VERSION or spec.config.get('rcore_terminal_revision')!=REVISION:raise ValueError('qualified common RCore version')
    if set(spec.config)-{'rcore_terminal_revision','terminal_public_binding','seed'}:raise ValueError('public RCore spec excludes private provider/cache fields')
    if type(spec.config.get('seed',0))is not int or spec.config.get('seed',0)<0:raise ValueError('nonnegative nonbool native environment seed')
    p=spec.config.get('terminal_public_binding')
    fields={'authority','profile_sha256','guard_sha256','worker_sha256','original_environment_sha256','original_environment_source_hash','snapshot_sha256','source_bundle_sha256','provider_resource_id','task_hashes'}
    if not isinstance(p,dict)or set(p)!=fields:raise ValueError('exact public terminal identity')
    for name in fields-{'task_hashes'}:sha(p[name])
    if not isinstance(p['task_hashes'],list)or len(p['task_hashes'])!=spec.num_samples or any(sha(v)!=v for v in p['task_hashes']):raise ValueError('original public task identity registry')
    if spec.num_samples!=64 or spec.max_turns!=1 or spec.max_output_tokens!=512 or type(spec.success_reward)not in(int,float)or spec.success_reward!=1.:raise ValueError('qualified original RCore budgets')
    return copy.deepcopy(p)

def admit_descriptor(spec,descriptor):
    from nacl.signing import VerifyKey
    p=validate_binding(spec)
    if not isinstance(descriptor,dict)or set(descriptor)!={'revision','audience','operator_role','stage'} or descriptor['revision']!='rcore-terminal-role-local-v1' or descriptor['audience']not in('miner','verifier','trainer','evaluator')or descriptor['operator_role']!='trusted-terminal-grader':raise ValueError('qualified role-local terminal descriptor')
    stage=Path(descriptor['stage']).resolve()
    profile_path=stage/'signed-resource-profile.json';raw=profile_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=p['profile_sha256']:raise ValueError('approved resource profile bytes')
    doc=json.loads(raw)
    if doc.get('signer')!=p['authority']:raise ValueError('known resource authority')
    VerifyKey(bytes.fromhex(p['authority'])).verify(canonical(doc['payload']),base64.b64decode(doc['signature'],validate=True))
    body=doc['payload']
    if body.get('revision')!='original-rcore-trusted-terminal-resources-v1' or body.get('role')!='trusted-terminal-grader' or body.get('CPU_only')is not True or body.get('model_execution')is not False or body.get('optimizer_ran')is not False or body.get('chain_transactions')is not False:raise ValueError('private terminal audience/scope')
    if body.get('remote')!=str(stage/'package')or body.get('environment_source_hash')!=p['original_environment_source_hash']:raise ValueError('original native spec identity')
    for name in ['source_bundle_sha256','provider_resource_id','guard_sha256']:
        if body.get(name)!=p[name]:raise ValueError('original resource/source identity')
    for path,key in [('preimport-terminal-worker.py','guard_sha256'),('package/worker.py','worker_sha256'),('package/operator/environment.json','original_environment_sha256'),('package/operator/affine_rcore.tasks.json','snapshot_sha256')]:
        f=stage/path
        if f.is_symlink()or hashlib.sha256(f.read_bytes()).hexdigest()!=p[key]:raise ValueError('qualified terminal executable/fixture bytes')
    return stage,p

class CommonRCoreSession:
    def __init__(self,spec,descriptor=None):
        from .environments import _source_hash
        validate_binding(spec)
        if _source_hash(spec)!=spec.source_hash:raise ValueError('common adapter trusted source')
        if descriptor is None:
            path=os.environ.get(BINDINGS)
            if not path:raise ValueError('operator role-local terminal descriptor missing')
            descriptor=json.loads(Path(path).read_bytes())
        self.stage,self.binding=admit_descriptor(spec,descriptor);self.spec=spec;self.process=None;self.actor=None;self.log=None
    def _call(self,request):
        if self.process.poll()is not None:raise ValueError('private terminal worker refused')
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        line=read_protocol_line(self.process.stdout,timeout=120)
        if not line:raise ValueError('private terminal worker rejected admission')
        value=json.loads(line)
        if value.get('ok')is not True:raise ValueError('original grader failure cannot be negative')
        return value['result']
    def reset(self,index,seed):
        if type(index)is not int or not 0<=index<self.spec.num_samples or type(seed)is not int or seed<0:raise ValueError('original task index/seed')
        self.close();self.log=(self.stage/'common-terminal.private.log').open('a')
        self.process=subprocess.Popen([sys.executable,'-I','-B',str(self.stage/'preimport-terminal-worker.py'),'--profile',str(self.stage/'signed-resource-profile.json'),'--authority',self.binding['authority'],'--root',str(self.stage/'package')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
        try:
            public=self._call({'op':'reset','index':index,'seed':seed});self.index=index;self.seed=seed
            if public.get('task_id')!=self.binding['task_hashes'][index]or public.get('provider_resource_id')!=self.binding['provider_resource_id']or public.get('source_bundle_sha256')!=self.binding['source_bundle_sha256']:raise ValueError('original reset binding')
            self.actor=PublicActor(public,self._finish)
            return dict(messages=copy.deepcopy(public['messages']),tools=[],task_hash=public['task_id'])
        except BaseException:self.close();raise
    def _finish(self,text):
        result=self._call({'op':'finish','text':text})
        native=dict(observations=[],done=True,reward=result.get('reward'),classification=result.get('classification'))
        trace=dict(public_descriptor_sha256=digest(self.actor.descriptor),index=self.index,seed=self.seed,action={'text':text},native_terminal=native)
        if result.get('trace_sha256')!=digest(trace):raise ValueError('terminal task/index/seed/action/spec commitment')
        return result
    def step(self,action):
        if isinstance(action,str):action={'text':action}
        if not isinstance(action,dict)or set(action)!={'text'}or self.actor is None:raise ValueError('original text-only terminal action')
        try:
            result=self.actor.finish(action['text'])
            expected='positive'if result['reward']>=self.spec.success_reward else'negative'
            if result['classification']!=expected:raise ValueError('original classification threshold')
            return dict(observations=[],done=True,reward=result['reward'],classification=result['classification'])
        except BaseException:self.close();raise
    def close(self):
        if self.process is not None:
            if self.process.poll()is None:
                try:self.process.stdin.write('{"op":"close"}\n');self.process.stdin.flush();self.process.wait(timeout=10)
                except (BrokenPipeError,subprocess.TimeoutExpired):self.process.terminate();self.process.wait(timeout=10)
            self.process.stdin.close();self.process.stdout.close();self.process=None
        if self.log is not None:self.log.close();self.log=None
        self.actor=None
