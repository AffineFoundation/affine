"""Controlled original Prolog public actor/private native grader prerequisite.

The actor receives only original public messages and starter facts. Private
metadata/reference answers remain in the operator's original task object.
"""
import hashlib,json,re,subprocess,uuid,asyncio,types,time,selectors,os
from pathlib import Path
REVISION='original-prolog-public-actor-native-grader-v1'
SHIM_SHA='ab0ec3e391015d233c827d35ce2924bb3cd6a9019f014014174a0db2a1a0add2'
BASE='sha256:4697e5fc9ca9fd4825a42144ee05545490c18385dc88a4ac0f99c598dafea0c2'
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'subnet/vendor/legacy/rollouts/envs/affine_prolog_v1/affine_prolog_v1/taskset.py'
BASE_TASK=ROOT/'subnet/vendor/research/environments/reasoning/prolog_v1/prolog_v1/taskset.py'
VERIFY=BASE_TASK.parent/'verify.py'


def public_task(task):
    if task.data.kind!='nqueens':raise ValueError('initial native controls support original NQueens only')
    starter=task.data.starter_file
    if len(starter)>65536 or not re.search(r'board_size\(\d+\)\.',starter):raise ValueError('original public starter')
    return {'revision':REVISION,'task_name':task.data.name,'original_index':task.data.idx,'kind':task.data.kind,
            'messages':[{'role':'system','content':task.data.system_prompt},{'role':'user','content':task.data.prompt}],
            'starter_file':starter,'starter_sha256':hashlib.sha256(starter.encode()).hexdigest(),
            'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ORIGINAL,BASE_TASK,VERIFY)},
            'tools':[{'type':'function','function':{'name':'bash','description':'Run a command in the isolated task container.','parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}]}


def build(directory,shim):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if hashlib.sha256(shim.encode()).hexdigest()!=SHIM_SHA:raise ValueError('exact original setup shim')
    subprocess.run(['docker','image','inspect',BASE],check=True,capture_output=True)
    (directory/'setup.sh').write_text('#!/bin/bash\n'+shim+'\n')
    (directory/'Dockerfile').write_text('FROM swipl@'+BASE+'\nCOPY setup.sh /opt/affine-prolog-setup.sh\nRUN bash /opt/affine-prolog-setup.sh\nUSER 65534:65534\nWORKDIR /workspace\nENTRYPOINT ["bash","-lc","sleep infinity"]\n')
    tag='affine-native-prolog-control:'+hashlib.sha256(shim.encode()).hexdigest()[:16]
    subprocess.run(['docker','build','--network','none','--pull=false','-t',tag,str(directory)],check=True,capture_output=True,timeout=120)
    image=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
    return {'revision':REVISION,'base_image':BASE,'image':image,'shim_sha256':hashlib.sha256(shim.encode()).hexdigest()}


def isolation_command(name,runtime):
    if not re.fullmatch(r'affine-prolog-native-[0-9a-f]{16}',name) or runtime.get('revision')!=REVISION or runtime.get('base_image')!=BASE or runtime.get('shim_sha256')!=SHIM_SHA or not re.fullmatch(r'sha256:[0-9a-f]{64}',runtime.get('image','')):raise ValueError('qualified actor runtime')
    return ['docker','run','-d','--name',name,'--label','affine.native-prolog='+REVISION,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--cpus','1','--memory','512m','--pids-limit','64','--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--tmpfs','/workspace:rw,nosuid,size=16m,uid=65534,gid=65534,mode=1777',runtime['image']]


class PublicActor:
    def __init__(self,runtime,public):
        expected={'revision','task_name','original_index','kind','messages','starter_file','starter_sha256','source_files','tools'}
        if set(public)!=expected or public['revision']!=REVISION or public['kind']!='nqueens' or hashlib.sha256(public['starter_file'].encode()).hexdigest()!=public['starter_sha256']:raise ValueError('public actor descriptor')
        self.runtime=runtime;self.public=public;self.name='affine-prolog-native-'+uuid.uuid4().hex[:16];self.started=False
    def start(self):
        subprocess.run(isolation_command(self.name,self.runtime),check=True,capture_output=True,timeout=30);self.started=True
        try:
            subprocess.run(['docker','exec','-i',self.name,'bash','-lc','cat > /workspace/solution.pl'],input=self.public['starter_file'].encode(),check=True,capture_output=True,timeout=10)
        except BaseException:self.close();raise
    def shell(self,command,timeout=65):
        if not self.started or not isinstance(command,str) or len(command)>32768 or not 0<timeout<=65:raise ValueError('bounded public command')
        p=subprocess.Popen(['docker','exec',self.name,'bash','-lc',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        streams=selectors.DefaultSelector();streams.register(p.stdout,selectors.EVENT_READ,'stdout');streams.register(p.stderr,selectors.EVENT_READ,'stderr')
        buffers={'stdout':bytearray(),'stderr':bytearray()};total=0;deadline=time.monotonic()+timeout
        try:
            while streams.get_map():
                remaining=deadline-time.monotonic()
                if remaining<=0:raise subprocess.TimeoutExpired('owned Prolog actor command',timeout)
                for key,_ in streams.select(min(remaining,.1)):
                    block=os.read(key.fileobj.fileno(),65536)
                    if not block:streams.unregister(key.fileobj);continue
                    total+=len(block)
                    if total>200000:raise ValueError('actor output budget')
                    target=buffers[key.data];target.extend(block[:max(0,32768-len(target))])
            code=p.wait(timeout=max(.1,deadline-time.monotonic()))
        except BaseException:
            # Release the exec client's output pipes before requesting container
            # removal: an undrained infinite-output exec can block Docker cleanup.
            if p.poll() is None:p.kill()
            p.wait(timeout=10)
            streams.close();p.stdout.close();p.stderr.close()
            self.close()
            raise
        finally:
            streams.close();p.stdout.close();p.stderr.close()
        return {'exit_code':code,'stdout':buffers['stdout'].decode(errors='replace'),'stderr':buffers['stderr'].decode(errors='replace')}
    def close(self):
        if not self.started:return
        result=subprocess.run(['docker','rm','-f',self.name],capture_output=True,timeout=30)
        if result.returncode and b'No such container' not in result.stderr:
            raise RuntimeError('owned Prolog actor removal failed')
        self.started=False


def nqueens_command(public,negative=False):
    # Only the public starter's board_size fact supplies the problem parameter.
    if public.get('kind')!='nqueens':raise ValueError('NQueens public policy')
    n=int(re.search(r'board_size\((\d+)\)\.',public['starter_file']).group(1))
    if not 4<=n<=20:raise ValueError('bounded board')
    valid=('solve(Qs) :- board_size(N), length(Qs,N), Qs ins 1..N, all_distinct(Qs), safe(Qs), once(labeling([ff],Qs)).\nsafe([]).\nsafe([Q|Qs]) :- diagonal(Q,Qs,1), safe(Qs).\ndiagonal(_,[],_).\ndiagonal(Q,[R|Rs],D) :- abs(Q-R) #\\= D, E #= D+1, diagonal(Q,Rs,E).\n')
    invalid='solve(Qs) :- board_size(N), length(Qs,N), maplist(=(1),Qs).\n'
    replacement=invalid if negative else valid
    starter=public['starter_file']
    if starter.count('solve(_) :- fail.  %% TODO: replace this')!=1:raise ValueError('original solve stub')
    content=starter.replace('solve(_) :- fail.  %% TODO: replace this',replacement)
    return "cat > /workspace/solution.pl <<'AFFINE_PUBLIC_PROLOG'\n"+content+"\nAFFINE_PUBLIC_PROLOG\ntimeout 60 swipl -g \"solve(X), write_canonical(X), nl, halt\" -t \"halt(1)\" /workspace/solution.pl"


async def grade_original(task,actor):
    from verifiers.v1.utils.decorators import invoke
    class Runtime:
        async def run(self,argv,environment):
            if argv[:2]!=['bash','-lc'] or len(argv)!=3:raise ValueError('original grading command')
            result=actor.shell(argv[2]);return types.SimpleNamespace(**result)
    trace=types.SimpleNamespace(errors=[],info={})
    reward=await invoke(task.solved,{'trace':trace,'runtime':Runtime()})
    return {'reward':float(reward),'original_grader_info':trace.info}
