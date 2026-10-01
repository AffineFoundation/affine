"""Public original Spider database/bash actor; reference SQL stays elsewhere.

Only a trusted operator builds/selects the image and public descriptor. This
is a separate controlled harness prerequisite, not shared-epoch admission.
"""
import ast
import hashlib
import json
import re
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from .native_sql_isolation import BASE,SOURCE

REVISION='controlled-original-spider-public-bash-v1'
RPC='''import json,os,selectors,signal,subprocess,sys,time
x=json.load(sys.stdin)
p=subprocess.Popen(['bash','-lc',x['command']],cwd='/workspace',stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
s=selectors.DefaultSelector();s.register(p.stdout,selectors.EVENT_READ,'stdout');s.register(p.stderr,selectors.EVENT_READ,'stderr')
buffers={'stdout':bytearray(),'stderr':bytearray()};end=time.monotonic()+55;expired=False
while s.get_map():
 if time.monotonic()>=end:
  expired=True;os.killpg(p.pid,signal.SIGKILL);break
 for key,event in s.select(min(1,max(0,end-time.monotonic()))):
  block=os.read(key.fileobj.fileno(),65536)
  if not block:s.unregister(key.fileobj);continue
  target=buffers[key.data];target.extend(block[:max(0,131072-len(target))])
if expired:
 try:p.wait(timeout=2)
 except subprocess.TimeoutExpired:pass
else:p.wait()
print(json.dumps({'exit_code':p.returncode,'stdout':buffers['stdout'].decode('utf-8',errors='replace')[:32768],'stderr':buffers['stderr'].decode('utf-8',errors='replace')[:32768],'timed_out':expired}))
'''
START='''import os,pathlib,shutil,signal
shutil.copyfile('/opt/public/database.sqlite','/workspace/'+os.environ['AFFINE_SQL_DBID']+'.sqlite')
signal.pause()
'''

def public_descriptor(private):
    """Original public schema/question renderer; never emits reference SQL."""
    identifier=private['db_id']
    if not isinstance(identifier,str) or not re.fullmatch('[A-Za-z0-9_-]{1,120}',identifier):raise ValueError('original database identifier')
    db=Path(private['db_path']);raw=db.read_bytes()
    if len(raw)>32*1024*1024 or hashlib.sha256(raw).hexdigest()!=private['database_sha256']:raise ValueError('original database closure')
    tree=ast.parse(SOURCE.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'WORKDIR','SYSTEM'} for t in node.targets):nodes.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=='schema_of':nodes.append(node)
    namespace={'Path':Path,'sqlite3':sqlite3};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-public-spider>','exec'),namespace)
    schema=namespace['schema_of'](db)
    prompt=(f"Database: `/workspace/{identifier}.sqlite` (SQLite)\n\nSchema:\n"
            f"```sql\n{schema}\n```\n\nQuestion: {private['question']}\n\n"
            "Write one SQLite query that answers the question. Explore the "
            "data first if the schema leaves room for doubt, then give the "
            "final query in a single ```sql block.")
    return {'revision':REVISION,'db_id':identifier,'database_sha256':private['database_sha256'],
            'original_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'messages':[{'role':'system','content':namespace['SYSTEM']},{'role':'user','content':prompt}],
            'tools':[{'type':'function','function':{'name':'bash','description':'Run a command in the isolated task container.','parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}]}

def build(private,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    public=public_descriptor(private)
    (directory/'database.sqlite').write_bytes(Path(private['db_path']).read_bytes())
    (directory/'start.py').write_text(START)
    (directory/'Dockerfile').write_text('FROM '+BASE+'\nRUN apt-get update -qq && apt-get install -y -qq --no-install-recommends sqlite3 && rm -rf /var/lib/apt/lists/*\nWORKDIR /workspace\nCOPY database.sqlite /opt/public/database.sqlite\nCOPY start.py /opt/public/start.py\nENV AFFINE_SQL_DBID='+public['db_id']+'\nUSER 65534:65534\nENTRYPOINT ["python","-I","/opt/public/start.py"]\n')
    tag='affine-native-spider-actor:'+public['database_sha256'][:16]
    subprocess.run(['docker','build','-t',tag,str(directory)],capture_output=True,check=True,timeout=180)
    image=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
    return {'revision':REVISION,'image':image,'base_image':BASE,'database_sha256':public['database_sha256'],'db_id':public['db_id'],'rpc_sha256':hashlib.sha256(RPC.encode()).hexdigest(),'start_sha256':hashlib.sha256(START.encode()).hexdigest()},public

def command(name,runtime):
    if not re.fullmatch('affine-sql-public-[0-9a-f]{16}',name):raise ValueError('owned actor identity')
    if runtime.get('revision')!=REVISION or runtime.get('base_image')!=BASE or not re.fullmatch('sha256:[0-9a-f]{64}',runtime.get('image','')):raise ValueError('approved public runtime')
    if runtime.get('rpc_sha256')!=hashlib.sha256(RPC.encode()).hexdigest() or runtime.get('start_sha256')!=hashlib.sha256(START.encode()).hexdigest():raise ValueError('actor source closure')
    return ['docker','run','-d','--name',name,'--label','affine.native-sql-public='+REVISION,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','64','--cpus','1','--memory','2g','--tmpfs','/tmp:rw,noexec,nosuid,size=128m','--tmpfs','/workspace:rw,nosuid,size=128m,uid=65534,gid=65534,mode=1777',runtime['image']]

class PublicSQLActor:
    def __init__(self,runtime,public):
        if public['revision']!=REVISION or public['db_id']!=runtime['db_id'] or public['database_sha256']!=runtime['database_sha256']:raise ValueError('public actor database binding')
        if set(public)!={'revision','db_id','database_sha256','original_source_sha256','messages','tools'}:raise ValueError('public descriptor fields')
        self.runtime=runtime;self.public=public;self.name='affine-sql-public-'+uuid.uuid4().hex[:16];self.started=False
    def start(self):
        if self.started:raise ValueError('actor already running')
        subprocess.run(command(self.name,self.runtime),capture_output=True,check=True,timeout=30);self.started=True
        try:
            deadline=time.monotonic()+15
            while True:
                response=subprocess.run(['docker','exec',self.name,'python','-c',"import os;assert os.path.exists('/workspace/'+os.environ['AFFINE_SQL_DBID']+'.sqlite')"],capture_output=True,timeout=5)
                if response.returncode==0:break
                if time.monotonic()>deadline:raise TimeoutError('original database setup')
                time.sleep(.1)
            return self.public
        except Exception:self.close();raise
    def call(self,name,arguments):
        if not self.started or name!='bash' or not isinstance(arguments,dict) or set(arguments)!={'command'}:raise ValueError('public bash action')
        text=arguments['command']
        if not isinstance(text,str) or len(text)>16384:raise ValueError('original command budget')
        try:
            result=subprocess.run(['docker','exec','-i',self.name,'python','-I','-c',RPC],input=json.dumps(arguments).encode(),capture_output=True,check=True,timeout=60)
            if len(result.stdout)>1024*1024:raise ValueError('tool observation budget')
            value=json.loads(result.stdout)
            if value.pop('timed_out'):raise TimeoutError('isolated original bash deadline')
            return value
        except Exception:self.close();raise
    def close(self):
        if self.started:
            subprocess.run(['docker','rm','-f',self.name],capture_output=True,timeout=20);self.started=False
