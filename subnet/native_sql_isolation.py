"""Operator-only original Spider reward bodies in a bounded private container.

This is a grader prerequisite, not miner/harness admission or model evidence.
The original reward/query/result-comparison bodies are preserved. Only their
framework decorator is omitted from this standalone execution boundary.
"""
import hashlib
import json
import re
import subprocess
import uuid
from pathlib import Path

REVISION = 'controlled-original-spider-grader-v1'
BASE = 'python@sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b'
SOURCE = Path(__file__).parent/'vendor/legacy/rollouts/envs/affine_sql_v1/affine_sql_v1/taskset.py'
RUNNER = '''from __future__ import annotations
import ast,asyncio,base64,json,re,sqlite3,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
source=Path('/opt/affine-sql/taskset.py').read_text()
tree=ast.parse(source)
functions={'last_sql_block','run_query','_norm','results_match'}
constants={'QUERY_TIMEOUT_S','MAX_ROWS','SQL_BLOCK_RE'}
nodes=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):nodes.append(node)
 if isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
 if isinstance(node,ast.ClassDef) and node.name=='SqlTask':
  for method in node.body:
   if isinstance(method,ast.AsyncFunctionDef) and method.name=='correct':
    method.decorator_list=[];nodes.append(method)
assert len(nodes)==8
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-spider-reward>','exec'),globals())
raw=sys.stdin.buffer.read(48*1024*1024+1)
if len(raw)>48*1024*1024:raise ValueError('input budget')
request=json.loads(raw)
with tempfile.TemporaryDirectory() as temporary:
 db=Path(temporary)/'original.sqlite'
 db.write_bytes(base64.b64decode(request['database_base64'],validate=True))
 task=SimpleNamespace(data=SimpleNamespace(db_path=str(db),gold_sql=request['gold_sql'],ordered=request['ordered']))
 trace=SimpleNamespace(last_reply=request['reply'],info={})
 reward=asyncio.run(correct(task,trace))
 print(json.dumps({'reward':reward,'info':trace.info},allow_nan=False))
'''

def build(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    raw=SOURCE.read_bytes();(directory/'taskset.py').write_bytes(raw)
    (directory/'grade.py').write_text(RUNNER)
    (directory/'Dockerfile').write_text('FROM '+BASE+'\nWORKDIR /opt/affine-sql\nCOPY taskset.py grade.py /opt/affine-sql/\nUSER 65534:65534\nENTRYPOINT ["python","-I","/opt/affine-sql/grade.py"]\n')
    tag='affine-native-spider-grader:'+hashlib.sha256(raw+RUNNER.encode()).hexdigest()[:16]
    subprocess.run(['docker','build','--network','none','-t',tag,str(directory)],check=True,capture_output=True,timeout=120)
    image=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
    return {'revision':REVISION,'image':image,'base_image':BASE,'original_source_sha256':hashlib.sha256(raw).hexdigest(),'runner_sha256':hashlib.sha256(RUNNER.encode()).hexdigest()}

def command(name,runtime):
    if not re.fullmatch('affine-sql-controlled-[0-9a-f]{16}',name):raise ValueError('container identity')
    if runtime.get('revision')!=REVISION or runtime.get('base_image')!=BASE or not re.fullmatch('sha256:[0-9a-f]{64}',runtime.get('image','')):raise ValueError('approved grader runtime')
    if runtime.get('original_source_sha256')!=hashlib.sha256(SOURCE.read_bytes()).hexdigest() or runtime.get('runner_sha256')!=hashlib.sha256(RUNNER.encode()).hexdigest():raise ValueError('original source/runtime closure')
    return ['docker','run','-i','--name',name,'--label','affine.native-sql-controlled='+REVISION,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','32','--cpus','1','--memory','512m','--tmpfs','/tmp:rw,noexec,nosuid,size=128m',runtime['image']]

def grade(private,reply,runtime,timeout=35):
    import base64
    if not isinstance(reply,str) or len(reply.encode())>1024*1024:raise ValueError('reply budget')
    if type(timeout) not in (int,float) or not 0<timeout<=35:raise ValueError('wall-clock budget')
    database=Path(private['db_path']).read_bytes()
    if len(database)>32*1024*1024 or hashlib.sha256(database).hexdigest()!=private['database_sha256']:raise ValueError('original database closure')
    name='affine-sql-controlled-'+uuid.uuid4().hex[:16]
    cmd=command(name,runtime)
    request={'database_base64':base64.b64encode(database).decode(),'gold_sql':private['gold_sql'],'ordered':private['ordered'],'reply':reply}
    try:
        result=subprocess.run(cmd,input=json.dumps(request).encode(),capture_output=True,timeout=timeout,check=True)
        if len(result.stdout)>2*1024*1024:raise ValueError('grader output budget')
        value=json.loads(result.stdout)
        if value.get('reward') not in (0.,1.):raise ValueError('original binary reward')
        inspect=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        host=inspect['HostConfig']
        if inspect['Mounts'] or host['NetworkMode']!='none' or not host['ReadonlyRootfs'] or inspect['Config']['User']!='65534:65534':raise ValueError('grader isolation')
        return dict(value,runtime=runtime,database_sha256=private['database_sha256'],isolation={'network':host['NetworkMode'],'read_only':host['ReadonlyRootfs'],'host_mounts':inspect['Mounts'],'memory':host['Memory'],'pids':host['PidsLimit'],'user':inspect['Config']['User']})
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
