"""Versioned public actor RPC and operator-only original EOG grader boundary.

Controlled co-located pilot: the model client receives no fixture, SQL, seed
bytes, Docker access, or operator capability. Host-root/process isolation is
not asserted; untrusted external miners need a separately hosted broker.
"""
import asyncio
import copy
import hashlib
import hmac
import json
import secrets
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from subnet.native_eog_isolation import NativeEOGSession, canonical, logical_state, sha, validate_runtime

REVISION='original-eog-public-actor-private-grader-v3-terminal'
LIMIT=2_000_000
PUBLIC_FIELDS={'schema','revision','task_id','messages','tools','runtime','seed_sha256','source_files'}

def validate_public(descriptor):
    if set(descriptor)!=PUBLIC_FIELDS or descriptor['schema']!=3 or descriptor['revision']!=REVISION:
        raise ValueError('public EOG descriptor fields/revision')
    validate_runtime(descriptor['runtime'])
    if not isinstance(descriptor['messages'],list) or [m.get('role') for m in descriptor['messages']]!=['system','user']:
        raise ValueError('original public messages')
    for message in descriptor['messages']:
        if set(message)!={'role','content'} or not isinstance(message['content'],str):raise ValueError('public message schema')
    if not isinstance(descriptor['tools'],list) or not descriptor['tools']:raise ValueError('public tools')
    names=[t.get('name') for t in descriptor['tools']]
    if len(set(names))!=len(names):raise ValueError('duplicate public tool schema')
    if len(canonical(descriptor))>LIMIT:raise ValueError('public descriptor budget')
    return descriptor

def source_files():
    paths=['subnet/native_eog_isolation.py','subnet/native_eog_clock.py','subnet/native_eog_split.py',
           'subnet/vendor/research/environments/tool_use/enterprise_ops_gym_v1/enterprise_ops_gym_v1/toolset.py',
           'subnet/vendor/legacy/rollouts/envs/affine_eog_v1/affine_eog_v1/taskset.py']
    return {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}

class OperatorBroker:
    """Only the trusted broker owner supplies private fixture/runtime config."""
    def __init__(self,private_task,runtime):
        self.session=NativeEOGSession(private_task,runtime=runtime)
        initial=self.session.start()
        self.public=validate_public({'schema':3,'revision':REVISION,'task_id':initial['task_id'],
             'messages':initial['messages'],'tools':initial['tools'],'runtime':copy.deepcopy(runtime),
             'seed_sha256':hashlib.sha256(Path(private_task['data']['services'][0]['seed_file']).read_bytes()).hexdigest(),
             'source_files':source_files()})
        self.private_grader={'revision':REVISION,'task_id':initial['task_id'],
             'public_descriptor_sha256':sha(self.public),'source_files':source_files(),
             'verifiers':copy.deepcopy(private_task['data']['verifiers'])}
        self.actor_capability=secrets.token_hex(32);self.operator_capability=secrets.token_hex(32)
        self.session_id=secrets.token_hex(16)
        self.lock=threading.Lock();self.sealed=False;self.calls=0;self.terminal=None
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    length=int(self.headers.get('Content-Length','-1'))
                    if not 0<=length<=LIMIT:raise ValueError('request budget')
                    value=json.loads(self.rfile.read(length))
                    auth=self.headers.get('Authorization','')
                    if self.path=='/actor' and hmac.compare_digest(auth,'Bearer '+owner.actor_capability):role='actor'
                    elif self.path=='/operator' and hmac.compare_digest(auth,'Bearer '+owner.operator_capability):role='operator'
                    else:
                        self.respond(403,{'error':'capability denied'});return
                    with owner.lock:result=owner.dispatch(role,value)
                    self.respond(200,{'result':result})
                except PermissionError:self.respond(403,{'error':'operation denied'})
                except ValueError:self.respond(400,{'error':'invalid actor/operator request'})
                except Exception:self.respond(500,{'error':'native broker operation failed'})
            def respond(self,status,value):
                data=canonical(value)
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start();self.endpoint='http://127.0.0.1:'+str(self.server.server_address[1])

    def dispatch(self,role,value):
        if not isinstance(value,dict):raise ValueError('RPC object')
        operation=value.get('operation')
        if role=='actor':
            if operation=='reset' and set(value)=={'operation'}:
                if self.calls or self.sealed:raise ValueError('cannot reset modified task')
                return copy.deepcopy(self.public)
            if operation=='close' and set(value)=={'operation'}:
                self.sealed=True;return {'sealed':True}
            if operation=='finish' and set(value)=={'operation'}:
                if self.terminal is None:
                    self.sealed=True
                    grade=self.session.grade()
                    self.terminal={'sealed':True,'reward':grade['reward'],'task_id':self.public['task_id'],
                        'session_id':self.session_id,'public_descriptor_sha256':sha(self.public),
                        'transcript_sha256':sha(self.session.events)}
                return copy.deepcopy(self.terminal)
            if operation=='call' and set(value)=={'operation','name','arguments'}:
                if self.sealed or self.calls>=32:raise ValueError('sealed actor/call budget')
                name=value['name'];arguments=value['arguments']
                if not isinstance(name,str) or len(name)>128 or not isinstance(arguments,dict):raise ValueError('tool action')
                if name not in self.session.tools:
                    # Return the exact pinned FastMCP unknown-tool error; no
                    # private HTTP route is invoked even for SQL-shaped names.
                    from mcp.server.fastmcp.tools.tool_manager import ToolManager
                    from mcp.server.fastmcp.exceptions import ToolError
                    try:asyncio.run(ToolManager().call_tool(name,arguments))
                    except ToolError as error:return {'tool_error':str(error)}
                    raise AssertionError('unexpected native tool resolution')
                observation=self.session.call(name,arguments);self.calls+=1
                return {'observation':observation}
            raise PermissionError('actor operation not allowed')
        if role=='operator' and operation=='grade' and set(value)=={'operation'}:
            result=self.session.grade()
            return {'grade':result,'logical_database_sha256':sha(logical_state(self.session.state())),
                    'public_descriptor_sha256':sha(self.public),'calls':self.calls,'sealed':self.sealed,
                    'session_id':self.session_id,'transcript_sha256':sha(self.session.events)}
        raise PermissionError('operator operation not allowed')

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5);self.session.close()

def request(endpoint,capability,role,value):
    if role not in ('actor','operator'):raise ValueError('RPC role')
    data=canonical(value)
    if len(data)>LIMIT:raise ValueError('RPC budget')
    req=urllib.request.Request(endpoint+'/'+role,data=data,
        headers={'Content-Type':'application/json','Authorization':'Bearer '+capability},method='POST')
    with urllib.request.urlopen(req,timeout=120) as response:raw=response.read(LIMIT+1)
    if len(raw)>LIMIT:raise ValueError('RPC response budget')
    return json.loads(raw)['result']

class PublicActor:
    """Miner-side client: stores only its endpoint, actor capability and task."""
    def __init__(self,endpoint,actor_capability,expected_descriptor_sha256):
        self.endpoint=endpoint;self.capability=actor_capability;self.expected=expected_descriptor_sha256
    def reset(self):
        value=validate_public(request(self.endpoint,self.capability,'actor',{'operation':'reset'}))
        if sha(value)!=self.expected:raise ValueError('approved public task descriptor mismatch')
        return value
    def call(self,name,arguments):
        value=request(self.endpoint,self.capability,'actor',{'operation':'call','name':name,'arguments':arguments})
        if 'tool_error' in value:
            from mcp.server.fastmcp.exceptions import ToolError
            raise ToolError(value['tool_error'])
        return value['observation']
    def close(self):return request(self.endpoint,self.capability,'actor',{'operation':'close'})
    def finish(self):
        value=request(self.endpoint,self.capability,'actor',{'operation':'finish'})
        if set(value)!={'sealed','reward','task_id','session_id','public_descriptor_sha256','transcript_sha256'} or value['sealed'] is not True or value['public_descriptor_sha256']!=self.expected:
            raise ValueError('approved native terminal outcome binding')
        return value
