"""Bounded CPU-only retries of a terminal, explicitly deferred original ACK.

No training job, scientific source, ACK payload, resource limit or retirement
guard changes. Unknown, failed and live actions are observed, never relaunched.
"""
import hashlib
import json
import shlex
import time

REASONS=frozenset(('workspace-role-in-flight','original-child-not-terminal'))

PROBE = r'''
import hashlib,json,os,stat
from pathlib import Path
def read(path):
 p=Path(path);s=p.lstat()
 if not stat.S_ISREG(s.st_mode)or s.st_uid!=os.geteuid()or s.st_nlink!=1 or s.st_mode&0o077:
  raise ValueError('owned private original ACK evidence')
 return json.loads(p.read_bytes())
root=Path(DATA['workspace'])
if root.resolve()!=root or root.is_symlink():raise ValueError('canonical original ACK workspace')
attempt=read(DATA['handle']+'-attempt.json');result=read(DATA['handle']+'-result.json')
guard=read(root/'.optimizer-state-cache/promotion.json')
if attempt!={'ack':DATA['ack']}or result!={'phase':'complete','receipt':DATA['receipt']}:
 raise ValueError('exact original terminal deferred ACK result')
if (guard.get('ack')!=DATA['ack']or guard.get('phase')!='complete'
    or guard.get('child_terminal_confirmed')is not True):
 print(json.dumps(dict(ready=False,reason='original-ACK-action-not-terminal')))
else:
 pid=guard.get('child_pid');ticks=guard.get('child_ticks')
 if type(pid)is not int or pid<1 or ticks is not None and not isinstance(ticks,str):raise ValueError('original terminal ACK child identity')
 try:fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 except FileNotFoundError:fields=None
 live=bool(fields and fields[0]!='Z'and(ticks is None or fields[19]==ticks))
 workspace_live=False
 for path in(root/'runner-status').glob('*.json'):
  status=read(path)
  for field in('runner_pid','child_pid'):
   role_pid=status.get(field);role_ticks=status.get(field+'_ticks')
   if not role_pid or role_ticks is None:continue
   try:role_fields=(Path('/proc')/str(role_pid)/'stat').read_text().rsplit(')',1)[1].split()
   except FileNotFoundError:continue
   if role_fields[0]!='Z'and role_fields[19]==str(role_ticks):workspace_live=True
 print(json.dumps(dict(ready=not(live or workspace_live),reason='original-ACK-child-live'if live else'workspace-role-in-flight'if workspace_live else'terminal-deferred-ACK')))
'''


def install(RemoteJobs, *, maximum_retries=4, retry_delay_seconds=1):
    if type(maximum_retries)is not int or not 1<=maximum_retries<=8:
        raise ValueError('bounded explicit ACK retry count')
    if type(retry_delay_seconds)not in(int,float)or not 0<=retry_delay_seconds<=10:
        raise ValueError('bounded ACK retry delay')
    original=RemoteJobs._durable_cache_ack
    def durable(self,script,ack,job_id):
        from subnet.distributed_roles import authenticate
        from subnet.storage import canonical
        from subnet.cache_lifecycle import identifier
        value=authenticate(ack,self.controller.authority.id)
        if (value.get('version')!='durable-original-trainer-cache-ACK-v1'
                or value.get('authority_state_committed')is not True or value.get('job_id')!=job_id):
            raise ValueError('same original committed training ACK')
        identifier(job_id)
        digest=hashlib.sha256(canonical(ack)).hexdigest()
        current=job_id
        for ordinal in range(maximum_retries+1):
            result=original(self,script,ack,current)
            if (not isinstance(result,dict)or result.get('status')!='deferred'
                    or result.get('reason')not in REASONS or ordinal==maximum_retries):
                return result
            handle=self.workspace+'/'+current+'-cache-ACK-'+digest
            data=dict(workspace=self.workspace,handle=handle,ack=ack,receipt=result)
            command=shlex.quote(self.python)+' -I -B -c '+shlex.quote('DATA='+repr(data)+'\n'+PROBE)
            evidence=json.loads(self.command(command,timeout=60))
            if evidence.get('ready')is not True:
                return dict(status='deferred',reason=evidence.get('reason','unknown-original-ACK-state'),
                    original_handle=handle,removed_checkpoints=[])
            time.sleep(retry_delay_seconds)
            current=job_id+'-retention-retry-'+str(ordinal+1)
            identifier(current)
        raise AssertionError('bounded ACK loop')
    RemoteJobs._durable_cache_ack=durable
    return original
