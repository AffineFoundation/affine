#!/usr/bin/env python3
"""Provision narrow verifier seeds; explicitly start only a reviewed new namespace.

The default only validates private configuration. This never deploys source,
changes a running pilot, extends signed deadlines or signals existing processes.
"""
import argparse
import json
import inspect
import os
import shlex
import secrets
import socket
import subprocess
import time
from pathlib import Path
from nacl.signing import SigningKey


def ticks(pid):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def process_identity(pid, proc='/proc'):
    """Read actual process bindings; a zombie cannot host a tunnel or worker."""
    root=Path(proc)/str(pid)
    try:
        fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X'):return None
        return dict(ticks=fields[19],argv=(root/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                    cwd=str((root/'cwd').resolve(strict=True)))
    except (FileNotFoundError,ProcessLookupError):return None


def matches_process(actual, marker, argv, cwd=None):
    return (actual is not None and actual['ticks']==marker.get('ticks') and
            actual['argv']==argv and (cwd is None or actual['cwd']==str(Path(cwd).resolve())))


def worker_arguments(python, coordinator, authority, key, workspace, caches):
    args=[python,'-B','-m','subnet.distributed_worker','--coordinator',coordinator,
          '--authority',authority,'--seed-file',key,'--workspace',workspace]
    for checkpoint,local in caches.items():args+=['--checkpoint-cache',checkpoint+'='+local]
    return args


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,sort_keys=True));temporary.chmod(0o600);temporary.replace(path)


def wait_for_coordinator(host, port, timeout=30):
    """Do not launch workers while their controller is still importing."""
    deadline=time.monotonic()+timeout
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('operator coordinator not listening; no verifier workers started')
        try:
            with socket.create_connection((host,port),timeout=min(1,remaining)):
                return
        except OSError:
            time.sleep(min(.1,max(0,deadline-time.monotonic())))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--seed-dir',required=True);parser.add_argument('--authority')
    parser.add_argument('--provision',action='store_true');parser.add_argument('--start',action='store_true')
    parser.add_argument('--forward-port',type=int,default=19081)
    args=parser.parse_args();configpath=Path(args.config);config=json.loads(configpath.read_text())
    remote=config['remote'];q=remote['verifier_queue'];seedroot=Path(args.seed_dir)
    if configpath.stat().st_mode & 0o077:raise ValueError('prospective config must remain private')
    if q.get('host','127.0.0.1')!='127.0.0.1':raise ValueError('operator coordinator must bind loopback')
    if args.start and (not args.authority or len(bytes.fromhex(args.authority))!=32):raise ValueError('new controller public authority required')
    if args.start:wait_for_coordinator('127.0.0.1',q['port'])
    for number,endpoint in enumerate(remote['roles']['verify'],1):
        seedpath=seedroot/('verifier-'+str(number)+'-worker.seed.private')
        if seedpath.stat().st_mode & 0o077:raise ValueError('verifier seed mode')
        identity=SigningKey(bytes.fromhex(seedpath.read_text().strip())).verify_key.encode().hex()
        if identity!=endpoint['worker_identity']:raise ValueError('verifier seed/public roster mismatch')
        peer=endpoint.get('user','root')+'@'+endpoint['host']
        options=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+endpoint['known_hosts']]
        ssh=['ssh',*options,'-p',str(endpoint['port']),peer]
        key='/root/affine-hopper-pilot-v1/private/verifier.seed'
        if args.provision:
            subprocess.run(ssh+['umask 077; mkdir -p /root/affine-hopper-pilot-v1/private'],check=True,timeout=30)
            incoming=key+'.incoming-'+secrets.token_hex(8)
            subprocess.run(['scp','-q',*options,'-P',str(endpoint['port']),str(seedpath),peer+':'+incoming],check=True,timeout=60)
            admission="from pathlib import Path;import os;p=Path("+repr(key)+");q=Path("+repr(incoming)+");data=q.read_bytes();q.chmod(0o600)\nif p.exists():\n if p.is_symlink() or p.read_bytes()!=data:raise ValueError('immutable verifier key collision')\n q.unlink()\nelse:q.replace(p)\np.chmod(0o600)"
            subprocess.run(ssh+[shlex.quote(endpoint['python'])+' -c '+shlex.quote(admission)],check=True,timeout=30)
        if args.start:
            marker=seedroot/('verifier-'+str(number)+'-forward.private.json')
            old=json.loads(marker.read_text()) if marker.exists() else None
            tunnel_args=['ssh',*options,'-o','ExitOnForwardFailure=yes',
                '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-N','-R',
                '127.0.0.1:'+str(args.forward_port)+':127.0.0.1:'+str(q['port']),
                '-p',str(endpoint['port']),peer]
            actual=process_identity(old['pid']) if old else None
            if not old or not matches_process(actual,old,tunnel_args):
                if old:
                    # Preserve a mismatched live tunnel: it may serve another
                    # operator namespace. Never signal it or erase its marker.
                    save(seedroot/('verifier-'+str(number)+'-forward-preserved-'+secrets.token_hex(8)+'.private.json'),old)
                logfile=seedroot/('verifier-'+str(number)+'-forward.private.log')
                with logfile.open('ab') as output:
                    logfile.chmod(0o600)
                    process=subprocess.Popen(tunnel_args,stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True)
                time.sleep(1)
                if process.poll() is not None:raise RuntimeError('review verifier forward log; SSH tunnel failed')
                save(marker,dict(pid=process.pid,ticks=ticks(process.pid),created_at=time.time(),role='verifier-'+str(number),
                    forward_port=args.forward_port,coordinator_port=q['port']))
            # Remote marker protects idempotent process activation. It never
            # stops or replaces an existing exact verifier worker.
            helpers='\n'.join(inspect.getsource(f) for f in (process_identity,matches_process,worker_arguments))
            code='''import json,os,subprocess,time,socket
from pathlib import Path
'''+helpers+'''
# Check the actual remote listener before starting any model worker.
with socket.create_connection(('127.0.0.1',FORWARD_PORT),timeout=5):pass
workspace=Path(WORKSPACE);workspace.mkdir(parents=True,exist_ok=True);workspace.chmod(0o700)
marker=workspace/'worker-process.json'
old=json.loads(marker.read_text()) if marker.exists() else None
args=worker_arguments(PYTHON,COORDINATOR,AUTHORITY,KEY,WORKSPACE,CACHEMAP)
actual=process_identity(old['pid']) if old else None
if actual and actual['ticks']==old.get('ticks'):
 if old.get('authority')!=AUTHORITY or not matches_process(actual,old,args,SOURCE):
  raise ValueError('refuse existing live worker binding replacement')
else:
 log=workspace/'worker-service.log'
 with log.open('ab') as output:
  log.chmod(0o600)
  p=subprocess.Popen(args,cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
 time.sleep(.2)
 if p.poll() is not None:raise RuntimeError('review verifier worker log; startup failed')
 observed=process_identity(p.pid)
 if observed is None:raise RuntimeError('worker disappeared before activation record')
 value=dict(pid=p.pid,ticks=observed['ticks'],started_at=time.time(),authority=AUTHORITY,
            coordinator=COORDINATOR,source=SOURCE)
 temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(value));temp.chmod(0o600);temp.replace(marker)
'''
            for name,value in {'WORKSPACE':endpoint['workspace'],'PYTHON':endpoint['python'],
                'COORDINATOR':'http://127.0.0.1:'+str(args.forward_port),'AUTHORITY':args.authority,
                'KEY':key,'SOURCE':endpoint['code'],'CACHEMAP':endpoint.get('checkpoint_caches',{}),
                'FORWARD_PORT':args.forward_port}.items():
                code=name+'='+repr(value)+'\n'+code
            subprocess.run(ssh+[shlex.quote(endpoint['python'])+' -c '+shlex.quote(code)],check=True,timeout=30)
    print('Verifier configuration validated.' if not (args.provision or args.start) else 'Requested narrow verifier preparation completed; inspect private process logs for runtime health.')

if __name__=='__main__':main()
