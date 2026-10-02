#!/usr/bin/env python3
"""Provision narrow verifier seeds; explicitly start only a reviewed new namespace.

The default only validates private configuration. This never deploys source,
changes a running pilot, extends signed deadlines or signals existing processes.
"""
import argparse
import json
import os
import shlex
import secrets
import subprocess
import time
from pathlib import Path
from nacl.signing import SigningKey


def ticks(pid):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,sort_keys=True));temporary.chmod(0o600);temporary.replace(path)


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
    for number,endpoint in enumerate(remote['roles']['verify'],1):
        seedpath=seedroot/('verifier-'+str(number)+'-worker.seed.private')
        if seedpath.stat().st_mode & 0o077:raise ValueError('verifier seed mode')
        identity=SigningKey(bytes.fromhex(seedpath.read_text().strip())).verify_key.encode().hex()
        if identity!=endpoint['worker_identity']:raise ValueError('verifier seed/public roster mismatch')
        peer=endpoint.get('user','root')+'@'+endpoint['host']
        options=['-o','BatchMode=yes','-o','UserKnownHostsFile='+endpoint['known_hosts']]
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
            if not old or ticks(old['pid'])!=old['ticks']:
                logfile=seedroot/('verifier-'+str(number)+'-forward.private.log')
                with logfile.open('ab') as output:
                    logfile.chmod(0o600)
                    process=subprocess.Popen(['ssh',*options,'-o','ExitOnForwardFailure=yes',
                        '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-N','-R',
                        '127.0.0.1:'+str(args.forward_port)+':127.0.0.1:'+str(q['port']),
                        '-p',str(endpoint['port']),peer],stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True)
                time.sleep(1)
                if process.poll() is not None:raise RuntimeError('review verifier forward log; SSH tunnel failed')
                save(marker,dict(pid=process.pid,ticks=ticks(process.pid),created_at=time.time(),role='verifier-'+str(number)))
            # Remote marker protects idempotent process activation. It never
            # stops or replaces an existing exact verifier worker.
            code='''import json,os,subprocess,time
from pathlib import Path
def ticks(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
workspace=Path(WORKSPACE);workspace.mkdir(parents=True,exist_ok=True);workspace.chmod(0o700)
marker=workspace/'worker-process.json'
old=json.loads(marker.read_text()) if marker.exists() else None
if not old or ticks(old['pid'])!=old['ticks']:
 log=workspace/'worker-service.log'
 with log.open('ab') as output:
  log.chmod(0o600)
  args=[PYTHON,'-B','-m','subnet.distributed_worker','--coordinator',COORDINATOR,'--authority',AUTHORITY,'--seed-file',KEY,'--workspace',WORKSPACE]
  for checkpoint,local in CACHEMAP.items():args+=['--checkpoint-cache',checkpoint+'='+local]
  p=subprocess.Popen(args,cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
 value=dict(pid=p.pid,ticks=ticks(p.pid),started_at=time.time(),authority=AUTHORITY)
 temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(value));temp.chmod(0o600);temp.replace(marker)
else:
 if old['authority']!=AUTHORITY:raise ValueError('refuse existing worker authority replacement')
'''
            for name,value in {'WORKSPACE':endpoint['workspace'],'PYTHON':endpoint['python'],
                'COORDINATOR':'http://127.0.0.1:'+str(args.forward_port),'AUTHORITY':args.authority,
                'KEY':key,'SOURCE':endpoint['code'],'CACHEMAP':endpoint.get('checkpoint_caches',{})}.items():
                code=name+'='+repr(value)+'\n'+code
            subprocess.run(ssh+[shlex.quote(endpoint['python'])+' -c '+shlex.quote(code)],check=True,timeout=30)
    print('Verifier configuration validated.' if not (args.provision or args.start) else 'Requested narrow verifier preparation completed; inspect private process logs for runtime health.')

if __name__=='__main__':main()
