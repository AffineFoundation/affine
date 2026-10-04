"""Observe one reviewed diagnostic child through its actual wait; never relaunch/kill.

This does not authorize scientific execution. The command and independently
authenticated runner request must already have operator approval. No credentials,
bucket/network access, CUDA imports, service control or changes to worker queues.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    path=Path(path)
    if not path.is_file()or path.is_symlink():raise ValueError('regular reviewed diagnostic file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ticks(pid):return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]


def validate(command,python):
    if (not isinstance(command,dict)or set(command)!={'argv','helper_path','helper_sha256','request_path','request_sha256',
            'root_scientific_execution_approved','diagnostic_only','normal_queue_allowed','live_admission_allowed','chain_transactions_allowed'}
            or command['root_scientific_execution_approved']is not True or command['diagnostic_only']is not True
            or any(command[name]is not False for name in('normal_queue_allowed','live_admission_allowed','chain_transactions_allowed'))):
        raise ValueError('explicit isolated diagnostic execution scope')
    helper=Path(command['helper_path']);request=Path(command['request_path']);argv=command['argv']
    if (not helper.is_absolute()or helper.name!='diagnostic_full_forward_cdf_gpu.py' or sha(helper)!=command['helper_sha256']
            or not request.is_absolute()or sha(request)!=command['request_sha256']
            or not isinstance(argv,list)or len(argv)!=8 or any(not isinstance(v,str)for v in argv)
            or argv[:6]!=[python,'-I','-B',str(helper),'--request',str(request)]or argv[6]!='--authority'):
        raise ValueError('exact original pinned diagnostic runner/request command')
    return argv


def main():
    p=argparse.ArgumentParser();p.add_argument('--command-file',required=True);p.add_argument('--command-file-sha256',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();os.umask(0o077)
    path=Path(args.command_file);out=Path(args.output)
    if sha(path)!=args.command_file_sha256 or out.exists():raise ValueError('reviewed command/fresh original wait output')
    command=json.loads(path.read_bytes());argv=validate(command,sys.executable)
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with(out/'stdout.private.log').open('xb')as stdout,(out/'stderr.private.log').open('xb')as stderr:
        child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=environment)
        process=dict(supervisor_pid=os.getpid(),supervisor_ticks=ticks(os.getpid()),child_pid=child.pid,child_ticks=ticks(child.pid),
            command=argv,command_file_sha256=args.command_file_sha256,helper_sha256=command['helper_sha256'],
            request_sha256=command['request_sha256'],started_at=time.time(),diagnostic_only=True,live_admission=False)
        (out/'process.private.json').write_text(json.dumps(process,sort_keys=True)+'\n')
        outer_timeout=False
        try:code=child.wait(timeout=3500)
        except subprocess.TimeoutExpired:
            outer_timeout=True;code=child.wait()  # SAME original child, no kill/retry.
    receipt=dict(process,actual_child_wait_completed=True,exit_code=code,outer_timeout=outer_timeout,completed_at=time.time(),
        stdout_sha256=sha(out/'stdout.private.log'),stderr_sha256=sha(out/'stderr.private.log'))
    (out/'completion.private.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')
    raise SystemExit(code)


if __name__=='__main__':main()
