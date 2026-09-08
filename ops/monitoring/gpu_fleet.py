"""Read-only GPU fleet telemetry, published through Arbos's panel document API.

Run with the workspace .venv Python (websockets), not inside a production pod.
Only affine-* / swarm-t-* inventory entries are probed. No production writes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time
from urllib.parse import urlsplit

from . import evalpods, infrastructure, remote_workers, swarm

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "panels/gpu-fleet.html"
INTERVAL = 30
STALE = 120

REMOTE = r'''
import csv, io, json, os, shutil, subprocess, time

def command(args):
    try:
        p = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=6, text=True)
        return p.stdout[:65536] if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None

def number(s):
    try:
        v = float(s.strip())
        return v if 0 <= v < 1e15 else None
    except (ValueError, AttributeError):
        return None

fields = ['index','name','utilization.gpu','memory.used','memory.total',
          'temperature.gpu','power.draw','power.limit','fan.speed','pstate','driver_version']
raw = command(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'])
gpus = []
if raw is not None:
    for row in csv.reader(io.StringIO(raw)):
        if len(row) != len(fields): continue
        row = [x.strip() for x in row]
        g = dict(zip(['index','name','util','memory_used_mib','memory_total_mib',
                      'temperature_c','power_w','power_limit_w','fan_pct','pstate','driver'],
                     [number(x) if i not in (1,9,10) else x for i,x in enumerate(row)]))
        gpus.append(g)
ecc = command(['nvidia-smi','--query-gpu=index,ecc.errors.uncorrected.volatile.total',
               '--format=csv,noheader,nounits'])
if ecc:
    values = {number(r[0]):number(r[1]) for r in csv.reader(io.StringIO(ecc)) if len(r)==2}
    for g in gpus: g['ecc_uncorrected'] = values.get(g['index'])
result = {'gpus':gpus, 'gpu_query_ok':raw is not None}
try:
    result['uptime_s'] = float(open('/proc/uptime').read().split()[0])
    mem = {k:int(v.split()[0])*1024 for k,v in
           (line.split(':',1) for line in open('/proc/meminfo'))}
    result.update(ram_total=mem['MemTotal'],ram_available=mem['MemAvailable'])
    disk = shutil.disk_usage('/')
    result.update(disk_total=disk.total,disk_free=disk.free,load_one=os.getloadavg()[0])
except (OSError,ValueError,KeyError): pass
workers = 0
try:
    for pid in os.listdir('/proc'):
        if not pid.isdigit(): continue
        try:
            args = open('/proc/'+pid+'/cmdline','rb').read(8192).split(b'\0')
            if any(args[i:i+2] == [b'-m',b'rollouts.run'] for i in range(len(args)-1)):
                workers += 1
        except (OSError,PermissionError): pass
    result['rollout_workers'] = workers
except OSError: pass
print(json.dumps(result,allow_nan=False))
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def num(value, ceiling=1e15):
    return value if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= ceiling else None


def safe_text(value, pattern=r"[A-Za-z0-9][A-Za-z0-9 _.()-]{0,79}"):
    return value if isinstance(value, str) and re.fullmatch(pattern, value) else None


def sanitize(data):
    if not isinstance(data, dict) or type(data.get('gpu_query_ok')) is not bool:
        raise ValueError('invalid probe')
    if not isinstance(data.get('gpus'), list) or len(data['gpus']) > 32:
        raise ValueError('invalid GPU list')
    result = {k:num(data.get(k)) for k in ('uptime_s','ram_total','ram_available',
               'disk_total','disk_free','load_one','rollout_workers')}
    result.update(gpu_query_ok=data['gpu_query_ok'], gpus={})
    for g in data['gpus']:
        if not isinstance(g, dict): raise ValueError('invalid GPU')
        index = num(g.get('index'), 31)
        if index is None or int(index) != index or str(int(index)) in result['gpus']:
            raise ValueError('invalid GPU index')
        clean = {k:num(g.get(k), cap) for k,cap in {
            'util':100,'memory_used_mib':1e7,'memory_total_mib':1e7,
            'temperature_c':150,'power_w':5000,'power_limit_w':5000,
            'fan_pct':100,'ecc_uncorrected':1e15}.items()}
        clean.update(index=int(index), name=safe_text(g.get('name')),
                     pstate=safe_text(g.get('pstate'),r'P[0-9]{1,2}'),
                     driver=safe_text(g.get('driver'),r'[0-9.]{3,30}'))
        if clean['memory_total_mib'] is not None and clean['memory_used_mib'] is not None and clean['memory_used_mib'] > clean['memory_total_mib']:
            clean['memory_used_mib'] = None
        result['gpus'][str(int(index))] = clean
    for total,free in [('ram_total','ram_available'),('disk_total','disk_free')]:
        if result[total] is not None and result[free] is not None and result[free] > result[total]:
            result[free] = None
    return result


def role(name):
    if name.startswith('swarm-t-'): return 'Teacher swarm'
    if name.startswith('affine-datagen'): return 'Data generation'
    return {'affine-eval':'Evaluation','affine-bench':'Benchmarks','affine-chat':'Public chat'}.get(name,'SN120 worker')


def probe(pod):
    name = pod['name']
    result = dict(name=name, role=role(name), inventory_status=infrastructure._status(pod.get('status'), infrastructure._POD_STATUSES),
                  expected_gpus=infrastructure._number(pod.get('gpu_count'), integer=True),
                  gpu_type=safe_text(pod.get('gpu_type')), price_hour=infrastructure._number(pod.get('price_per_hour')),
                  sampled_at=now(), reachable=False, gpus={})
    if result['inventory_status'] != 'running':
        return dict(result, probe_error='Pod is not listed as running; no SSH probe attempted.')
    try:
        args = remote_workers._ssh_args(pod.get('ssh_cmd'))
        args[-1] = 'python3 -B -c ' + shlex.quote(REMOTE)
    except (TypeError, ValueError):
        return dict(result, probe_error='No supported SSH connection in inventory.')
    try:
        p = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=19)
        result['sampled_at'] = now()
        if p.returncode:
            return dict(result, probe_error='SSH unreachable or remote probe failed.')
        result['reachable'] = True
        if len(p.stdout) > 65536: raise ValueError('oversize')
        result.update(sanitize(json.loads(p.stdout)))
    except subprocess.TimeoutExpired:
        result.update(sampled_at=now(), probe_error='SSH / telemetry probe timed out.')
    except OSError:
        result['probe_error'] = 'SSH probe unavailable.'
    except (ValueError, TypeError):
        result['probe_error'] = 'Invalid GPU telemetry response.'
    return result


def service_health():
    services = {}
    for key,panel in evalpods.collect().items():
        name = {'eval':'affine-eval','bench-engine':'affine-bench','chat':'affine-chat'}[key]
        metrics = {m['label']:m['value'] for m in panel['metrics']}
        services[name] = dict(status=panel['status'], label=metrics.get('Readiness','Unknown'),
                              busy=metrics.get('Busy'), state=metrics.get('State'),
                              sampled_at=panel['collected_at'], details=panel['notes'],
                              slots=panel['sections'][0]['rows'])
    backends = swarm._backends(swarm._load(swarm._METRICS, remote=True))
    groups = {}
    for b in backends or []:
        name = infrastructure._name(b.get('pod'))
        if name: groups.setdefault(name,[]).append(b)
    for name, rows in groups.items():
        complete = all(type(b.get('healthy')) is bool for b in rows)
        eligible = sum(b.get('healthy') is True for b in rows)
        services[name] = dict(status=('ok' if eligible == len(rows) else 'warn') if complete else 'unknown',
                             label=f'{eligible}/{len(rows)} replicas eligible' if complete else 'Unknown circuits',
                             in_flight=swarm._sum(rows,'in_flight'), sampled_at=now(),
                             details=['Router circuit eligibility, not a GPU diagnostic.'], slots=[])
    return services


def assess(pod):
    issues = []
    severity = 'ok'
    def issue(text, level='warn'):
        nonlocal severity
        issues.append(text)
        if level == 'error' or severity == 'ok': severity = level
    if not pod.get('reachable') or pod.get('probe_error'):
        return 'unknown', [pod.get('probe_error','No hardware telemetry.')]
    gpus = pod.get('gpus',{})
    if not pod.get('gpu_query_ok') or not gpus:
        return 'unknown', ['NVIDIA telemetry unavailable; SSH alone is not GPU health.']
    if len(gpus) != pod.get('expected_gpus'): issue('Observed GPU count differs from inventory.')
    for g in gpus.values():
        temp = g.get('temperature_c')
        if temp is not None and temp >= 80:
            issue(f"GPU {g['index']} temperature {temp:g}°C.", 'error' if temp >= 90 else 'warn')
        if (g.get('ecc_uncorrected') or 0) > 0:
            issue(f"GPU {g['index']} reports volatile uncorrected ECC errors; investigate historical counter.")
        if any(g.get(k) is None for k in ('temperature_c','util','memory_used_mib','memory_total_mib')):
            issue(f"GPU {g['index']} has incomplete core telemetry.")
    if pod.get('disk_total') and pod.get('disk_free') is not None:
        used = 100*(1-pod['disk_free']/pod['disk_total'])
        if used >= 90: issue(f'Root disk is {used:.1f}% used.', 'error' if used >= 97 else 'warn')
    service = pod.get('service',{})
    if service.get('status') not in ('ok',):
        issue('Application: '+service.get('label','not independently observed.')+'.',
              'error' if service.get('status') == 'error' else 'warn')
    return severity, issues


def collect():
    started = time.monotonic()
    inventory,error = infrastructure._inventory(['lium','ps','--format','json'],timeout=20,wrappers=('pods','data'))
    pods = [dict(p,name=infrastructure._name(p.get('name') or p.get('pod_name')))
            for p in inventory or [] if infrastructure._name(p.get('name') or p.get('pod_name'))]
    with ThreadPoolExecutor(max_workers=9) as pool:
        health_future = pool.submit(service_health)
        records = list(pool.map(probe, pods[:32]))
        try: services = health_future.result()
        except Exception: services = {}
    for p in records:
        service = services.get(p['name'])
        if p['role'] == 'Data generation':
            n = p.get('rollout_workers')
            service = dict(status='ok' if n is not None and n > 0 else 'unknown',
                           label=f'{int(n)} rollout supervisors' if n is not None else 'Worker state unknown',
                           sampled_at=p['sampled_at'], details=['Process presence only; not proof of task progress.'],slots=[])
        p['service'] = service or dict(status='unknown',label='No matching live service observation',sampled_at=now(),details=[],slots=[])
        p['status'],p['issues'] = assess(p)
    return dict(collected_at=now(), refresh_seconds=INTERVAL, stale_after_seconds=STALE,
                collection_seconds=round(time.monotonic()-started,2), inventory_error=error,
                excluded_pods=len(inventory)-len(pods) if inventory is not None else None,
                truncated=len(pods)>32, pods={p['name']:p for p in sorted(records,key=lambda p:p['name'])})


def publish(gateway, snapshot):
    from websockets.sync.client import connect
    url = urlsplit(gateway)
    if url.scheme != 'http' or url.hostname not in ('localhost','127.0.0.1'):
        raise ValueError('Use the local Arbos gateway only.')
    with connect('ws://'+url.netloc+'/api/board/ws', origin=gateway,
                 open_timeout=5, close_timeout=2, proxy=None) as ws:
        ws.send(json.dumps(dict(type='state_set',panel=str(PANEL),patch={'telemetry':snapshot}),allow_nan=False))
        deadline = time.monotonic()+8
        while time.monotonic()<deadline:
            reply = json.loads(ws.recv(timeout=max(.1,deadline-time.monotonic())))
            if reply.get('type') == 'state' and reply.get('panel') == str(PANEL):
                if reply.get('doc',{}).get('telemetry',{}).get('collected_at') == snapshot['collected_at']:
                    return
        raise TimeoutError('No panel state acknowledgement')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway',required=True)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    stop = threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop.set())
    with (ROOT/'panels/data/.gpu-fleet.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('GPU fleet collector already running.')
        while not stop.is_set():
            try:
                snapshot = collect()
            except Exception:
                snapshot = dict(collected_at=now(),refresh_seconds=INTERVAL,stale_after_seconds=STALE,
                                inventory_error='Collection failed; current state unknown.',pods={})
            try:
                publish(args.gateway,snapshot)
                print(f"Published {len(snapshot['pods'])} pods at {snapshot['collected_at']}",flush=True)
            except Exception as exc:
                print(f'Publication failed ({type(exc).__name__}); dashboard will age to stale.',flush=True)
                if args.once: raise SystemExit(1)
            if args.once or stop.wait(INTERVAL): break


if __name__ == '__main__': main()
