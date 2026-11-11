"""A probe box on local disk, for tests.

The code that decides how a probe runs is the real code: the pinned
``podctl.py``, ``serve.sh`` and ``probe_worker.py`` are copied from
``miner/validator/probe`` and only their path constants point at a temp
directory. What a rented box brings from outside the repository is replaced by
small stand-ins:

* ``nvidia-smi``: prints the GPUs the test asks for;
* ``vllm``: a server with the vLLM command line and the three HTTP routes the
  probe uses. An answer is a fixed function of (prompt, request seed), so the
  same task and seed give the same answer under every layout;
* the evaluator source: one text environment where the answer ``7`` passes.
"""
from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

from ops.parallel_eval.live.pe_probe_plan import build_probe_plan

from . import probe_layout as PL
from . import probe_serve as PS

REPO = Path(__file__).resolve().parents[3]
PINNED = REPO / 'miner' / 'validator' / 'probe'
ENV_ID = 'affine_gatepool'
GPU_NAME = 'NVIDIA GeForce RTX 5090'
HARNESS = {
    'harness': {'version': 'fixture-v1', 'enable_thinking': True, 'max_context_tokens': 69632,
                'temperature': 1.0, 'top_p': 1.0, 'top_k': -1},
    'mimo_answer_projection': {'mode': 'fixture'},
    'sampling_contract': {'n': 1},
}

_SMI = r'''#!%(python)s
import json, os, sys
cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpus.json')))
n, args = cfg['count'], sys.argv[1:]
query = [a.split('=', 1)[1].split(',') for a in args if a.startswith('--query-gpu=')]
if args[:1] == ['-L']:
    for i in range(n): print('GPU %%d: %%s (UUID: GPU-%%04d)' %% (i, cfg['name'], i))
elif query:
    row = {'name': cfg['name'], 'memory.used': str(cfg['used_mib']), 'memory.total': '32607', 'utilization.gpu': '0'}
    for i in range(n): print(', '.join(row[k] for k in query[0]))
elif not any(a.startswith('--query-compute-apps=') for a in args):
    sys.exit(2)
'''

_VLLM = r'''#!%(python)s
import hashlib, importlib.metadata, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
if argv[:1] != ['serve']: sys.exit(2)
model, flags, i = argv[1], {}, 2
while i < len(argv):
    if argv[i] in ('--enable-prefix-caching', '--no-enable-prefix-caching'): flags[argv[i]] = True; i += 1
    else: flags[argv[i]] = argv[i + 1]; i += 2
state = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(model))), 'fake-vllm')
os.makedirs(state, exist_ok=True)
try: control = json.load(open(os.path.join(state, 'control.json')))
except OSError: control = {}
version = importlib.metadata.version('vllm')
with open(os.path.join(state, 'started.jsonl'), 'a') as f:
    f.write(json.dumps(dict(pid=os.getpid(), argv=sys.argv, version=version, port=int(flags['--port']),
                            env={k: os.environ.get(k) for k in (
                                'CUDA_VISIBLE_DEVICES', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                                'VLLM_LOGGING_LEVEL', 'HF_HUB_OFFLINE', 'VLLM_NO_USAGE_STATS', 'DO_NOT_TRACK')})) + '\n')
if any(flag in flags for flag in control.get('fail_if', [])):
    print('fake vllm: cannot start with %%s' %% control['fail_if'], file=sys.stderr); sys.exit(3)
OPEN_T, CLOSE_T, EOS = 248068, 248069, 248046

def unit(text): return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / 16.0 ** 12

class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def _send(self, doc):
        body = json.dumps(doc).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == '/v1/models':
            self._send(dict(object='list', data=[dict(id=flags['--served-model-name'], object='model', root=model,
                                                      max_model_len=int(flags['--max-model-len']))]))
        elif self.path == '/version': self._send(dict(version=control.get('report_version') or version))
        else: self.send_error(404)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if control.get('delay_ms'): time.sleep(control['delay_ms'] / 1000.0)
        prompt = json.dumps(req['prompt'])
        rate = 0.15 + 0.7 * unit('task:' + prompt) + float(control.get('bias', 0.0))
        ok = unit('%%s:%%s' %% (prompt, req['seed'])) < rate
        self._send(dict(choices=[dict(token_ids=[OPEN_T, 5, CLOSE_T, 7 if ok else 8, EOS], finish_reason='stop', stop_reason=EOS)]))
    def log_message(self, *a): pass

ThreadingHTTPServer.request_queue_size = 1024
ThreadingHTTPServer(('127.0.0.1', int(flags['--port'])), H).serve_forever()
'''

_UV = r'''#!%(python)s
import json, os, re, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
with open(os.path.join(here, 'uv-calls.jsonl'), 'a') as f:
    f.write(json.dumps(dict(args=args, env={k: os.environ.get(k) for k in ('VIRTUAL_ENV', 'UV_CACHE_DIR')})) + '\n')
if args[:1] == ['venv']:
    sys.exit(subprocess.call([sys.executable, '-m', 'venv', '--without-pip', '--symlinks', args[-1]]))
if args[:2] != ['pip', 'install']: sys.exit(2)
spec = open(args[args.index('-r') + 1]).read() if '-r' in args else ' '.join(args)
version = re.search(r'vllm==([0-9.]+)', spec).group(1)
venv = os.environ['VIRTUAL_ENV']
site = os.path.join(venv, 'lib', os.listdir(os.path.join(venv, 'lib'))[0], 'site-packages')
for name, ver in [('vllm', version)] + re.findall(r'^(?!vllm==)([a-z0-9-]+)==([0-9.]+)$', spec, re.M):
    info = os.path.join(site, '%%s-%%s.dist-info' %% (name, ver))
    os.makedirs(info)
    open(os.path.join(info, 'METADATA'), 'w').write('Metadata-Version: 2.1\nName: %%s\nVersion: %%s\n' %% (name, ver))
script = os.path.join(venv, 'bin', 'vllm')
open(script, 'w').write(open(os.path.join(here, 'vllm.tmpl')).read().replace('@PYTHON@', os.path.join(venv, 'bin', 'python')))
os.chmod(script, 0o755)
'''

# The live clean-up of the eval stage ends every process that matches a pattern. On a test machine it must find none.
_PGREP = '#!/bin/sh\nexit 1\n'

_ENVIRONMENTS = '''class EnvironmentSpec:
    def __init__(self, d): self.id = d['id']; self.source_hash = d['source_hash']
    @classmethod
    def from_dict(cls, d): return cls(d)
def _source_hash(spec): return spec.source_hash
class _Session:
    def reset(self, index, seed):
        return dict(messages=[dict(role='user', content='task %d' % index)], task_hash='h%d' % index)
    def close_runtime(self): pass
    def close(self): pass
    def step(self, action):
        return dict(done=True, classification='positive' if action['text'] == '7' else 'negative')
def create_session(spec): return _Session()
'''

_TRANSFORMERS = '''class _Tok:
    all_special_ids = ()
    def apply_chat_template(self, messages, **kw): return [100, int(messages[0]['content'].split()[1])]
    def decode(self, ids, skip_special_tokens=False): return ' '.join(str(i) for i in ids)
class AutoTokenizer:
    @staticmethod
    def from_pretrained(path): return _Tok()
'''


def _swap(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError('pinned probe file changed: %r occurs %d times' % (old, text.count(old)))
    return text.replace(old, new)


def _script(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _free_port_block(count: int = 16) -> int:
    for _ in range(200):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            base = probe.getsockname()[1]
        if base + count >= 65535:
            continue
        socks = []
        try:
            for port in range(base, base + count):
                sock = socket.socket()
                socks.append(sock)
                sock.bind(('127.0.0.1', port))
            return base
        except OSError:
            continue
        finally:
            for sock in socks:
                sock.close()
    raise RuntimeError('no free port block')


@dataclass
class FakeBox:
    root: Path
    bin: Path
    port: int
    cid: str

    def box(self) -> PS.Box:
        return PS.Box(root=str(self.root), python=sys.executable, port=self.port,
                      poll_s=0.05, probe_poll_s=0.1, settle_s=0.1, drain_s=0.5, gpu_wait_s=0.5)

    def path_env(self) -> Dict[str, str]:
        """PATH with this box's ``nvidia-smi`` first; patch ``os.environ`` with it for the test."""
        return {'PATH': '%s%s%s' % (self.bin, os.pathsep, os.environ.get('PATH', ''))}

    def set_gpus(self, count: int, *, name: str = GPU_NAME, used_mib: int = 12) -> None:
        (self.bin / 'gpus.json').write_text(json.dumps(dict(count=count, name=name, used_mib=used_mib)))

    def add_vllm(self, version: str, venv_dir: Path) -> Path:
        """A Python environment whose ``vllm`` console script is the stand-in server of ``version``."""
        venv.EnvBuilder(with_pip=False, symlinks=True).create(str(venv_dir))
        site = next((venv_dir / 'lib').glob('python*')) / 'site-packages'
        info = site / ('vllm-%s.dist-info' % version)
        info.mkdir(parents=True)
        (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: vllm\nVersion: %s\n' % version)
        _script(venv_dir / 'bin' / 'vllm', _VLLM % dict(python=venv_dir / 'bin' / 'python'))
        return venv_dir

    def add_uv(self) -> None:
        """Put an installer named ``uv`` on this box's PATH. It builds the stand-in server and records its calls."""
        (self.bin / 'vllm.tmpl').write_text(_VLLM % dict(python='@PYTHON@'))
        _script(self.bin / 'uv', _UV % dict(python=sys.executable))

    def uv_calls(self) -> List[Dict[str, Any]]:
        try:
            return [json.loads(line) for line in (self.bin / 'uv-calls.jsonl').read_text().splitlines()]
        except OSError:
            return []

    def control(self, **settings: Any) -> None:
        """Steer the stand-in servers that start after this call.

        ``fail_if``: flags that make a server exit at start. ``delay_ms``: time per answer. ``bias``: added to every
        task's pass rate. ``report_version``: what ``/version`` says.
        """
        state = self.root / 'fake-vllm'
        state.mkdir(exist_ok=True)
        (state / 'control.json').write_text(json.dumps(settings))

    def started(self) -> List[Dict[str, Any]]:
        try:
            return [json.loads(line) for line in (self.root / 'fake-vllm' / 'started.jsonl').read_text().splitlines()]
        except OSError:
            return []

    def podctl(self, *args: str) -> str:
        return subprocess.run([sys.executable, str(self.root / 'podctl.py'), *args],
                              capture_output=True, text=True, timeout=600).stdout

    def close(self) -> None:
        runs = self.root / 'runs'
        for run in (sorted(p.name for p in runs.iterdir()) if runs.is_dir() else []):
            self.podctl('abort-run', run)
        self.podctl('stop-serve')


def make_box(tmp: Path, *, gpus: int = 8, live_vllm: str = '0.32.0') -> FakeBox:
    """Build the box under ``tmp``: pinned probe code, one checkpoint, the live vLLM bundle, ``gpus`` GPUs."""
    root, bindir = tmp / 'affine-env-probe', tmp / 'bin'
    root.mkdir(parents=True)
    bindir.mkdir()
    port = _free_port_block()
    cid = 'c' * 64

    podctl = (PINNED / 'podctl.py').read_text()
    podctl = _swap(podctl, "D = '/root/affine-env-probe'; PORT = 18555;", "D = %r; PORT = %d;" % (str(root), port))
    podctl = _swap(podctl, "EVPY = '/root/affine-mimo-20261013/venv/bin/python'", 'EVPY = %r' % sys.executable)
    podctl = _swap(podctl, "source_root='/root/affine-mimo-public-evaluator-20261013/source-final-v822b'",
                   'source_root=%r' % str(root / 'src'))
    (root / 'podctl.py').write_text(podctl)
    _script(root / 'serve.sh', _swap((PINNED / 'serve.sh').read_text(), 'D=/root/affine-env-probe;', 'D=%s;' % root))
    (root / 'probe_worker.py').write_bytes((PINNED / 'probe_worker.py').read_bytes())

    (root / 'envs.json').write_text(json.dumps({ENV_ID: {'spec': {'id': ENV_ID, 'source_hash': 'abc'}, 'kind': 'text'}}))
    for name, text in (('subnet/environments.py', _ENVIRONMENTS), ('subnet/__init__.py', ''),
                       ('transformers/__init__.py', _TRANSFORMERS)):
        path = root / 'src' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    ckpt = root / 'ckpt' / cid
    ckpt.mkdir(parents=True)
    (ckpt / 'config.json').write_text('{}')

    _script(bindir / 'nvidia-smi', _SMI % dict(python=sys.executable))
    _script(bindir / 'pgrep', _PGREP)
    box = FakeBox(root=root, bin=bindir, port=port, cid=cid)
    box.set_gpus(gpus)
    box.add_vllm(live_vllm, root / 'venv')
    return box


class LocalPod:
    """The SSH link to a box, replaced by bash on this machine."""

    def __init__(self, env: Dict[str, str]) -> None:
        self.env = env

    def run(self, cmd: str, timeout: float = 60, check: bool = False, **kwargs: Any) -> str:
        proc = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=timeout, env=self.env)
        if check and proc.returncode != 0:
            raise RuntimeError('rc=%d %s' % (proc.returncode, proc.stderr[-300:]))
        return proc.stdout


class BoxCase(unittest.TestCase):
    """A test with one local box. ``GPUS`` is how many GPUs the box reports."""

    GPUS = 4

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.fb = make_box(self.tmp, gpus=self.GPUS)
        self.addCleanup(self.fb.close)
        patch = mock.patch.dict(os.environ, self.fb.path_env())
        patch.start()
        self.addCleanup(patch.stop)
        self.v033 = self.fb.add_vllm('0.33.0', self.fb.root / 'venv-vllm-0.33.0')
        self.box = self.fb.box()
        self.cid = self.fb.cid

    def probe_plan(self, tasks: int = 16, seed: str = 'seed-1', layout: Optional[PL.ProbeLayout] = None) -> Dict[str, Any]:
        """The plan the host builds for a probe of ``tasks`` tasks, with fewer grading processes."""
        layout = layout or PL.PRESETS['rtx5090-tp2']
        plan = build_probe_plan(self.cid, list(range(tasks)), seed, k=8, max_tokens=65536,
                                max_num_seqs=layout.concurrency(), harness=HARNESS)
        plan['grade_workers'] = 2
        return plan

    def alive(self) -> List[Dict[str, Any]]:
        """Servers of this box that still run."""
        return [server for server in self.fb.started() if PS.running(server['pid'])]

    def pod(self) -> LocalPod:
        """This box as the host reaches it: ``python3`` is this interpreter, ``pgrep`` and ``nvidia-smi`` are the box's."""
        return LocalPod({'PATH': '%s:%s:/usr/bin:/bin' % (self.fb.bin, os.path.dirname(sys.executable)),
                         'HOME': str(self.tmp)})
