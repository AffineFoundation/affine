"""Fakepass second step ON THE BOX: load the promoted head C exactly as train_pit would.

Runs under the box venv against host-v834e (``host.checked`` ROOT admission,
no overlay) and stops after the trainer's own ``_fp32_prepare``: the fp32
master fetched through the promoted index must load from disk
(``FP32_STORE/<bf16 digest>/meta.json`` with ``tensor_file``) and match
``AFFINE_FP32_EXPECTED_MASTER_HASH``. A full GRPO step needs a signed dispatch
bound to C, which only the live lifecycle can issue.

argv[1]: input dir holding head.descriptor.json, manifest.json,
train_config.json, meta.json and actor/<C>/.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

T = time.time()
W = Path('/workspace/affine-mimo-trainer-prod')
SRC = W / 'source-final-v823u'
HOST = W / 'host-v834e'
IN = Path(sys.argv[1]).resolve()


def ev(event, **kw):
    print(json.dumps(dict(event=event, seconds=round(time.time() - T, 1), **kw), default=str), flush=True)


exp = os.environ.get('AFFINE_FP32_EXPECTED_MASTER_HASH') or ''
assert len(exp) == 64, 'expected master hash required (fail closed)'
assert not os.environ.get('AFFINE_FP32_BOOTSTRAP_OVERRIDE'), 'ROOT bootstrap override forbidden in PIT'
assert HOST.is_dir(), 'host-v834e missing (fail closed)'
os.chdir(SRC)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HOST))
spec = importlib.util.spec_from_file_location('mimo_host_trainer_v83', HOST / 'mimo_host_trainer_v83.py')
host = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = host
spec.loader.exec_module(host)
assert host.FP32_BOOTSTRAP_OVERRIDE is None
cfg = host.checked(HOST / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json')
ev('root-admission-checked', engine_fp32=cfg['engine'].get('sgd_fp32_master'), host=HOST.name)
host.install(cfg, start_dp=False)
ev('native-installed')

# subnet.* must import after host.install patches the runtime (train_pit order).
from subnet.mimo_grpo.config import TrainerConfig  # noqa: E402
from subnet.mimo_grpo.optimizer import CPUAdamW  # noqa: E402
from subnet.mimo_runtime import MimoRuntime  # noqa: E402


def rd(name):
    return json.loads((IN / name).read_text())


head = {k: v for k, v in rd('head.descriptor.json').items() if k in ('id', 'files')}
manifest = rd('manifest.json')
tcfg = rd('train_config.json')
step = int(rd('meta.json')['optimizer_step'])
config = TrainerConfig(**tcfg['settings'])
assert config.document() == tcfg
runtime = MimoRuntime.load(
    str(IN / 'actor' / head['id']), head, manifest['payload']['mimo_runtime'],
    device='cuda', trainable=True,
)
ev('actor-loaded', checkpoint=head['id'])
opt = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=head['id'])
opt.global_step = step
ev('optimizer-built', step=step)
info = host._fp32_prepare(opt, None, lambda row: print(json.dumps(row, default=str), flush=True))
ev(
    'fp32-prepared',
    checkpoint=head['id'],
    source=info.get('source'),
    input_bf16_sha256=info.get('input_bf16_sha256'),
    input_master_digest=info.get('input_master_digest'),
    input_store_entry=info.get('input_store_entry'),
    expected_master_digest=info.get('expected_master_digest'),
    version=info.get('version'),
    prepare_seconds=info.get('prepare_seconds'),
    host=HOST.name,
)
os._exit(0)
