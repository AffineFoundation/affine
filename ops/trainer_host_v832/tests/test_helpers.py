"""CPU sanity test for v8.3.2 drift control (synthetic tensors, real CPUAdamW + real v8.3.1 PipelinedCPUAdamW step).
usage: python test_v832_drift.py <source_root> <stage_dir> <settings.json>   (+ rank-agreement test via gloo world 8)"""
import sys, json, hashlib, math, os, importlib.util, copy
SRC, STAGE, SETTINGS = os.environ["V832_SRC"], os.environ["V832_STAGE"], os.environ["V832_SETTINGS"]
sys.path.insert(0, SRC); sys.path.insert(0, STAGE)
import torch
torch.manual_seed(0)
from subnet.mimo_grpo.config import TrainerConfig
from subnet.mimo_grpo.optimizer import CPUAdamW
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
L = load('launcher832', os.path.join(STAGE, 'mimo_host_trainer_v83.py'))
FO = load('fast_opt', os.path.join(STAGE, 'v824_optimizer_fast.py'))
cfg = TrainerConfig(**json.load(open(SETTINGS)))
CK = 'a' * 64
SHAPES = {'layers.0.mlp.weight': (256, 512), 'layers.0.gate.weight': (64, 256), 'layers.0.norm.weight': (512,), 'layers.0.tiny.bias': (32,), 'embed.weight': (1000, 64)}
SCALES = {'layers.0.mlp.weight': 0.02, 'layers.0.gate.weight': 3e-4, 'layers.0.norm.weight': 1.0, 'layers.0.tiny.bias': 1e-3, 'embed.weight': 0.05}
def model(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(n, torch.nn.Parameter((torch.randn(s, generator=g) * SCALES[n]).to(torch.bfloat16))) for n, s in SHAPES.items()]
def grads(step, seed=100):
    g = torch.Generator().manual_seed(seed + step)
    return {n: torch.randn(s, generator=g) * 1e-3 for n, s in SHAPES.items()}
def make(params):
    o = CPUAdamW(params, cfg, input_checkpoint=CK); o.step_threads = 4
    peak = max(p.numel() for _, p in o.parameters)
    o._pinned = [torch.empty(peak, dtype=torch.float32) for _ in range(2)]; o._copy_stream = None
    return o
inner = FO.PipelinedCPUAdamW.step
H = lambda o: hashlib.sha256(b''.join(o.masters[n].detach().numpy().tobytes() for n, _ in o.parameters)).hexdigest()
rel = lambda a, b: math.sqrt(sum(float(((a[n] - b[n]).double() ** 2).sum()) for n in a)) / math.sqrt(sum(float((b[n].double() ** 2).sum()) for n in b))
snap = lambda o: {n: o.masters[n].detach().clone() for n, _ in o.parameters}
