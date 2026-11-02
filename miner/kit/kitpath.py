"""Import the published live validator code (miner/validator/) without the validator's infrastructure.

The engine package `subnet.mimo_grpo` is published verbatim, but its __init__ imports the R2 publication / signed-job
modules (botocore, subnet.backend_jobs, ...) that a miner does not need. We register `subnet` and `subnet.mimo_grpo` as
plain namespace modules so `from subnet.mimo_grpo.config import ...` loads the verbatim submodules directly. Nothing in the
published files is changed; `subnet.backend_jobs.signed` is a stub that refuses to run (only the validator verifies signed
dispatches)."""
import sys, types
from pathlib import Path

KIT = Path(__file__).resolve().parent
MINER = KIT.parent
VAL = MINER / 'validator'
ENGINE = VAL / 'engine'
HOST = VAL / 'trainer-v833l'
DATA = MINER / 'data'
POOL_FILE = DATA / 'original-math7496.tasks.json'
POOL_SHA256 = '77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3'


def install():
    if 'subnet.mimo_grpo' in sys.modules and getattr(sys.modules['subnet.mimo_grpo'], '_kit', False):
        return
    for p in (str(HOST), str(VAL / 'batch'), str(VAL / 'gate'), str(KIT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    sub = types.ModuleType('subnet'); sub.__path__ = [str(ENGINE / 'subnet')]; sub._kit = True
    grpo = types.ModuleType('subnet.mimo_grpo'); grpo.__path__ = [str(ENGINE / 'subnet' / 'mimo_grpo')]; grpo._kit = True
    bj = types.ModuleType('subnet.backend_jobs')
    def signed(*a, **k):
        raise RuntimeError('signed dispatch verification is validator-only (not part of the miner kit)')
    bj.signed = signed
    sys.modules.update({'subnet': sub, 'subnet.mimo_grpo': grpo, 'subnet.backend_jobs': bj})
    sub.mimo_grpo = grpo; sub.backend_jobs = bj


install()
