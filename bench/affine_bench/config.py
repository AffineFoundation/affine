"""Paths and constants. Host = affinevalidator (runs the daemon, holds the R2 presign credentials, serves affine.io);
node = the rented GPU pod (vLLM + verifiers). Override any value with an env var of the same name prefixed BENCH_."""
import os
def _e(k, d): return os.environ.get("BENCH_" + k, d)
HOME = os.path.expanduser("~")
RW = _e("RW", HOME + "/subnet120-rewrite")
STATE = _e("STATE", RW + "/state/bench-results")                       # results + daemon state (host)
LIFECYCLE_EPOCHS = _e("LIFECYCLE_EPOCHS", RW + "/state/mimo-mainnet-v1/epochs")
GATE_LEDGER = _e("GATE_LEDGER", RW + "/state/root-audits/20261026-promotion-gate/state/LEDGER.public.jsonl")
PRESIGN = _e("PRESIGN", RW + "/state/root-audits/20261026-promotion-gate/code/presign.py")
HOST_PYTHON = _e("HOST_PYTHON", RW + "/.venv/bin/python")
PERFORMANCE_JSON = _e("PERFORMANCE_JSON", "https://affine.io/performance.json")   # king list (kings[]: king, epoch, era, crowned_at, parent)
SITE = _e("SITE", HOME + "/subnet120/affine/website")                   # affine.io document root (affine-dash)
NODE_SSH = _e("NODE_SSH", HOME + "/bench-work/bp.sh")                   # ssh wrapper to the node (written by deploy/rent.sh)
NODE_ROOT = _e("NODE_ROOT", "/data/bench")
NODE_REPO = NODE_ROOT + "/repo"
NODE_KINGS = NODE_ROOT + "/models/kings"
NODE_REFS = NODE_ROOT + "/models/ref"
