"""Host settings of the benchmark service. Only the host imports this module; the node never does.

Every host value is a required environment variable. A missing one stops the service at start: it never guesses a path,
a price cap or a store. The paths on the rented node are fixed, because the node is a fresh machine this service sets up.

  BENCH_STATE      directory for results and daemon state
  BENCH_SITE       directory for the local copy of benchmarks.json (a cache; the record is layout.BENCHMARKS in the store)
  BENCH_NODE_SSH   file where deploy/rent.sh writes the ssh wrapper to the node
  BENCH_LIUM       path of the Lium CLI (its key stays in the operator's Lium config)
  BENCH_MAX_PRICE  highest price in $/h this service may rent at
  BENCH_REPO_REF   git commit of this repository that the node checks out (the pinned runtime)
  BENCH_STORE      factory of the v2 object store, as "module:Class.method" (sn120_v2.r2:R2Store.from_env reads SN120_R2_*)
  BENCH_JOURNAL    the controller's journal file, <state_dir>/journal.jsonl; this service only reads it
"""
from sn120_v2.params import require_env

STATE = require_env("BENCH_STATE")
SITE = require_env("BENCH_SITE")
NODE_SSH = require_env("BENCH_NODE_SSH")
LIUM = require_env("BENCH_LIUM")
MAX_PRICE = float(require_env("BENCH_MAX_PRICE"))
REPO_REF = require_env("BENCH_REPO_REF")
STORE = require_env("BENCH_STORE")
JOURNAL = require_env("BENCH_JOURNAL")
REPO_URL = "https://github.com/AffineFoundation/affine.git"
NODE_ROOT = "/data/bench"
NODE_CLONE = NODE_ROOT + "/repo"
NODE_REPO = NODE_CLONE + "/v2"  # the lifted scripts address themselves as <NODE_REPO>/bench/...
NODE_KINGS = NODE_ROOT + "/models/kings"
