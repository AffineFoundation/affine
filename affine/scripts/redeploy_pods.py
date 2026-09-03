"""Push the current tree + .eval_env to the LIVE eval/bench/chat pods and
relaunch bootstrap — no new rental, weights in HF_HOME survive.

Use after a code change the pods need (e.g. the R2 checkpoint path) or after
adding pod secrets (AFFINE_EVAL_R2_*). A duel in flight on the eval pod dies
and is requeued by the validator as an infra fault (no miner is burned).

    cd affine && source ../.venv/bin/activate && set -a && source ../.env && set +a
    python scripts/redeploy_pods.py            # eval pod only
    python scripts/redeploy_pods.py --all      # eval + bench + chat
    python scripts/redeploy_pods.py --role bench

Reads the machine records from the validator's state.json (read-only).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from affine.config import load_config
from affine.provisioner import (BenchMachineManager, ChatMachineManager,
                                EvalMachineManager)
from affine.state import State

FACTORIES = {"eval": EvalMachineManager, "bench": BenchMachineManager,
             "chat": ChatMachineManager}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--role", choices=sorted(FACTORIES), action="append",
                    help="pod role(s) to redeploy (default: eval)")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    roles = sorted(FACTORIES) if args.all else (args.role or ["eval"])
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    state = State(cfg.state_dir)
    state.load()
    repo_root = Path(__file__).resolve().parents[1]
    rc = 0
    for role in roles:
        mgr = FACTORIES[role](cfg, state, repo_root)
        ok = mgr.redeploy()
        print(f"{role}: {'relaunched' if ok else 'FAILED'}")
        rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
