#!/usr/bin/env python
"""The suite lockfile: everything that can change a benchmark number, pinned in
one place (`ops/benchsuite/suite.lock.json`) and stamped into every run
manifest and scorecard, so a Prime run and a Lium run of the same king are
comparable — or visibly not.

  python lock.py write  --bench-home ~/benchsuite     # (re)generate from suite.toml + the installed env
  python lock.py check  --bench-home ~/benchsuite     # exit 1 if the installed env differs from the lock
  python lock.py show

What is pinned:
  code      verifiers commit, research-environments commit, the local patches (sha256 of each patched file)
  envs      per benchmark: taskset id, package version, dataset revision(s) hard-coded in the taskset,
            harness, temperature(s), max_tokens, task subset rule (n + shuffle seed), reward name
  python    versions of the packages that grade or drive (verifiers, math-verify, datasets, bfcl-eval,
            openai, httpx, nltk, langdetect, spacy, prime)
  serving   vLLM version, parsers, max_model_len, max_num_seqs, max_num_batched_tokens, gpu util,
            dtype (bf16), prefix caching, teacher revision
  seeds     verifiers shuffle seed (0), GPQA option shuffle seed (0), sampling seeds (none: T=0 greedy,
            T=0.8 unseeded — the secondary row is stochastic by design)
  sandbox   docker image for chat cells, runtimes per env
Hardware (GPU model, TP) is recorded per run in the manifest, not locked: it is the one thing a
Prime-vs-Lium comparison is allowed to vary, and the parity check measures its effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
LOCK_PATH = HERE / "suite.lock.json"
PATCHED_FILES = [
    "verifiers/verifiers/v1/runtimes/docker/egress.py",
    "verifiers/verifiers/v1/harnesses/utils/core.py",
    "verifiers/verifiers/v1/tasksets/lean/taskset.py",
    "research-environments/environments/tool_use/bfcl_v3/bfcl_v3/taskset.py",
]
PY_PACKAGES = ["verifiers", "math-verify", "datasets", "bfcl-eval", "openai", "httpx", "nltk",
               "langdetect", "spacy", "prime", "prime-sandboxes", "harbor", "huggingface-hub"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=9", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def pip_versions(venv_python: Path) -> dict:
    out = subprocess.run([str(venv_python), "-m", "pip", "list", "--format=json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        out = subprocess.run(["uv", "pip", "list", "--format=json", "--python", str(venv_python)],
                             capture_output=True, text=True)
    pkgs = {p["name"].lower().replace("_", "-"): p["version"] for p in json.loads(out.stdout or "[]")}
    return {k: pkgs.get(k) for k in PY_PACKAGES}


def dataset_pins(taskset_dir: Path) -> dict:
    """Dataset names / revisions hard-coded in a taskset module (best effort)."""
    pins = {}
    for py in taskset_dir.rglob("taskset.py"):
        s = py.read_text()
        for m in re.finditer(r'(?:dataset_name|DATASET_NAME|DATASET)\s*[:=]\s*(?:str\s*=\s*)?"([^"]+)"', s):
            pins.setdefault("dataset", m.group(1))
        for m in re.finditer(r'(?:dataset_revision|DATASET_REVISION)\s*[:=]\s*(?:str\s*=\s*)?"([0-9a-f]{40})"', s):
            pins.setdefault("revision", m.group(1))
        for m in re.finditer(r'load_dataset\("([^"]+)"', s):
            pins.setdefault("dataset", m.group(1))
    return pins


def build(bench_home: Path) -> dict:
    vf = bench_home / "verifiers"
    re_dir = bench_home / "research-environments"
    envs = {}
    for e in SUITE["envs"]:
        inst = e["install"]
        tdir = (HERE / inst) if inst.startswith("envs/") else (re_dir / "environments" / inst)
        version = None
        pyproj = tdir / "pyproject.toml"
        if pyproj.exists():
            version = tomllib.loads(pyproj.read_text())["project"].get("version")
        envs[e["id"]] = {
            "taskset": e["taskset"], "package_version": version, "install": inst,
            "harness": e["harness"], "runtime": e["runtime"],
            "temperatures": [SUITE["sampling"]["primary_temperature"]]
                            + ([SUITE["sampling"]["secondary_temperature"]] if e.get("secondary") else []),
            "secondary_rollouts": e.get("secondary_rollouts", 1) if e.get("secondary") else None,
            "max_tokens": e["max_tokens"], "reward": e["reward"],
            "subset": {"n": e["n"], "shuffle_seed": 0} if int(e.get("n", -1)) > 0 else {"n": "all"},
            "taskset_config": e.get("taskset_config"), "max_turns": e.get("max_turns"),
            "rollout_timeout_s": e.get("rollout_timeout"),
            "datasets": dataset_pins(tdir) if tdir.exists() else None,
        }
    lock = {
        "lock_version": 1,
        "suite": {"name": SUITE["suite"]["name"], "version": SUITE["suite"]["version"]},
        "code": {
            "verifiers": {"repo": SUITE["suite"]["verifiers_repo"], "commit": git_commit(vf) if vf.exists() else SUITE["suite"]["verifiers_commit"]},
            "research_environments": {"repo": SUITE["suite"]["research_envs_repo"], "commit": git_commit(re_dir) if re_dir.exists() else SUITE["suite"]["research_envs_commit"]},
            "patched_files_sha256": {f: (sha256_file(bench_home / f) if (bench_home / f).exists() else None) for f in PATCHED_FILES},
        },
        "python": pip_versions(vf / ".venv" / "bin" / "python") if (vf / ".venv").exists() else {},
        "serving": {**SUITE["serving"], "dtype": "bfloat16", "enable_prefix_caching": True,
                    "teacher": SUITE["teacher"]},
        "seeds": {"verifiers_shuffle_seed": 0, "gpqa_option_shuffle_seed": 0,
                  "sampling_seed": None, "note": "T=0 is greedy; the T=0.8 row is unseeded and stochastic by design"},
        "sandbox": {"chat_cell_image": "python:3.11-slim"},
        "envs": envs,
    }
    lock["lock_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in lock.items()}, sort_keys=True).encode()).hexdigest()[:16]
    return lock


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}


def diff(a: dict, b: dict, path: str = "") -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        p = f"{path}.{k}" if path else k
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff(va, vb, p)
        elif va != vb:
            out.append(f"{p}: locked={va!r} installed={vb!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "check", "show"])
    ap.add_argument("--bench-home", default=str(Path.home() / "benchsuite"))
    a = ap.parse_args()
    home = Path(a.bench_home).expanduser()
    if a.cmd == "show":
        print(json.dumps(load_lock(), indent=1))
        return 0
    cur = build(home)
    if a.cmd == "write":
        LOCK_PATH.write_text(json.dumps(cur, indent=1))
        print(f"wrote {LOCK_PATH} (lock_sha256 {cur['lock_sha256']})")
        return 0
    locked = load_lock()
    if not locked:
        print("no lockfile; run `lock.py write` first")
        return 1
    # hardware-independent fields only; python versions compared for the graders
    d = diff({k: locked[k] for k in ("code", "python", "serving", "seeds", "sandbox", "envs")},
             {k: cur[k] for k in ("code", "python", "serving", "seeds", "sandbox", "envs")})
    if d:
        print("LOCK MISMATCH:")
        for line in d:
            print("  " + line)
        return 1
    print(f"lock ok ({locked['lock_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
