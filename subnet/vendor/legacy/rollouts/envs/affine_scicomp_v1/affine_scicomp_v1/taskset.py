"""affine-scicomp-v1: scientific-computing Python functions with hidden tests, single turn.

aa-gap-fill-plan §2.1 (Jacob "go", 2026-09-22 15:24 UTC). SciCode (10 % of
the AA Intelligence Index) asks for a Python function against a
scientist-written spec and background, graded by hidden numerical tests; D
had no scientific-computing task at all (affine_i3code is competitive
programming). Tasks here are TEACHER-GENERATED (`generate.py`) from public
method inventories (SciPy / NumPy docs, OpenStax chapters, Astropy /
Biopython / ChemPy algorithms) — SciCode's own problems are never used
(decontamination: uids carry `[GEN:e<epoch>]`; the fold drops anything
without it for this source).

Shape (mirrors affine_i3code): a real system message; the reply is graded
as a whole (`text` dialect), the LAST ```python block is the solution; the
hidden pytest file runs IN the rollout runtime (docker `runner =
"verifiers"`, never `verifiers_chat`) with numpy / scipy / sympy available.
`solved` = every test passes; `pass_rate` metric.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import verifiers.v1 as vf

from affine_gen_v1.store import GenTaskStore

SOURCE = "affine_scicomp"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_EPOCH = 1
IMAGE = "python:3.12-slim"
TIMEOUT_S = 120

SYSTEM = (
    "You write scientific-computing code in Python. Read the background and "
    "the task, think it through, then give exactly one complete solution in a "
    "single ```python code block that defines the requested function with "
    "the exact signature given (plus any helpers). numpy, scipy and sympy are "
    "available; do not read input or print. Your visible reply is graded as a "
    "whole by running hidden tests against the last code block."
)

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

# Runs inside the rollout runtime: solution.py + test_solution.py -> JSON report on the last line.
VERIFY = r'''# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "sympy>=1.12", "pytest>=8"]
# ///
import json, os, subprocess, sys
workdir, timeout = sys.argv[1], float(sys.argv[2])
try:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no",
                        "-rN", "test_solution.py"], cwd=workdir, capture_output=True, text=True, timeout=timeout)
    out = p.stdout
    import re
    m_pass = re.search(r"(\d+) passed", out); m_fail = re.search(r"(\d+) failed", out); m_err = re.search(r"(\d+) error", out)
    n_pass = int(m_pass.group(1)) if m_pass else 0
    n_fail = (int(m_fail.group(1)) if m_fail else 0) + (int(m_err.group(1)) if m_err else 0)
    print(json.dumps({"passed": p.returncode == 0 and n_pass > 0 and n_fail == 0, "n_pass": n_pass,
                      "n_total": n_pass + n_fail, "timed_out": False}))
except subprocess.TimeoutExpired:
    print(json.dumps({"passed": False, "n_pass": 0, "n_total": 0, "timed_out": True}))
'''


def extract_code(text: str) -> str:
    blocks = CODE_BLOCK_RE.findall(text or "")
    return blocks[-1].strip() if blocks else ""


class SciCompData(vf.TaskData):
    uid: str
    domain: str
    signature: str
    tests_code: str
    n_tests: int
    teacher_pass: int
    """Blind teacher passes out of 3 at generation time (band: 1-3)."""


class SciCompTask(vf.Task[SciCompData]):
    NEEDS_CONTAINER = True

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        completion = "\n\n".join(m.content for m in trace.assistant_messages if m.content)
        code = extract_code(completion)
        if not code:
            trace.record_metrics({"pass_rate": 0.0, "n_tests": float(self.data.n_tests)})
            return 0.0
        workdir = f"/tmp/scicomp/{trace.id}"
        await runtime.run(["mkdir", "-p", workdir], {})
        await runtime.write(f"{workdir}/solution.py", code.encode())
        await runtime.write(f"{workdir}/test_solution.py", self.data.tests_code.encode())
        result = await runtime.run_uv_script(VERIFY, args=[workdir, str(TIMEOUT_S)])
        if result.exit_code != 0:
            raise RuntimeError(f"scicomp verify failed: {result.stderr.strip()[-1000:]}")
        report = json.loads(result.stdout.strip().splitlines()[-1])
        total = report["n_total"] or self.data.n_tests
        trace.record_metrics({"pass_rate": (report["n_pass"] / total) if total else 0.0,
                              "n_tests": float(total), "timed_out": float(report["timed_out"])})
        return 1.0 if report["passed"] else 0.0


class SciCompConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task uids to load (empty = every task of the epoch)."""
    epoch: int = DEFAULT_EPOCH


def build_prompt(rec: dict) -> str:
    return (f"## Background\n\n{rec['background'].strip()}\n\n## Task\n\n{rec['task'].strip()}\n\n"
            f"Implement this function (exact signature):\n\n```python\n{rec['signature'].strip()}\n```\n\n"
            "Return the value(s) described; do not print. Give the complete solution in one ```python block.")


def list_catalog(epoch: int = DEFAULT_EPOCH) -> list[dict]:
    """Rows for rollouts/catalog.py `genenv`: uid + grouping fields."""
    store = GenTaskStore(SOURCE, PACKAGE_DIR, epoch)
    return [{"uid": r["uid"], "domain": r["domain"], "topic": r.get("topic", ""),
             "teacher_pass": r.get("teacher_pass", 0)} for r in store.tasks()]


class SciCompTaskset(vf.Taskset[SciCompTask, SciCompConfig]):
    def load(self) -> list[SciCompTask]:
        want = set(self.config.tasks)
        store = GenTaskStore(SOURCE, PACKAGE_DIR, self.config.epoch)
        tasks: list[SciCompTask] = []
        for i, rec in enumerate(store.tasks()):
            if want and rec["uid"] not in want:
                continue
            tasks.append(SciCompTask(
                SciCompData(
                    idx=i, name=rec["uid"], image=IMAGE, system_prompt=SYSTEM, prompt=build_prompt(rec),
                    uid=rec["uid"], domain=rec["domain"], signature=rec["signature"],
                    tests_code=rec["tests_code"], n_tests=int(rec.get("n_tests") or 0),
                    teacher_pass=int(rec.get("teacher_pass") or 0),
                ),
                self.config.task,
            ))
        if want and not tasks:
            raise ValueError(f"no scicomp task matched {sorted(want)[:5]}...")
        return tasks
