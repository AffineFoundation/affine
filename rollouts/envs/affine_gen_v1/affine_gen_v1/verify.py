"""Verification helpers for generated tasks (host side, generator time)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:  # the verifiers venv has math-verify; plain generator venvs may not
    from math_verify import parse as _mv_parse, verify as _mv_verify
except Exception:  # noqa: BLE001
    _mv_parse = _mv_verify = None

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
BOXED_RE = re.compile(r"\\boxed\{")


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def run_python(code: str, *, timeout: float = 60.0, python: str | None = None,
               argv: list[str] | None = None, cwd: str | None = None) -> RunResult:
    """Run `code` as a script in a fresh process. Generator-time only: the
    duel-time graders run inside the rollout runtime (see each taskset)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([python or sys.executable, path, *(argv or [])], capture_output=True, text=True,
                               timeout=timeout, cwd=cwd or d, env={**os.environ, "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired as e:
            return RunResult(124, (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
                             (e.stderr or "")[-4000:] if isinstance(e.stderr, str) else "", timed_out=True)
        return RunResult(p.returncode, p.stdout[-20000:], p.stderr[-4000:])


def extract_code(text: str) -> str:
    """The LAST fenced code block (what the graders take)."""
    blocks = CODE_BLOCK_RE.findall(text or "")
    return blocks[-1].strip() if blocks else ""


def boxed_answer(text: str) -> str | None:
    """Content of the last `\\boxed{...}` with balanced braces, or None."""
    text = text or ""
    starts = [m.end() for m in BOXED_RE.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i:j - 1].strip() if depth == 0 else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower().replace("$", "").replace("\\left", "").replace("\\right", ""))


def answers_equal(pred: str | None, gold: str, *, rel_tol: float = 1e-2) -> bool:
    """math-verify when available; else numeric with tolerance, else normalised text."""
    if pred is None:
        return False
    if _mv_parse is not None:
        try:
            g = _mv_parse(f"${gold}$") if not gold.strip().startswith("$") else _mv_parse(gold)
            p = _mv_parse(f"${pred}$") if not pred.strip().startswith("$") else _mv_parse(pred)
            if g and p and _mv_verify(g, p):
                return True
        except Exception:  # noqa: BLE001
            pass
    num = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    gm = num.search(_norm(gold).replace(",", ""))
    pm = num.search(_norm(pred).replace(",", "").replace("\\times10^", "e").replace("×10^", "e"))
    if gm and pm and gm.group(0) == _norm(gold).replace(",", "").strip("$"):
        # gold is a bare number: compare the first number in the prediction (units may follow)
        return abs(float(pm.group(0)) - float(gm.group(0))) <= rel_tol * max(1.0, abs(float(gm.group(0))))
    return _norm(pred) == _norm(gold)
