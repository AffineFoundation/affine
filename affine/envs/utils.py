from __future__ import annotations

import re
from typing import Tuple, Optional
import affine as af
from affine import quixand as qx
import atexit


config = qx.Config(timeout=600, image="python:3.11-slim")
executors = af.singleton('slim-playground', lambda: qx.Playground(n=2, config=config))

def strip_fences(text: str) -> str:
    """Extract code from ```python ...``` or generic ``` ``` fences; fallback to raw text.
    Returns the last fenced block if multiple are present.
    """
    py_blocks = re.findall(r"```python\s*\n([\s\S]*?)```", text, re.DOTALL)
    if py_blocks:
        return py_blocks[-1].strip()
    code_blocks = re.findall(r"```(?:\w*)\s*\n([\s\S]*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    return (m.group(1) if m else text).strip()


async def run_in_sandbox(
    program: str,
    stdin_text: str = "",
    *,
    template: str = "python:3.11-slim",
) -> Tuple[str, str]:
    """Execute Python code inside a Quixand sandbox with optional stdin.

    - Strips code fences
    - Autoruns solve() when no __main__ guard exists
    Returns (stdout, stderr) as text.
    """
    src = strip_fences(program)
    if ("def solve" in src) and ("__name__" not in src):
        src = (
            src
            + "\n\nif __name__ == \"__main__\":\n"
            + "    res = solve()\n"
            + "    if res is not None:\n"
            + "        import sys\n"
            + "        if isinstance(res, (list, tuple)):\n"
            + "            print(*res)\n"
            + "        else:\n"
            + "            print(res)\n"
        )

    ply = executors()
    atexit.register(ply.close)
    sbx: Optional[qx.Sandbox] = ply.create()
    try:
        sbx.files.write("/workspace/main.py", src)
        redir = ""
        if stdin_text:
            if not stdin_text.endswith("\n"):
                stdin_text = stdin_text + "\n"
            sbx.files.write("/workspace/input.txt", stdin_text)
            redir = "< /workspace/input.txt"

        cmd = f"bash -lc 'python /workspace/main.py {redir} 1>/workspace/_out.txt 2>/workspace/_err.txt'"
        sbx.run(cmd)
        try:
            out = sbx.files.read("/workspace/_out.txt")
        except Exception:
            out = ""
        try:
            err = sbx.files.read("/workspace/_err.txt")
        except Exception:
            err = ""
        return out, err
    finally:
        if sbx is not None:
            try:
                sbx.shutdown()
            except Exception:
                pass


