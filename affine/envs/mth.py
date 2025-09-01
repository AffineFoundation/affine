from __future__ import annotations
import re
import affine as af
import quixand as qx
from typing import Tuple, Optional

config = qx.Config(timeout=600, image="python:3.11-slim")
executors = af.singleton('math', lambda: qx.Playground(n=2, config=config))
executors().prewarm()


dataset = af.singleton('rl-python', lambda: af.utils.R2BufferedDataset(
        dataset_name="satpalsr/rl-python",
        buffer_size=5,
        max_batch=5,
))

class MTH(af.BaseEnv):
    __version__: str = "0.0.0"
    def __init__(self):
        super().__init__()
        
    async def generate(self):
        sol = {i: random.choice([True, False]) for i in range(1, self.n+1)}
        cls = []
        for _ in range(self.m):
            vs = random.sample(list(sol), self.k)
            sv = random.choice(vs)
            cls.append([(lit := (v if sol[v] else -v)) if v==sv else (v if random.choice([True,False]) else -v) for v in vs])
        formula = " ∧ ".join("(" + " ∨ ".join(f"{'' if l>0 else '¬'}x{abs(l)}" for l in c) + ")" for c in cls)
        prompt = (
            f"Find a satisfying assignment for the following {self.k}-SAT formula over variables x1..x{self.n}:\n"
            f"{formula}\n"
            "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, "
            "or respond `UNSAT` if it has no solution."
        )
        return af.Challenge(env=self, prompt=prompt, extra={"sol": sol, "cls": cls, 'timestamp': time.time()})        

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        sol, cls = challenge.extra["sol"], challenge.extra["cls"]
        got = {int(v): val.lower() in ("true","1")
               for v, val in re.findall(r"x(\d+)=(True|False|1|0)", (response.response or ""))}
        ok = all(any((lit>0)==got.get(abs(lit), None) for lit in c) for c in cls)
        return af.Evaluation(env=self, score=float(ok), extra={"expected": sol, "got": got})

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

    sbx: Optional[qx.Sandbox] = executors().create()
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
