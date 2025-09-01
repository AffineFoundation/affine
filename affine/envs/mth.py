from __future__ import annotations
import re
import math
import json
import affine as af
import quixand as qx
from typing import Tuple, Optional
from .utils import strip_fences
from typing import Optional, Tuple, Any, Dict

# ----------------------------
# Config / Singletons
# ----------------------------
LEAN_IMAGE = "leanprovercommunity/lean:latest"
lean_config = qx.Config(timeout=600, image=LEAN_IMAGE, workdir="/workspace")
lean_executors = af.singleton('lean-proof', lambda: qx.Playground(n=2, config=lean_config))
lean_executors().prewarm()

math_dataset = af.singleton('rl-math', lambda: af.utils.R2BufferedDataset(
    dataset_name="satpalsr/rl-math",
    buffer_size=10,
    max_batch=1,
))

# ----------------------------
# Helpers
# ----------------------------
def _strip_fences(text: str) -> str:
    """Extract code from ```lang ...``` fences; fallback to raw text. Last block wins."""
    blocks = re.findall(r"```(?:\w+)?\s*\n([\s\S]*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip()

def _parse_response(payload: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse miner response.
    Preference: JSON with keys {"final_answer": <str|num>, "proof_lean": <code>}.
    Fallbacks: detect 'Final Answer: ...' line and ```lean``` fenced block.
    Returns (final_answer_str, lean_proof_src_or_None).
    """
    if not payload:
        return None, None

    # Try JSON
    try:
        data = json.loads(payload)
        ans = str(data.get("final_answer", "")).strip() or None
        proof = data.get("proof_lean", None)
        if isinstance(proof, str):
            proof = proof.strip()
        return ans, proof
    except Exception:
        pass

    # Try "Final Answer:" style
    m = re.search(r"(?i)final\s*answer\s*:\s*([^\n\r]+)", payload)
    ans = m.group(1).strip() if m else None

    # Try ```lean fenced code
    lean_blocks = re.findall(r"```lean\s*\n([\s\S]*?)```", payload, re.DOTALL | re.IGNORECASE)
    proof = lean_blocks[-1].strip() if lean_blocks else None

    return ans, proof

def _answers_equal(expected: str, got: str, *, tol: float = 1e-9) -> bool:
    """
    Compare answers leniently:
      - exact string match (trimmed)
      - numeric equality (float-ish) within tolerance
      - integer string equivalence
    """
    if expected is None or got is None:
        return False

    e = str(expected).strip()
    g = str(got).strip()

    if e == g:
        return True

    # Try numeric comparison
    def _num(s: str):
        # allow things like "4", "4.0", "+4", "-3/2" (fraction fallback)
        s = s.replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            # fraction a/b
            frac = re.match(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$", s)
            if frac:
                a, b = int(frac.group(1)), int(frac.group(2))
                if b != 0:
                    return a / b
        return None

    en = _num(e)
    gn = _num(g)
    if en is not None and gn is not None:
        return math.isfinite(en) and math.isfinite(gn) and abs(en - gn) <= tol

    return False


def _lean_wrapper(expected_expr: str, user_proof_body: str) -> str:
    """
    Wrap the miner's Lean proof inside a theorem that asserts the expected result.
    We require miners to prove:  result_is_correct : result = <expected_expr>
    The user proof body should conclude this equality.

    NOTE: keep wrapper minimal to avoid extra dependencies.
    """
    # Very lightweight Lean 4 file; users can 'by' finish the proof they supply.
    return f"""\
import Std

open Std

-- Prove that the computed result equals the expected expression.
-- Miners must provide a proof of:
--   theorem result_is_correct : result = ({expected_expr}) := by
-- where `result` is defined below (miners can redefine or give a direct proof).
-- They can also directly prove: theorem result_is_correct : ({expected_expr}) = ({expected_expr}) := rfl

/-- Miner-defined result (OPTIONAL). They may also ignore and provide a direct proof. -/
def result : Int := 0

/-- The required correctness statement. -/
theorem result_is_correct : result = ({expected_expr}) := by
{user_proof_body}
"""


def _normalize_expected_for_lean(expected: str) -> str:
    """
    Convert expected answer string into a Lean Int or Rat-like literal where possible.
    For simplicity, treat as Int if it parses, else as Int division if 'a/b'.
    (This keeps the wrapper minimal and avoids importing mathlib rational machinery.)
    """
    s = expected.strip()
    # int?
    try:
        iv = int(s)
        return str(iv)
    except Exception:
        pass
    # fraction?
    m = re.match(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            # fallback — Lean can't handle division by zero; leave as 0 to force failure
            return "0"
        # use Int division if exact (a % b == 0), else leave as an expression with Int.div?
        if a % b == 0:
            return str(a // b)
        # non-integer: represent as (a / b) over Int -> not legal equality for Int.
        # Better: keep numerator and expect miners to convert; but to keep it simple,
        # we encode expected as an Int if it's integral; otherwise, force them to set `result`
        # to the exact same expression with Int.div (which is truncating). We'll hint in prompt.
        return f"Int.ediv ({a}) ({b})"
    # float?
    try:
        fv = float(s)
        # Represent as Int via rounding if exact integer; else floor to avoid mismatches
        if math.isfinite(fv) and abs(fv - round(fv)) < 1e-12:
            return str(int(round(fv)))
        return str(int(math.floor(fv)))
    except Exception:
        pass
    # last resort: 0 (this will likely fail unless miners adjust)
    return "0"


def _compile_lean_in_container(lean_src: str) -> Tuple[bool, str]:
    """
    Write Lean file into a fresh Lean sandbox and run `lean --make Main.lean`.
    Returns (ok, compiler_stderr_or_log).
    """
    sbx: Optional[qx.Sandbox] = lean_executors().create()
    try:
        sbx.files.write("/workspace/Main.lean", lean_src)
        # quick compilation
        r = sbx.run("bash -lc 'lean --make /workspace/Main.lean 1>/workspace/_out.txt 2>/workspace/_err.txt || true'")
        # Read logs
        out = ""
        err = ""
        try:
            out = sbx.files.read("/workspace/_out.txt")
        except Exception:
            pass
        try:
            err = sbx.files.read("/workspace/_err.txt")
        except Exception:
            pass
        # If `lean --make` exited with 0, we consider ok.
        ok = (r.exit_code == 0)
        # lean --make may return 0 even with warnings; we accept as success.
        return ok, (err or out or "")
    finally:
        if sbx is not None:
            try:
                sbx.shutdown()
            except Exception:
                pass
# ----------------------------
# Env
# ----------------------------
class MTHLean(af.BaseEnv):
    """
    Math + Lean-proof environment.

    - generate(): pulls one math item and asks miner for BOTH final answer and a Lean proof
      that the answer equals the ground truth.
    - evaluate(): checks numeric equality and compiles the Lean proof in an isolated container.
    """
    __version__: str = "0.1.0"

    def __init__(self):
        super().__init__()

    async def _next_item(self) -> Dict[str, Any]:
        """
        Pulls one item from the math dataset.
        Expected keys:
          - question: str
          - answer: str|number
          - id: optional
        """
        # R2BufferedDataset usually exposes an async iterator / next()
        item = await math_dataset().next()
        # Normalize
        q = str(item.get("question", "")).strip()
        a = item.get("answer", None)
        if isinstance(a, (int, float)):
            a_str = str(a)
        elif a is None:
            a_str = ""
        else:
            a_str = str(a).strip()

        return {
            "id": item.get("id", f"{int(time.time()*1000)}"),
            "question": q,
            "answer": a_str,
        }

    async def generate(self):
        item = await self._next_item()

        expected = item["answer"]
        qtxt = item["question"]

        prompt = f"""\
            You are given a math problem and must do TWO things:

            1) Compute the **Final Answer** (a single canonical value).
            2) Provide a **Lean 4 proof** that the answer equals the expected value.

            Return your result as JSON with two fields:
            {{
            "final_answer": "<value>",
            "proof_lean": "lean\\n<LEAN PROOF CODE HERE>\\n"
            }}

            markdown
            Copy code

            ### Problem
            {qtxt}

            ### Requirements
            - `final_answer` must be a single value (e.g., `4`, `-3/2`, `42.0`).
            - In `proof_lean`, include a theorem named `result_is_correct` that proves:
            `result = ({expected})`
            You may define `def result : Int := ...` if helpful, or prove equality directly.
            - If the expected answer is a non-integer rational a/b, use integer division `Int.ediv a b`
            in the proof equality (this env checks Int equality by default). If your answer is integral,
            prove equality to that integer.
            - Put the Lean code **inside** a ```lean fenced block as shown above.

            ### Example Skeleton
            ```lean
            -- You can ignore result and prove the equality directly if you prefer.
            -- Or define:
            def result : Int := 4
            theorem result_is_correct : result = (4) := by
            -- your proof here (e.g., rfl if trivially defined)
            rfl
        """
        extra = {
        "id": item["id"],
        "expected": expected,
        "question": qtxt,
        "ts": time.time(),
        }
        return af.Challenge(env=self, prompt=prompt, extra=extra)

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        expected = str(challenge.extra.get("expected", "")).strip()
        payload = response.response or ""

        # Parse miner output
        got_ans, lean_code = _parse_response(payload)

        # 1) Check numeric/lenient equality
        ans_ok = _answers_equal(expected, got_ans or "")

        # 2) Compile Lean proof inside docker
        proof_ok = False
        compile_log = ""
        if lean_code:
            user_body = _strip_fences(lean_code)
            # Allow users who pasted full file with theorem name present already:
            # If they already declared theorem result_is_correct, accept their file as-is.
            if re.search(r"theorem\s+result_is_correct\s*:", user_body):
                lean_src = user_body
            else:
                lean_expected = _normalize_expected_for_lean(expected)
                lean_src = _lean_wrapper(lean_expected, user_body)

            proof_ok, compile_log = _compile_lean_in_container(lean_src)

        score = float(ans_ok and proof_ok)

        extra = {
            "expected": expected,
            "got_answer": got_ans,
            "answer_ok": bool(ans_ok),
            "proof_ok": bool(proof_ok),
            "compile_log": compile_log[-4000:],  # keep it reasonable
        }
        return af.Evaluation(env=self, score=score, extra=extra)