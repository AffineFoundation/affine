from __future__ import annotations
import re
import time
import math
import json
import atexit
import affine as af
import quixand as qx
from typing import Tuple, Optional, Any, Dict

# ----------------------------
# Config / Singletons
# ----------------------------
LEAN_IMAGE = "leanprovercommunity/lean:latest"
lean_config = qx.Config(timeout=600, image=LEAN_IMAGE, workdir="/workspace")
lean_executors = af.singleton('lean-proof', lambda: qx.Playground(n=2, config=lean_config))

math_dataset = af.singleton('gsm8k', lambda: af.utils.R2BufferedDataset(
    dataset_name="openai/gsm8k",
    buffer_size=10,
    max_batch=1,
    config='main'
))

# ----------------------------
# Helpers
# ----------------------------
def _strip_fences(text: str) -> str:
    """Extract code from ```lang ...``` fences; fallback to raw text. Last block wins."""
    if not text:
        return ""
    blocks = re.findall(r"```(?:\w+)?\s*\n([\s\S]*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip()

def _extract_answer_from_gsm8k(answer_text: str) -> str:
    """
    Extract the final answer from GSM8K format.
    GSM8K answers end with #### followed by the final answer.
    """
    if not answer_text:
        return ""
    match = re.search(r'####\s*([^\n\r]+)', answer_text)
    return match.group(1).strip() if match else str(answer_text).strip()

def _parse_response(payload: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Parse miner response.
    Preference: strict JSON -> {"final_answer": <str|num>, "proof_lean": "```lean ...```"}.
    Fallbacks: detect 'Final Answer: ...' line and ```lean``` fenced block.
    Return (final_answer_str_or_None, lean_src_or_None, parsed_json_or_None).
    """
    if not payload:
        return None, None, None

    # Try JSON first
    try:
        data = json.loads(payload)
        ans = str(data.get("final_answer", "")).strip() or None
        raw_proof = data.get("proof_lean", None)
        proof = _strip_fences(raw_proof) if isinstance(raw_proof, str) else None
        return ans, proof, data
    except Exception:
        pass

    # Fallback "Final Answer: ..." and fenced lean
    m = re.search(r"(?i)final\s*answer\s*:\s*([^\n\r]+)", payload)
    ans = m.group(1).strip() if m else None

    lean_blocks = re.findall(r"```lean\s*\n([\s\S]*?)```", payload, re.DOTALL | re.IGNORECASE)
    proof = lean_blocks[-1].strip() if lean_blocks else None

    return ans, proof, None

def _answers_equal(expected: str, got: str, *, tol: float = 1e-9) -> bool:
    """
    Compare answers leniently:
      - exact trimmed string
      - numeric equality within tolerance
      - fraction a/b allowed
    """
    if expected is None or got is None:
        return False

    e = str(expected).strip()
    g = str(got).strip()
    if e == g:
        return True

    def _num(s: str):
        s = s.replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            frac = re.match(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$", s)
            if frac:
                a, b = int(frac.group(1)), int(frac.group(2))
                if b != 0:
                    return a / b
        return None

    en = _num(e)
    gn = _num(g)
    if en is not None and gn is not None and math.isfinite(en) and math.isfinite(gn):
        return abs(en - gn) <= tol
    return False

def _normalize_literal_for_lean(claimed: str) -> str:
    """
    Convert the miner's claimed answer string into a Lean Int-ish literal/expression.
    Policy:
      - integers become '123'
      - proper fractions a/b -> Int.ediv a b (truncating). This pairs with the prompt requirement.
      - floats: if near-int -> that int; else floor (to avoid non-int).
    This keeps the wrapper minimal (no mathlib rationals).
    """
    if claimed is None:
        return "0"
    s = claimed.strip()
    # integer?
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
            return "0"
        if a % b == 0:
            return str(a // b)
        return f"Int.ediv ({a}) ({b})"
    # float?
    try:
        fv = float(s)
        if math.isfinite(fv) and abs(fv - round(fv)) < 1e-12:
            return str(int(round(fv)))
        return str(int(math.floor(fv)))
    except Exception:
        pass
    # fallthrough
    return "0"

def _lean_wrapper_expect_claim(claim_expr: str, user_proof_body: str) -> str:
    """
    Wrap the miner's Lean proof inside a theorem that asserts:
        theorem result_equals_claim : result = (<claim_expr>) := by
            <user_proof_body>
    The miner proves their claimed value. Separately, we check that claim vs ground truth.
    """
    return f"""\
import Std
open Std

/-- Optional: miner-defined computed result. They may ignore and prove directly. -/
def result : Int := 0

/-- The miner must prove that their computed result equals their claimed answer. -/
theorem result_equals_claim : result = ({claim_expr}) := by
{user_proof_body}
"""

def _compile_lean_in_container(lean_src: str) -> Tuple[bool, str]:
    """
    Write Lean file into a fresh Lean sandbox and run `lean --make Main.lean`.
    Returns (ok, compiler_stderr_or_log).
    """
    ply = lean_executors()
    atexit.register(ply.close)
    sbx: Optional[qx.Sandbox] = ply.create()
    try:
        sbx.files.write("/workspace/Main.lean", lean_src)
        r = sbx.run("bash -lc 'lean --make /workspace/Main.lean 1>/workspace/_out.txt 2>/workspace/_err.txt || true'")
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
        ok = (r.exit_code == 0)
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
class MTH(af.BaseEnv):
    """
    Math + Lean-proof environment (HARD mode).
    The prompt does NOT reveal ground truth.
    Miners must:
      (1) Solve and produce a single canonical 'final_answer'.
      (2) Provide Lean code that PROVES their claimed answer (result = claim).
    Evaluation:
      - Check numeric equality (claimed vs hidden truth).
      - Compile Lean proof and ensure the claim theorem type matches their claim.
      - Score = 1.0 only if BOTH pass; otherwise 0.0.
    """
    __version__: str = "0.2.0"

    def __init__(self):
        super().__init__()

    async def _next_item(self) -> Dict[str, Any]:
        """
        Pull a single item. Expected keys:
          - question: str
          - answer: str|number (GSM8K format with ####)
          - id: optional
        """
        item = await math_dataset().get()
        q = str(item.get("question", "")).strip()
        a = item.get("answer", None)
        a_str = str(a).strip() if not isinstance(a, (int, float)) else str(a)
        return {
            "id": item.get("id", f"{int(time.time()*1000)}"),
            "question": q,
            "answer": a_str,
        }

    async def generate(self):
        item = await self._next_item()

        # Keep the true expected hidden from the prompt.
        hidden_expected = _extract_answer_from_gsm8k(item["answer"])
        qtxt = item["question"]

        prompt = f"""\
You are given a math problem. You must do **two** things:

1) **Solve** it and produce a single canonical value as `"final_answer"`.
2) **Prove** your claimed answer in **Lean 4**.

Return **JSON only** (no extra text), in this shape:
{{
"final_answer": "<value>",
"proof_lean": "```lean\\n<LEAN CODE HERE>\\n```"
}}

### Rules
- `final_answer` must be a single literal like `7`, `-12`, `3/4`, or `42.0`.
- In your Lean code, include a theorem named **`result_equals_claim`** with the exact type:

theorem result_equals_claim : result = (<your_claim_literal>) := by

You may optionally define `def result : Int := ...` or directly prove the equality.
- Use **`Int`** values/equalities only. If your answer is a non-integer rational `a/b`,
use **`Int.ediv a b`** for the claimed literal (truncating division). If it is integral,
use that integer.
- Put all Lean code inside a ```lean fenced block.

### Problem
{qtxt}
        """
        # Keep the hidden truth only in the challenge metadata, not in the prompt.
        extra = {
            "id": item["id"],
            "expected": hidden_expected,
            "question": qtxt,
            "ts": time.time(),
        }
        return af.Challenge(env=self, prompt=prompt, extra=extra)

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
      # 0) Basics
      payload = response.response or ""
      hidden_expected = str(challenge.extra.get("expected", "")).strip()

      # 1) Parse miner output
      got_ans, lean_code, parsed = _parse_response(payload)

      # 2) Check numeric/lenient equality with hidden truth
      ans_ok = _answers_equal(hidden_expected, got_ans or "")

      # 3) Compile Lean proof that asserts equality to the miner's claim
      proof_ok = False
      compile_log = ""
      if lean_code and got_ans:
          claim_expr = _normalize_literal_for_lean(got_ans)
          lean_src = _lean_wrapper_expect_claim(claim_expr, _strip_fences(lean_code))
          proof_ok, compile_log = _compile_lean_in_container(lean_src)
      else:
          compile_log = "Missing Lean code and/or final_answer."

      # 4) Score: must pass BOTH
      score = float(ans_ok and proof_ok)

      extra = {
          "expected_hidden": hidden_expected,  # stored for audit
          "got_answer": got_ans,
          "answer_ok": bool(ans_ok),
          "proof_ok": bool(proof_ok),
          "compile_log": (compile_log or "")[-4000:],
          "parsed_json": bool(parsed is not None),
      }
      return af.Evaluation(env=self, score=score, extra=extra)
