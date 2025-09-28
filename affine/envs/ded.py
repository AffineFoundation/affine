from __future__ import annotations
import ast
import time
import json
import asyncio
import subprocess
import affine as af
from typing import Any, Dict, List, Tuple

# -------------------------------- Helpers -------------------------------- #
def _to_str(x) -> str:
    """
    Canonicalise any JSON‑serialisable test‑case payload to a single
    newline‑delimited string suitable for feeding to `stdin`.
    """
    if isinstance(x, str):
        return x                     # already a single line
    if isinstance(x, (bytes, bytearray)):
        return x.decode()            # rare, but be safe
    if isinstance(x, list):
        # Recursively stringify nested lists and join with newlines
        return "\n".join(_to_str(e) for e in x)
    # Dicts / numbers / other scalars → JSON text
    return json.dumps(x, ensure_ascii=False)

# --------------------------------------------------------------------------- #
#                           Utility functions                                 #
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Trim trailing blank lines and per‑line trailing spaces."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


# --------------------------------------------------------------------------- #
#                              Affine Env                                     #
# --------------------------------------------------------------------------- #
dataset = af.singleton('rl-python', lambda: af.utils.R2BufferedDataset(
        dataset_name="satpalsr/rl-python",
        buffer_size=5,
        max_batch=5,
))

class DED(af.BaseEnv):
    __version__: str = "0.0.0"
    def __init__(self):
        super().__init__()
        self._executor = af.utils.ProgramExecutor()

    # ----------------------------- Env API -------------------------------- #
    async def generate(self) -> af.Challenge:
        af.logger.trace("Generating new coding challenge")
        sample = await dataset().get()
        if sample is None:
            raise RuntimeError("Failed to fetch dataset row")

        # Mild redundancy to guarantee correct formatting.
        extra_hint = (
            "\n\n---\n"
            "⚠️ **Instructions** ⚠️\n"
            "Write a complete **Python 3** program that\n"
            "• reads *all* input from **STDIN** (using `input()` / `sys.stdin`),\n"
            "• writes *only* the required answer(s) to **STDOUT** using `print`,\n"
            "• contains no additional prompts or debug text, and\n"
            "• is returned as a single ```python … ``` fenced block.\n"
        )
        prompt = sample["prompt"].rstrip() + extra_hint
        return af.Challenge(env=self, prompt=prompt, extra=sample)

    async def evaluate(
        self, challenge: af.Challenge, response: af.Response
    ) -> af.Evaluation:
        af.logger.trace("Starting evaluation of the challenge.")
        raw_reply = response.response
        program = self._executor._strip_fences(raw_reply)
        af.logger.trace(f"Stripped program from response: {program[:50]}...")

        # ---------------- Verification info ---------------------------- #
        sample = challenge.extra or {}
        ver_raw = sample.get("verification_info") or sample.get("test_cases")
        af.logger.trace(f"Verification raw data: {str(ver_raw)[:50]}...")

        # try JSON first, then Python‐literal
        try:
            if isinstance(ver_raw, str):
                try:
                    ver_json = json.loads(ver_raw)
                    af.logger.trace("Parsed verification info via json.loads")
                except json.JSONDecodeError:
                    ver_json = ast.literal_eval(ver_raw)
                    af.logger.trace("Parsed verification info via ast.literal_eval")
            else:
                ver_json = ver_raw
        except Exception as err:
            af.logger.trace(f"Failed to parse verification info: {err}")
            return af.Evaluation(
                env=self,
                score=0.0,
                feedback=f"Invalid verification_info format: {err}",
            )

        # extract test cases list
        cases = ver_json.get("test_cases")
        if not cases:
            af.logger.trace("No test_cases found in verification info.")
            return af.Evaluation(
                env=self, score=0.0, feedback="No public test cases available"
            )
        af.logger.trace(f"Found {len(cases)} test cases.")

        loop = asyncio.get_running_loop()
        passed, total = 0, len(cases)
        details = []

        for i, case in enumerate(cases, start=1):
            ctype = case.get("type")
            raw_inp = case.get("input")
            raw_exp = case.get("output")

            if ctype == "stdin_stdout":
                inp = _to_str(raw_inp)
                if not inp.endswith("\n"):
                    inp += "\n"
                exec_prog = program
                exp = _to_str(raw_exp)
            elif ctype == "function_call":
                fn = case.get("fn_name")
                # input is a list of args
                args = case.get("input", [])
                # wrap program with a call to fn(...) and print its result
                exec_prog = (
                    program
                    + "\n"
                    + f"if __name__ == '__main__':\n"
                    + f"    result = {fn}(*{args!r})\n"
                    + "    print(result)"
                )
                inp = ""  # no stdin
                exp = _to_str(raw_exp[0]) if isinstance(raw_exp, list) and raw_exp else _to_str(raw_exp)
            else:
                af.logger.trace(f"Unknown test case type '{ctype}', skipping.")
                total -= 1
                continue

            try:
                out, err = await loop.run_in_executor(
                    None, self._executor.execute, exec_prog, inp
                )
            except subprocess.TimeoutExpired:
                out, err = "", "TIMEOUT"

            ok_run = not err.strip()
            out_norm = _normalize(out)
            exp_norm = _normalize(exp) if exp is not None else None
            correct = ok_run and (exp_norm is None or out_norm == exp_norm)
            if correct:
                passed += 1
                af.logger.trace(f"Test case {i} passed.")
            else:
                af.logger.trace(
                    f"Test case {i} failed. Got: {out_norm!r}, Expected: {exp_norm!r}"
                )

            details.append(
                {
                    "input": inp,
                    "expected": exp_norm,
                    "got": out_norm,
                    "stderr": err.strip(),
                    "passed": correct,
                }
            )

        score = 1.0 if passed == total else 0.0
        feedback = json.dumps(
            {"passed": passed, "total": total, "tests": details}, ensure_ascii=False
        )
        af.logger.trace(f"Evaluation completed with score: {score}")
        return af.Evaluation(env=self, score=score, feedback=feedback)