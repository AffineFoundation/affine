from __future__ import annotations

import ast
import json
import asyncio
import subprocess
from typing import Any, Dict, List, Tuple


def _normalize(text: str) -> str:
    """Trim trailing blank lines and per‑line trailing spaces."""
    return "\n".join(line.rstrip() for line in (text or "").rstrip().splitlines())


def _to_str(x) -> str:
    """
    Canonicalise any JSON‑serialisable test‑case payload to a newline‑delimited
    string suitable for feeding to `stdin`.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    if isinstance(x, list):
        return "\n".join(_to_str(e) for e in x)
    return json.dumps(x, ensure_ascii=False)


def _parse_verification_info(ver_raw: Any) -> Dict[str, Any] | None:
    """
    Accept flexible verification info formats from datasets:
    - dict already shaped
    - JSON string
    - Python-literal string
    Returns a dict with a `test_cases` list when possible.
    """
    if ver_raw is None:
        return None
    try:
        if isinstance(ver_raw, str):
            try:
                return json.loads(ver_raw)
            except json.JSONDecodeError:
                return ast.literal_eval(ver_raw)
        return ver_raw
    except Exception:
        return None


def _compat_extract_tests(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Try multiple common keys to obtain a list of test cases.
    Each test case should have at least:
      - type: "stdin_stdout" or "function_call"
      - input: data for stdin or args (list)
      - output: expected output (string or list)
    """
    # 1) Direct `tests` or `public_tests` field
    tests = row.get("tests") or row.get("public_tests") or row.get("test_cases")
    if tests:
        return list(tests)

    # 2) Nested under `verification_info`
    ver = _parse_verification_info(row.get("verification_info"))
    if isinstance(ver, dict) and ver.get("test_cases"):
        return list(ver.get("test_cases"))

    # 3) Nothing found → return empty list
    return []


async def evaluate_program(
    executor,
    program: str,
    row: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a Python program against dataset-provided tests.
    Returns (score, details_dict).
    First version uses all-or-nothing scoring: 1.0 if all required tests pass, else 0.0.
    """
    tests = _compat_extract_tests(row)
    details: List[Dict[str, Any]] = []
    if not tests:
        return 0.0, {"error": "no_tests", "message": "No public tests available"}

    loop = asyncio.get_running_loop()
    passed = 0
    total = 0

    for i, case in enumerate(tests, start=1):
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
            fn = case.get("fn_name") or case.get("function") or "solve"
            args = case.get("input", [])
            exec_prog = (
                program
                + "\n"
                + f"if __name__ == '__main__':\n"
                + f"    result = {fn}(*{args!r})\n"
                + "    print(result)\n"
            )
            inp = ""
            exp = _to_str(raw_exp[0]) if isinstance(raw_exp, list) and raw_exp else _to_str(raw_exp)
        else:
            # Unknown case type → skip it from denominator
            details.append({"skipped": True, "reason": f"unknown type: {ctype}"})
            continue

        total += 1
        try:
            out, err = await loop.run_in_executor(None, executor.execute, exec_prog, inp)
        except subprocess.TimeoutExpired:
            out, err = "", "TIMEOUT"

        ok_run = not (err or "").strip()
        out_norm = _normalize(out)
        exp_norm = _normalize(exp) if exp is not None else None
        correct = ok_run and (exp_norm is None or out_norm == exp_norm)
        if correct:
            passed += 1
        details.append(
            {
                "i": i,
                "type": ctype,
                "input": inp,
                "expected": exp_norm,
                "got": out_norm,
                "stderr": (err or "").strip(),
                "passed": correct,
            }
        )

    score = 1.0 if total > 0 and passed == total else 0.0
    return score, {"passed": passed, "total": total, "tests": details}
