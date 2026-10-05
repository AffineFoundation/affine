# /// script
# requires-python = "==3.12.3"
# dependencies = ["math-verify==0.9.0", "latex2sympy2-extended==1.11.0", "sympy==1.14.0", "antlr4-python3-runtime==4.13.2", "mpmath==1.3.0"]
# ///
"""Grade one \\boxed{} answer inside the rollout's runtime (`uv run`).

argv[1] = gold answer (already the bare content of the reference \\boxed{}),
argv[2] = the model's full reply. Prints 1.0 when the LAST complete
\\boxed{...} in the reply is equivalent to the gold under math-verify, else
0.0. The last-boxed rule and brace balancing mirror affine/dialects.py
`boxed` (the duel scores the same span), kept inline because this script
runs with a pinned math-verify dependency closure. Timeouts and execution
errors exit 75 as indeterminate, rather than certifying a negative sample.

The taskset uses --json-arguments followed by an escaped JSON [gold, reply]
array so arbitrary model text, including null characters, can cross the process
argument boundary unchanged. The original two-argument CLI remains supported.
"""

import json
import sys

import hashlib
import importlib.metadata
import sysconfig
from pathlib import Path

# Prospective approved runtime profile; never derive expected values from the worker.
RUNTIME_LOCK = {'python_version': [3, 12, 3], 'profiles': [{'python_executable_sha256': 'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f', 'stdlib': {'files_count': 619, 'files_sha256': '5e966d14bdf07aa30b321f03ba2c29a29460140d996589436e38224cf5762d52'}}, {'python_executable_sha256': 'a92f0f95e883390c7256b2e441484aac06b1002dbe1d924141a77c8d82f96223', 'stdlib': {'files_count': 621, 'files_sha256': '109f64e0d5d5e1fec32e74a10667dd5e81da7ab14f2a36960cd66b5872de3e4f'}}, {'python_executable_sha256': '1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118', 'stdlib': {'files_count': 621, 'files_sha256': '98ad5250fc683808dd72d414fcab6b0faf5a644b2f7dba9dee8818da81d38955'}}], 'distributions': {'antlr4-python3-runtime': {'python_files_count': 58, 'python_files_sha256': '3d111a9f253f21e31bce604ea6e1eb6aca43a1de5ebbfea64a04282c7689a26e', 'version': '4.13.2'}, 'latex2sympy2-extended': {'python_files_count': 18, 'python_files_sha256': 'c020e4b8c48b3741f8bce7f005700d918e313f29845d6c0de5e8360f48139e27', 'version': '1.11.0'}, 'math-verify': {'python_files_count': 8, 'python_files_sha256': '3d94972b1f30fb376eb2237db446ccfa26e0831f87dd72695497f41576b53455', 'version': '0.9.0'}, 'mpmath': {'python_files_count': 87, 'python_files_sha256': 'fb985c249a036cd33d770c63a3932d237aff8af16e58b47d90daeff4ad8986b5', 'version': '1.3.0'}, 'sympy': {'python_files_count': 1533, 'python_files_sha256': 'dce9dcf83bed93f3e6bcde5c968759a8967c82c6613f34f4fe6136e39c1c7425', 'version': '1.14.0'}}}


def runtime_fingerprint():
    result = {"python_version": list(sys.version_info[:3]),
              "python_executable_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
              "distributions": {}}
    stdlib = Path(sysconfig.get_path("stdlib"))
    files = {str(path.relative_to(stdlib)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(stdlib.rglob("*"))
             if path.is_file() and path.suffix in (".py", ".so")
             and not any(part in path.parts for part in ("site-packages", "dist-packages", "__pycache__"))}
    result["stdlib"] = {"files_count": len(files), "files_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    for name in RUNTIME_LOCK["distributions"]:
        distribution = importlib.metadata.distribution(name)
        files = {}
        for entry in distribution.files or []:
            name_in_wheel = str(entry)
            if name_in_wheel.endswith(".py") and not name_in_wheel.startswith("../"):
                files[name_in_wheel] = hashlib.sha256(Path(distribution.locate_file(entry)).read_bytes()).hexdigest()
        result["distributions"][name] = {"version": distribution.version,
            "python_files_count": len(files),
            "python_files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return result


def indeterminate(reason):
    print(json.dumps({"status": "indeterminate", "reason": reason}), file=sys.stderr)
    sys.exit(75)


try:
    observed = runtime_fingerprint()
    profile = {name: observed[name] for name in ("python_executable_sha256", "stdlib")}
    if (observed["python_version"] != RUNTIME_LOCK["python_version"]
            or observed["distributions"] != RUNTIME_LOCK["distributions"]
            or profile not in RUNTIME_LOCK["profiles"]):
        indeterminate("grader_runtime_mismatch")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment and sys.flags.no_user_site):
        indeterminate("grader_import_isolation_missing")
except Exception:
    indeterminate("grader_runtime_unavailable")

if sys.argv[1:] == ["--runtime-check"]:
    print(json.dumps({"status": "ready", "runtime": RUNTIME_LOCK, "selected_profile": profile, "isolated": True, "site_disabled": True}, sort_keys=True))
    sys.exit(0)

# Close Python imports to the pinned standard library and grader package closure.
# -I -S bootstrap prevents .pth, user site, PYTHONPATH and sitecustomize execution.
import importlib.abc
import importlib.machinery
import importlib.util
stdlib_root = Path(sysconfig.get_path("stdlib")).resolve()
package_roots = [Path(importlib.metadata.distribution(distribution).locate_file(package)).resolve()
                 for distribution, package in (("math-verify", "math_verify"),
                    ("latex2sympy2-extended", "latex2sympy2_extended"), ("sympy", "sympy"),
                    ("antlr4-python3-runtime", "antlr4"), ("mpmath", "mpmath"))]


class PinnedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None:
            return None
        if found.origin is None:
            raise ModuleNotFoundError("grader namespace imports are not approved: " + fullname)
        origin = Path(found.origin).resolve()
        standard = origin.is_relative_to(stdlib_root) and not any(
            part in origin.parts for part in ("site-packages", "dist-packages"))
        if not standard and not any(origin.is_relative_to(root) for root in package_roots):
            raise ModuleNotFoundError("grader import outside pinned closure: " + fullname)
        return found


sys.meta_path.insert(0, PinnedImports())

# Imports follow runtime validation, before any untrusted prediction is parsed.
from math_verify import parse, verify
from math_verify.errors import TimeoutException

OPEN = "\\boxed{"


def last_boxed(text: str) -> str | None:
    found = None
    start = text.find(OPEN)
    while start != -1:
        depth = 0
        for i in range(start + len(OPEN) - 1, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    found = text[start:i + 1]
                    break
        start = text.find(OPEN, start + len(OPEN))
    return found


if len(sys.argv) == 3 and sys.argv[1] == "--json-arguments":
    arguments = json.loads(sys.argv[2])
    if (not isinstance(arguments, list) or len(arguments) != 2
            or any(not isinstance(value, str) for value in arguments)):
        raise ValueError("expected JSON gold/reply string pair")
    gold, reply = arguments
else:
    gold, reply = sys.argv[1], sys.argv[2]
pred = last_boxed(reply)
try:
    # A bad trusted reference is an infrastructure error, never a negative sample.
    parsed_gold = parse(OPEN + gold + "}", raise_on_error=True)
    if not parsed_gold:
        indeterminate("grader_reference_unparseable")
    if pred is None:
        print(0.0)
        sys.exit(0)
    parsed_pred = parse(pred, raise_on_error=True)
    score = 1.0 if verify(parsed_gold, parsed_pred, raise_on_error=True) else 0.0
except TimeoutException:
    indeterminate("grader_timeout")
except Exception:
    indeterminate("grader_execution_error")
print(score)
