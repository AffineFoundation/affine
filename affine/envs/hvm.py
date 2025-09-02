# affine/affine/envs/hvm.py
from __future__ import annotations
import json, random, re
from typing import Any, Dict, List, Optional, Tuple

import affine as af

# --------------------------------------------------------------------------- #
# HVM: Hole-filled Virtual Machine
# Miners receive:
#   • A small stack-VM program with unknown constants (?a, ?b, …)
#   • Several input vectors and their expected outputs
# They must return a mapping for the holes so that the program reproduces
# the expected outputs on ALL cases.
#
# Miner answer format:
#   <HOLES>
#   ?a=3
#   ?b=-1
#   ?c=42
#   </HOLES>
# --------------------------------------------------------------------------- #

class HVM(af.BaseEnv):
    __version__: str = "0.1.1"  # newline-robust compare + consistent emit

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        # Use the repo's hardened sandbox runner (same one used by ABD/DED).
        self._executor = af.utils.ProgramExecutor()

    # --------------------------- Public API -------------------------------- #
    async def generate(self) -> af.Challenge:
        prog = self._make_program(hard=True)
        inputs, expected = self._forge_io(prog, n_cases=3)
        prompt = self._render_prompt(prog, inputs, expected)
        extra = {
            "program": prog,          # JSON-serializable spec (no secret hole values)
            "inputs": inputs,
            "expected": expected,
        }
        return af.Challenge(env=self, prompt=prompt, extra=extra)

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        resp_text = response.response or ""
        # Parse <HOLES> block
        holes = self._parse_holes(resp_text)
        if holes is None:
            return af.Evaluation(env=self, score=0.0, extra={"error": "Missing or invalid <HOLES> block"})

        # Pull program & cases back out
        spec = challenge.extra.get("program") or {}
        inputs: List[List[int]] = challenge.extra.get("inputs") or []
        expected: List[str] = challenge.extra.get("expected") or []

        # Domain & completeness checks
        hole_domains: Dict[str, List[int]] = spec.get("hole_domains", {})
        hole_names: List[str] = spec.get("holes", [])
        if set(holes.keys()) != set(hole_names):
            return af.Evaluation(env=self, score=0.0, extra={"error": "Not all holes provided"})

        for h, v in holes.items():
            dom = hole_domains.get(h) or []
            if v not in dom:
                return af.Evaluation(env=self, score=0.0, extra={"error": f"value {v} for {h} outside domain {dom}"})

        # Run each case inside the sandboxed ProgramExecutor (like ABD/DED)
        details: List[Dict[str, Any]] = []
        passed = 0
        for idx, (inp, exp) in enumerate(zip(inputs, expected)):
            ok, out = self._run_vm_sandbox(spec, holes, inp)
            exp_c = self._canon(exp)
            out_c = self._canon(out) if ok else ""
            correct = ok and (out_c == exp_c)
            details.append({
                "input": inp,
                "expected": exp, "expected_repr": repr(exp),
                "got": out, "got_repr": repr(out),
                "expected_canon": exp_c, "got_canon": out_c,
                "passed": bool(correct), "sandbox_ok": bool(ok),
            })
            if correct:
                passed += 1

        score = 1.0 if passed == len(inputs) else 0.0
        return af.Evaluation(env=self, score=score, extra={"passed": passed, "total": len(inputs), "details": details})

    # --------------------------- Generators -------------------------------- #

    def _make_program(self, hard: bool) -> Dict[str, Any]:
        """Build a small stack-VM program with holes & domains."""
        rng = self._rng
        holes: List[str] = []
        hole_domains: Dict[str, List[int]] = {}
        code: List[Tuple[str, Optional[str]]] = []

        def new_hole(domain: List[int]) -> str:
            name = f"?{chr(ord('a') + len(holes))}"
            holes.append(name)
            hole_domains[name] = domain[:]
            return name

        dom_small = list(range(-9, 10))
        dom_pos   = list(range(0, 13))
        dom_mod   = [m for m in range(3, 31)]

        # Base arithmetic mix
        code.append(("LOAD", "0"))                  # a
        code.append(("LOAD", "1"))                  # b
        code.append(("PUSH", new_hole(dom_small)))  # ?a
        code.append(("MUL",  None))
        code.append(("PUSH", new_hole(dom_small)))  # ?b
        code.append(("MUL",  None))
        code.append(("ADD",  None))
        code.append(("DUP",  None))
        code.append(("PUSH", new_hole(dom_mod)))    # ?c
        code.append(("MOD",  None))
        code.append(("PRINT", None))

        if hard:
            # Add a small loop over k steps: x = (x*?d + ?e) % ?f
            code.append(("LOAD", "2"))                 # k
            d = new_hole(dom_pos)                      # ?d
            e = new_hole(dom_small)                    # ?e
            f = new_hole(dom_mod)                      # ?f

            loop_start = len(code)
            code.append(("DUP", None))                 # k
            j_end = len(code); code.append(("JMPZ", None))   # -> ?j_end
            code.append(("SWAP", None))                # bring x to top
            code.append(("PUSH", d)); code.append(("MUL", None))
            code.append(("PUSH", e)); code.append(("ADD", None))
            code.append(("PUSH", f)); code.append(("MOD", None))
            code.append(("SWAP", None))
            code.append(("PUSH", "1")); code.append(("SUB", None))
            j_back = len(code); code.append(("JMP", None))   # -> ?j_back
            end_addr = len(code)
            code.append(("POP", None))
            code.append(("PRINT", None))
            code.append(("HALT", None))

            # lock jump targets as holes (single-value domain)
            j1 = new_hole([end_addr])
            j2 = new_hole([loop_start])
            code[j_end]  = ("JMPZ", j1)
            code[j_back] = ("JMP",  j2)
        else:
            code.append(("HALT", None))

        return {
            "code": code,
            "holes": holes,
            "hole_domains": hole_domains,
            "max_steps": 8000 if hard else 4000,
            "stack_cap": 256,
        }

    def _forge_io(self, prog: Dict[str, Any], n_cases: int) -> Tuple[List[List[int]], List[str]]:
        """Create inputs and expected outputs by sampling concrete hole values in-domain."""
        rng = self._rng
        chosen = {h: rng.choice(dom) for h, dom in prog["hole_domains"].items()}
        inputs: List[List[int]] = []
        expected: List[str] = []

        for _ in range(n_cases):
            a = rng.randint(-8, 8)
            b = rng.randint(-8, 8)
            if any(op == "LOAD" and arg == "2" for op, arg in prog["code"]):
                k = rng.randint(1, 8)
                case = [a, b, k]
            else:
                case = [a, b]

            ok, out = self._run_vm_local(prog, chosen, case)
            if not ok or out == "":
                # resample on degenerate
                return self._forge_io(prog, n_cases)
            inputs.append(case)
            expected.append(out)
        return inputs, expected

    def _render_prompt(self, prog: Dict[str, Any], inputs: List[List[int]], expected: List[str]) -> str:
        def render_program() -> str:
            lines = []
            for i, (op, arg) in enumerate(prog["code"]):
                lines.append(f"{i:03d}: {op}" + (f" {arg}" if arg is not None else ""))
            return "\n".join(lines)

        hole_lines = []
        for h in prog["holes"]:
            dom = prog["hole_domains"][h]
            if len(dom) > 15:
                hole_lines.append(f"{h} ∈ [{min(dom)}, {max(dom)}] (integers)")
            else:
                hole_lines.append(f"{h} ∈ {{{', '.join(map(str, dom))}}}")

        case_lines = []
        for i, (inp, out) in enumerate(zip(inputs, expected)):
            case_lines.append(f"Case #{i}: input={inp}  expected stdout=\n---\n{out}\n---")

        return f"""You are given a small stack-based Virtual Machine program with UNKNOWN constants (holes).
Instruction set:
  PUSH n       ; push integer n (n can also be a hole like ?a)
  LOAD i       ; push i-th input integer (0-based)
  ADD SUB MUL DIV MOD
  DUP SWAP POP
  JMP k        ; absolute jump
  JMPZ k       ; jump if top==0
  JMPNZ k      ; jump if top!=0
  PRINT
  HALT

Program:
{render_program()}

Holes and domains:
- """ + "\n- ".join(hole_lines) + """

Test cases:
""" + "\n".join(case_lines) + """

Return ONLY the hole mapping in this exact format:

<HOLES>
?a=3
?b=-1
?c=42
</HOLES>
"""

    # --------------------------- VM runners -------------------------------- #

    def _run_vm_local(self, prog: Dict[str, Any], holes: Dict[str, int], inputs: List[int]) -> Tuple[bool, str]:
        """Deterministic interpreter for gold-label forging (no sandbox). Emits NO trailing newline."""
        ip = 0
        steps = 0
        stack: List[int] = []
        out: List[str] = []
        code = prog["code"]
        max_steps = int(prog["max_steps"])
        cap = int(prog["stack_cap"])

        def push(v: int) -> bool:
            if len(stack) >= cap:
                return False
            stack.append(int(v))
            return True

        while True:
            if steps > max_steps or ip < 0 or ip >= len(code):
                return (False, "")
            op, arg = code[ip]
            steps += 1

            if op == "PUSH":
                if arg is None:
                    return (False, "")
                if isinstance(arg, str) and arg.startswith("?"):
                    if arg not in holes or not push(holes[arg]):
                        return (False, "")
                else:
                    if not push(int(arg)):
                        return (False, "")
                ip += 1
            elif op == "LOAD":
                idx = int(arg or -1)
                if idx < 0 or idx >= len(inputs):
                    return (False, "")
                if not push(inputs[idx]):
                    return (False, "")
                ip += 1
            elif op in ("ADD", "SUB", "MUL", "DIV", "MOD"):
                if len(stack) < 2:
                    return (False, "")
                b = stack.pop()
                a = stack.pop()
                if op == "ADD":
                    c = a + b
                elif op == "SUB":
                    c = a - b
                elif op == "MUL":
                    c = a * b
                elif op == "DIV":
                    if b == 0:
                        return (False, "")
                    c = int(a / b)
                else:
                    if b == 0:
                        return (False, "")
                    c = a % b
                if not push(c):
                    return (False, "")
                ip += 1
            elif op == "DUP":
                if not stack or not push(stack[-1]):
                    return (False, "")
                ip += 1
            elif op == "SWAP":
                if len(stack) < 2:
                    return (False, "")
                stack[-1], stack[-2] = stack[-2], stack[-1]
                ip += 1
            elif op == "POP":
                if not stack:
                    return (False, "")
                stack.pop()
                ip += 1
            elif op in ("JMP", "JMPZ", "JMPNZ"):
                tgt_raw = arg
                if isinstance(tgt_raw, str) and tgt_raw.startswith("?"):
                    if tgt_raw not in holes:
                        return (False, "")
                    tgt = holes[tgt_raw]
                else:
                    tgt = int(tgt_raw)
                if op == "JMP":
                    ip = tgt
                else:
                    if not stack:
                        return (False, "")
                    top = stack.pop()
                    cond = (top == 0)
                    if (op == "JMPZ" and cond) or (op == "JMPNZ" and not cond):
                        ip = tgt
                    else:
                        ip += 1
            elif op == "PRINT":
                if not stack:
                    return (False, "")
                out.append(str(int(stack.pop())))
                ip += 1
            elif op == "HALT":
                break
            else:
                return (False, "")
        # NO trailing newline
        return (True, "\n".join(out))

    def _run_vm_sandbox(self, prog: Dict[str, Any], holes: Dict[str, int], inputs: List[int]) -> Tuple[bool, str]:
        """Run inside the repo's sandbox using ProgramExecutor.execute(code, stdin)."""
        runner = r"""
import sys, json
data = json.loads(sys.stdin.read())
code = data["code"]; holes = data["holes"]; inputs = data["inputs"]
max_steps = int(data["max_steps"]); stack_cap = int(data["stack_cap"])
ip=0; steps=0; stack=[]; out=[]
def push(v):
    if len(stack) >= stack_cap: raise SystemExit(2)
    stack.append(int(v))
while True:
    if steps > max_steps or ip < 0 or ip >= len(code): raise SystemExit(3)
    op, arg = code[ip]; steps += 1
    if op == "PUSH":
        if arg is None: raise SystemExit(4)
        if isinstance(arg, str) and arg.startswith("?"):
            v = holes.get(arg); 
            if v is None: raise SystemExit(5)
            push(v)
        else:
            push(int(arg))
        ip += 1
    elif op == "LOAD":
        idx = int(arg if arg is not None else -1)
        if idx < 0 or idx >= len(inputs): raise SystemExit(6)
        push(inputs[idx]); ip += 1
    elif op in ("ADD","SUB","MUL","DIV","MOD"):
        if len(stack) < 2: raise SystemExit(7)
        b = stack.pop(); a = stack.pop()
        if op == "ADD": c = a + b
        elif op == "SUB": c = a - b
        elif op == "MUL": c = a * b
        elif op == "DIV":
            if b == 0: raise SystemExit(8)
            c = int(a / b)
        else:
            if b == 0: raise SystemExit(9)
            c = a % b
        push(c); ip += 1
    elif op == "DUP":
        if not stack: raise SystemExit(10)
        push(stack[-1]); ip += 1
    elif op == "SWAP":
        if len(stack) < 2: raise SystemExit(11)
        stack[-1], stack[-2] = stack[-2], stack[-1]; ip += 1
    elif op == "POP":
        if not stack: raise SystemExit(12)
        stack.pop(); ip += 1
    elif op in ("JMP","JMPZ","JMPNZ"):
        tgt = holes[arg] if (isinstance(arg,str) and arg.startswith("?")) else int(arg)
        if op == "JMP":
            ip = tgt
        else:
            if not stack: raise SystemExit(13)
            top = stack.pop()
            if (op == "JMPZ" and top == 0) or (op == "JMPNZ" and top != 0):
                ip = tgt
            else:
                ip += 1
    elif op == "PRINT":
        if not stack: raise SystemExit(14)
        out.append(str(int(stack.pop()))); ip += 1
    elif op == "HALT":
        break
    else:
        raise SystemExit(15)
# NO trailing newline
sys.stdout.write("\n".join(out))
"""
        payload = json.dumps({
            "code": prog["code"],
            "holes": holes,
            "inputs": inputs,
            "max_steps": prog["max_steps"],
            "stack_cap": prog["stack_cap"],
        })
        out, err = self._executor.execute(runner, stdin=payload)
        ok = (err.strip() == "")
        return (ok, out if ok else "")

    def _parse_holes(self, text: str) -> Optional[Dict[str, int]]:
        m = re.findall(r"<HOLES>\s*(.*?)\s*</HOLES>", text, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            return None
        block = m[-1]
        out: Dict[str, int] = {}
        for line in block.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mm = re.match(r"(\?[a-zA-Z]\w*)\s*=\s*(-?\d+)$", line)
            if not mm:
                return None
            out[mm.group(1)] = int(mm.group(2))
        return out

    @staticmethod
    def _canon(s: str) -> str:
        """Normalize for robust comparison: unify newlines, strip one trailing newline, rstrip each line."""
        if s is None:
            return ""
        # unify CRLF/CR to LF
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # strip exactly one trailing newline
        if s.endswith("\n"):
            s = s[:-1]
        # rstrip each line (avoid trailing spaces mismatches)
        return "\n".join(line.rstrip() for line in s.split("\n"))
