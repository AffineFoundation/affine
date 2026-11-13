"""Record the verdicts of the v1 math grader, as test data for the v2 grader.

This script imports v1 code, so it lives outside ``v2/``. It writes only
inputs and verdicts. The v2 test ``v2/tests/test_envs_math_grader.py`` reads
the file and requires the v2 grader to give the same verdict for every case.

What it runs: ``miner/kit/grade.py::grade``, the importable copy of the grade
path of the v1 probe worker (final-channel projection, math-verify 0.9.0 with
the ``verify.py`` rules, then the lenient rules of
``miner/validator/probe/allan_src.py``), wrapped the way
``probe_worker.grade_job`` wraps it (any escaped error is ``indeterminate``).
It also checks that ``allan_src.py`` is the text the probe worker runs, and
that the projection equals ``subnet.mimo_answer_projection.project``.

Use, from the repository root, in a virtual environment with the pinned
grader packages plus ``transformers`` and ``numpy``:

    PYTHONPATH=. python tools/record_v1_math_verdicts.py --tokenizer DIR --out v2/tests/data/math_grader_v1_verdicts.jsonl

``DIR`` holds ``tokenizer.json`` and ``tokenizer_config.json`` of the head
(public at huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from miner.kit import grade as kit
from subnet import complete_group_distribution, mimo_answer_projection

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "miner/data/gate-pool-v2.jsonl"
SOURCES = (
    "miner/validator/probe/allan_src.py",
    "miner/validator/probe/probe_worker.py",
    "miner/kit/grade.py",
    "subnet/mimo_answer_projection.py",
    "miner/data/gate-pool-v2.jsonl",
)
PACKAGES = ("math-verify", "latex2sympy2-extended", "sympy", "antlr4-python3-runtime", "mpmath")
EOS = kit.EOS[0]
MANIFEST = {
    mimo_answer_projection.POLICY_KEY: mimo_answer_projection.POLICY,
    "sampling_contract": {"version": complete_group_distribution.MIMO_VERSION},
}
HARNESS = {"version": "text-tools-mimo-complete-group-v1", "enable_thinking": True}
T = "<think>\nLet me work it out step by step.\n</think>\n\n"
OPTIONS = "What is $2+5$? $\\textbf{(A)}\\ 5 \\qquad \\textbf{(B)}\\ 7 \\qquad \\textbf{(C)}\\ \\frac{1}{2} \\qquad \\textbf{(D)}\\ 12$"
OPTIONS_TEXT = "Pick one. $$\\text A. \\ 5 \\qquad \\text B. \\ 7 \\qquad \\text C. \\ 9$$"

# Output formats. "@" is where the boxed answer text goes.
FORMATS = {
    "strict": T + "\\boxed{@}",
    "strict-sentence": T + "The final answer is \\boxed{@}",
    "strict-space": "<think>\nwork\n</think>\\boxed{@}\n\n",
    "strict-display": T + "\\[\\boxed{@}\\]",
    "strict-display-spaces": T + "So\n\\[ \\boxed{@} \\]\n",
    "strict-inline": T + "\\(\\boxed{@}\\)",
    "strict-dollar": T + "$\\boxed{@}$",
    "strict-dollars": T + "Answer: $$\\boxed{@}$$",
    "strict-box-in-think": "<think>\nI guess \\boxed{0}, no.\n</think>\n\n\\boxed{@}",
    "strict-empty-think": "<think></think>\\boxed{@}",
    "p1-closed-dollar": T + "We get $x$\\boxed{@}",
    "p1-double-dollar": T + "$$\\boxed{@}",
    "p1-escaped-dollar": T + "It costs \\$5 and $y$\\boxed{@}",
    "p2-open-dollar": T + "$\\boxed{@}",
    "p2-open-display": T + "\\[\\boxed{@}",
    "p2-open-inline": T + "\\(\\boxed{@}",
    "p2-tail-period": T + "\\boxed{@}.",
    "p2-tail-words": T + "\\boxed{@} is the answer.",
    "p2-tail-wrong-wrapper": T + "$\\boxed{@}$$",
    "p2-tail-no-opener": T + "\\boxed{@}\\]",
    "p2-two-boxes-last": T + "First \\boxed{0}, then \\boxed{@}",
    "p2-two-boxes-first": T + "\\boxed{@} and not \\boxed{0}",
    "p2-same-box-twice": T + "\\boxed{@} so \\boxed{@}",
    "p2-no-think": "\\boxed{@}",
    "p2-no-think-words": "The answer is \\boxed{@}.",
    "p2-close-only": "work\n</think>\n\n\\boxed{@}",
    "p2-open-only": "<think>\nwork \\boxed{@}",
    "p2-two-think": "<think>a</think>\n<think>b</think>\n\\boxed{@}",
    "p2-two-close": "<think>a</think> \\boxed{0} </think> \\boxed{@}",
    "p2-text-before-think": "Hello <think>a</think>\\boxed{@}",
    "p2-think-reversed": "</think>a<think>\\boxed{@}",
    "p2-last-box-unclosed": T + "\\boxed{@} and \\boxed{12",
    "p2-last-box-empty": T + "\\boxed{@} and \\boxed{ }",
    "p2-control-token-tail": T + "\\boxed{@}<|im_end|>",
    "control-token-in-think": "<think>a<|vision_start|>b</think>\\boxed{@}",
    "control-token-before-box": T + "<|endoftext|>\\boxed{@}",
    "none-unclosed": T + "\\boxed{@",
    "none-box-only-in-think": "<think>\\boxed{@}</think>\n\nno answer here",
    "none-no-box": T + "The answer is @.",
    "none-not-a-box": T + "\\boxed {@}",
    "none-empty-output": "",
}
ANSWERS = (("391", "391"), ("391", "390"), ("\\frac{1}{2}", "\\frac{2}{4}"), ("\\frac{1}{2}", "0.25"))

# (group, reference answer, boxed answer text, problem), each graded in a strict and in a lenient-only format.
PAIRS = [
    *[("native", gold, answer, "p") for gold, answer in (
        ("1/2", "0.5"), ("2\\sqrt{2}", "\\sqrt{8}"), ("x^2+2x+1", "(x+1)^2"), ("10\\%", "10"), ("5", "x=5"), ("x=5", "5"),
        ("5", "\\{5\\}"), ("5", "5,"), ("5", "\\text{five}"), ("\\text{five}", "five"), ("\\frac12", "\\int_0^1 x dx"),
        ("5", "1/0"), ("5", "\\begin{pmatrix}"), ("5", "\\frac{"), ("5", "}{"), ("(1,2)", "(1, 2)"), ("(1,2)", "[1,2]"),
        ("\\pi", "3.14"), ("3.14", "3.140"), ("1,2,3", "3,2,1"), ("\\{1,2\\}", "\\{2,1\\}"), ("5", "5^\\circ"), ("90^\\circ", "90"),
        ("\\$5", "5"), ("5", "5 \\text{ dollars}"), ("x \\in [1,2]", "[1,2]"), ("y=2x+1", "2x+1"), ("2x+1", "1+2x"),
    )],
    *[("E1", gold, answer, "p") for gold, answer in (
        ("12", "12\\text{ cm}"), ("12", "12 \\mathrm{cm}"), ("1000000", "1,\\!000,\\!000"), ("1{,}000", "1000"),
        ("1{,}000{,}000", "1,\\!000,\\!000"), ("\\dfrac{1}{2}", "\\tfrac{1}{2}"), ("\\dfrac{3}{4}", "0.75"), ("12", "\\textbf{12}"),
        ("\\text{Monday}", "\\mbox{Monday}"), ("\\text{Monday}", "\\text{Tuesday}"), ("12\\text{ cm}", "13"), ("12,\\!345", "12345"),
        ("\\text{odd}", "odd"), ("\\text{yes}", "\\text{Yes}"), ("3\\text{ and }4", "3, 4"), ("\\mathrm{e}", "e"),
    )],
    *[("E2", gold, answer, "p") for gold, answer in (
        ("A", "(A)"), ("\\text{(B)}", "B"), ("A, C", "C, A"), ("(A), (C)", "A,C"), ("\\textbf{(D)}", "D"), ("A", "B"),
        ("A, C", "A"), ("\\text{A}", "\\textbf{A}"), ("B", "\\ B"), ("(E)", "$E$"), ("A,B,C", "(C), (B), (A)"), ("AB", "BA"),
    )],
    *[("E3", gold, answer, OPTIONS) for gold, answer in (
        ("7", "B"), ("7", "\\textbf{(B)}"), ("7", "(B) 7"), ("7", "\\text{(B)}\\ 7"), ("7", "A"), ("7", "(A) 7"), ("\\frac{1}{2}", "C"),
        ("\\frac12", "\\textbf{(C)}\\ \\frac{1}{2}"), ("12", "(D)"), ("12", "E"), ("B", "7"), ("B", "(B) 7"), ("\\textbf{(B)}", "B"),
    )],
    *[("E3", gold, answer, OPTIONS_TEXT) for gold, answer in (("9", "C"), ("9", "\\text{C}"), ("9", "B"), ("7", "\\text B. 7"))],
    *[("E4", gold, answer, "p") for gold, answer in (
        ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "(1, 2)"), ("(1, 2)", "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}"),
        ("\\begin{bmatrix} 1 \\\\ 2 \\\\ 3 \\end{bmatrix}", "(1,2,3)"), ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "(2, 1)"),
        ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}"),
        ("\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}", "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}"),
        ("\\begin{pmatrix} 1 & 2 \\end{pmatrix}", "(1, 2)"), ("\\begin{pmatrix} \\frac{1}{2} \\\\ -3 \\end{pmatrix}", "\\left(\\frac12, -3\\right)"),
    )],
    *[("E5", gold, answer, "p") for gold, answer in (
        ("(-\\infty,5)", "x<5"), ("(5,\\infty)", "x>5"), ("(-\\infty,5]", "x\\le 5"), ("(-\\infty,5]", "x\\leq 5"),
        ("[5,\\infty)", "x\\ge 5"), ("[5,\\infty)", "x\\geq 5"), ("(2,5)", "2<x<5"), ("(2,5)", "2 < y < 6"),
        ("(-\\infty,3)\\cup(3,\\infty)", "x\\neq 3"), ("(-\\infty,3)\\cup(3,\\infty)", "x\\ne3"), ("(-\\infty,5)", "x>5"),
        ("(-\\infty,\\frac{1}{2})", "x<\\frac{1}{2}"), ("(-\\infty,-2.5)", "t < -2.5"), ("[2,5]", "2\\le x\\le 5"), ("(5,\\infty)", "5<x"),
    )],
    *[("E6", gold, answer, "p") for gold, answer in (
        ("\\begin{pmatrix} 2 & 0 \\\\ 0 & 2 \\end{pmatrix}", "2I"), ("\\begin{pmatrix} 2 & 0 \\\\ 0 & 2 \\end{pmatrix}", "2\\mathbf{I}"),
        ("\\begin{pmatrix} -3 & 0 & 0 \\\\ 0 & -3 & 0 \\\\ 0 & 0 & -3 \\end{pmatrix}", "-3 I"),
        ("\\begin{pmatrix} 2 & 0 \\\\ 0 & 2 \\end{pmatrix}", "3I"), ("\\begin{pmatrix} 2 & 1 \\\\ 0 & 2 \\end{pmatrix}", "2I"),
        ("\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}", "I"), ("\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}", "1I"),
    )],
    *[("E7", gold, answer, "p") for gold, answer in (
        ("1+2i", "2i+1"), ("\\frac{1}{2}+\\frac{\\sqrt{3}}{2}i", "\\frac{1+\\sqrt{3}i}{2}"), ("1+2i", "1-2i"), ("i", "\\sqrt{-1}"),
        ("-i", "\\frac{1}{i}"), ("2i", "i+i"), ("(1+i)^2", "2i"), ("3+4i", "4i+3"), ("\\text{finite}", "\\text{infinite}"),
        ("e^{i\\pi}", "-1"), ("-1", "i^2"), ("1+i", "\\sqrt{2}e^{i\\pi/4}"),
    )],
    *[("indeterminate", gold, answer, "p") for gold, answer in (
        ("", "5"), ("\\text{}", "5"), ("5", "9^{9^{9^{9}}}"), ("9^{9^{9^{9}}}", "5"), ("\\text{}", "\\text{}"),
    )],
]


def mutations(gold: str, n: int) -> tuple[str, str]:
    """One (format name, boxed answer text) for a pool reference answer, chosen by ``n``."""
    names = sorted(FORMATS)
    choices = [
        ("strict", gold),
        (names[n % len(names)], gold),
        ("strict", "\\text{" + gold + "}"),
        ("strict", gold + "\\text{ units}"),
        ("strict", gold.replace("\\frac", "\\dfrac").replace("\\dfrac", "\\tfrac", 1)),
        ("strict", "x = " + gold),
        ("strict", gold + "+1"),
        (names[(7 * n + 3) % len(names)], gold + "0"),
        ("strict", gold.replace("\\left", "").replace("\\right", "").replace(" ", "")),
        ("strict", "\\textbf{(" + gold + ")}"),
        ("strict", gold.replace("\\begin{pmatrix}", "(").replace("\\end{pmatrix}", ")").replace("\\\\", ",")),
        ("strict", gold.replace(",", ",\\!")),
    ]
    return choices[n % len(choices)]


def fuzz_cases(count: int, seed: int = 120) -> list[dict]:
    """Random cases near every rule: the same answer written in two random ways, in a random output shape."""
    rng = random.Random(seed)
    pick = rng.choice

    def number() -> list[str]:
        value = pick((12, 1000, 43210, 1234567))
        grouped = f"{value:,}"
        forms = [str(value), grouped, grouped.replace(",", ",\\!"), grouped.replace(",", "{,}"), str(value + 1)]
        unit = pick(("", "", "\\text{ cm}", " \\mathrm{cm}", "\\mbox{ cm}", "\\text{ units}", "\\%", "^\\circ"))
        wrap = pick(("@", "@", "\\text{@}", "\\textbf{@}", "\\mathrm{@}", "\\$@"))
        return [wrap.replace("@", form) + unit for form in forms]

    def fraction() -> list[str]:
        top, bottom = pick((1, 3, 7)), pick((2, 4, 8))
        return [f"\\{kind}{{{top}}}{{{bottom}}}" for kind in ("frac", "dfrac", "tfrac")] + [f"{top}/{bottom}", str(top / bottom)]

    def letters() -> list[str]:
        chosen = rng.sample("ABCDE", pick((1, 1, 2, 3)))
        forms = []
        for _ in range(5):
            item = pick(("@", "(@)", "\\text{@}", "\\text{(@)}", "\\textbf{(@)}", "\\textbf{@}", "\\mathrm{@}"))
            order = rng.sample(chosen, len(chosen))
            text = pick((", ", ",", " , ", "\\ ", " and ", "")).join(item.replace("@", letter) for letter in order)
            forms.append(pick(("@", "@", "$@$", "\\text{@}", "@.")).replace("@", text))
        return forms + [pick("ABCDE"), pick("ABCDE").lower()]

    def choice() -> tuple[list[str], str]:
        values = rng.sample(["5", "7", "\\frac{1}{2}", "12", "2\\sqrt{3}", "0.5", "\\pi"], 4)
        if rng.random() < 0.7:
            problem = "Which one? $" + " \\qquad ".join(f"\\textbf{{({letter})}}\\ {value}" for letter, value in zip("ABCD", values)) + "$"
        else:
            problem = "Which one? $$" + " \\qquad ".join(f"\\text {letter}. \\ {value}" for letter, value in zip("ABCD", values)) + "$$"
        at = rng.randrange(4)
        letter, value = "ABCD"[at], values[at]
        other = "ABCD"[(at + 1) % 4]
        forms = [value, letter, f"({letter})", f"\\textbf{{({letter})}}", f"\\text{{({letter})}}", f"({letter}) {value}",
                 f"\\textbf{{({letter})}}\\ {value}", f"\\text{{({letter})}} {value}", f"({other})", f"({other}) {value}", "E", f"{letter}) {value}"]
        return forms, problem

    def vector() -> list[str]:
        entries = [pick(("1", "-2", "\\frac{1}{2}", "0", "x", "3\\sqrt{2}")) for _ in range(pick((2, 3)))]
        column = " \\\\ ".join(entries)
        return [f"\\begin{{pmatrix}} {column} \\end{{pmatrix}}", f"\\begin{{bmatrix}} {column} \\end{{bmatrix}}", "(" + ", ".join(entries) + ")",
                "\\left(" + ",".join(entries) + "\\right)", "[" + ", ".join(entries) + "]", f"\\begin{{pmatrix}} {' & '.join(entries)} \\end{{pmatrix}}",
                "(" + ", ".join(reversed(entries)) + ")", "\\langle " + ", ".join(entries) + " \\rangle"]

    def inequality() -> list[str]:
        variable, low, high = pick("xyt"), pick(("5", "-2.5", "\\frac{1}{2}", "\\sqrt{2}", "0")), pick(("7", "10", "3\\pi"))
        space = pick(("", " "))
        forms = [f"{variable}{space}{op}{space}{low}" for op in ("<", ">", "\\le", "\\leq", "\\ge", "\\geq", "\\neq", "\\ne")]
        forms += [f"{low}{space}<{space}{variable}{space}<{space}{high}", f"{low}\\le {variable}<{high}", f"{variable}\\in({low},{high})"]
        forms += [f"(-\\infty,{space}{low})", f"(-\\infty,{low}]", f"({low},{space}\\infty)", f"[{low},\\infty)", f"({low},{space}{high})",
                  f"(-\\infty,{low})\\cup({low},\\infty)", f"\\left(-\\infty,{low}\\right)"]
        return forms

    def diagonal() -> list[str]:
        scalar, size = pick((-3, 1, 2, 5)), pick((2, 3))
        def matrix(value: int, kind: str = "pmatrix") -> str:
            rows = [" & ".join(str(value) if row == column else "0" for column in range(size)) for row in range(size)]
            return f"\\begin{{{kind}}} " + " \\\\ ".join(rows) + f" \\end{{{kind}}}"
        return [matrix(scalar), matrix(scalar), matrix(scalar, "bmatrix"), matrix(scalar + 1), f"{scalar}I", f"{scalar} I", f"{scalar}\\mathbf{{I}}",
                f"{scalar}\\mathbf{{I}}_{size}", f"{scalar}I_{size}", f"{scalar + 1}I", f"{scalar}\\text{{I}}", "I"]

    def complex_number() -> list[str]:
        real, imaginary = pick((1, -2, 3)), pick((2, -1, 4))
        return [f"{real}+{imaginary}i".replace("+-", "-"), f"{imaginary}i+{real}".replace("+-", "-"), f"{real} + {imaginary} i".replace("+ -", "- "),
                f"\\frac{{{2 * real}+{2 * imaginary}i}}{{2}}".replace("+-", "-"), f"{real}+{imaginary}\\sqrt{{-1}}".replace("+-", "-"),
                f"{real}-{imaginary}i".replace("--", "+"), f"({real},{imaginary})", f"{real}+{imaginary}j".replace("+-", "-"), f"i({imaginary}) + {real}"]

    def shape() -> str:
        think = pick((T, T, T, T, "", "<think>a", "a</think>", "<think>a</think><think>b</think>", "Hi <think>a</think>", " \n<think>\na\n</think>"))
        prefix = pick(("", "", "", "The answer is ", "\\[", "\\(", "$", "$$", "$x$", "$x$ $", "\\$ $", "so $y$ and \\(", "\\[ ", "$ "))
        box = pick(("\\boxed{@}",) * 6 + ("\\boxed{@} \\boxed{@}", "\\boxed{0} \\boxed{@}", "\\boxed{@} \\boxed{0}", "\\boxed{@", "\\boxed{}", "\\boxed{{@}}", "boxed{@}", "\\boxed{ @ }"))
        tail = pick(("", "", "", ".", "\\]", "\\)", "$", "$$", " ", "\n", "$ ", " \\] ", " \\)", "<|im_end|>"))
        return think + prefix + box + tail

    families = (number, fraction, letters, vector, inequality, diagonal, complex_number)
    cases = []
    for n in range(count):
        if n % 8 == 7:
            forms, problem = choice()
            name = "choice"
        else:
            family = families[n % 8]
            forms, problem, name = family(), "p", family.__name__
        gold, answer = (forms[0], pick(forms)) if name == "choice" else (pick(forms), pick(forms))
        template = shape() if n % 3 == 0 else FORMATS[pick(("strict", "strict", "p2-tail-period", "p1-closed-dollar", "p2-no-think"))]
        cases.append({"group": "fuzz:" + name, "gold": gold, "problem": problem, "output": template.replace("@", answer), "stop": "eos"})
    return cases


def build_cases(pool_step: int, fuzz: int) -> tuple[list[dict], dict[str, dict]]:
    cases = []
    for name, template in FORMATS.items():
        for gold, answer in ANSWERS:
            cases.append({"group": "format:" + name, "gold": gold, "problem": "p", "output": template.replace("@", answer), "stop": "eos"})
    for name in ("strict", "p2-tail-period", "none-unclosed"):
        cases.append({"group": "capped:" + name, "gold": "391", "problem": "p", "output": FORMATS[name].replace("@", "391"), "stop": "length"})
    cases.append({"group": "capped:mid-think", "gold": "391", "problem": "p", "output": "<think>\nstill working", "stop": "length"})
    for group, gold, answer, problem in PAIRS:
        for name in ("strict", "p2-tail-period"):
            cases.append({"group": f"{group}:{name}", "gold": gold, "problem": problem, "output": FORMATS[name].replace("@", answer), "stop": "eos"})
    rows = [json.loads(line) for line in POOL.read_text().splitlines()]
    for n, row in enumerate(rows[::pool_step]):
        for k in (0, 1 + n % 11):
            name, answer = mutations(row["answer"], k if k == 0 else n * 12 + k)
            cases.append({"group": f"pool:{name}", "task_id": row["id"], "output": FORMATS[name].replace("@", answer), "stop": "eos"})
    return cases + fuzz_cases(fuzz), {row["id"]: row for row in rows}


def v1_verdict(tokenizer, case: dict, gold: str, problem: str) -> tuple[str, str, str, float]:
    """Run the v1 grader. Returns the text v1 graded, the kind, how, and the seconds it took."""
    body = tokenizer.encode(case["output"], add_special_tokens=False)
    out = body + [EOS] if case["stop"] == "eos" else body
    text = tokenizer.decode(body, skip_special_tokens=False)
    started = time.monotonic()
    try:
        kind, how = kit.grade(out, case["stop"], gold, problem)
    except BaseException as error:  # probe_worker.grade_job: an escaped error is indeterminate
        if isinstance(error, KeyboardInterrupt):
            raise
        kind, how = "indeterminate", "grade-crash " + type(error).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    seconds = time.monotonic() - started
    probe = mimo_answer_projection.project(MANIFEST, HARNESS, tokenizer, out, case["stop"])
    if case["stop"] == "eos" and (probe["grading_text"] or None) != kit.project(out):
        sys.exit(f"the kit projection differs from subnet.mimo_answer_projection for {case!r}")
    return text, kind, how, seconds


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="directory with the head tokenizer files")
    parser.add_argument("--out", required=True, help="the JSONL file to write")
    parser.add_argument("--pool-step", type=int, default=96, help="take every n-th pool task")
    parser.add_argument("--fuzz", type=int, default=1600, help="how many random cases to add")
    args = parser.parse_args()

    worker = ast.parse((ROOT / SOURCES[1]).read_text())
    embedded = [node.value.value for node in worker.body if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_ALLAN_SRC"]
    if embedded != [(ROOT / SOURCES[0]).read_text()]:
        sys.exit("allan_src.py is not the lenient source that probe_worker.py runs")
    kit.init(args.tokenizer)
    tokenizer = kit._G["tok"]

    cases, rows = build_cases(args.pool_step, args.fuzz)
    lines, kinds, dropped = [], {}, 0
    for number, case in enumerate(cases):
        row = rows.get(case.get("task_id"), case)
        text, kind, how, seconds = v1_verdict(tokenizer, case, row.get("gold", row.get("answer")), row["problem"])
        timed_out = how.endswith("TimeoutException")
        if case["group"].startswith("fuzz:") and (seconds > 1.0 or timed_out):
            dropped += 1  # a random case near the time limit could flip on a slower box
            continue
        if seconds > 1.0 and not timed_out:
            sys.exit(f"case {number} took {seconds:.1f} s with no timeout; it is too close to the limit to record: {case!r}")
        kinds[kind] = kinds.get(kind, 0) + 1
        lines.append({"id": len(lines), **case, "output": text, "kind": kind, "how": how})
    meta = {
        "what": "verdicts of the v1 math grader (miner/kit/grade.py::grade wrapped like probe_worker.grade_job)",
        "v1_commit": subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", *SOURCES], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "sources_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "tokenizer_json_sha256": sha256(Path(args.tokenizer) / "tokenizer.json"),
        "python": sys.version.split()[0],
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "cases": len(lines),
        "kinds": kinds,
        "random_cases_dropped_near_time_limit": dropped,
    }
    with open(args.out, "w", encoding="utf-8") as file:
        for line in [{"meta": meta}, *lines]:
            file.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
