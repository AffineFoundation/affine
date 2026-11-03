"""arxivmath grader tests (needs verifiers + math-verify, i.e. the node's venv-vf: python -m pytest bench/tests/test_arxivmath.py -q)."""
import pytest
from arxivmath.taskset import check, implicit_mult

R = lambda a: "<think>work</think>\nSo the answer is \\boxed{%s}." % a
Y = [
    ("\\frac{r(2d-r-1)}{2}", "\\frac{2dr-r^2-r}{2}"), ("\\frac{r(2d-r-1)}{2}", "\\frac{2 d r - r^2 - r}{2}"),
    ("\\min(\\frac{4a}{2+d}, 1) \\cdot d", "d\\min(1,\\frac{4a}{d+2})"), ("2^{-d}(d+1)^{-1}-4^{-d}", "\\frac{1}{2^d(d+1)}-\\frac{1}{4^d}"), ("\\frac{3}{7}", "3/7"), ("\\tfrac{1}{2}", "0.5"), ("(14,16,16)", "(14, 16, 16)"),
    ("\\frac{\\ln(2s-r-1)}{\\ln s}", "\\log_s(2s-r-1)"), ("2^{n-1}", "2^{n}/2"), ("\\frac{15}{11}n", "\\frac{15n}{11}"),
    ("m^{-1}", "\\frac{1}{m}"), ("\\frac{\\sqrt{5 \\sqrt{5} - 11}}{2}", "\\sqrt{\\frac{5\\sqrt5-11}{4}}"), ("5.5", "\\frac{11}{2}"),
    ("q^{n-2}d(2,q)", "d(2,q)q^{n-2}"), ("\\frac{1}{6}(q-5)", "\\frac{q-5}{6}"), ("2e^{-H^2}", "2\\exp(-H^2)"),
    ("\\binom{q}{2}\\left\\lfloor\\frac{n}{2}\\right\\rfloor", "\\frac{q(q-1)}{2}\\lfloor n/2\\rfloor"),
    ("\\frac{1}{\\sqrt{3}L}", "\\frac{\\sqrt3}{3L}"), ("14634", "14634"), ("\\frac{491411}{268435456}", "\\frac{491411}{268435456}"),
]
N = [("3", "4"), ("\\frac{3}{7}", "\\frac{3}{8}"), ("n", "n+1"), ("(4,5)", "(5,4)"), ("12\\pi", "37.699"),
     ("\\frac{r(2d-r-1)}{2}", "\\frac{r(2d-r+1)}{2}"), ("2n-3", "2n-1"), ("e^{-c^2}", "e^{-c}"), ("\\frac{1}{\\sqrt{3}L}", "\\frac{1}{\\sqrt{3}}L"), ("dx", "d")]


@pytest.mark.parametrize("gold,pred", Y)
def test_equivalent(gold, pred):
    assert check(R(pred), gold) == 1.0


@pytest.mark.parametrize("gold,pred", N)
def test_different(gold, pred):
    assert check(R(pred), gold) == 0.0


def test_unfinished_thinking_scores_zero():
    assert check("<think>... \\boxed{3}", "3") == 0.0


def test_rewrite_leaves_commands():
    assert implicit_mult("\\ln(dr) + \\min(d, 1)") == "\\ln(\\vartheta  r) + \\min(\\vartheta , 1)"
