"""Unit tests for affine-math-grader (run: python -m pytest bench/tests -q)."""
import pytest
from affine_math_grader import grade, last_boxed, normalize, split_top

Y = [
    # numbers, fractions, latex variants
    ("\\frac{1}{2}", "0.5"), ("\\dfrac{3}{4}", "\\frac34"), ("1,000", "1000"), ("50\\%", "50"), ("\\sqrt{8}", "2\\sqrt{2}"),
    ("x = 3", "3"), ("3.", "3"), ("\\text{(B)}", "\\text{(B)}"),
    # sympy equivalence
    ("(x+1)^2", "x^2+2x+1"), ("\\frac{\\pi}{2}", "\\pi/2"), ("2^{10}", "1024"), ("\\sin^2 x + \\cos^2 x", "1"),
    # units
    ("12\\text{ cm}", "12"), ("45^\\circ", "45"), ("30 \\text{ degrees}", "30"), ("5\\text{ m}^2", "5"),
    # intervals / unions
    ("(-\\infty, 2) \\cup (3, \\infty)", "(-\\infty,2)\\cup(3,\\infty)"), ("[1, \\frac{5}{2}]", "[1,2.5]"), ("x \\in (0,1]", "(0,1]"),
    # tuples / sets
    ("(1, 2)", "(1,2)"), ("\\{3, 1, 2\\}", "\\{1,2,3\\}"), ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\pi/2)"),
]
N = [
    ("\\frac{1}{3}", "0.33"), ("(1, 2)", "(2,1)"), ("[0,1]", "(0,1]"), ("(-\\infty,2)", "(-\\infty,3)"), ("\\{1,2\\}", "\\{1,2,3\\}"),
    ("7", "8"), ("x^2", "x^3"),
]


@pytest.mark.parametrize("pred,gold", Y)
def test_equal(pred, gold):
    assert grade(pred, gold).score == 1.0, grade(pred, gold)


@pytest.mark.parametrize("pred,gold", N)
def test_not_equal(pred, gold):
    assert grade(pred, gold).score == 0.0, grade(pred, gold)


def test_numeric_tolerance():
    assert grade("3.1416", "3.14159", tolerance=1e-3).score == 1.0
    assert grade("3.2", "3.14159", tolerance=1e-3).score == 0.0


def test_multi_part_unordered_and_ordered():
    assert grade("2, 3", ["3", "2"], multiple=True).score == 1.0
    assert grade("2 \\text{ and } 3", ["2", "3"], multiple=True).score == 1.0
    assert grade("2, 3", ["2", "4"], multiple=True).score == 0.0
    assert grade("2", ["2", "3"], multiple=True).score == 0.0
    assert grade("3, 2", ["2", "3"], multiple=True, ordered=True).score == 0.0


def test_needs_judge_flag():
    g = grade("\\text{all odd primes}", "\\text{every prime greater than two}")
    assert g.score == 0.0 and g.needs_judge
    assert not grade("7", "8").needs_judge


def test_last_boxed_after_think():
    assert last_boxed("<think>\\boxed{1}</think> so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("<think> unfinished \\boxed{3}") is None
    assert last_boxed("no box") is None
    assert last_boxed("\\boxed{a{b}c}") == "a{b}c"


def test_helpers():
    assert split_top("(1,2), \\{3,4\\}, 5") == ["(1,2)", "\\{3,4\\}", "5"]
    assert normalize("\\$1,234.") == "1234" or normalize("\\$1,234.").endswith("1234")
    assert grade(None, "1").score == 0.0
