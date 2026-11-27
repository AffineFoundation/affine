"""The guard's deterministic checks: what they reject on their own, and what they only flag."""
from __future__ import annotations

from sn120_v2.envs.math_pool import Task
from sn120_v2.guard.checks import (MAX_PROBLEM_CHARS, NearIndex, Universe, check_task, content_hash,
                                   key_components, normalise, soft_flags)

PROBLEM = "Let $X$ be a random variable. Compute its mean and variance."


def task(task_id: str = "gen:1", problem: str = PROBLEM, answer: str = "42") -> Task:
    return Task(task_id=task_id, slice="math", problem=problem, answer=answer)


def empty_universe() -> Universe:
    return Universe([])


def test_normalise_folds_latex_and_case_but_keeps_cjk():
    assert normalise(r"\frac{1}{2}  AB") == "1 2 ab"
    assert normalise("如图, AB = 3") == "如图 ab 3"
    # NFKC: full-width digits and letters fold onto their ascii forms.
    assert normalise("ＡＢ１") == normalise("AB1")


def test_normalise_ignores_whitespace_and_punctuation_differences():
    assert normalise("What is $1+1$?") == normalise("what   is  1 + 1 ???")


def test_key_components_splits_only_at_the_top_level():
    assert key_components(r"E=90, \operatorname{Var}=990") == ["E=90", "90", r"\operatorname{Var}=990", "990"]
    assert key_components("(1, 2)") == []          # a bracketed pair is one value
    assert key_components(r"\frac{1}{2}") == []    # a brace group is not a split point
    assert key_components("x=3") == ["3"]
    assert key_components("2 and 3") == ["2", "3"]


def test_partial_key_rejects_a_key_whose_component_the_grader_accepts():
    # Verified against the production grader: math-verify keeps only the last
    # relation, so a model that boxes just the variance passes this key.
    result = check_task(task(answer=r"E=90, \operatorname{Var}=990"), empty_universe())
    assert result.category == "partial_key"


def test_partial_key_also_catches_a_named_single_value_key():
    assert check_task(task(answer="x=3"), empty_universe()).category == "partial_key"


def test_a_plain_key_is_not_a_partial_key():
    for answer in ("42", r"\frac{3}{4}", "(1, 2)"):
        assert check_task(task(answer=answer), empty_universe()).category is None


def test_multiple_choice_letter_keys_are_rejected():
    for answer in ("C", "(B)", r"\textbf{D}"):
        assert check_task(task(answer=answer), empty_universe()).category == "mc_letter_key"


def test_empty_and_oversized_are_rejected_before_anything_else():
    assert check_task(task(problem="   "), empty_universe()).category == "empty"
    assert check_task(task(answer=""), empty_universe()).category == "empty"
    assert check_task(task(problem="x " * MAX_PROBLEM_CHARS), empty_universe()).category == "oversized"
    assert check_task(task(answer="1" * 300), empty_universe()).category == "oversized"


def test_an_unparseable_key_is_rejected():
    # math-verify's parser is lenient, so only a key it cannot read at all is
    # ``key_unparseable``; the rest fall to the self-grade, which is stricter.
    assert check_task(task(answer="}}}}}"), empty_universe()).category == "key_unparseable"
    assert check_task(task(answer=r"\frac{1"), empty_universe()).category == "self_grade_fail"


def test_a_duplicate_of_another_task_is_rejected_whatever_its_formatting():
    universe = Universe([("gen:7", "What is $1+1$?", "active")])
    assert check_task(task(problem="what is   1 + 1 ?"), universe).category == "duplicate"
    # The owner itself is not a duplicate of itself.
    assert check_task(task(task_id="gen:7", problem="What is $1+1$?"), universe).category is None


def test_the_duplicate_check_runs_before_the_partial_key_grade():
    universe = Universe([("gen:7", PROBLEM, "active")])
    assert check_task(task(answer="x=3"), universe).category == "duplicate"


def test_soft_flags_never_reject_and_name_what_the_judge_should_look_at():
    flags = soft_flags("As shown in the figure, (1) find $x$. (2) find $y$", r"x \text{ is the long answer} \pm 1", None)
    assert {"key_prose", "key_multivalue", "multipart", "figure_ref"} <= set(flags)
    assert check_task(task(problem="As shown in the figure, compute $x$.", answer="42"),
                      empty_universe()).category is None


def test_possible_truncation_and_unbalanced_math_are_flags():
    assert "possible_truncation" in soft_flags("The value of the sum is clearly", "42", None)
    assert "unbalanced_math" in soft_flags("Compute $x + 1.", "42", None)
    assert "possible_truncation" not in soft_flags("Compute $x+1$.", "42", None)


def test_near_duplicates_are_flagged_and_distant_problems_are_not():
    base = "Let n be a positive integer and compute the sum of the first n odd numbers for n equal to seven"
    near = base.replace("seven", "eight")
    far = "A train leaves the station at noon travelling east at sixty kilometres per hour towards the city"
    index = NearIndex({"gen:1": base, "gen:9": far})
    assert index.nearest("gen:2", near) == "gen:1"
    assert index.nearest("gen:2", far) == "gen:9"
    assert index.nearest("gen:1", base) is None  # its own entry never matches itself


def test_content_hash_matches_the_reserve_content_sha():
    from sn120_v2.canonical import canonical_json, sha256_hex
    assert content_hash(PROBLEM, "42") == sha256_hex(canonical_json({"problem": PROBLEM, "answer": "42"}))
    assert check_task(task(), empty_universe()).content_sha == content_hash(PROBLEM, "42")
