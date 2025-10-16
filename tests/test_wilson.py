from affine.core import wilson


def test_wilson_interval_symmetry() -> None:
    lower, upper = wilson.wilson_interval(5, 10, 0.95)
    assert 0 <= lower < 0.5 < upper <= 1


def test_beats_and_loses() -> None:
    assert wilson.beats_target(18, 20, 0.5, 0.95)
    assert wilson.loses_to_target(2, 20, 0.5, 0.95)
