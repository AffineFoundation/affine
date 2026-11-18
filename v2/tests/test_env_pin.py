from sn120_v2.controller import check_env_pin
from sn120_v2.errors import ConfigError
from sn120_v2.params import Params
import pytest


def test_an_empty_env_pin_keeps_the_previous_behavior():
    params = Params()
    assert params.env_pin == ""
    check_env_pin(params)


def test_a_nonempty_env_pin_must_equal_the_local_pool_and_grader():
    pytest.importorskip("math_verify")
    from sn120_v2.envs.math_grader import env_pin

    actual = env_pin()
    check_env_pin(Params(env_pin=actual))
    with pytest.raises(ConfigError) as failure:
        check_env_pin(Params(env_pin="0" * 64))
    assert "0" * 64 in str(failure.value) and actual in str(failure.value)
