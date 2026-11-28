"""Miner-facing Params model merge_after_confirm so live mining.json params reproduce the published params_hash."""

import dataclasses
import json
from pathlib import Path

import pytest

from sn120_v2.errors import ConfigError
from sn120_v2.params import Params

LIVE = Path(__file__).parent / "fixtures" / "live_params_92676f30.json"
HASH_FLAG_OFF = "92676f30aff26748285f49a4c37638938fb7ac2bc346217142c76d62c67b4c80"
HASH_FLAG_ON = "7838317829e36ebbf769b0b150fbbd45e18984da4d7fe9b97d7ebca1bf7ff6f7"


def _live() -> dict:
    return json.loads(LIVE.read_text())


def test_default_params_leave_the_flag_out_and_keep_their_hash():
    default = Params()
    assert default.merge_after_confirm is False
    assert "merge_after_confirm" not in default.to_dict()
    assert dataclasses.replace(default, merge_after_confirm=False).params_hash == default.params_hash


def test_live_shaped_params_reproduce_the_published_hashes():
    live = _live()
    assert "merge_after_confirm" not in live
    off = Params.from_public_dict(live)
    assert off.params_hash == HASH_FLAG_OFF
    on = Params.from_public_dict({**live, "merge_after_confirm": True})
    assert on.merge_after_confirm is True and on.to_dict()["merge_after_confirm"] is True
    assert on.params_hash == HASH_FLAG_ON
    assert Params.from_public_dict({**live, "merge_after_confirm": False}).params_hash == HASH_FLAG_OFF


def test_merge_after_confirm_validation():
    live = Params.from_public_dict(_live())
    with pytest.raises(ConfigError):
        dataclasses.replace(live, confirm_next_round=False, merge_after_confirm=True)
    with pytest.raises(ConfigError):
        dataclasses.replace(live, merge_after_confirm=True, fold_every_k=2)
    with pytest.raises(ConfigError):
        dataclasses.replace(live, merge_after_confirm=1)
