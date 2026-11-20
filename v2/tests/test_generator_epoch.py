"""Four-hour epoch boundaries and isolated Generator object names."""

from __future__ import annotations

import datetime as dt

import pytest

from sn120_v2.errors import ContractError
from sn120_v2.generator import constants, epoch

UTC = dt.timezone.utc
DEADLINE = dt.datetime(2026, 11, 20, tzinfo=UTC)


def test_epoch_id_is_deadline_scoring_start_not_opening():
    name = "20261120T0000Z"
    start, deadline = epoch.epoch_bounds(name)
    assert start == DEADLINE - dt.timedelta(hours=4)
    assert deadline == DEADLINE
    assert epoch.epoch_id(start) == name
    assert epoch.epoch_id(start + dt.timedelta(hours=2)) == name
    assert epoch.epoch_id(deadline - dt.timedelta(microseconds=1)) == name
    assert epoch.epoch_id(deadline) == "20261120T0400Z"
    assert epoch.epoch_id(deadline + dt.timedelta(microseconds=1)) == "20261120T0400Z"
    assert epoch.epoch_start(name) == start
    assert epoch.epoch_deadline(name) == deadline
    assert (deadline - start).total_seconds() == 4 * 3600


def test_epoch_midnight_day_month_and_leap_year_boundaries():
    cases = [
        (dt.datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC), "20270101T0000Z"),
        (dt.datetime(2027, 1, 1, 0, 0, tzinfo=UTC), "20270101T0400Z"),
        (dt.datetime(2028, 2, 28, 20, 0, tzinfo=UTC), "20280229T0000Z"),
        (dt.datetime(2028, 2, 29, 20, 0, tzinfo=UTC), "20280301T0000Z"),
        (dt.datetime(2026, 11, 19, 23, 59, tzinfo=UTC), "20261120T0000Z"),
    ]
    for stamp, expected in cases:
        assert epoch.epoch_id(stamp) == expected


def test_unix_timestamps_and_timezone_normalization():
    expected = "20261120T0000Z"
    local = DEADLINE.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    assert epoch.epoch_id(local) == "20261120T0400Z"
    assert epoch.epoch_id(DEADLINE.timestamp()) == "20261120T0400Z"
    assert epoch.epoch_id(int(DEADLINE.timestamp()) - 1) == expected
    assert epoch.epoch_id(epoch.epoch_start(expected).timestamp()) == expected
    with pytest.raises(ValueError, match="timezone"):
        epoch.epoch_id(dt.datetime(2026, 11, 19, 20))


def test_epoch_contains_is_half_open():
    name = "20261120T0000Z"
    start, deadline = epoch.epoch_bounds(name)
    assert epoch.epoch_contains(name, start)
    assert epoch.epoch_contains(name, deadline - dt.timedelta(microseconds=1))
    assert not epoch.epoch_contains(name, deadline)
    assert not epoch.epoch_contains(name, start - dt.timedelta(microseconds=1))
    assert not epoch.epoch_contains(name, DEADLINE.timestamp())


@pytest.mark.parametrize(
    "name",
    [
        "",
        "20261120T0000",
        "20261120T0000z",
        "2026-11-20T00:00Z",
        "20261120T000000Z",
        "20261120T0100Z",
        "20261120T0001Z",
        "20260230T0000Z",
        "19700101T0000Z",
        "20261120T0000Z ",
    ],
)
def test_epoch_strings_are_canonical_real_and_aligned(name):
    with pytest.raises(ValueError):
        epoch.epoch_id(name)


@pytest.mark.parametrize("value", [True, None, [], "20261120T0000Zjunk"])
def test_epoch_input_types_fail(value):
    with pytest.raises((TypeError, ValueError)):
        epoch.epoch_id(value)


def test_layout_is_complete_and_no_submission_can_be_public():
    names = epoch.layout()
    assert names["pool_pointer"] == "public/generator/pool"
    assert names["payout"] == "public/generator/payout.json"
    assert names["state"] == "state.json"
    assert epoch.link_object("hotkey.test") == "public/generator/links/hotkey.test.json"
    assert epoch.submission_object("20261120T0000Z", "hotkey") == "private/generator/sub/20261120T0000Z/hotkey.json"
    prefix = "public/generator/epochs/20261120T0000Z"
    assert epoch.epoch_manifest("20261120T0000Z") == f"{prefix}/manifest.json"
    assert epoch.epoch_results("20261120T0000Z") == f"{prefix}/results.json"
    assert epoch.accepted_tasks("20261120T0000Z") == f"{prefix}/accepted.jsonl"
    assert epoch.epoch_rollout_prefix("20261120T0000Z") == f"{prefix}/rollouts"
    assert epoch.rollout_object("20261120T0000Z", "sample.jsonl.gz") == f"{prefix}/rollouts/sample.jsonl.gz"


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".", "..", "white space", "a\\b", None])
def test_path_components_cannot_escape(bad):
    with pytest.raises(ContractError):
        epoch.link_object(bad)
    with pytest.raises(ContractError):
        epoch.submission_object("20261120T0000Z", bad)


def test_rollout_suffix_is_data_only():
    with pytest.raises(ContractError):
        epoch.rollout_object("20261120T0000Z", "code.py")


def test_public_identity_and_mechanism_are_fixed():
    assert constants.PUBLIC_NAME == "Generator"
    assert constants.PUBLIC_URL == "https://affine.io/generator"
    assert constants.SCHEMA_VERSION == "affine-generator-v1"
    assert constants.NETUID == 120 and constants.MECH_ID == 1
    assert constants.MAX_SUBMISSION_BYTES == 16 * 1024
