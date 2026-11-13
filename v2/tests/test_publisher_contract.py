"""The publisher: llms.txt and mining.json come from one snapshot, and only from public facts."""

from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2 import layout, publisher
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import ContractError, InfraError, VerifyError
from sn120_v2.params import MinerBound, Recipe
from sn120_v2.types import King, Verdict


class MemStore(dict):
    put, get, exists = dict.__setitem__, dict.__getitem__, dict.__contains__

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


def snapshot(build, params=None, **changes) -> dict:
    head = build.head()
    base = {
        "generated_at": 1763000000,
        "params": (params or build.params()).to_dict(),
        "head": head.to_dict(),
        "round": {"round_id": "r000007", "state": "TRAIN", "seats_filled": 1, "round_seed": "seed-7"},
        "queue": [{"position": 0, "uid": 11, "hotkey": "hk11", "file_valid": True, "squat_trimmed": False}],
        "kings": [King(5, "hk5", "r000003", head.version_id, 900, 1.7).to_dict()],
        "verdicts": {"hk11": Verdict("hk11", True, "", "affine-batch-v2", 3, 9, "ab" * 32, 1762999000).to_dict()},
        "irt": {"snapshot_id": "uniform-1", "method": "uniform placeholder"},
    }
    return base | changes


def published(build, snap=None, store=None) -> tuple[dict, str, MemStore]:
    snap = snap or snapshot(build)
    store = MemStore() if store is None else store
    store[layout.head_manifest(snap["head"]["version_id"])] = b"{}"
    publisher.StorePublisher(store, "https://files.example/").publish(snap)
    return json.loads(store[layout.MINING_JSON]), store[layout.LLMS_TXT].decode(), store


def test_both_files_come_from_one_snapshot_and_change_together(build):
    doc, text, store = published(build)
    assert doc["snapshot_digest"] in text and doc["round"]["state"] == "TRAIN" and "- state: TRAIN" in text
    assert doc["head"] == build.head().to_dict() and build.head().version_id in text
    assert {key: doc[key] for key in snapshot(build)} == snapshot(build)  # the website reads the snapshot's own keys
    moved = snapshot(build, round={"round_id": "r000007", "state": "EVAL"})
    doc2, text2, _ = published(build, moved, store)
    assert doc2["snapshot_digest"] != doc["snapshot_digest"] and doc2["snapshot_digest"] in text2
    assert "- state: EVAL" in text2 and doc["snapshot_digest"] not in text2


def test_every_miner_bound_is_published_from_the_records(build):
    bounds = tuple(
        MinerBound("steps", 2, 7, integer=True) if bound.key == "steps" else bound for bound in Recipe().miner_bounds
    )
    recipe = Recipe(image_digests=(build.image,), steps=3, miner_bounds=bounds)
    doc, text, _ = published(build, snapshot(build, build.params(recipe=recipe)))
    rows = {row["key"]: row for row in doc["contract"]["miner_settable"]}
    assert list(rows) == ["lora_lr", "steps", "group_size", "sample_temperature", "rank", "alpha", "target_modules"]
    for bound in bounds:
        row = rows[bound.key]
        assert (row["low"], row["high"], row["choices"], row["decided"]) == (bound.low, bound.high, list(bound.choices), bound.decided)
        assert row["default"] == json.loads(json.dumps(getattr(recipe, bound.key)))
        assert f"| `train.{bound.key}` | " in text and row["allowed"] in text
    assert "| `train.steps` | 3 | a whole number from 2 to 7, both ends included | placeholder: it can change |" in text
    assert "a non-empty subset of q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | decided" in text
    assert doc["contract"]["recipe_id"] == recipe.recipe_id and recipe.recipe_id in text


def test_every_number_follows_the_pinned_params(build):
    params = build.params(max_prompts=123, max_rollouts=4567, z_bar=1.25, kings_window=7, n_tasks=400, looks=(0.25, 1.0))
    doc, text, _ = published(build, snapshot(build, params))
    assert doc["contract"]["params_hash"] == params.params_hash
    assert "Caps: 123 prompts and 4567 rollouts" in text and "strictly above 1.25" in text and "last 7 distinct king" in text
    assert "after 100, 400 tasks" in text and "after 100 tasks." in text  # looks and the early-drop bar are computed
    for field in dataclasses.fields(params) + dataclasses.fields(params.recipe):  # the full table and the full recipe
        assert f"`{field.name}`" in text or field.name in ("recipe", "miner_bounds")
    for key in doc["contract"]["objects"].values():
        assert key.startswith("public/") and key in text


@pytest.mark.parametrize(
    "bad",
    [
        {"queue": [{"hotkey": "hk11", "groups": [{"prompt": "p"}]}]},
        {"round": {"batch": "private/rounds/r000007/batches/hk11.json"}},
        {"round": {"lora": "pending/rounds/r000007/l0/loras/000"}},
        {"queue": [{"hotkey": "hk11", "put_url": "https://signed"}]},
        {"verdicts": {"hk11": {"hotkey": "hk11", "valid": True, "train": {"rank": 64}}}},
        {"contract": {}},
    ],
)
def test_private_content_never_reaches_the_public_files(build, bad):
    store = MemStore()
    with pytest.raises(ContractError):
        published(build, snapshot(build, **bad), store)
    assert layout.MINING_JSON not in store and layout.LLMS_TXT not in store


def test_a_head_is_named_only_after_its_manifest_is_public(build):
    store = MemStore()
    with pytest.raises(VerifyError, match="not public yet"):
        publisher.StorePublisher(store).publish(snapshot(build))
    assert not store


def test_a_bad_write_is_an_error_not_a_silent_stale_contract(build):
    class Corrupting(MemStore):
        def sha256(self, key: str) -> str:
            return "0" * 64

    class Down(MemStore):
        def put(self, key: str, data: bytes) -> None:
            raise OSError("network")

    with pytest.raises(VerifyError, match="read back"):
        published(build, store=Corrupting())
    with pytest.raises(InfraError):
        published(build, store=Down())
    with pytest.raises(ContractError):
        published(build, snapshot(build) | {"params": {"netuid": 120, "no_such_param": 1}})
