"""Tiny CPU end-to-end of merge_after_confirm: SimWorld with the real store publisher and the public head chain.

Every round runs the whole controller path (open, train, score, judge, confirm, close, publish). The checks read the
public files a miner reads: head_now.json, mining.json, llms.txt, and the check_pair of the last two.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2 import layout
from sn120_v2.controller import HeadPublish
from sn120_v2.headchain import ChainWriter
from sn120_v2.payout import paid_set
from sn120_v2.publisher import StorePublisher, check_pair
from sn120_v2.sim import SimWorld, sim_params


def wire(world: SimWorld) -> list[dict]:
    """Real publisher plus head chain; returns the list of fast-path head publications (head_now.json, mining.json, llms.txt)."""
    publisher = StorePublisher(world.store, "https://files.example")
    world.controller.publisher = publisher
    world.controller.head = HeadPublish(chain=ChainWriter(world.store))
    fast: list[dict] = []
    real = publisher.publish_head_now

    def capture(manifest, chain, now):
        real(manifest, chain, now)
        fast.append({"head_now": json.loads(world.store.get(layout.HEAD_NOW)), "mining": json.loads(world.store.get(layout.MINING_JSON)),
                     "llms": world.store.get(layout.LLMS_TXT)})

    publisher.publish_head_now = capture
    world.controller._publish(world.clock())
    return fast


def assert_public_head(world: SimWorld, head_now: dict, mining: dict, llms: bytes) -> None:
    head = world.controller.state.head
    assert head_now["version_id"] == head.version_id and head_now["head"] == world.controller.state.manifest.to_dict()
    assert mining["head"]["head"]["version_id"] == head.version_id
    assert mining["head_chain"]["head_version_id"] == head.version_id and head_now["head_chain"] == mining["head_chain"]
    assert mining["heads"][-1]["version_id"] == head.version_id
    text = llms.decode()
    assert head.version_id in text and head.base_hash in text
    assert check_pair(json.dumps(mining, indent=1, sort_keys=True, ensure_ascii=False).encode() + b"\n", llms) == []


def published(world: SimWorld) -> tuple[dict, dict, bytes]:
    return (json.loads(world.store.get(layout.HEAD_NOW)), json.loads(world.store.get(layout.MINING_JSON)), world.store.get(layout.LLMS_TXT))


def merging(tmp_path) -> SimWorld:
    return SimWorld(tmp_path, sim_params(confirm_next_round=True, merge_after_confirm=True))


def test_e2e_crown_then_confirm_pass_advances_the_head_and_pays(tmp_path):
    world = merging(tmp_path)
    fast = wire(world)
    root = world.controller.state.head
    world.register(1, 0.3)
    first = world.round()
    pending = world.controller.state.pending_king
    assert first.outcome == "applied" and world.controller.state.head == root and fast == []
    head_now, mining, _ = published(world)
    assert head_now["version_id"] == root.version_id and mining["head_chain"]["head_version_id"] == root.version_id
    world.register(2, -1.0)
    second = world.round()
    state = world.controller.state
    assert second.confirmation.merged and state.head == pending.candidate.head
    assert [k.round_id for k in paid_set(state.kings, state.null_rounds, world.params)] == [first.round_id]
    assert len(fast) == 1  # the merge went public at once, before the close
    assert_public_head(world, fast[0]["head_now"], fast[0]["mining"], fast[0]["llms"])
    assert_public_head(world, *published(world))
    lora = pending.stacked.head.loras[0].lora_hash
    step = published(world)[1]["head_chain"]["steps"][-1]
    assert step["stacked_version_id"] == pending.new_version_id and lora in json.dumps(step["loras"])


def test_e2e_crown_then_confirm_fail_leaves_every_public_head_file_unchanged(tmp_path):
    world = merging(tmp_path)
    fast = wire(world)
    root = world.controller.state.head
    world.register(1, 0.3)
    world.round()
    pending = world.controller.state.pending_king
    before_now, before_mining, _ = published(world)
    world.evaluator.candidate_quality = -0.5
    world.register(2, -1.0)
    second = world.round()
    assert not second.confirmation.confirmed and world.controller.state.head == root and fast == []
    after_now, after_mining, llms = published(world)
    assert after_now == before_now
    for key in ("head", "head_chain", "heads"):
        assert after_mining[key] == before_mining[key]
    lora = pending.stacked.head.loras[0].lora_hash
    assert lora not in json.dumps(after_mining["head_chain"]) and lora not in json.dumps(after_now)
    assert pending.candidate.head.version_id not in json.dumps(after_mining["heads"])
    assert world.controller.state.kings == []
    assert_public_head(world, after_now, after_mining, llms)


def test_e2e_legacy_pending_revert_publishes_the_reverted_head(tmp_path):
    world = SimWorld(tmp_path, sim_params(confirm_next_round=True))
    wire(world)
    root = world.controller.state.head
    world.register(1, 0.3)
    world.round()
    tip = world.controller.state.head
    assert tip != root and published(world)[0]["version_id"] == tip.version_id
    world.controller.set_params(dataclasses.replace(world.params, merge_after_confirm=True), world.clock())
    world.reopen()
    fast = wire(world)
    world.evaluator.parent_quality = 0.5  # the legacy crown fails its confirmation
    world.register(2, -1.0)
    second = world.round()
    state = world.controller.state
    assert not second.confirmation.confirmed and state.head == root and state.reverts[-1]["round_id"] == second.round_id
    assert len(fast) == 1  # the revert went public through the fast head path
    for head_now, mining, llms in ((fast[0]["head_now"], fast[0]["mining"], fast[0]["llms"]), published(world)):
        assert_public_head(world, head_now, mining, llms)
        assert tip.version_id not in {row["version_id"] for row in mining["heads"]}
        assert mining["head_chain"]["head_version_id"] == root.version_id
