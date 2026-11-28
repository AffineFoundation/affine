"""The small, leak-free head publication between journal commit and full close."""

import json

import pytest

from sn120_v2 import layout
from sn120_v2.publisher import StorePublisher, check_pair
from sn120_v2.controller import HeadPublish
from sn120_v2.headchain import ChainWriter
from sn120_v2.sim import SimWorld, sim_params


@pytest.fixture
def world(tmp_path):
    return SimWorld(tmp_path, sim_params())


def register(world, quality):
    """Allocate an unused simulated registration without naming any real-world identifier."""
    return world.register(max(world.chain, default=0) + 1, quality)


def test_apply_and_fold_publish_only_the_head_before_the_full_contract(world):
    publisher = StorePublisher(world.store, "https://files.example")
    world.controller.publisher = publisher
    world.controller._publish(world.clock())  # the previously published contract has no open round
    for quality in (0.3,):
        register(world, quality)
    assert world.round().outcome == "applied"
    previous = json.loads(world.store.get(layout.MINING_JSON))
    assert previous["round"] is None and previous["history"]
    world.controller.open_round(world.clock(), "previous-open")
    for ticket in world.controller.tickets():
        world.controller.submit_result(world.runner.run(ticket), world.clock())
    world.register(max(world.chain) + 1, 0.0)  # a waiting seat in the previously published contract
    # Start from the last closed contract, then use the real journal apply/fold path below.
    world.controller.state.round = None
    world.controller._publish(world.clock())
    previous = json.loads(world.store.get(layout.MINING_JSON))
    assert previous["round"] is None and previous["history"]
    published = []
    real = publisher.publish_head_now

    def capture(manifest, chain, now):
        real(manifest, chain, now)
        published.append(json.loads(world.store.get(layout.HEAD_NOW)))

    publisher.publish_head_now = capture
    world.controller._publish = lambda now: None  # isolate the fast path from open and close
    for quality in (0.3, -0.2):
        register(world, quality)
    assert world.round().outcome == "applied"
    assert len(published) == 2

    applied, folded = published
    manifest = world.controller.state.manifest
    assert applied["schema"] == "sn120-head-now-v1"
    assert set(applied) == {"schema", "round_id", "version_id", "parent_version_id", "applied_at", "folded", "head"}
    assert applied["head"] == world.controller.state.manifests[applied["version_id"]].to_dict()
    assert applied["folded"] is False and folded["folded"] is True
    assert folded["head"] == manifest.to_dict()
    assert folded["version_id"] == manifest.head.version_id
    assert folded["parent_version_id"] == applied["version_id"]
    assert not {"queue", "history", "entries", "decision", "judge", "waiting", "batches", "rollouts"} & set(applied)
    assert world.store.get(layout.CURRENT) == manifest.to_json()

    mining = json.loads(world.store.get(layout.MINING_JSON))
    assert mining["head"] == folded["head"]
    assert mining["heads"][-1] == {key: folded[key] for key in ("version_id", "round_id", "applied_at", "folded")}
    assert mining["round"] == previous["round"]
    assert mining["history"] == previous["history"]
    assert mining["queue"] == previous["queue"]
    assert previous["round"] is None and previous["history"] and previous["queue"]["length"] > 0
    assert mining["history"] == previous["history"]
    assert mining["queue"] == previous["queue"]
    assert "lost" not in json.dumps(applied)
    assert json.dumps(world.controller.state.rounds[-1].decision.to_dict()) not in json.dumps(mining)
    assert check_pair(world.store.get(layout.MINING_JSON), world.store.get(layout.LLMS_TXT), world.params) == []


def test_head_chain_is_in_the_apply_and_fold_documents(world):
    world.controller.head = HeadPublish(chain=ChainWriter(world.store))
    publisher = StorePublisher(world.store)
    world.controller.publisher = publisher
    world.controller._publish(world.clock())
    world.controller._publish = lambda now: None
    register(world, 0.3)
    assert world.round().outcome == "applied"
    doc = json.loads(world.store.get(layout.HEAD_NOW))
    assert doc["head_chain"] == json.loads(world.store.get(layout.MINING_JSON))["head_chain"]
    assert doc["head_chain"]["head_version_id"] == doc["version_id"]
    assert check_pair(world.store.get(layout.MINING_JSON), world.store.get(layout.LLMS_TXT)) == []


def test_bootstrap_head_publication_does_not_invent_a_contract(world):
    publisher = StorePublisher(world.store)
    world.controller.publisher = publisher
    world.controller._publish = lambda now: None
    register(world, 0.3)
    assert world.round().outcome == "applied"
    assert world.store.exists(layout.HEAD_NOW)
    assert not world.store.exists(layout.MINING_JSON)
    assert not world.store.exists(layout.LLMS_TXT)


def test_fast_publication_failure_does_not_stop_apply_or_fold(world, capsys):
    class Down:
        def publish_head_now(self, manifest, chain, now):
            raise RuntimeError("the store is down")

    world.controller.publisher = Down()
    register(world, 0.3)
    assert world.round().outcome == "applied"
    assert world.controller.state.head.folded
    assert capsys.readouterr().err.count("publish failed: the store is down") == 2
