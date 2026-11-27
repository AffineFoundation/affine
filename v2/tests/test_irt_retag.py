"""IRT retagging after an operator head revert; all fixtures are synthetic."""
from __future__ import annotations

import json

import pytest

from sn120_v2 import layout
from sn120_v2.cli import main as cli_main
from sn120_v2.controller import Controller
from sn120_v2.errors import V2Error
from sn120_v2.irt import IRTSampler
from sn120_v2.journal import JOURNAL_FILE, Journal
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import load


def world_with_irt(root, rounds=2):
    world = SimWorld(root, sim_params())
    for uid in range(1, rounds + 1):
        world.register(uid, 0.3 + uid)
        assert world.round().outcome == "applied"
    target, stale_head = world.controller.state.heads[0], world.controller.state.head
    sampler = IRTSampler(world.pool, world.params)
    draw = sampler.draw(target, "target", 20)
    sampler.update(target, draw, {"null": ("F" * 20,)})
    draw = sampler.draw(stale_head, "stale", 20)
    sampler.update(stale_head, draw, {"null": ("P" * 20,), "winner": ("P" * 20,)},
                   new_head_version_id=stale_head.version_id, winner_lora_hash="winner")
    stale = sampler.public_snapshot()
    world.controller.sampler = sampler
    world.controller._store_snapshot(sampler.snapshot_id())
    return world, target, stale_head, stale


def test_revert_retags_masses_publishes_and_old_journal_replay_heals(tmp_path, monkeypatch):
    world, target, stale_head, stale = world_with_irt(tmp_path, rounds=3)
    stacks = []

    from sn120_v2 import cli as command
    from sn120_v2.sim import stack as sim_stack

    def irt_stack(context):
        stack = sim_stack(context)
        stack.sampler = IRTSampler(world.pool, context.params, snapshot=stale)
        stacks.append(stack)
        return stack

    monkeypatch.setattr(command, "load_stack", lambda spec, context: irt_stack(context))
    assert cli_main(["revert-head", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack",
                     "--to", target.version_id, "--reason", "operator restore"]) == 0
    healed = stacks[-1].sampler.public_snapshot()
    assert healed["head_version_id"] == target.version_id
    assert healed["head_ability"] == healed["heads"][target.version_id]["ability"]
    assert stale_head.version_id in healed["heads"]
    assert healed["tasks"] != stale["tasks"]
    assert world.store.exists(layout.irt_snapshot(healed["snapshot_id"]))
    published = stacks[-1].publisher.snapshots[-1]
    assert published["irt_snapshot_id"] == healed["snapshot_id"] and published["irt_head_ability"] == healed["head_ability"]

    # The old journal has HEAD_REVERTED, while its restored IRT object names the old head.
    replayed_sampler = IRTSampler(world.pool, world.params, snapshot=stale)
    controller = Controller(Journal(world.root / JOURNAL_FILE, "replay-irt", world.clock()), world.store,
                            world.admission, replayed_sampler, world.params)
    assert controller.state.reverts
    assert controller.sampler.public_snapshot()["snapshot_id"] == healed["snapshot_id"]
    assert controller.sampler.retag(target.version_id) == healed["snapshot_id"], "repeat is idempotent"


def test_retag_cli_dry_run_refusals_and_open_fail_safe(tmp_path, monkeypatch, capsys):
    world, target, stale_head, stale = world_with_irt(tmp_path)

    from sn120_v2 import cli as command
    from sn120_v2.sim import stack as sim_stack

    def irt_stack(context):
        stack = sim_stack(context)
        stack.sampler = IRTSampler(world.pool, context.params, snapshot=stale)
        return stack

    monkeypatch.setattr(command, "load_stack", lambda spec, context: irt_stack(context))
    argv = ["retag-irt", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack", "--to", target.version_id]
    before_journal = (world.root / JOURNAL_FILE).read_bytes()
    before_store = {key: world.store.get(key) for key in world.store.list("")}
    assert cli_main([*argv, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["current"]["head_version_id"] == stale_head.version_id
    assert preview["head_version_id"] == target.version_id and preview["head_ability"] != preview["current"]["head_ability"]
    assert preview["snapshot_id"] != stale["snapshot_id"]
    assert (world.root / JOURNAL_FILE).read_bytes() == before_journal
    assert {key: world.store.get(key) for key in world.store.list("")} == before_store
    unknown = ["retag-irt", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack", "--to", "0" * 64]
    assert cli_main(unknown) == 2
    assert (world.root / JOURNAL_FILE).read_bytes() == before_journal
    capsys.readouterr()

    assert cli_main(["revert-head", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack",
                     "--to", target.version_id, "--reason", "operator restore"]) == 0
    capsys.readouterr()
    healed = json.loads(world.store.get(layout.irt_snapshot(preview["snapshot_id"])))
    world.controller.sampler = IRTSampler(world.pool, world.params, snapshot=stale)
    world.controller.journal = Journal(world.root / JOURNAL_FILE, "open-retag", world.clock())
    world.controller.state = load(world.root / JOURNAL_FILE)
    world.register(7, 0.3)
    tickets = world.controller.open_round(world.clock(), "open-retag")
    assert tickets and tickets[0].draw.snapshot_id == healed["snapshot_id"]
    assert world.controller.state.round.snapshot_id == healed["snapshot_id"]
    assert world.controller.sampler.public_snapshot()["draw"]["for_head_version_id"] == target.version_id
    pinned = tickets[0].draw.snapshot_id
    world.controller.sampler = IRTSampler(world.pool, world.params, snapshot=stale)
    before_lock = (world.root / JOURNAL_FILE).read_bytes()
    with pytest.raises(V2Error, match="open"):
        world.controller.retag_irt(target.version_id, world.clock())
    assert world.controller.state.round.snapshot_id == pinned
    assert (world.root / JOURNAL_FILE).read_bytes() == before_lock
    assert cli_main(argv) == 2
    assert (world.root / JOURNAL_FILE).read_bytes() == before_lock
