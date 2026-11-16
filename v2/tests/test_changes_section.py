"""Section 12 of llms.txt and ``changes`` of mining.json: the history of the parameters, from the journal."""

from __future__ import annotations

import dataclasses
import functools
import json
import re
from dataclasses import replace

import pytest

from sn120_v2 import layout, public, publisher, sim
from sn120_v2 import state as state_module
from sn120_v2.errors import ConfigError
from sn120_v2.gpu import plan
from sn120_v2.journal import JOURNAL_FILE, JournalRecord, read_journal
from sn120_v2.params import MinerBound, Params, Recipe
from sn120_v2.state import CHANGES_KEPT, PARAMS_SET, ROUND_OPENED, JournalCorrupt, replay
from sn120_v2.store import put_record

SITE, LINKS = "https://example.org", "https://example.org/links"
DAY = re.compile(r"\d{4}-\d\d-\d\d")
TIME_OR_DAY = re.compile(r"\d{4}-\d\d-\d\d(?: \d\d:\d\d:\d\d)? UTC")
CLOCK = re.compile(r"\d\d:\d\d")
OTHER_ZONE = re.compile(r"\b(?:GMT|Z|[A-Z]{2,4}T|local)\b|[+-]\d\d:?\d\d\b")
WIDTH = 112


def with_bounds(params: Params, **changed: MinerBound | None) -> Params:
    """The params with these bounds of a miner's fields replaced; None takes the bound away."""
    kept = tuple(changed.get(bound.key, bound) for bound in params.recipe.miner_bounds if changed.get(bound.key, bound) is not None)
    new = tuple(bound for key, bound in changed.items() if bound is not None and key not in {one.key for one in params.recipe.miner_bounds})
    return replace(params, recipe=replace(params.recipe, miner_bounds=kept + new))


def a_world(tmp_path) -> sim.SimWorld:
    return sim.SimWorld(tmp_path, sim.sim_params())


def rounds(world: sim.SimWorld, count: int = 1) -> str:
    """Run this many rounds, each with one new seat. The id of the last one."""
    for _ in range(count):
        world.register(len(world.chain) + 1, 0.3)
        world.round()
    return world.controller.state.rounds[-1].round_id


def pin(world: sim.SimWorld, params: Params) -> None:
    world.controller.set_params(params, world.clock())
    world.params = params


def changes(world: sim.SimWorld) -> list[dict]:
    return public.changes_view(world.controller.state.params_changes)


def pair(world: sim.SimWorld) -> tuple[bytes, bytes]:
    state = world.controller.state
    return publisher.render(public.snapshot(state, state.params, world.clock()), SITE, LINKS)


def section(text: str) -> str:
    return text.split("## 12. Changes\n")[1]


def lines(world: sim.SimWorld) -> list[str]:
    """The lines of the parameters in section 12, each on one line."""
    part = section(pair(world)[1].decode()).split("### This text")[0]
    return [line for line in " ".join(part.split()).split(" - ")[1:]]


def opened(world: sim.SimWorld, round_id: str) -> state_module.RoundRecord:
    return next(record for record in world.controller.state.rounds if record.round_id == round_id)


def step_1(params: Params) -> Params:
    return with_bounds(replace(params, max_prompts=params.max_prompts * 4), steps=MinerBound("steps", 1, 9, integer=True, decided=True))


def moved_twice(tmp_path) -> tuple[sim.SimWorld, Params, Params]:
    """A round, then two changes with a round after each."""
    world = a_world(tmp_path)
    rounds(world)
    first = replace(world.params, max_prompts=world.params.max_prompts + 1)
    second = replace(first, z_bar=first.z_bar + 0.5, recipe=replace(first.recipe, global_norm_cap=2.5))
    pin(world, first)
    rounds(world, 2)
    pin(world, second)
    rounds(world, 2)
    return world, first, second


def test_the_first_parameters_of_a_journal_are_no_change(tmp_path):
    world = a_world(tmp_path)
    assert [record.kind for record in read_journal(tmp_path / JOURNAL_FILE)].count(PARAMS_SET) == 1, "bootstrap writes the first record"
    rounds(world, 2)
    mining, llms = pair(world)
    assert json.loads(mining)["changes"] == []
    assert "No change of the parameters is recorded." in section(llms.decode())
    assert publisher.check_pair(mining, llms, world.params) == []


def test_new_parameters_before_the_first_round_are_no_change(tmp_path):
    world = a_world(tmp_path)
    pin(world, step_1(world.params))
    rounds(world, 2)
    assert changes(world) == [], "no round ran with the parameters of the bootstrap"


def test_a_line_is_a_round_that_opened_with_other_values_and_it_has_the_time_and_the_hash_of_that_round(tmp_path):
    world, first, second = moved_twice(tmp_path)
    base = sim.sim_params()
    newest, older = changes(world)
    for row, round_id, params in ((older, "r000002", first), (newest, "r000004", second)):
        record = opened(world, round_id)
        assert (row["round_id"], row["opened_at"], row["params_hash"]) == (round_id, record.opened_at, record.params_hash)
        assert record.params_hash == params.params_hash
    assert older["values"] == [{"name": "max_prompts", "before": base.max_prompts, "after": first.max_prompts}]
    assert newest["values"] == [
        {"name": "z_bar", "before": first.z_bar, "after": second.z_bar},
        {"name": "recipe.global_norm_cap", "before": first.recipe.global_norm_cap, "after": 2.5},
    ]

    mining, llms = pair(world)
    doc = json.loads(mining)
    assert doc["changes"] == [newest, older]
    shown = {view["round_id"]: (view["opened_at"], view["params_hash"]) for view in doc["history"]}
    assert all(shown[row["round_id"]] == (row["opened_at"], row["params_hash"]) for row in doc["changes"]), "the same round in `history`"
    first_line, second_line = lines(world)
    assert first_line == (
        f"Round `r000004`, opened {publisher.utc(newest['opened_at'])}: `z_bar` {first.z_bar:g} to {second.z_bar:g}; "
        f"`recipe.global_norm_cap` {publisher.show(first.recipe.global_norm_cap)} to 2.5. `params_hash` of the round: `{second.params_hash}`."
    )
    assert second_line.startswith(f"Round `r000002`, opened {publisher.utc(older['opened_at'])}: `max_prompts` {base.max_prompts} to {first.max_prompts}.")
    assert publisher.check_pair(mining, llms, world.controller.state.params) == []


def test_the_history_comes_back_from_the_journal(tmp_path):
    world, *_ = moved_twice(tmp_path)
    again = replay(read_journal(tmp_path / JOURNAL_FILE))
    assert again.params_changes == world.controller.state.params_changes and len(again.params_changes) == 2
    assert again.opened_with == again.params
    assert public.changes_view(world.reopen().state.params_changes) == public.changes_view(again.params_changes)


def test_the_way_back_and_the_same_step_again_each_name_their_own_round(tmp_path):
    world = a_world(tmp_path)
    before = world.params
    rounds(world, 2)
    for params in (step_1(before), before, step_1(before)):
        pin(world, params)
        rounds(world, 2)
    rows = changes(world)
    assert [row["round_id"] for row in rows] == ["r000007", "r000005", "r000003"]
    assert rows[0]["params_hash"] == rows[2]["params_hash"] == step_1(before).params_hash and rows[1]["params_hash"] == before.params_hash
    assert opened(world, "r000001").params_hash == rows[1]["params_hash"], "the hash of the way back is the hash of the first round, and the line is round 5"
    assert rows[0]["values"] == rows[2]["values"] and [value["name"] for value in rows[1]["values"]] == ["max_prompts", "train.steps"]
    assert (rows[1]["values"][0]["before"], rows[1]["values"][0]["after"]) == (before.max_prompts * 4, before.max_prompts)


def test_a_line_stays_when_the_state_no_longer_keeps_its_round(tmp_path, monkeypatch):
    monkeypatch.setattr(state_module, "HISTORY", 3)
    world = a_world(tmp_path)
    rounds(world)
    pin(world, step_1(world.params))
    rounds(world, 6)
    state = world.controller.state
    assert [record.round_id for record in state.rounds] == ["r000005", "r000006", "r000007"]
    assert [row["round_id"] for row in changes(world)] == ["r000002"]
    assert public.changes_view(replay(read_journal(tmp_path / JOURNAL_FILE)).params_changes) == changes(world)


def test_a_record_that_changes_nothing_and_values_that_no_round_ran_with_have_no_line(tmp_path):
    world = a_world(tmp_path)
    before = world.params
    rounds(world)
    pin(world, before)  # the operator sets the pinned values again
    assert rounds(world) == "r000002" and changes(world) == []
    pin(world, step_1(before))
    assert changes(world) == [] and "No change of the parameters is recorded." in section(pair(world)[1].decode()), "no round has opened with them yet"
    assert json.loads(pair(world)[0])["params"] == step_1(before).to_dict(), "`params` shows them"
    pin(world, before)  # taken back before a round
    assert rounds(world) == "r000003" and changes(world) == [], "no round ran with other values"


def test_several_records_before_one_round_are_one_line(tmp_path):
    world = a_world(tmp_path)
    before = world.params
    rounds(world)
    last = replace(step_1(before), z_bar=before.z_bar + 0.5)
    for params in (replace(before, z_bar=before.z_bar + 0.5), replace(before, z_bar=before.z_bar + 0.5, max_prompts=7), last):
        pin(world, params)
    rounds(world, 2)
    (row,) = changes(world)
    assert row["round_id"] == "r000002" and row["params_hash"] == last.params_hash
    assert [value["name"] for value in row["values"]] == ["z_bar", "max_prompts", "train.steps"]
    assert row["values"][1] == {"name": "max_prompts", "before": before.max_prompts, "after": last.max_prompts}, "the value 7 was pinned and no round ran with it"


def test_new_digests_for_the_fold_of_a_decided_round_are_a_line_of_the_next_round(tmp_path, monkeypatch):
    old_code, new_code = "code-before-the-fix", "code-after-the-fix"
    world = sim.SimWorld(tmp_path, sim.sim_params(recipe=Recipe(image_digests=(sim.SIM_IMAGE,), code_digest=old_code)))
    monkeypatch.setattr(plan, "code_digest", lambda: new_code)
    world.controller.pins = functools.partial(plan.check_pins, sim.SIM_IMAGE)
    world.register(1, 0.3)
    (ticket,) = world.controller.open_round(world.clock(), "entropy", clock=world.clock)
    put_record(world.store, layout.sealed_result(ticket.round_id, ticket.lease_id), world.runner.run(ticket))
    with pytest.raises(ConfigError, match="so the fold was not started"):
        world.round()
    params = world.controller.params
    pin(world, replace(params, recipe=replace(params.recipe, code_digest=new_code)))
    assert world.controller.state.round.decision is not None and changes(world) == []
    world.round()
    assert world.controller.state.rounds[-1].round_id == "r000001" and changes(world) == [], "round 1 pinned the old digest"
    assert rounds(world) == "r000002"
    (row,) = changes(world)
    assert row["round_id"] == "r000002" and row["values"] == [{"name": "recipe.code_digest", "before": old_code, "after": new_code}]
    assert lines(world)[0].startswith(f"Round `r000002`, opened {publisher.utc(row['opened_at'])}: `recipe.code_digest` `{old_code}` to `{new_code}`. ")


def test_the_hash_of_a_line_is_the_one_that_the_round_record_holds(tmp_path):
    world = a_world(tmp_path)
    rounds(world)
    pin(world, step_1(world.params))
    rounds(world)
    older_form = "0" * 64  # as a record of older code: not the hash that this code computes for the values

    def with_hash(record: JournalRecord) -> JournalRecord:
        if record.kind != ROUND_OPENED or record.data["round"]["round_id"] != "r000002":
            return record
        return dataclasses.replace(record, data={"round": record.data["round"] | {"params_hash": older_form}})

    state = replay([with_hash(record) for record in read_journal(tmp_path / JOURNAL_FILE)])
    (row,) = public.changes_view(state.params_changes)
    assert row["params_hash"] == older_form == state.rounds[-1].params_hash != state.params.params_hash
    assert [value["name"] for value in row["values"]] == ["max_prompts", "train.steps"]


def test_two_hashes_for_the_same_values_are_a_line_with_no_value_and_the_same_hash_is_no_line(tmp_path):
    world = a_world(tmp_path)
    rounds(world, 2)
    older_form = "0" * 64

    def first_round_of_older_code(record: JournalRecord) -> JournalRecord:
        if record.kind != ROUND_OPENED or record.data["round"]["round_id"] != "r000001":
            return record
        return dataclasses.replace(record, data={"round": record.data["round"] | {"params_hash": older_form}})

    state = replay([first_round_of_older_code(record) for record in read_journal(tmp_path / JOURNAL_FILE)])
    snapshot = public.snapshot(state, state.params, world.clock())
    assert snapshot["changes"] == [{"round_id": "r000002", "opened_at": state.rounds[-1].opened_at, "params_hash": world.params.params_hash, "values": []}]
    text = " ".join(section(publisher.render(snapshot, SITE, LINKS)[1].decode()).split())
    assert f"Round `r000002`, opened {publisher.utc(state.rounds[-1].opened_at)}: its `params_hash` differs, and no value that this text can name. " in text


def test_a_journal_that_is_not_as_the_controller_writes_it_replays_as_before(tmp_path):
    world = a_world(tmp_path)
    rounds(world, 2)
    records = list(read_journal(tmp_path / JOURNAL_FILE))
    kept = [record for record in records if record.kind != PARAMS_SET]
    no_params = [dataclasses.replace(record, seq=records[0].seq + index) for index, record in enumerate(kept)]
    state = replay(no_params)
    assert state.params is None and state.opened_with is None and len(state.rounds) == 2 and state.params_changes == []
    for kind, data in ((PARAMS_SET, {}), (ROUND_OPENED, {})):
        with pytest.raises(JournalCorrupt):
            replay([dataclasses.replace(record, data=data) if record.kind == kind else record for record in records])


def test_a_bound_of_a_miners_field_is_a_line_of_its_own(tmp_path):
    world = a_world(tmp_path)
    before = with_bounds(world.params, global_norm_cap=None)
    pin(world, before)
    rounds(world)
    wider = MinerBound("steps", 1, 9, integer=True, decided=True)
    cap = MinerBound("global_norm_cap", 0.01, 6.0, decided=True)
    pin(world, with_bounds(before, steps=wider, rank=None, global_norm_cap=cap))
    rounds(world)
    (row,) = changes(world)
    old_steps = next(bound for bound in before.recipe.miner_bounds if bound.key == "steps")
    old_rank = next(bound for bound in before.recipe.miner_bounds if bound.key == "rank")
    assert row["values"] == [
        {"name": "train.steps", "bound": True, "before": old_steps.to_dict(), "after": wider.to_dict()},
        {"name": "train.rank", "bound": True, "before": old_rank.to_dict(), "after": None},
        {"name": "train.global_norm_cap", "bound": True, "before": None, "after": cap.to_dict()},
    ]
    (line,) = lines(world)
    assert f"`train.steps` was {publisher.bound_text(old_steps)}, and is now {publisher.bound_text(wider)}; " in line
    assert "`train.rank` is no longer a field that a miner may set; " in line
    assert f"`train.global_norm_cap` is new: {publisher.bound_text(cap)}. `params_hash` of the round" in line


def test_a_list_is_in_brackets_a_text_is_in_backticks_and_no_line_is_wider_than_the_text(tmp_path):
    world = a_world(tmp_path)
    before = replace(world.params, recipe=replace(world.params.recipe, code_digest="a" * 64))
    pin(world, before)
    rounds(world)
    after = replace(before, looks=(0.25, 0.5, 1.0), env_pin="math-2", recipe=replace(before.recipe, code_digest="b" * 64, image_digests=("c" * 64, "d" * 64)))
    pin(world, after)
    rounds(world)
    (line,) = lines(world)
    assert "`looks` [0.5, 1] to [0.25, 0.5, 1]; " in line and "`env_pin` none to `math-2`; " in line
    assert f"`recipe.code_digest` `{'a' * 64}` to `{'b' * 64}`" in line
    assert f"`recipe.image_digests` [`{sim.SIM_IMAGE}`] to [`{'c' * 64}`, `{'d' * 64}`]" in line
    text = section(pair(world)[1].decode())
    assert max(len(one) for one in text.splitlines()) <= WIDTH and "c" * 64 in text, "a digest stays whole on one line"


def test_the_state_keeps_only_the_changes_that_are_published_the_newest_first(tmp_path):
    world = a_world(tmp_path)
    base = world.params
    rounds(world)
    for step in range(1, CHANGES_KEPT + 4):
        pin(world, replace(base, max_prompts=base.max_prompts + step))
        rounds(world)
    state = world.controller.state
    assert len(state.params_changes) == CHANGES_KEPT
    rows = changes(world)
    assert [row["round_id"] for row in rows] == [f"r{number:06d}" for number in range(CHANGES_KEPT + 4, 4, -1)]
    assert [row["opened_at"] for row in rows] == sorted((row["opened_at"] for row in rows), reverse=True)
    assert replay(read_journal(tmp_path / JOURNAL_FILE)).params_changes == state.params_changes
    assert public.changes_view(state_module.State().params_changes) == []


def test_each_time_of_the_section_is_utc_and_no_other_zone_is_named(tmp_path):
    world, *_ = moved_twice(tmp_path)
    text = section(pair(world)[1].decode())
    days = [found.start() for found in DAY.finditer(text)]
    stamped = [found.span() for found in TIME_OR_DAY.finditer(text)]
    assert len(days) >= 2 + len(publisher.TEXT_CHANGES) and days == [start for start, _ in stamped], "each day is followed by UTC"
    assert all(any(low <= found.start() < high for low, high in stamped) for found in CLOCK.finditer(text)), "each time of day is in a UTC stamp"
    assert not OTHER_ZONE.findall(text), "UTC is the only zone word"
    assert "Each time is UTC." in text
    parameter_lines, text_lines = (part.splitlines() for part in text.split("### This text\n"))
    times = [found.group() for line in parameter_lines if line.startswith("- ") for found in TIME_OR_DAY.finditer(line)]
    assert times == sorted(times, reverse=True) and len(times) == 2, "the newest change of the parameters is the first line"
    days_of_text = [line[2:12] for line in text_lines if line.startswith("- ")]
    assert days_of_text == sorted(days_of_text, reverse=True) == [day for day, _, _ in publisher.TEXT_CHANGES]


def test_a_typed_line_names_no_field_of_train():
    # The lines are the same for each set of parameters, and a `train.` key is in the text only while a miner may set it.
    assert not [sentence for _, _, sentence in publisher.TEXT_CHANGES if "`train." in sentence]


def test_the_section_has_no_table_and_a_row_after_the_parameter_table_is_no_parameter(tmp_path):
    world, *_ = moved_twice(tmp_path)
    mining, llms = pair(world)
    text = llms.decode()
    assert not [line for line in section(text).splitlines() if line.startswith("|")], "a reader of the parameter table of older code reads every row after its heading"
    stated = publisher.stated_params(text)
    assert publisher.stated_params(text + "\n## 13. Later\n\n| `not_a_parameter` | 1 |\n") == stated and "not_a_parameter" not in stated
    assert publisher.check_pair(mining, llms) == [] and publisher.check_pair(mining, llms, whole_text=False) == []


def test_a_mining_json_of_older_code_renders_with_no_line(tmp_path):
    world, *_ = moved_twice(tmp_path)
    doc = json.loads(pair(world)[0])
    del doc["changes"]
    assert "No change of the parameters is recorded." in section(publisher.llms_text(doc))


def test_the_new_key_moves_no_schema_no_hash_of_the_parameters_and_no_other_key(tmp_path):
    world, _, second = moved_twice(tmp_path)
    state = world.controller.state
    snapshot = public.snapshot(state, state.params, world.clock())
    assert snapshot["schema"] == public.SCHEMA == "sn120-v2-public-1"
    assert (snapshot["params_hash"], snapshot["recipe_id"], snapshot["params"]) == (second.params_hash, second.recipe_id, second.to_dict())
    doc = publisher.mining_document(snapshot, SITE, LINKS)
    without = publisher.mining_document({key: value for key, value in snapshot.items() if key != "changes"}, SITE, LINKS)
    assert {key for key in doc if doc[key] != without.get(key)} == {"changes", "snapshot_digest"}
