from __future__ import annotations

import dataclasses
import io
import json
import random
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from sn120_v2 import public, publisher
from sn120_v2.cli import summary
from sn120_v2.errors import ConfigError
from sn120_v2.params import Params
from sn120_v2.payout import paid_set, payees, pruned_by_null, weights
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import NULL, ROUND_CLOSED, RoundRecord, State, load
from sn120_v2.types import King

PROD_HASH = "ab4e228c961058a29ed062bc67f6da6881cf3c13396eb03730d54badb4f261c0"
GPU_DIGEST = "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1"
NEW_HASH = "1449210f89f30a65dc3f6e8d4ac956f5f2c1c8d37364a46b20e5cfb01c24eace"
ROOT = Path(__file__).parents[2]


def king(uid: int, round_id: str, hotkey: str | None = None) -> King:
    return King(uid, hotkey or f"hk{uid}", round_id, f"head-{round_id}-{uid}", 1000 + uid)


def crowns(n: int = 10, start: int = 0) -> list[King]:
    return [king(uid, f"r{start + uid:06d}") for uid in range(1, n + 1)]


def params(**changes) -> Params:
    return Params(null_prune_from_round="r000011", **changes)


def uids(kings: list[King]) -> list[int]:
    return [king.uid for king in kings]


def test_judged_nulls_prune_oldest_until_the_one_king_floor():
    kings = crowns()
    for removed in range(10):
        nulls = [f"r{n:06d}" for n in range(11, 11 + removed)]
        assert uids(paid_set(kings, nulls, params())) == list(range(10, removed, -1))
    nulls = [f"r{n:06d}" for n in range(11, 30)]
    assert uids(paid_set(kings, nulls, params())) == [10]
    assert pruned_by_null(kings, nulls, params()) == 9
    assert weights(kings, {uid: f"hk{uid}" for uid in range(1, 11)}, params(), nulls) == {10: 1.0}


def test_crowns_after_pruning_regrow_one_at_a_time_and_full_crowns_evict_oldest():
    initial = crowns()
    assert uids(paid_set([*initial, king(11, "r000013")], ["r000011", "r000012"], params())) == [11, 10, 9, 8, 7, 6, 5, 4, 3]
    assert uids(paid_set([*initial, king(11, "r000013"), king(12, "r000014")], ["r000011", "r000012"], params())) == [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    assert uids(paid_set([*initial, king(11, "r000011")], [], params())) == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]


def test_a_recrown_moves_the_newest_hotkey_to_the_front_without_regrowing():
    kings = crowns(3, 1) + [king(2, "r000011", "new-key")]
    paid = paid_set(kings, [], params())
    assert uids(paid) == [2, 3, 1] and paid[0].hotkey == "new-key"
    assert weights(kings, {2: "new-key", 3: "hk3", 1: "hk1"}, params()) == {2: 1 / 3, 3: 1 / 3, 1: 1 / 3}


def test_only_judged_null_rounds_prune_and_deregistration_renormalizes():
    kings = crowns(3, 1)
    events = ["r000011", "r000012"]  # no_decision and abandoned never enter state.null_rounds
    assert uids(paid_set(kings, events, params())) == [3]
    assert weights(kings, {1: "hk1", 2: "hk2", 3: "hk3"}, params(), events) == {3: 1.0}
    changed = {1: "hk1", 2: "hk2", 3: "another-hotkey"}
    assert weights(kings, changed, Params(), events[:1]) == {1: 0.5, 2: 0.5}, "feature off ignores the event list"
    assert weights(kings, changed, params(), events[:1]) == {2: 1.0}
    assert weights(kings, {0: "hk0"}, Params(), []) == {0: 1.0}, "the pure fallback remains; the writer still refuses it"


def test_events_before_the_cutoff_and_imported_v1_kings_are_not_pruned():
    kings = [king(99, "v1-e1"), *crowns()]
    nulls = [f"r{n:06d}" for n in range(2, 12)]
    paid = paid_set(kings, nulls, params())
    assert uids(paid) == [10, 9, 8, 7, 6, 5, 4, 3, 2]
    assert pruned_by_null(kings, nulls, params()) == 1


@pytest.mark.parametrize("value", ["r1", "r1234567", "R000098", "v1-e1", "r00a098", " r000098", None, 98])
def test_the_cutoff_must_be_empty_or_canonical(value):
    with pytest.raises(ConfigError):
        Params(null_prune_from_round=value)


def test_feature_off_is_byte_identical_on_randomized_histories(build):
    rng = random.Random(98120)
    for _ in range(250):
        kings = [king(rng.randrange(1, 30), f"r{i:06d}", f"key-{i}") for i in range(rng.randrange(60))]
        nulls = [f"r{i:06d}" for i in sorted(rng.sample(range(60), rng.randrange(20)))]
        registered = {uid: f"key-{uid}" for uid in range(1, 30) if rng.random() < 0.85}
        params = build.params(kings_window=rng.randrange(1, 15))
        assert paid_set(kings, nulls, params) == payees(kings, params.kings_window)
        assert pruned_by_null(kings, nulls, params) == 0
        assert weights(kings, registered, params, nulls) == weights(kings, registered, params)


def test_current_prod_hash_is_unchanged_at_the_default_and_ops_file_changes_only_the_cutoff():
    current = json.loads((ROOT / "v2/tools/dedup-deploy/new-params.json").read_text())
    selected = json.loads((ROOT / "v2/ops/null_prune_params.json").read_text())
    assert Params.from_dict(current).params_hash == PROD_HASH
    assert "null_prune_from_round" not in current
    assert "null_prune_from_round" not in Params.from_dict(current).to_dict()
    assert selected == current | {"null_prune_from_round": "r000098"}
    made = Params.from_dict(selected)
    assert made.params_hash == NEW_HASH and made.recipe.code_digest == GPU_DIGEST
    assert Params.from_dict(made.to_dict()).params_hash == NEW_HASH


def test_public_views_and_publisher_render_the_current_paid_set():
    params = sim_params(null_prune_from_round="r000011")
    state = State(params=params)
    state.kings = crowns()
    state.null_rounds = ["r000011", "r000012"]
    state.rounds = [dataclasses.replace(RoundRecord("r000011", "head", "params", "recipe", "seed", "snapshot", 1, 1, ()), outcome=NULL, closed_at=2)]
    from conftest import Build
    state.heads = [Build.head()]
    snap = public.snapshot(state, params, 2000)
    assert [row["uid"] for row in snap["kings"]] == list(range(10, 0, -1))
    expected = [{"uid": uid, "round_id": f"r{uid:06d}"} for uid in range(10, 2, -1)]
    assert snap["paid_kings"] == expected and snap["null_prune_from_round"] == "r000011"
    assert summary(state, params)["paid_kings"] == expected
    # The renderer takes the same paid set as mining.json, rather than recomputing the old window.
    doc = {"params": params.to_dict(), "kings": snap["kings"], "paid_kings": expected, "contract": {"public_url": ""}}
    text = " ".join(publisher._payout_lines(doc, params))
    rule = "each round with no new king (the king held) removes the oldest king from the paid set, down to one."
    assert rule in text and "Pruned kings return only by winning again." in text
    assert "Rounds without a decision (operator stop, failure) do not prune. Paid kings split the weight equally." in text
    assert "A round with no winner changes nothing in the payout" not in text
    table = publisher.rows_table(expected, ("uid", "round_id"))
    assert all(row in text for row in table) and "| 10 | r000010 |" in text and "| 2 | r000002 |" not in text


def test_synthetic_journal_replay_preserves_untrimmed_nulls_and_weights(tmp_path):
    params = sim_params(null_prune_from_round="r000001")
    world = SimWorld(tmp_path, params)
    with redirect_stdout(io.StringIO()):
        # Three crowns interleaved with judged nulls.
        for uid, quality in ((1, 0.4), (2, 0.4), (3, 0.4)):
            world.register(uid, quality)
            assert world.round().outcome == "applied"
            world.register(100 + uid, 0.0)
            assert world.round().outcome == "null"
        # More than HISTORY nulls: public round history trims, payout history does not.
        for uid in range(200, 402):
            world.register(uid, 0.0)
            assert world.round().outcome == "null"
    state = world.controller.state
    assert len(state.rounds) == 200
    assert len(state.null_rounds) == 205 and len(set(state.null_rounds)) == 205
    assert [king.uid for king in state.kings] == [1, 2, 3]
    replayed = load(world.root / "journal.jsonl")
    assert replayed.null_rounds == state.null_rounds
    chain = {uid: f"hk{uid}" for uid in range(500)}
    assert weights(replayed.kings, chain, replayed.params, replayed.null_rounds) == weights(state.kings, chain, params, state.null_rounds)
    assert weights(replayed.kings, chain, replayed.params, replayed.null_rounds) == {3: 1.0}


def test_no_decision_and_abandoned_journal_closes_do_not_prune(tmp_path):
    world = SimWorld(tmp_path, sim_params(null_prune_from_round="r000001"))
    world.register(1, 0.4)
    world.register(2, 0.4)
    assert world.round().outcome == "applied"
    # Corrupt the winner before its durable apply check: abandoned, never a judged null.
    world.register(3, 0.4)
    ticket, = world.controller.open_round(world.clock(), "entropy-abandoned")
    result = world.runner.run(ticket)
    winner = result.entries[0]
    world.store.put(f"{winner.lora.prefix}/adapter_model.bin", b"corrupted")
    world.controller.submit_result(result, world.clock())
    assert world.controller.state.rounds[-1].outcome == "abandoned"
    # No training finishes: the lease fails and the round closes without a decision.
    world.trainer.fail_hotkeys = {"hk1", "hk2", "hk3"}
    world.now += world.params.apply_retry_s
    assert world.round().outcome == "no_decision"
    assert world.controller.state.null_rounds == []
    replayed = load(world.root / "journal.jsonl")
    assert replayed.null_rounds == []
    assert paid_set(replayed.kings, replayed.null_rounds, replayed.params) == payees(replayed.kings, replayed.params.kings_window)
