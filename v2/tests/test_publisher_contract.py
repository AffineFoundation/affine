"""The publisher: llms.txt and mining.json come from one snapshot, and only from public facts."""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

import pytest

from sn120_v2 import layout, links, public, publisher, r2, sim
from sn120_v2.admission import RULES, TIME_LIMIT, TRANSIENT, check_batch
from sn120_v2.canonical import sha256_hex, to_jsonable
from sn120_v2.envs.math_pool import MANDATE, POOL_V2_0_SHA256
from sn120_v2.errors import ConfigError, ContractError, InfraError
from sn120_v2.params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, MinerBound, Recipe
from sn120_v2.types import HeadVersion


class MemStore(dict):
    put, get, exists = dict.__setitem__, dict.__getitem__, dict.__contains__

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


def world(tmp_path, play: bool = False, **changes) -> sim.SimWorld:
    """The real controller on CPU. Its recorded snapshots are what the publisher gets in production."""
    made = sim.SimWorld(tmp_path / "world", sim.sim_params(**changes))
    if play:
        made.register(11, 0.4)  # wins the round
        made.register(12, None)  # no file: the seat waits
        made.round()
    return made


def published(snap, store=None) -> tuple[dict, str, MemStore]:
    store = MemStore() if store is None else store
    if snap["head"]:
        store[layout.head_manifest(snap["head"]["head"]["version_id"])] = b"{}"
    publisher.StorePublisher(store, "https://files.example/").publish(snap)
    return json.loads(store[layout.MINING_JSON]), store[layout.LLMS_TXT].decode(), store


def test_the_controller_publishes_through_the_real_store(tmp_path):
    made = world(tmp_path)
    made.controller.publisher = publisher.StorePublisher(made.store, "https://files.example/")
    made.register(11, 0.4)
    made.register(12, None)
    made.round()
    assert made.controller.last_error == ""  # each head manifest was public before the contract named the head
    doc, text = json.loads(made.store.get(layout.MINING_JSON)), made.store.get(layout.LLMS_TXT).decode()
    controller = made.controller
    snap = public.snapshot(controller.state, made.params, doc["generated_at"], controller._verdicts, controller._verdicts_at,
                           stop_asked=controller._stop_asked, close_publish_pending=controller._close_publish_pending,
                           waiting_for_seats=controller._waiting_for_seats)  # the website reads the snapshot's own keys
    assert doc["snapshot_digest"] in text and doc["head"]["head"]["version_id"] in text and doc["irt_snapshot_id"] in text
    assert [king["uid"] for king in doc["kings"]] == [11] and "| 11 | hk11 |" in text
    assert doc["queue"]["length"] == 1 and "| 0 | 12 | hk12 | 1012 | no |" in text


def test_winner_stats_contain_only_applied_crowns_with_public_batch_facts(tmp_path):
    made = world(tmp_path)
    made.register(11, 0.4, train={"lora_lr": 0.0001, "steps": 2})
    first = made.round()
    made.register(12, 0.0)
    second = made.round()
    made.register(13, 0.4)
    third = made.round()
    assert [record.outcome for record in (first, second, third)] == ["applied", "null", "applied"]
    snap = made.publisher.snapshots[-1]
    doc, _, store = published(snap)
    stats = json.loads(store[layout.WINNER_STATS])
    assert "winner_stats" not in doc
    assert stats["schema"] == public.WINNER_STATS_SCHEMA and type(stats["generated_at"]) is int
    assert stats["generated_at"] == snap["generated_at"] and [row["uid"] for row in stats["winners"]] == [13, 11]
    for row in stats["winners"]:
        expected = next(king for king in made.controller.state.kings if king.round_id == row["round_id"])
        assert row["head"] == expected.head_version_id and row["applied_at"] == expected.applied_at
        assert row["batch_url"] == f"https://files.example/{layout.winner_batch(row['round_id'])}"
        assert row["batch_prompts"] == 1 and row["batch_rollouts"] == 1 and row["batch_format"] == "sim-batch"
        assert (row["confirmed"], row["paid"], row["payout_status"]) == (True, True, "paid")
        assert set(row) <= public.WINNER_FIELDS | {"batch_url"} | public.ENRICHMENT_FIELDS
    assert store[layout.WINNER_STATS_ALIAS] == store[layout.WINNER_STATS]
    assert stats["winners"][1]["train"] == {"lora_lr": 0.0001, "steps": 2}
    assert second.round_id not in json.dumps(stats) and "hk12" not in json.dumps(stats)
    assert all(word not in json.dumps(stats) for word in ("private/", "pending/", "put_url", '"groups"', '"rollouts": ['))
    external = json.loads(store[layout.MINING_JSON])
    assert "winner_stats" not in external
    public.validate_winner_stats(external, stats, "https://files.example")
    before = json.dumps(external, sort_keys=True)
    public.winner_stats_from_snapshot(external, "https://files.example")
    assert json.dumps(external, sort_keys=True) == before


def crowned_public_batch_fixture(tmp_path):
    """A real admitted batch, real IRT bytes, and the controller's crowned-only public transport."""
    from sn120_v2.envs.math_pool import MathPool, Task
    from sn120_v2.irt import IRTSampler
    from sn120_v2.params import Params

    made = world(tmp_path)
    made.register(11, 0.4)
    made.register(12, 0.0)  # loser: its private batch must never be read
    made.round()
    snapshot = json.loads(json.dumps(made.publisher.snapshots[-1]))
    tasks = (Task("gen:a", "affine_math", "public reserve A", "1"),
             Task("math:b", "affine_math", "public base B", "2"),
             Task("other:c", "affine_math", "public other C", "3"))
    pool = MathPool(tasks, "fixture-catalog")
    groups = [
        {"prompt": tasks[0].batch_prompt(), "rollouts": [{"output": "work \\boxed{1}", "reward": 1.0},
                                                          {"output": "no box", "reward": 0.0}]},
        {"prompt": tasks[1].problem, "rollouts": [{"output": "abcdef", "reward": 1.0}]},
        {"prompt": "custom unmapped prompt", "rollouts": [{"output": "abcdefghij", "reward": 0.0}]},
        {"prompt": tasks[2].batch_prompt(), "rollouts": [{"output": "\\boxed{3} trailing", "reward": 1.0}]},
    ]
    raw = json.dumps({"version": "affine-batch-v2", "hotkey": "hk11", "groups": groups}).encode()
    winner = snapshot["winner_stats"]["winners"][0]
    winner["batch_sha256"] = sha256_hex(raw)
    snapshot["history"][0]["entries"][0]["batch_sha256"] = winner["batch_sha256"]
    irt = IRTSampler([task.task_id for task in tasks], Params(irt_draw_rule="band")).public_snapshot()
    snapshot["history"][0]["irt_snapshot_id"] = irt["snapshot_id"]
    store = MemStore({layout.winner_batch(winner["round_id"]): raw,
                      layout.irt_snapshot(irt["snapshot_id"]): json.dumps(irt).encode()})
    return snapshot, winner, store, pool


def test_crowned_public_batch_enrichment_and_privacy(tmp_path, caplog):
    snapshot, winner, store, pool = crowned_public_batch_fixture(tmp_path)
    reads = []

    class PublicOnly(MemStore):
        def get(self, key):
            reads.append(key)
            assert not key.startswith(("private/", "pending/"))
            if "/batch.json" in key:
                assert key == layout.winner_batch(winner["round_id"])
            return super().get(key)

    store = PublicOnly(store)
    stats = {"schema": public.WINNER_STATS_SCHEMA, "generated_at": snapshot["generated_at"], "winners": [winner]}
    enriched = public.enrich_winner_stats(snapshot, stats, store, pool)["winners"][0]
    assert enriched["group_count"] == 4 and enriched["boxed_frac"] == 0.2 and enriched["output_chars_p50"] == 10
    assert enriched["prompt_sources"] == {"gen": 1, "dapo": 0, "math": 1, "nmv2": 0, "other": 1, "unmapped": 1}
    assert sum(enriched["prompt_band"].values()) == 3  # only mapped prompts, from the crown round's real IRT
    assert reads == [layout.winner_batch(winner["round_id"]), layout.irt_snapshot(snapshot["history"][0]["irt_snapshot_id"])]
    public.validate_winner_enrichment(snapshot, enriched, store, pool)
    forged = dict(enriched, group_count=999)
    with pytest.raises(ValueError, match="group_count"):
        public.validate_winner_enrichment(snapshot, forged, store, pool)
    irt_key = layout.irt_snapshot(snapshot["history"][0]["irt_snapshot_id"])
    irt = json.loads(store[irt_key])
    store[irt_key] = b"not an IRT snapshot"
    basic = public.enrich_winner_stats(snapshot, stats, store, pool)["winners"][0]
    assert "prompt_band" not in basic and basic["group_count"] == 4 and "prompt_band omitted" in caplog.text
    store[irt_key] = json.dumps(irt | {"snapshot_id": "forged"}).encode()
    basic = public.enrich_winner_stats(snapshot, stats, store, pool)["winners"][0]
    assert "prompt_band" not in basic and basic["group_count"] == 4  # VerifyError also preserves the basic fields


def test_v4_crowned_batch_enriches_grpo_groups_and_sft_without_admission_caps(tmp_path):
    snapshot, winner, store, pool = crowned_public_batch_fixture(tmp_path)
    tasks = [pool.task(task_id) for task_id in pool.task_ids]
    document = {
        "version": "affine-batch-v4", "hotkey": winner["hotkey"],
        "stages": [
            {"train": {"steps": 2}, "groups": [
                {"prompt": tasks[0].batch_prompt(), "rollouts": [{"output": "work \\boxed{1}", "reward": 1.0}]},
            ]},
            {"sft": [
                {"prompt": tasks[1].problem, "output": "abcdef"},
                {"prompt": "custom public prompt", "output": "no box"},
            ]},
        ],
    }
    raw = json.dumps(document).encode()
    winner["batch_sha256"] = sha256_hex(raw)
    store[layout.winner_batch(winner["round_id"])] = raw
    stats = {"schema": public.WINNER_STATS_SCHEMA, "generated_at": snapshot["generated_at"], "winners": [winner]}
    enriched = public.enrich_winner_stats(snapshot, stats, store, pool)["winners"][0]
    assert enriched["group_count"] == 3
    assert enriched["boxed_frac"] == 1 / 3 and enriched["output_chars_p50"] == 6
    assert enriched["prompt_sources"] == {"gen": 1, "dapo": 0, "math": 1, "nmv2": 0, "other": 0, "unmapped": 1}
    assert sum(enriched["prompt_band"].values()) == 2
    public.validate_winner_enrichment(snapshot, enriched, store, pool)


def test_real_controller_enriches_only_its_crowned_public_batch(tmp_path):
    from sn120_v2.envs.math_pool import MathPool

    made = world(tmp_path)
    reads = []
    winner_batch = layout.winner_batch("r000001")

    # A real public crowned batch has already exercised enrichment above.
    # Here the real controller publishes one winner beside one loser. Their
    # simulator batches are deliberately non-affine; only the crowned public
    # object may be read, and its malformed shape must not stop publication.
    class GuardedStore:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def get(self, key):
            reads.append(key)
            assert not key.startswith(("private/", "pending/"))
            if "/batch.json" in key:
                assert key == winner_batch
            return self.inner.get(key)

    guarded = GuardedStore(made.store)
    made.controller.publisher = publisher.StorePublisher(guarded, "https://files.example/", prompt_pool=MathPool((), "fixture"))
    made.register(11, 0.4)
    made.register(12, 0.0)
    made.round()
    assert made.controller.last_error == ""
    stats = json.loads(made.store.get(layout.WINNER_STATS))
    assert len(stats["winners"]) == 1 and stats["winners"][0]["hotkey"] == "hk11"
    assert not set(stats["winners"][0]) & public.ENRICHMENT_FIELDS
    assert made.store.get(layout.WINNER_STATS_ALIAS) == made.store.get(layout.WINNER_STATS)
    assert reads.count(winner_batch) == 1
    assert all(not key.startswith(("private/", "pending/")) and (key == winner_batch or "/batch.json" not in key) for key in reads)


def test_unreadable_crowned_batch_never_blocks_publication(tmp_path, caplog):
    snapshot, winner, source, pool = crowned_public_batch_fixture(tmp_path)
    _, _, store = published(snapshot)  # head manifest and all ordinary public artifacts
    store[layout.winner_batch(winner["round_id"])] = source[layout.winner_batch(winner["round_id"])]
    reads = []

    class Guarded(MemStore):
        def get(self, key):
            reads.append(key)
            assert not key.startswith(("private/", "pending/"))
            if key == layout.winner_batch(winner["round_id"]):
                raise RuntimeError("public winner object temporarily unreadable")
            return super().get(key)

    guarded = Guarded(store)
    publisher.StorePublisher(guarded, "https://files.example/", prompt_pool=pool).publish(snapshot)
    stats = json.loads(guarded[layout.WINNER_STATS])
    assert not set(stats["winners"][0]) & public.ENRICHMENT_FIELDS
    assert guarded[layout.WINNER_STATS_ALIAS] == guarded[layout.WINNER_STATS]
    assert layout.MINING_JSON in guarded and layout.LLMS_TXT in guarded
    assert all(not key.startswith(("private/", "pending/")) for key in reads)
    assert "enrichment omitted" in caplog.text


def test_winner_transport_is_matched_and_reduced_to_safe_fields(tmp_path):
    made = world(tmp_path)
    made.register(11, 0.4)
    made.register(12, 0.0)
    made.round()
    snap = json.loads(json.dumps(made.publisher.snapshots[-1]))
    row = snap["winner_stats"]["winners"][0]
    loser = next(entry for entry in snap["history"][0]["entries"] if entry["status"] != "winner")
    snap["winner_stats"]["winners"] += [
        dict(row, uid=loser["uid"], hotkey=loser["hotkey"], batch_sha256=loser["batch_sha256"]),
        {"round_id": "r999999", "uid": 99, "hotkey": "hk99", "batch_sha256": "f" * 64},
    ]
    row |= {"uri": "private/rounds/r000001/batches/hk11.json", "reason": "private waiting verdict", "train": {"lora_lr": 0.00005}}
    doc, _, store = published(snap)
    stats = json.loads(store[layout.WINNER_STATS])
    assert len(stats["winners"]) == 1 and stats["winners"][0]["train"] == {"lora_lr": 0.00005}
    assert "uri" not in stats["winners"][0] and "reason" not in stats["winners"][0]
    assert "private/" not in json.dumps(stats) and "hk99" not in json.dumps(stats) and "r999999" not in json.dumps(stats)
    assert "winner_stats" not in doc
    external = {key: value for key, value in snap.items() if key != "winner_stats"}
    fallback = public.winner_stats_from_snapshot(external, "https://files.example")
    assert fallback["winners"][0]["batch_prompts"] is None and fallback["winners"][0]["train"] is None
    public.validate_winner_stats(external, fallback, "https://files.example")


def test_winner_file_validation_rejects_forged_or_extra_public_content(tmp_path):
    made = world(tmp_path)
    made.register(11, 0.4, train={"steps": 2})
    made.round()
    snap = json.loads(json.dumps(made.publisher.snapshots[-1]))
    _, _, store = published(snap)
    stats = json.loads(store[layout.WINNER_STATS])
    good = stats["winners"][0]
    bad_rows = [
        dict(good, uri="private/a-batch"),
        dict(good, z=999),
        dict(good, confirmed=False),
        dict(good, batch_url="https://attacker.example/public/rounds/r000001/batch.json"),
        dict(good, batch_url="/public/rounds/r000001/batch.json"),
        dict(good, train={"steps": 2, "uri": "private/a-batch"}),
        dict(good, train={"groups": ["private prompt"]}),
        {name: value for name, value in good.items() if name != "lora"},
    ]
    for bad in (
        stats | {"private": "pending/batch"},
        *(stats | {"winners": [row]} for row in bad_rows),
    ):
        with pytest.raises(ValueError):
            public.validate_winner_stats(snap, bad, "https://files.example")
    external = {key: value for key, value in snap.items() if key != "winner_stats"}
    fallback = public.winner_stats_from_snapshot(external, "https://files.example")
    public.validate_winner_stats(external, fallback, "https://files.example")
    rich = stats | {"winners": [good]}
    public.validate_winner_stats(external, rich, "https://files.example")
    for name, value in (("batch_prompts", "one"), ("batch_rollouts", True), ("batch_format", 1), ("train", [])):
        with pytest.raises(ValueError):
            public.validate_winner_stats(external, stats | {"winners": [good | {name: value}]}, "https://files.example")


def test_standalone_stats_accept_real_public_heads_and_type_checked_crowned_batch_facts(tmp_path):
    made = world(tmp_path)
    made.register(11, 0.4, train={"steps": 2})
    made.round()
    made.register(12, 0.4)
    made.round()
    doc, _, store = published(made.publisher.snapshots[-1])
    stats = json.loads(store[layout.WINNER_STATS])
    row = next(row for row in stats["winners"] if row["uid"] == 11)
    actual = row["head"]
    other = next(head["version_id"] for head in doc["heads"] if head["round_id"] != row["round_id"])
    row["head"] = other
    public.validate_winner_stats(doc, stats, "https://files.example")
    for name, value in (("batch_prompts", "one"), ("batch_rollouts", True), ("batch_format", 1), ("train", [])):
        bad = stats | {"winners": [dict(item, **{name: value}) if item["round_id"] == row["round_id"] else item for item in stats["winners"]]}
        with pytest.raises(ValueError):
            public.validate_winner_stats(doc, bad, "https://files.example")
    with pytest.raises(ValueError):
        public.validate_winner_stats(doc, stats | {"winners": [dict(item, head="not-a-public-head") for item in stats["winners"]]}, "https://files.example")


def test_pending_crown_is_not_reported_as_confirmed_or_paid(tmp_path):
    made = world(tmp_path, confirm_next_round=True)
    made.register(11, 0.4)
    made.round()
    assert made.controller.state.pending_king is not None and not made.controller.state.kings
    stats = json.loads(published(made.publisher.snapshots[-1])[2][layout.WINNER_STATS])
    row = stats["winners"][0]
    assert row["head"] == made.controller.state.pending_king.new_version_id
    external = {key: value for key, value in made.publisher.snapshots[-1].items() if key != "winner_stats"}
    public.validate_winner_stats(external, stats, "https://files.example")
    assert row["applied_at"] == made.controller.state.pending_king.applied_at
    assert (row["confirmed"], row["paid"], row["payout_status"]) == (None, False, "unpaid")
    legacy = json.loads(json.dumps(stats))
    legacy["winners"][0]["confirmed"] = False  # the already-written live object, before the next-publish writer fix
    public.validate_winner_stats(external, legacy, "https://files.example")
    legacy["winners"][0]["confirmed"] = True
    with pytest.raises(ValueError):
        public.validate_winner_stats(external, legacy, "https://files.example")


def test_a_failed_resolved_confirmation_is_unpaid_in_winner_stats_and_fallback(tmp_path):
    made = world(tmp_path, confirm_next_round=True)
    made.register(11, 0.4)
    first = made.round()
    made.evaluator.parent_quality = 0.5
    made.register(12, 0.0)
    second = made.round()
    assert second.confirmation is not None and not second.confirmation.confirmed
    assert not made.controller.state.kings and made.controller.state.pending_king is None
    stats = json.loads(published(made.publisher.snapshots[-1])[2][layout.WINNER_STATS])
    row = next(row for row in stats["winners"] if row["round_id"] == first.round_id)
    assert (row["confirmed"], row["paid"], row["payout_status"]) == (False, False, "unpaid")
    external = {key: value for key, value in made.publisher.snapshots[-1].items() if key != "winner_stats"}
    fallback = public.winner_stats_from_snapshot(external, "https://files.example")
    assert [(row["round_id"], row["uid"]) for row in fallback["winners"]] == [(row["round_id"], row["uid"]) for row in stats["winners"]]
    projected = next(row for row in fallback["winners"] if row["round_id"] == first.round_id)
    assert (projected["confirmed"], projected["paid"], projected["payout_status"]) == (False, False, "unpaid")
    public.validate_winner_stats(external, fallback, "https://files.example")



def test_both_files_come_from_one_snapshot_and_change_together(tmp_path):
    running, idle = world(tmp_path, play=True).publisher.snapshots[-2:]
    doc, text, store = published(running)
    assert doc["phase"] == "RUNNING" and "- phase: RUNNING" in text and f"- round_seed: {doc['round']['round_seed']}" in text
    assert doc["snapshot_digest"] in text and f"| 0 | 11 | hk11 | {doc['round']['entries'][0]['batch_sha256']} |" in text
    doc2, text2, _ = published(idle, store)
    assert doc2["snapshot_digest"] != doc["snapshot_digest"] and doc2["snapshot_digest"] in text2
    assert "- phase: IDLE" in text2 and doc["snapshot_digest"] not in text2


def test_local_delta_se_is_measured_and_bounded(tmp_path):
    """One stable docs line, sourced from public history, plus a UTC-only changelog."""
    doc, text, _ = published(world(tmp_path).publisher.snapshots[-1])
    flat = " ".join(text.split())
    assert flat.count(publisher.LOCAL_DELTA_SE) == 1
    assert "about 5.3 percentage points" in publisher.LOCAL_DELTA_SE
    assert "public 2000-task SE scaled by √(n/100)" in publisher.LOCAL_DELTA_SE
    changelog = next(line for line in publisher.RULE_CHANGELOG if "local-delta measurement guidance" in line)
    assert "2026-11-25" in changelog and "5.3 percentage points" in changelog
    assert "Costa Rica" not in changelog and "UTC-6" not in changelog


def test_service_state_reasons_coarse_utc_and_public_vocabulary(tmp_path):
    made = world(tmp_path, play=True)
    running, idle = made.publisher.snapshots[-2:]
    doc, text, _ = published(running)
    assert doc["phase"] == "RUNNING"
    assert doc["service"] == {"state": "running", "reason": None, "next_open_at": None}
    assert idle["phase"] == "IDLE"
    assert idle["service"] == {"state": "idle", "reason": "between rounds", "next_open_at": idle["service"]["next_open_at"]}
    closed = made.controller.state.rounds[-1]
    made.controller.state.round = dataclasses.replace(closed, closed_at=0)
    assert public.service_view(made.controller.state, made.now) == {"state": "applying", "reason": None, "next_open_at": None}
    made.controller.state.round = None

    cases = [
        ({}, ("idle", "waiting for seats", None)),
        ({"waiting_for_seats": False}, ("idle", "between rounds", "2023-11-14T22:20:00Z")),
        ({"close_publish_pending": True}, ("stalled", "publish retrying", "2023-11-14T22:20:00Z")),
        ({"stop_asked": True}, ("idle", "deploying fix", "2023-11-14T22:20:00Z")),
        ({"stop_asked": True, "close_publish_pending": True, "waiting_for_seats": False}, ("stalled", "deploying fix", "2023-11-14T22:20:00Z")),
        ({"close_publish_pending": True, "waiting_for_seats": False}, ("stalled", "publish retrying", "2023-11-14T22:20:00Z")),
    ]
    for flags, expected in cases:
        shown = public.snapshot(made.controller.state, made.params, 1_700_000_001, **flags)["service"]
        assert set(shown) == {"state", "reason", "next_open_at"}
        assert (shown["state"], shown["reason"], shown["next_open_at"]) == expected
        if shown["next_open_at"] is not None:
            parsed = datetime.datetime.strptime(shown["next_open_at"], "%Y-%m-%dT%H:%M:%SZ")
            assert int(parsed.replace(tzinfo=datetime.UTC).timestamp()) % 300 == 0
        assert shown["state"] in public.SERVICE_STATES
        assert shown["reason"] in public.SERVICE_REASONS | {None}
        assert not set(shown) & {"idle_sleep", "max_idle", "retry", "tries", "stop_file", "last_error", "n_valid"}
    assert public.service_view(made.controller.state, 1_700_000_000, waiting_for_seats=False)["next_open_at"] == "2023-11-14T22:20:00Z"
    assert "service.state" in text and "deploying fix" in text and "publish retrying" in text
    assert "2026-11-25 / `service`" in text


def test_service_seats_and_close_retry_use_actual_controller_facts(tmp_path):
    made = world(tmp_path, n_min_start=2, batch_cap=4)
    made.register(1, 0.0)  # one valid file is insufficient at n_min_start=2, even though batch_cap is four
    assert not made.controller.open_round(made.clock(), "entropy")
    assert made.controller.publish(made.clock())
    assert made.publisher.snapshots[-1]["service"]["reason"] == "waiting for seats"
    assert made.publisher.snapshots[-1]["service"]["next_open_at"] is None
    made.register(2, 0.0)
    assert made.controller.run_round(made.runner, made.clock, "entropy")
    assert made.publisher.snapshots[-1]["service"]["reason"] == "between rounds"

    controller = made.reopen()  # an already-published close needs no write; this fresh controller has not published it
    real, attempted = controller.publisher.publish, []

    def down(snapshot):
        attempted.append(snapshot)
        raise InfraError("store down")

    controller.publisher.publish = down
    problem = controller.close_in_public(made.clock())
    assert problem and controller._close_publish_pending
    assert attempted[-1]["service"]["state"] == "stalled"
    assert attempted[-1]["service"]["reason"] == "publish retrying"
    assert "store down" not in json.dumps(attempted[-1]["service"])
    controller.publisher.publish = real
    assert controller.close_in_public(made.clock()) == ""
    assert made.publisher.snapshots[-1]["service"]["state"] == "stalled"  # the successful retry exposes its last status
    assert not controller._close_publish_pending
    assert controller.publish(made.clock())
    assert made.publisher.snapshots[-1]["service"]["state"] == "idle"
    assert controller.close_in_public(made.clock()) == ""
    assert not controller._close_publish_pending


def test_every_miner_bound_is_published_from_the_records(tmp_path):
    bounds = tuple(
        MinerBound("steps", 2, 7, integer=True) if bound.key == "steps" else bound for bound in Recipe().miner_bounds
    )
    recipe = Recipe(image_digests=(sim.SIM_IMAGE,), steps=3, miner_bounds=bounds)
    doc, text, _ = published(world(tmp_path, recipe=recipe).publisher.snapshots[-1])
    rows = {row["key"]: row for row in doc["contract"]["miner_settable"]}
    assert list(rows) == ["lora_lr", "adam_eps", "steps", "minibatches", "group_size", "sample_temperature", "rank", "alpha", "target_modules", "global_norm_cap"]
    for bound in bounds:
        row = rows[bound.key]
        assert (row["low"], row["high"], row["choices"], row["decided"]) == (bound.low, bound.high, list(bound.choices), bound.decided)
        assert row["default"] == json.loads(json.dumps(getattr(recipe, bound.key)))
        assert f"| `train.{bound.key}` | " in text and row["allowed"] in text
    assert "| `train.steps` | 3 | a whole number from 2 to 7, both ends included |" in text
    subset = "a non-empty subset of q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj_qkv, in_proj_z, out_proj"
    assert subset + "; a list with in_proj_z must hold in_proj_qkv too |" in text
    assert "placeholder" not in text.lower()  # the miner's text states the launch values as they are
    assert doc["contract"]["recipe_id"] == recipe.recipe_id and recipe.recipe_id in text


def test_stored_prompt_contract_and_recommendation_are_published(tmp_path):
    text = published(world(tmp_path).publisher.snapshots[-1])[1]
    flat = " ".join(text.split())
    assert "Validator training uses exactly the stored `prompt` field of each batch row" in flat
    assert "Sampling context such as a separate system message is not kept unless it was written into that stored `prompt` field" in flat
    assert "official instruction of section 6 (`MANDATE`), a blank line, then the problem, both in that one `prompt` field, as recent crowned kings do" in flat
    assert '`MathPool.task.batch_prompt()` = `MANDATE + "\\n\\n" + problem`' in flat
    assert "For your sampler, match the trainer's render: strip a leading official instruction + blank line from the stored `prompt` if present; render system = the official instruction, user = the problem; then apply the head's chat template with `add_generation_prompt=True` and `enable_thinking=True`" in flat
    assert "The template is not applied to the raw stored `prompt` as a single user message" in flat
    assert "Prompt input: exactly the stored `prompt` field" not in flat
    assert "Both are the same task for the grader and give exactly the same model tokens" not in flat
    assert "Scoring renders a task in another way than training" not in flat


def test_the_pool_drop_list_and_non_pool_meaning_are_published(tmp_path):
    _, text, _ = published(world(tmp_path).publisher.snapshots[-1])
    flat = " ".join(text.split())
    assert "Base task-body catalog: 20,120 tasks" in flat
    assert "1,454 tasks of the earlier pool" in flat and POOL_V2_0_SHA256[:8] in flat
    assert "https://github.com/AffineFoundation/affine/blob/v2/v2/sn120_v2/envs/math_pool_v2_1.dropped.json" in flat
    assert "removed tasks is no longer a pool task" in flat and "min_pool_prompt_frac" in flat


def test_entry_outcomes_do_not_require_a_round_decision(tmp_path, build):
    from sn120_v2.state import Lease, RoundRecord
    from sn120_v2.types import RoundDecision
    ticket = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF"})
    lease = Lease(ticket.lease_id, ticket.box_id, ticket.issued_at, ticket.expires_at, ticket.digest(), ticket.entries)
    record = RoundRecord(ticket.round_id, ticket.head.version_id, ticket.params.params_hash, ticket.params.recipe_id,
                         ticket.draw.round_seed, ticket.draw.snapshot_id, len(ticket.draw.task_ids), ticket.issued_at, (lease,))
    report = public.round_report(record, [(ticket, result)])
    assert [row["status"] for row in report["entry_outcomes"]] == [None, None]
    for row, seat in zip(report["entry_outcomes"], ticket.entries):
        assert (row["queue_pos"], row["uid"], row["hotkey"]) == (seat.queue_pos, seat.reg.uid, seat.reg.hotkey)
        assert row["outcomes"] == list(next(entry.outcomes for entry in result.entries if entry.queue_pos == seat.queue_pos))
    partial = dataclasses.replace(record, decision=RoundDecision(
        ticket.round_id, ticket.head.version_id, None, (), len(ticket.draw.task_ids), 0.0, 0.0))
    assert [row["status"] for row in public.round_report(partial, [(ticket, result)])["entry_outcomes"]] == [None, None]


def test_all_scored_entry_eval_outputs_are_public(tmp_path):
    made = world(tmp_path)
    made.register(11, 0.4)
    made.register(12, 0.0)
    record = made.round()
    report = json.loads(made.store.get(layout.round_file(record.round_id)))
    assert [row["queue_pos"] for row in report["entry_outcomes"]] == [0, 1]
    assert [row["status"] for row in report["entry_outcomes"]] == ["winner", "lost"]
    assert all(len(row["outcomes"]) == made.params.samples_per_task and row["lora_hash"] and row["hotkey"] for row in report["entry_outcomes"])
    assert report["rollouts_meta"] == layout.public_rollout_gate_meta(record.round_id)
    flat = " ".join(published(made.publisher.snapshots[-1])[1].split())
    assert "The outcome strings of the other entries are not published." not in flat
    for text in (
        "every other entry that was scored (`entry_outcomes`: queue position, UID, hotkey, LoRA hash, one string per sample)",
        "null M (`public/rollouts/gate/<round_id>/head.jsonl.gz`)",
        "every scored entry (`entries/<queue_pos>.jsonl.gz`)",
        "listed with sizes and sha256 in `public/rollouts/gate/<round_id>/meta.json`",
        "only the winner's batch is published (`public/rounds/<round_id>/batch.json`)",
    ):
        assert text in flat


def test_every_number_follows_the_pinned_params(tmp_path):
    recipe = Recipe(image_digests=(sim.SIM_IMAGE,), train_max_output_tokens=777)
    changes = dict(max_prompts=123, max_rollouts=4567, z_bar=1.25, kings_window=7, n_tasks=400, looks=(0.25, 1.0), recipe=recipe)
    made = world(tmp_path, **changes)
    doc, text, _ = published(made.publisher.snapshots[-1])
    assert doc["contract"]["params_hash"] == made.params.params_hash
    assert "Caps: 123 prompts and 4567 rollouts" in text and "strictly above 1.25" in text and "last 7 distinct king" in text
    assert "after 100, 400 tasks" in text and "after 100 tasks." in text  # looks and the early-drop bar are computed
    flat = " ".join(text.split())
    assert "Training reads at most 777 output tokens" in flat and "A longer rollout is valid: the trainer" in flat
    header = doc["contract"]["batch_file"]["upload_headers"]
    assert header == r2.PUT_HEADERS and "`Content-Type: application/octet-stream`" in " ".join(text.split())  # what the presigner signs
    assert f"rule code `{TIME_LIMIT}`" in text
    for field in dataclasses.fields(made.params) + dataclasses.fields(recipe):  # the full table and the full recipe
        # Optional king-count, confirmation and default rotation fields are deliberately absent;
        # a selected king count and every dedup/seed field are stated.
        assert f"`{field.name}`" in text or field.name.startswith("_") or field.name in ("recipe", "miner_bounds") or (field.name in made.params.LEFT_OUT_AT_DEFAULT and (field.name == "null_samples_per_task" or field.name == "min_kl_tok" or field.name.startswith("rotation_") or field.name == "confirm_next_round") and getattr(made.params, field.name) == field.default)
    selected = world(tmp_path / "selected", null_samples_per_task=2)
    selected_text = published(selected.publisher.snapshots[-1])[1]
    assert "`null_samples_per_task`" in selected_text
    for key in doc["contract"]["objects"].values():
        assert key.startswith("public/") and key in text


def test_the_trainer_rules_and_the_batch_rule_are_stated_from_the_recipe(tmp_path):
    recipe = Recipe(image_digests=(sim.SIM_IMAGE,), train_max_output_tokens=777, train_max_context_tokens=888, sample_temperature=0.75)
    made = world(tmp_path, temperature=0.9, batch_cap=3, kings_window=7, recipe=recipe)
    text = published(made.publisher.snapshots[-1])[2].get(layout.LLMS_TXT).decode()
    flat = " ".join(text.split())  # the rules as sentences, whatever the line breaks
    assert "A value G above 0 cuts the rollouts of each prompt, in file order, into groups of G" in flat
    assert "its reward minus the mean reward of its group. The default is 0." in flat
    assert "divides the logits of the head by T" in flat and "always samples at temperature 0.9. The default T is 0.75." in flat
    assert "every B matrix is multiplied by cap / norm" in flat
    assert "The recipe value is the cap of a file that sets none. It is 6 now. `train.global_norm_cap` sets the cap of your LoRA, from 0.01 to 6." in flat
    assert "`max_grad_norm` | 1 | gradient clip of each update; it does not cap the size of an update" in flat and "step-size cap" not in flat
    assert "The clip does not cap the size of an update." in flat and "The one cap on it is `global_norm_cap`." in flat
    assert "Row k of the shuffled list (k from 0) goes to minibatch k modulo N." in flat and "steps x N updates in all" in flat
    assert "The cost does not count `minibatches`" in flat and "Each update also has a fixed cost of its own, about 0.1 s" in flat
    old = dataclasses.replace(recipe, global_norm_cap=None, miner_bounds=tuple(bound for bound in recipe.miner_bounds if bound.key not in ("minibatches", "global_norm_cap")))
    before = " ".join(published(world(tmp_path / "before", recipe=old).publisher.snapshots[-1])[2].get(layout.LLMS_TXT).decode().split())
    assert "This recipe has none: there is no cap. A file may not set a cap of its own now." in before, "a set from before the two new bounds"
    assert "`train.minibatches`" not in before and "`train.global_norm_cap`" not in before
    assert "states the norm before the cap and the norm after it (`step_norm` is the norm after the cap)" in flat
    assert "`train_max_output_tokens` (777) or `train_max_context_tokens` (888, prompt + output) is truncated, not rejected" in flat
    assert "A rollout with no output token left is dropped and counted." in flat
    assert "keeps its queue position, and may replace its file as often as it likes until the entry is evaluated. There is no fixed window." in flat
    assert "copies the latest file when a round takes the entry" in flat and "The admission checks run again on each new file." in flat
    assert "that only v1 had (`lr`, `kl`, `step_mult`, the knobs) is invalid. The seat is skipped and not charged, as with any invalid file (section 4)" in flat
    assert "A round takes at most 3 batches" in flat
    assert "The only rule that moves a waiting seat back is squat-trim (section 4)" in flat
    assert "A round with no winner changes nothing in the payout: the window of the last 7 kings keeps being paid." in flat
    assert "No burn, in any case: no weight goes to UID 0 or to the owner as a fallback." in flat
    assert "The share of a king that is no longer registered is split among the other registered kings of the window." in flat
    assert "goes to UID 0." not in flat and "the writer sets nothing and the last valid weights stay on chain" in flat
    capped = Recipe(image_digests=(sim.SIM_IMAGE,), global_norm_cap=2.5)
    assert "It is 2.5 now." in published(world(tmp_path / "capped", recipe=capped).publisher.snapshots[-1])[2].get(layout.LLMS_TXT).decode()


def test_the_times_of_a_round_and_the_queue_place_follow_the_params(tmp_path):
    changes = dict(round_time_limit_s=4800, lease_timeout_s=9000, promote_reserve_s=300, train_time_budget_s=1500)
    changes |= dict(eval_chunk_tasks=40, n_gpus=4, n_tasks=400, looks=(0.25, 1.0))
    changes |= dict(max_infra_attempts=5, squat_min_wait_s=7200, squat_age_factor=3.0, max_train_token_steps=123456)
    flat = " ".join(published(world(tmp_path, **changes).publisher.snapshots[-1])[1].split())
    assert "Time limit: 80 minutes from the lease to the promotion of the winner." in flat
    assert "The box stops scoring 5 minutes before it, at minute 75 (the stop time)" in flat  # worker.LocalRunner.run
    assert "ends early, as soon as every LoRA is dropped or decided" in flat
    assert "Training time: the first 25 minutes." in flat and "Batches train longest-expected-first" in flat
    assert "Such a batch is `not_trained`: you are not charged, your seat is first in the next round" in flat
    assert "the training time of the round (25 minutes for all batches, section 3) and this cost cap for one file (123456 token-steps)" in flat
    assert "`train_cost`: for v1-v3, cost = steps x the tokens" in flat and "A file with a cost above 123456 token-steps is invalid." in flat
    no_cap = " ".join(published(world(tmp_path / "no-cap", max_train_token_steps=0).publisher.snapshots[-1])[1].split())
    assert "There is no cap on the cost now" in no_cap and "this cost cap for one file (no cap now)" in no_cap
    assert "Scoring streams over the draw with no chunk barrier." in flat
    assert "at minute 75 (the stop time). The stop is hard." in flat and "The stop time cuts every look, the first one too." in flat
    assert "At the stop time, with the first look (100 tasks) complete: each open LoRA is judged by the win rule of section 6 on the complete cutoff blocks that null M and every open LoRA finished" in flat
    assert "At the stop time, with fewer than 100 tasks scored: each trained LoRA is `not_scored`. You are not charged, your seat is first in the next round, and no failure is counted for the seat." in flat
    assert "always scored in full" not in flat and "longer than 80 minutes only when" not in flat, "no exemption for the first look"
    assert "`not_scored`: the time limit of the round came before the first look of the scoring (section 3)." in flat
    assert "With fewer tasks than the first look, there is no decision: `not_scored` (section 3)." in flat
    assert "The lease ends 150 minutes after it starts." in flat
    assert "When 5 failures are counted for a seat, the seat is held" in flat and "so it never holds a seat" in flat
    assert "A seat that has a valid file stored keeps its place and is never moved back." in flat
    assert "It applies only to a seat that has no valid file stored when the validator picks the entries of a round" in flat
    assert "older than 3 x 7200 s = 21600 s (`squat_age_factor` x `squat_min_wait_s`). No seat is moved when no turn was charged in the last 24 hours. A seat with a valid file is never moved." in flat
    assert "The two values are parameters (section 11). The validator can change them." in flat and "is announced" not in flat
    assert "seats ahead x seconds per turn" not in flat
    assert "seconds per turn = 86400 /" not in flat.lower()
    assert "No seat is moved when no turn was charged in the last 24 hours." in flat


def test_the_queue_text_gives_the_wait_in_days_and_says_what_removes_a_seat(tmp_path):
    for wait, first, total in ((31536000, "365 days", "730 days"), (86400, "1 day", "2 days"), (129600, "129600 s", "3 days"), (7201, "7201 s", "14402 s")):
        flat = " ".join(published(world(tmp_path / str(wait), squat_min_wait_s=wait).publisher.snapshots[-1])[1].split())
        assert f"older than 2 x {first} = {total} (`squat_age_factor` x `squat_min_wait_s`)." in flat
        assert "e+" not in flat.split("## 4.")[1].split("## 5.")[0]
        assert f"| `squat_min_wait_s` | {wait} |" in flat  # the table of section 11 keeps the value of the params
    assert publisher.duration(2.5 * 7201) == "18002.5 s" and publisher.duration(0) == "0 s"
    assert "a round takes it when it has room after the front seats and the normal seats that have a valid file" in flat
    assert "A seat that v1 had moved to the back of its queue starts at the back here." in flat
    assert "`demoted` = yes: the seat is at the back, unless `front` is also yes." in flat  # queue.zone
    assert "`position` counts every seat ahead of you, also a seat with no valid file" in flat and "`held` = yes: the seat takes no turn" in flat
    assert "A seat exists only while its registration is on chain." in flat  # queue.plan_sync
    assert "(the UID is deregistered, or the UID goes to a newer registration), the seat is removed and its turn is lost" in flat
    assert "A seat that is in an open round stays until that round closes." in flat
    assert "the seat keeps its place. The validator then reads the file of the new hotkey" in flat
    assert "When the new hotkey already had its turn, the seat is removed." in flat


def test_the_text_on_no_decision_says_what_the_controller_does_and_what_the_round_file_shows(tmp_path, out_of_time):
    """Each sentence here stands next to the case that it describes, played by the real controller."""
    made = world(tmp_path, max_infra_attempts=2, apply_retry_s=420)
    state = made.controller.state
    for uid in (1, 2, 3):
        made.register(uid, 0.0)
    made.trainer.fail_hotkeys = {"hk1", "hk2", "hk3"}  # no training finishes: a round with no result (Controller.run_round)
    report = json.loads(made.store.get(layout.round_file(made.round().round_id)))
    doc, text, _ = published(made.publisher.snapshots[-1])
    flat, last = " ".join(text.split()), doc["history"][-1]
    assert "A round with no result: the box failed, not a seat." in flat
    assert "when the training failed for at least one batch and finished for no batch, when the box stopped with an error, when its result did not arrive before the end of the lease, or when the operator ended the round" in flat
    assert "nobody is charged, no failure is counted for a seat, and each seat of the round goes to the front of the queue" in flat
    assert not state.served_regs and all((seat.front, seat.infra_attempts, seat.held) == (True, 0, False) for seat in state.queue.values())
    assert "The next round starts 420 s after the failure, or later." in flat and made.round() is None
    assert "In the round file (section 9) you see it thus: `outcome` is `no_decision`, no entry has a `status`, and `note` gives the cause." in flat
    assert report["outcome"] == "no_decision" and report["note"].startswith("InfraError: no training finished") and not any("status" in entry for entry in report["entries"])
    assert "In `history` of mining.json the round has the same `outcome` and its entries have no `status`; the `note` is only in the round file." in flat
    assert last["outcome"] == "no_decision" and "note" not in last and not any("status" in entry for entry in last["entries"])
    assert "A box result that arrives after the end of the lease is never used: the round then has no result (section 4)." in flat
    assert "each entry of that box is" not in flat and "has reported `no_decision`" not in flat, "the words of the rule before Controller.run_round failed the lease"

    made.trainer.fail_hotkeys = {"hk3"}  # one training job fails in each of its tries and the others finish: never counted
    for other in (None, 4, 5):
        made.now += made.params.apply_retry_s
        if other:
            made.register(other, 0.0)
        report = json.loads(made.store.get(layout.round_file(made.round().round_id)))
        mine = next(entry for entry in report["entries"] if entry["uid"] == 3)
        assert report["outcome"] == "null" and mine["status"] == "no_decision" and "delta" not in mine and mine["reason"].startswith("training failed")
        assert (state.queue[(3, 1003)].front, state.queue[(3, 1003)].infra_attempts, state.queue[(3, 1003)].held) == (True, 0, False)
    assert "A training job that failed is never counted for your seat: out of memory on the box, a crash of the job, any other error of the training." in flat
    assert "The box tries the training again (section 3). If each try fails, your entry has the status `no_decision` and no scores, and its `reason` starts with `training failed`. You are not charged and you go to the front of the queue." in flat

    out_of_time(made, {"hk3"})  # the batch had the whole training time and did not finish, and the others finish: the one case that is counted
    for count, other in ((1, 6), (2, 7)):
        made.register(other, 0.0)
        report = json.loads(made.store.get(layout.round_file(made.round().round_id)))
        mine = next(entry for entry in report["entries"] if entry["uid"] == 3)
        assert report["outcome"] == "null" and report["note"] == "" and mine["status"] == "no_decision" and "delta" not in mine
        assert (state.queue[(3, 1003)].infra_attempts, state.queue[(3, 1003)].held) == (count, count == 2)
    assert "A failure is counted for your seat in one case only: your batch had the whole training time of the round (it started at the lease, and the box started its work on the round at most 120 s after the start of the lease), the first try of its training did not finish in that time (section 3), and at least one other batch of the same round finished its training." in flat
    assert "Your entry then has the status `no_decision` and no scores. When 2 failures are counted for a seat, the seat is held: it takes no turn until the operator releases it." in flat
    made.controller.release_seat((3, 1003), made.now)
    assert "A release also sets the count to zero." in flat and (state.queue[(3, 1003)].infra_attempts, state.queue[(3, 1003)].held) == (0, False)

    assert "A `no_decision` that comes from the scoring (section 6) has scores and is not counted." in flat  # Controller._close: only a status of the box
    assert "If that is so for null M, every scored entry of the round is `no_decision`. Neither is counted as a failure for the seat (section 4)." in flat
    assert "A round that has a result also gets the outcome `no_decision` when none of its entries was charged." in flat
    assert "`no_decision` as the status of an entry means: not charged, and the seat goes to the front of the queue (section 4)." in flat
    assert "The `note` is empty unless the round had no result or has the outcome `abandoned` (section 4)." in flat
    markers = (public.CUT_SECRET, public.CUT_LOGIN, public.CUT_QUERY, public.CUT_ADDRESS)  # public.public_text
    assert "a marker replaces each part that it takes for private (`{}`, `{}`, `{}` for the query string of a link, `{}` for an IP address)".format(*markers) in flat
    assert f"a text of more than {public.TEXT_LIMIT} characters is cut there and ends with a note of how many characters were cut" in flat
    long = public.public_text("InfraError: 403 for url: https://user:pw@store.example/x.json?X-Amz-Signature=0123abcd from 203.0.113.7:443 " + "x" * public.TEXT_LIMIT)
    assert all(marker in long for marker in markers[1:]) and long.endswith(" more characters]") and long.startswith("InfraError: 403 for url: https://")
    assert "The same rule applies to the `reason` of a round in mining.json and in section 3." in flat
    assert "To decide what applies to your seat, read `outcome` and `status`, not the words of a text." in flat


def test_the_public_address_of_the_link_files_is_a_setting(tmp_path):
    snap = world(tmp_path).publisher.snapshots[-1]
    doc, text, store = published(snap)
    assert doc["contract"]["upload"] is None and doc["contract"]["links_url"] == "" and "Where to get the link: not published yet." in text
    publisher.StorePublisher(store, "https://files.example/", "https://site.example/ff/links/").publish(snap)
    doc, flat = json.loads(store[layout.MINING_JSON]), " ".join(store[layout.LLMS_TXT].decode().split())
    upload = doc["contract"]["upload"]
    assert doc["contract"]["links_url"] == "https://site.example/ff/links" and upload == links.upload_entry("https://site.example/ff/links")
    assert "Where to get the link: https://site.example/ff/links/<hotkey>.json." in flat
    assert f"A new file is written every {upload['reissue_every_seconds'] // 60} minutes. Each link works for {upload['link_ttl_seconds'] // 3600} hours." in flat
    assert "links open --hotkey <ss58> --key <seed file> --links-url https://site.example/ff/links)" in flat
    assert "links activate --wallet <wallet name> --hotkey <hotkey name>" in flat and "The opened URL is a secret." in flat
    with pytest.raises(ConfigError):
        publisher.StorePublisher(store, "", "site.example/ff/links")


def test_the_head_names_the_genesis_model_only_while_its_base_is_that_model(tmp_path):
    made = world(tmp_path).controller.state.head
    fields = dict(base_fp32_master_digest=made.base_fp32_master_digest, loras=(), round_id=made.round_id, applied_at=made.applied_at)
    genesis = HeadVersion.make(parent_version_id=None, base_hash=GENESIS_BASE_HASH, **fields)
    later = HeadVersion.make(parent_version_id=genesis.version_id, base_hash=GENESIS_BASE_HASH, **fields)
    assert publisher._genesis_lines(made) == []  # the sim's base is not the launch model
    text = " ".join(publisher._genesis_lines(genesis))
    assert GENESIS_MODEL_REPO in text and GENESIS_MODEL_REVISION in text and "No winner is applied yet" in text
    assert "No winner is applied yet" not in " ".join(publisher._genesis_lines(later))


def test_the_example_file_trains_and_every_rule_code_is_listed(tmp_path):
    made = world(tmp_path)
    doc, text, _ = published(made.publisher.snapshots[-1])
    example = json.loads(text.split("```json\n")[1].split("```")[0])
    rewards = [rollout["reward"] for rollout in example["groups"][0]["rollouts"]]
    assert len(set(rewards)) > 1  # one rollout, or equal rewards, trains nothing
    check_batch(json.dumps(example | {"hotkey": "hk1"}).encode(), "hk1", made.params)
    rows = text.split("### Rule codes")[1].split("### How to upload")[0].splitlines()
    assert [row for row in rows if row.startswith("| `")] == [f"| `{code}` | {meaning} |" for code, meaning in RULES.items()]  # each rule, with its text, in run order
    assert all(meaning.strip() for meaning in RULES.values()) and set(TRANSIENT) <= set(RULES)
    assert "Not a final verdict: " + ", ".join(f"`{code}`" for code in TRANSIENT) + "." in " ".join(text.split())
    assert "- `no_signal`: your file must train something." in text
    assert publisher.SAMPLING_GUIDANCE in " ".join(text.split())
    assert "960 prompts produced 97 mixed-reward groups, a yield of about 10%" in " ".join(text.split())
    assert '--limit-mm-per-prompt \'{"image":0,"video":0}\'' in text
    assert "Only an output of exactly `params.max_output_tokens`" in " ".join(text.split())
    flat = " ".join(text.split())
    assert f"are in https://files.example/{publisher.READ_LINKS}." in flat and doc["contract"]["read_links"] == publisher.READ_LINKS
    assert "also at https://files.example/llms.txt and https://files.example/mining.json." in flat
    assert f'"{MANDATE}"' in flat and json.dumps(MANDATE) not in flat  # the instruction as it is sent: one backslash, not the JSON form
    assert "the hotkey never enters again, also with a new registration" in flat and "To enter again, register a new hotkey." in flat
    assert "The validator does not publish the code or the count" in flat and "the seat is skipped as with any invalid file (section 4)" in flat
    assert f'LINKS = "{publisher.READ_LINKS}"' in (Path(__file__).parents[1] / "site" / "pull.py").read_text()  # the writer's path
    assert all(f'"{key}"' in (Path(__file__).parents[1] / "site" / "pull.py").read_text() for key in ("expires_at", "head_version_id", "base", "loras", "master", "sha256", "size"))


def test_the_irt_snapshot_is_named_by_id_and_key_and_never_embedded(tmp_path):
    made = world(tmp_path)
    doc, text, _ = published(made.publisher.snapshots[-1])
    assert doc["irt_snapshot_id"] == "" and "No round has used a snapshot yet." in text  # bootstrap: nothing to name
    made.register(11, 0.4)
    made.round()
    running, idle = made.publisher.snapshots[-2:]
    for snap, name, found in ((running, "the open round", running["round"]["irt_snapshot_id"]), (idle, "the last closed round", idle["irt_snapshot_id"])):
        doc, text, _ = published(snap)
        key = layout.irt_snapshot(found)
        assert made.store.exists(key) and f"- Snapshot of {name}: id `{found}`, object `{key}`" in " ".join(text.split())
        assert "tasks" not in doc and "tasks" not in json.dumps(doc["contract"])  # the table stays in its own object
    store = MemStore({layout.head_manifest(idle["head"]["head"]["version_id"]): b"{}"})
    publisher.StorePublisher(store, "https://files.example/").publish(idle)
    assert f"https://files.example/{layout.irt_snapshot(idle['irt_snapshot_id'])}" in store.get(layout.LLMS_TXT).decode()


@pytest.mark.parametrize(
    "bad",
    [
        {"queue": {"length": 1, "next": [{"hotkey": "hk11", "groups": [{"prompt": "p"}]}]}},
        {"round": {"batch": "private/rounds/r000007/batches/hk11.json"}},
        {"history": [{"entries": [{"lora": "pending/rounds/r000007/l0/loras/000"}]}]},
        {"queue": {"length": 1, "next": [{"hotkey": "hk11", "put_url": "https://signed"}]}},
        {"contract": {}},
        {"head": None},
        {"params": {"netuid": 120, "no_such_param": 1}},
    ],
)
def test_private_or_malformed_content_never_reaches_the_public_files(tmp_path, bad):
    store = MemStore()
    with pytest.raises(ContractError):
        published(world(tmp_path).publisher.snapshots[-1] | bad, store)
    assert layout.MINING_JSON not in store and layout.LLMS_TXT not in store and layout.WINNER_STATS not in store


def test_a_head_is_named_only_after_its_manifest_is_public(tmp_path):
    store = MemStore()
    with pytest.raises(InfraError, match="not public yet"):
        publisher.StorePublisher(store).publish(world(tmp_path).publisher.snapshots[-1])
    assert not store


def test_a_bad_write_is_an_error_not_a_silent_stale_contract(tmp_path):
    class Corrupting(MemStore):
        def sha256(self, key: str) -> str:
            return "0" * 64

    class Down(MemStore):
        def put(self, key: str, data: bytes) -> None:
            raise OSError("network")

    snap = world(tmp_path).publisher.snapshots[-1]
    with pytest.raises(InfraError, match="read back"):
        published(snap, Corrupting())
    with pytest.raises(InfraError, match="cannot publish"):
        published(snap, Down())
