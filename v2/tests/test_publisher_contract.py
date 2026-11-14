"""The publisher: llms.txt and mining.json come from one snapshot, and only from public facts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from sn120_v2 import layout, links, public, publisher, r2, sim
from sn120_v2.admission import RULES, TIME_LIMIT, TRANSIENT, check_batch
from sn120_v2.canonical import sha256_hex, to_jsonable
from sn120_v2.envs.math_pool import MANDATE
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
    snap = public.snapshot(controller.state, made.params, doc["generated_at"], controller._verdicts, controller._verdicts_at)
    assert {key: doc[key] for key in snap} == to_jsonable(snap)  # the website reads the snapshot's own keys
    assert doc["snapshot_digest"] in text and doc["head"]["head"]["version_id"] in text and doc["irt_snapshot_id"] in text
    assert [king["uid"] for king in doc["kings"]] == [11] and "| 11 | hk11 |" in text
    assert doc["queue"]["length"] == 1 and "| 0 | 12 | hk12 | 1012 | no |" in text


def test_both_files_come_from_one_snapshot_and_change_together(tmp_path):
    running, idle = world(tmp_path, play=True).publisher.snapshots[-2:]
    doc, text, store = published(running)
    assert doc["phase"] == "RUNNING" and "- phase: RUNNING" in text and f"- round_seed: {doc['round']['round_seed']}" in text
    assert doc["snapshot_digest"] in text and f"| 0 | 11 | hk11 | {doc['round']['entries'][0]['batch_sha256']} |" in text
    doc2, text2, _ = published(idle, store)
    assert doc2["snapshot_digest"] != doc["snapshot_digest"] and doc2["snapshot_digest"] in text2
    assert "- phase: IDLE" in text2 and doc["snapshot_digest"] not in text2


def test_every_miner_bound_is_published_from_the_records(tmp_path):
    bounds = tuple(
        MinerBound("steps", 2, 7, integer=True) if bound.key == "steps" else bound for bound in Recipe().miner_bounds
    )
    recipe = Recipe(image_digests=(sim.SIM_IMAGE,), steps=3, miner_bounds=bounds)
    doc, text, _ = published(world(tmp_path, recipe=recipe).publisher.snapshots[-1])
    rows = {row["key"]: row for row in doc["contract"]["miner_settable"]}
    assert list(rows) == ["lora_lr", "steps", "group_size", "sample_temperature", "rank", "alpha", "target_modules"]
    for bound in bounds:
        row = rows[bound.key]
        assert (row["low"], row["high"], row["choices"], row["decided"]) == (bound.low, bound.high, list(bound.choices), bound.decided)
        assert row["default"] == json.loads(json.dumps(getattr(recipe, bound.key)))
        assert f"| `train.{bound.key}` | " in text and row["allowed"] in text
    assert "| `train.steps` | 3 | a whole number from 2 to 7, both ends included |" in text
    assert "a non-empty subset of q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |" in text
    assert "placeholder" not in text.lower()  # the miner's text states the launch values as they are
    assert doc["contract"]["recipe_id"] == recipe.recipe_id and recipe.recipe_id in text


def test_every_number_follows_the_pinned_params(tmp_path):
    recipe = Recipe(image_digests=(sim.SIM_IMAGE,), train_max_output_tokens=777)
    changes = dict(max_prompts=123, max_rollouts=4567, z_bar=1.25, kings_window=7, n_tasks=400, looks=(0.25, 1.0), recipe=recipe)
    made = world(tmp_path, **changes)
    doc, text, _ = published(made.publisher.snapshots[-1])
    assert doc["contract"]["params_hash"] == made.params.params_hash
    assert "Caps: 123 prompts and 4567 rollouts" in text and "strictly above 1.25" in text and "last 7 distinct king" in text
    assert "after 100, 400 tasks" in text and "after 100 tasks." in text  # looks and the early-drop bar are computed
    assert "Training reads at most 777 output tokens" in text and "A longer rollout is valid: the trainer" in text
    header = doc["contract"]["batch_file"]["upload_headers"]
    assert header == r2.PUT_HEADERS and "`Content-Type: application/octet-stream`" in " ".join(text.split())  # what the presigner signs
    assert f"rule code `{TIME_LIMIT}`" in text
    for field in dataclasses.fields(made.params) + dataclasses.fields(recipe):  # the full table and the full recipe
        assert f"`{field.name}`" in text or field.name in ("recipe", "miner_bounds")
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
    assert "every B matrix is multiplied by cap / norm" in flat and "unset means no cap. It is unset now: there is no cap." in flat
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
    assert "The cap is now 2.5." in published(world(tmp_path / "capped", recipe=capped).publisher.snapshots[-1])[2].get(layout.LLMS_TXT).decode()


def test_the_times_of_a_round_and_the_queue_place_follow_the_params(tmp_path):
    changes = dict(round_time_limit_s=4800, lease_timeout_s=9000, promote_reserve_s=300, train_time_budget_s=1500)
    changes |= dict(eval_chunk_tasks=40, n_gpus=4, n_tasks=400, looks=(0.25, 1.0))
    changes |= dict(max_infra_attempts=5, squat_min_wait_s=7200, squat_age_factor=3.0, max_train_token_steps=123456)
    flat = " ".join(published(world(tmp_path, **changes).publisher.snapshots[-1])[1].split())
    assert "Time limit: 80 minutes from the lease to the promotion of the winner." in flat
    assert "The box stops scoring 5 minutes before it, at minute 75 (the stop time)" in flat  # worker.LocalRunner.run
    assert "ends early, as soon as every LoRA is dropped or decided" in flat
    assert "Training time: the first 25 minutes." in flat and "A slot is floor(4 / batches) GPUs, and at least 1." in flat
    assert "Such a batch is `not_trained`: you are not charged, your seat is first in the next round" in flat
    assert "the training time of the round (25 minutes for all batches, section 3) and this cost cap for one file (123456 token-steps)" in flat
    assert "`train_cost`: cost = steps x the tokens" in flat and "A file with a cost above 123456 token-steps is invalid." in flat
    no_cap = " ".join(published(world(tmp_path / "no-cap", max_train_token_steps=0).publisher.snapshots[-1])[1].split())
    assert "There is no cap on the cost now" in no_cap and "this cost cap for one file (no cap now)" in no_cap
    assert "Scoring runs in chunks of 40 tasks." in flat
    assert "at minute 75 (the stop time). The stop is hard." in flat and "The stop time cuts every look, the first one too." in flat
    assert "At the stop time, with the first look (100 tasks) complete: each open LoRA is judged by the win rule of section 6 on the whole chunks that null M and every open LoRA finished" in flat
    assert "At the stop time, with fewer than 100 tasks scored: each trained LoRA is `not_scored`. You are not charged, your seat is first in the next round, and no failure is counted for the seat." in flat
    assert "always scored in full" not in flat and "longer than 80 minutes only when" not in flat, "no exemption for the first look"
    assert "`not_scored`: the time limit of the round came before the first look of the scoring (section 3)." in flat
    assert "With fewer tasks than the first look, there is no decision: `not_scored` (section 3)." in flat
    assert "The lease ends 150 minutes after it starts." in flat
    assert "When 5 failures are counted for a seat, the seat is held" in flat and "so it never holds a seat" in flat
    assert "A seat that has a valid file stored keeps its place and is never moved back." in flat
    assert "older than 3 x its expected wait. Expected wait = max(7200 s, seats ahead x seconds per turn)." in flat
    assert "keeps its place for at least 21600 s after its registration, and for as long as no entry was charged in the last 24 hours" in flat
    assert "(`squat_min_wait_s`, `squat_age_factor`) are parameters (section 11)" in flat and "is announced" not in flat


def test_the_queue_text_gives_the_wait_in_days_and_says_what_removes_a_seat(tmp_path):
    for wait, first, total in ((31536000, "365 days", "730 days"), (86400, "1 day", "2 days"), (129600, "129600 s", "3 days"), (7201, "7201 s", "14402 s")):
        flat = " ".join(published(world(tmp_path / str(wait), squat_min_wait_s=wait).publisher.snapshots[-1])[1].split())
        assert f"older than 2 x its expected wait. Expected wait = max({first}, seats ahead x seconds per turn)." in flat
        assert f"keeps its place for at least {total} after its registration" in flat and "e+" not in flat.split("## 4.")[1].split("## 5.")[0]
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


def test_the_text_on_no_decision_says_what_the_controller_does_and_what_the_round_file_shows(tmp_path):
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

    made.trainer.fail_hotkeys = {"hk3"}  # one training fails and the others finish: the one case that is counted
    for count, other in ((1, None), (2, 4)):
        made.now += made.params.apply_retry_s
        if other:
            made.register(other, 0.0)
        report = json.loads(made.store.get(layout.round_file(made.round().round_id)))
        mine = next(entry for entry in report["entries"] if entry["uid"] == 3)
        assert report["outcome"] == "null" and report["note"] == "" and mine["status"] == "no_decision" and "delta" not in mine
        assert (state.queue[(3, 1003)].infra_attempts, state.queue[(3, 1003)].held) == (count, count == 2)
    assert "A failure is counted for your seat in one case only: the box could not train your batch (the training failed, or the batch had the whole training time and did not finish), and at least one other batch of the same round finished its training." in flat
    assert "Your entry then has the status `no_decision` and no scores. When 2 failures are counted for a seat, the seat is held: it takes no turn until the operator releases it." in flat
    made.controller.release_seat((3, 1003), made.now)
    assert "A release also sets the count to zero." in flat and (state.queue[(3, 1003)].infra_attempts, state.queue[(3, 1003)].held) == (0, False)

    assert "A `no_decision` that comes from the scoring (section 6) has scores and is not counted." in flat  # Controller._close: only a status of the box
    assert "If that is so for null M, every scored entry of the round is `no_decision`. Neither is counted as a failure for the seat (section 4)." in flat
    assert "A round that has a result also gets the outcome `no_decision` when none of its entries was charged." in flat
    assert "`no_decision` as the status of an entry means: not charged, and the seat goes to the front of the queue (section 4)." in flat
    assert "The `note` is empty unless the round had no result or has the outcome `abandoned` (section 4)." in flat
    assert "Do not rely on their exact words: the validator is free to cut a text short, and to put a marker in square brackets in the place of a part of it" in flat
    assert "To decide what applies to your seat, read `outcome` and `status`." in flat


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
    assert layout.MINING_JSON not in store and layout.LLMS_TXT not in store


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
