"""``queue.files`` of the public snapshot: the verdict on the stored file of every hotkey that has a waiting seat.

``queue.next`` shows the first 50 seats. A reader that counts the valid files of the whole line needs the other
seats too. ``queue.files`` has one row for each hotkey, and the counts.

What these tests hold:

* the counts over the whole line, against files that are valid, not valid and not there;
* a file that is replaced after the check shows the verdict of the check until the next check;
* the time of the list is the start of the check, and no check runs inside a round;
* the id of the params is unchanged, and waiting publication remains SHA256 + verdict check time only;
* the full ordered line gives valid-file seats ahead and a coarse public-parameter round bound;
* pending chain registrations have identity only, never file fields or URLs.
"""

from __future__ import annotations

import json
from pathlib import Path

from sn120_v2 import layout, public, publisher
from sn120_v2.admission import StoreAdmission
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.params import ATTENTION_AND_MLP_PROJECTIONS, MinerBound, Params, Recipe
from sn120_v2.public import NOT_JUDGED, NOT_VALID, VALID, file_verdict, files_view
from sn120_v2.queue import waiting
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import State
from sn120_v2.store import LocalStore
from sn120_v2.types import ChainSnapshot, Registration, Verdict

SEATS = 60  # more than ``public.QUEUE_PREVIEW``
ROW = ("uid", "hotkey", "file_valid", "file_verdict", "batch_sha256", "checked_at", "valid_ahead", "eta_rounds")
FILES = ("checked_at", "seats", "hotkeys", VALID, NOT_VALID, NOT_JUDGED, "verdicts")
ROW_OF_NEXT = ("position", "uid", "hotkey", "registration_block", "file_valid", "file_verdict", "batch_sha256", "checked_at",
               "front", "demoted", "held", "valid_ahead", "eta_rounds")

# Both values are from the code of v2 at 580aee91, the commit before ``queue.files``. This change adds one key and
# changes nothing else. A later change that changes a param or the snapshot on purpose puts its new value here.
PARAMS_HASH_BEFORE = "65ce6ef3bc867172eb87673bdb39461d03a8c85aac743c61e8b3d233b9f26d54"
PARAMS_BEFORE = Path(__file__).parent / "data" / "params_default_a5c2b6a6.json"
SNAPSHOT_BEFORE = "76a2ad84c7f938fb6deca1e5597a4d523380e33cde166a49c7e60ae198361716"  # sha256 of the snapshot of ``line_world``


def kind(uid: int) -> str:
    """The file of a seat in ``line_world``: one seat in three has a valid file, one a file that is not valid, one no file."""
    return (NOT_JUDGED, VALID, NOT_VALID)[uid % 3]


def params_before() -> Params:
    """The sim params with the defaults of that time: the 4 caps of a file, no cap on the update, the 7 bounds of a miner.

    ``target_modules`` had the 7 names as its choices.
    """
    targets = MinerBound("target_modules", choices=ATTENTION_AND_MLP_PROJECTIONS, decided=True)
    today = (bound for bound in Recipe().miner_bounds if bound.key not in ("minibatches", "global_norm_cap"))
    bounds = tuple(targets if bound.key == "target_modules" else bound for bound in today)
    recipe = Recipe(image_digests=sim_params().recipe.image_digests, global_norm_cap=None, miner_bounds=bounds)
    return sim_params(max_batch_bytes=64 * 1024 * 1024, max_prompts=250, max_rollouts=2000, max_train_token_steps=8_000_000, recipe=recipe)


def line_world(root: Path, params: Params | None = None) -> SimWorld:
    world = SimWorld(root, params or sim_params())
    for uid in range(1, SEATS + 1):
        world.register(uid, None if kind(uid) == NOT_JUDGED else 0.0, **({"valid": False} if kind(uid) == NOT_VALID else {}))
    assert world.controller.publish(world.clock(), check_files=True)
    return world


def last(world: SimWorld) -> dict:
    return world.publisher.snapshots[-1]


def verdicts(world: SimWorld) -> dict[str, str]:
    return {row["hotkey"]: row["file_verdict"] for row in last(world)["queue"]["files"]["verdicts"]}


def counts(files: dict) -> tuple[int, int, int]:
    return files[VALID], files[NOT_VALID], files[NOT_JUDGED]


# ------------------------------------------------------------ the counts and the rows


def test_a_reader_counts_the_valid_files_of_the_whole_line(tmp_path):
    queue = last(line_world(tmp_path))["queue"]
    files = queue["files"]
    assert tuple(files) == FILES and all(tuple(row) == ROW for row in files["verdicts"])
    assert files["seats"] == queue["length"] == SEATS and files["hotkeys"] == len(files["verdicts"]) == SEATS
    assert counts(files) == (20, 20, 20), "hotkeys with a valid file, with a file that is not valid, with no verdict"
    assert {row["hotkey"]: row["file_verdict"] for row in files["verdicts"]} == {f"hk{uid}": kind(uid) for uid in range(1, SEATS + 1)}
    assert all(row["file_valid"] is (row["file_verdict"] == VALID) for row in files["verdicts"]), "file_valid is true only with a valid file"
    assert len(queue["next"]) == public.QUEUE_PREVIEW and sum(row["file_valid"] for row in queue["next"]) == 17, "the first 50 seats alone give a smaller count"
    assert queue["line"][:public.QUEUE_PREVIEW] == queue["next"] and len(queue["line"]) == SEATS
    assert [row["position"] for row in queue["line"]] == list(range(SEATS))
    assert [row["valid_ahead"] for row in queue["line"][:6]] == [0, 1, 1, 1, 2, 2]
    assert queue["line"][-1]["position"] == 59 and queue["line"][-1]["valid_ahead"] == 20
    assert queue["line"][-1]["eta_rounds"] == 5  # ceil(20 / sim batch_cap=4), no private scheduling input
    assert all(row["eta_rounds"] == (row["valid_ahead"] + 3) // 4 for row in queue["line"])
    assert files["checked_at"] == queue["files_checked_at"] is not None


def test_the_three_verdicts():
    assert file_verdict(None) == NOT_JUDGED == "not_judged"
    assert file_verdict(Verdict("hk", True)) == VALID == "valid"
    assert file_verdict(Verdict("hk", False, "breaks a rule", rule="json")) == NOT_VALID == "not_valid"
    # A fault of our machine shows as file_valid shows it today: not valid. Which rule a file breaks is not published.
    assert file_verdict(Verdict("hk", False, "the grader gave no answer", rule="grader_fault")) == NOT_VALID


def test_the_rows_are_in_uid_order_while_valid_ahead_uses_real_queue_order(build):
    """The queue order puts a seat at the front first and a demoted seat last; files stay UID ordered."""
    line = waiting(build.queue(build.seat(7, front=True), build.seat(2, demoted=True), build.seat(5, held=True), build.seat(3)))
    assert [seat.reg.uid for seat in line] == [7, 3, 5, 2], "the queue order"
    flagged = files_view(line, build.verdicts("hk3", "hk7"), 9, 4)
    plain = files_view(waiting(build.queue(*(build.seat(uid) for uid in (7, 2, 5, 3)))), build.verdicts("hk3", "hk7"), 9, 4)
    assert [row["uid"] for row in flagged["verdicts"]] == [2, 3, 5, 7]
    assert [row["valid_ahead"] for row in flagged["verdicts"]] == [2, 1, 2, 0]
    assert [row["valid_ahead"] for row in plain["verdicts"]] == [0, 0, 1, 1]
    assert counts(flagged) == (2, 0, 2) and flagged["checked_at"] == 9


def test_a_waiting_row_has_only_its_own_sha_and_verdict_time(build):
    valid = Verdict("hk1", True, "never public", format="v2", n_prompts=8, n_rollouts=16, batch_sha256="a" * 64, checked_at=123, rule="")
    invalid = Verdict("hk2", False, "never public", format="v2", n_prompts=8, n_rollouts=16, batch_sha256="b" * 64, checked_at=124, rule="json")
    files = files_view(waiting(build.queue(build.seat(1), build.seat(2), build.seat(3))), {"hk1": valid, "hk2": invalid}, 124, 2)
    assert files["verdicts"] == [
        {"uid": 1, "hotkey": "hk1", "file_valid": True, "file_verdict": VALID, "batch_sha256": "a" * 64, "checked_at": 123, "valid_ahead": 0, "eta_rounds": 0},
        {"uid": 2, "hotkey": "hk2", "file_valid": False, "file_verdict": NOT_VALID, "batch_sha256": "b" * 64, "checked_at": 124, "valid_ahead": 1, "eta_rounds": 1},
        {"uid": 3, "hotkey": "hk3", "file_valid": False, "file_verdict": NOT_JUDGED, "batch_sha256": None, "checked_at": None, "valid_ahead": 1, "eta_rounds": 1},
    ]
    text = json.dumps(files)
    assert all(value not in text for value in ("never public", "format", "n_prompts", "n_rollouts", "reason", "rule", "uri", "url"))


def test_a_hotkey_with_two_waiting_seats_has_one_row_and_its_file_is_counted_once(build):
    line = waiting(build.queue(build.seat(4, hotkey="same"), build.seat(1, hotkey="same"), build.seat(3)))
    files = files_view(line, {"same": Verdict("same", True, batch_sha256="c" * 64, checked_at=9)}, 9, 4)
    assert files["seats"] == 3 and files["hotkeys"] == 2 and counts(files) == (1, 0, 1)
    assert files["verdicts"] == [
        {"uid": 1, "hotkey": "same", "file_valid": True, "file_verdict": VALID, "batch_sha256": "c" * 64, "checked_at": 9, "valid_ahead": 0, "eta_rounds": 0},
        {"uid": 3, "hotkey": "hk3", "file_valid": False, "file_verdict": NOT_JUDGED, "batch_sha256": None, "checked_at": None, "valid_ahead": 1, "eta_rounds": 1},
    ]


def test_an_empty_line_and_a_line_before_the_first_check(build):
    assert files_view([], {}, None) == {"checked_at": None, "seats": 0, "hotkeys": 0, VALID: 0, NOT_VALID: 0, NOT_JUDGED: 0, "verdicts": []}
    files = files_view(waiting(build.queue(build.seat(1), build.seat(2))), {}, None)
    assert files["checked_at"] is None and counts(files) == (0, 0, 2), "no check yet: no file is counted as valid or as not valid"


# ------------------------------------------------------------ the real checker's verdicts


class WordTokens:
    prompt_tokens = output_tokens = staticmethod(lambda text: len(text.split()))


def real_file(hotkey: str, answer: str = "2") -> bytes:
    output = "<think>" + "one and one is two " * 10 + f"</think> \\boxed{{{answer}}}"
    rollouts = [{"output": output, "reward": 1.0}, {"output": output + " again", "reward": 0.0}]
    return json.dumps({"version": "affine-batch-v2", "hotkey": hotkey, "groups": [{"prompt": "What is 1+1?", "rollouts": rollouts}]}).encode()


def test_the_counts_from_the_verdicts_of_the_real_checker(build, tmp_path):
    """Files in a store: valid, broken JSON, too large, and none. Then a valid file is replaced after the check."""
    store, params = LocalStore(tmp_path), build.params()
    admission = StoreAdmission(store, params, tokens=WordTokens(), grader=None, clock=lambda: 5)
    line = waiting(build.queue(*(build.seat(uid) for uid in (1, 2, 3, 4))))
    hotkeys = [seat.reg.hotkey for seat in line]
    store.put(layout.miner_object("hk1"), real_file("hk1"))
    store.put(layout.miner_object("hk2"), b"{ not json")
    store.put(layout.miner_object("hk3"), b"x" * (params.max_batch_bytes + 1))
    checked = dict(admission.verdicts(hotkeys))
    files = files_view(line, checked, 5)
    assert [row["file_verdict"] for row in files["verdicts"]] == [VALID, NOT_VALID, NOT_VALID, NOT_JUDGED], checked
    assert counts(files) == (1, 2, 1) and files["hotkeys"] == files["seats"] == 4

    store.put(layout.miner_object("hk1"), b"{ not json")  # the file is replaced after the check
    assert files_view(line, checked, 5) == files, "the verdicts of the check: a snapshot reads no file"
    after = files_view(line, dict(admission.verdicts(hotkeys)), 6)
    assert counts(after) == (0, 3, 1) and after["verdicts"][0]["file_verdict"] == NOT_VALID
    assert after["verdicts"][0]["checked_at"] == 5 and after["verdicts"][-1]["checked_at"] is None

    text = json.dumps(files) + json.dumps(after)
    assert all(verdict.batch_sha256 in text for verdict in checked.values() if verdict.batch_sha256), "the authorized SHA of each judged file"
    assert checked["hk2"].rule not in text.replace("hotkey", "") and "reason" not in text
    assert '"checked_at":5' in text.replace(" ", "") and '"checked_at":6' in text.replace(" ", "")  # outer second check starts at 6; its verdict clock is 5


# ------------------------------------------------------------ pending chain registrations


def test_pending_is_from_the_existing_chain_read_and_has_no_file_fields(tmp_path):
    world = SimWorld(tmp_path, sim_params(batch_cap=1))
    world.register(1, 0.0)
    tickets = world.controller.open_round(world.clock(), "entropy")
    assert tickets and world.controller.state.queue[(1, 1001)].leased
    chain = ChainSnapshot(world.params.netuid, 9, world.now, (
        Registration(0, "burn", 1, coldkey="private-owner"),
        Registration(1, "hk1", 1001, coldkey="private-owner"),  # leased, not pending
        Registration(2, "hk2", 1002, registered_at=123, coldkey="private-owner"),
    ))
    before = len(world.publisher.snapshots)
    # Before the ordinary registration sync seats UID 2, the existing validated chain read gives
    # exactly the identity-only pending projection; it adds no chain query or second queue.
    pending = public.pending_registrations(world.controller.state, chain, world.params)
    assert pending == [{"uid": 2, "hotkey": "hk2", "registration_block": 1002, "status": "pending"}]
    assert list(pending[0]) == ["uid", "hotkey", "registration_block", "status"]
    assert not any(key in json.dumps(pending) for key in ("file", "sha", "checked", "coldkey", "registered_at", "url", "uri"))
    world.controller.sync_registrations(chain, world.clock())
    assert len(world.publisher.snapshots) == before, "sync does not add a publication or change publication timing"
    assert world.controller.publish(world.clock())
    assert world.publisher.snapshots[-1]["queue"]["pending"] == []  # the ordinary sync seated UID 2
    assert public.snapshot(world.controller.state, world.params, world.now)["queue"]["pending"] == []
    assert public.pending_registrations(world.controller.state, chain, world.params) == []  # the ordinary sync seated UID 2
    world.controller.state.served_hotkeys.update(("hk1", "hk2"))
    world.controller.state.queue.clear()
    assert public.pending_registrations(world.controller.state, chain, world.params) == []


def test_pending_does_not_count_leased_seats_or_change_valid_ahead(tmp_path, build):
    state = State(queue=build.queue(build.seat(1), build.seat(2)))
    state.queue = build.queue(build.seat(1), build.seat(2))
    state.queue[(1, 101)] = build.seat(1, leased=True)
    rows = public.line_view(waiting(state.queue), {"hk1": Verdict("hk1", True), "hk2": Verdict("hk2", True)}, 4)
    assert [row["uid"] for row in rows] == [2] and rows[0]["valid_ahead"] == 0


# ------------------------------------------------------------ the time of the list


def test_a_file_that_is_replaced_after_the_check_shows_the_verdict_of_the_check_until_the_next_check(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.0)
    world.register(2, 0.0, valid=False)
    world.register(3, None)
    assert world.controller.publish(world.clock(), check_files=True)
    checked = last(world)["queue"]["files"]["checked_at"]
    assert verdicts(world) == {"hk1": VALID, "hk2": NOT_VALID, "hk3": NOT_JUDGED}

    world.admission.upload("hk1", 0.0, valid=False)  # a valid file is replaced by one that is not valid
    world.admission.upload("hk3", 0.0)  # a first file
    world.register(4, 0.0)  # a seat that comes after the check, with a valid file
    assert world.controller.publish(world.clock())
    files = last(world)["queue"]["files"]
    assert files["checked_at"] == checked == last(world)["queue"]["files_checked_at"], "no new check: the time of the list is the time of the old check"
    assert verdicts(world) == {"hk1": VALID, "hk2": NOT_VALID, "hk3": NOT_JUDGED, "hk4": NOT_JUDGED}
    assert counts(files) == (1, 1, 2) and files["seats"] == 4, "the counts are those of the check, with the new seat not judged"

    assert world.controller.publish(world.clock(), check_files=True)
    files = last(world)["queue"]["files"]
    assert files["checked_at"] > checked and counts(files) == (2, 2, 0)
    assert verdicts(world) == {"hk1": NOT_VALID, "hk2": NOT_VALID, "hk3": VALID, "hk4": VALID}


def test_the_time_of_the_list_is_the_start_of_the_check(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.0)
    real, started = world.admission.verdicts, []

    def slow(hotkeys):
        started.append(world.now)
        world.now += 1000  # reading every waiting file takes time
        return real(hotkeys)

    world.admission.verdicts = slow
    assert world.controller.publish(world.clock(), check_files=True)
    files = last(world)["queue"]["files"]
    assert files["checked_at"] == started[0] < world.now - 999, "the clock is read before the files are read"


def test_no_check_runs_inside_a_round_so_the_verdicts_are_as_old_as_the_round(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    for uid in range(1, 7):
        world.register(uid, 0.0)
    world.controller.open_round(world.clock(), "entropy")
    assert world.controller.publish(world.clock())
    snap = last(world)
    files, taken = snap["queue"]["files"], {entry["hotkey"] for entry in snap["round"]["entries"]}
    assert snap["phase"] == "RUNNING" and len(taken) == world.params.batch_cap
    assert [row["hotkey"] for row in files["verdicts"]] == ["hk5", "hk6"], "a seat that the round took does not wait: it has no row"
    assert files["seats"] == snap["queue"]["length"] == 2 and counts(files) == (2, 0, 0)
    assert files["checked_at"] <= snap["round"]["opened_at"], "the check that picked the round"

    world.admission.upload("hk5", 0.0, valid=False)  # a new file while the round runs
    world.now += 5000
    assert world.controller.publish(world.clock())
    later = last(world)["queue"]["files"]
    assert later == files and last(world)["generated_at"] > snap["generated_at"] + 5000, "the same list with the same time, however long the round runs"


def test_a_new_process_shows_no_verdict_before_its_first_check(tmp_path):
    world = line_world(tmp_path)
    world.reopen()
    assert world.controller.publish(world.clock())
    files = last(world)["queue"]["files"]
    assert files["checked_at"] is None and counts(files) == (0, 0, SEATS)
    assert all(row["file_verdict"] == NOT_JUDGED and not row["file_valid"] for row in files["verdicts"])


# ------------------------------------------------------------ what does not change


def test_the_snapshot_keeps_its_schema_and_explicit_safe_queue_keys(tmp_path):
    snap = last(line_world(tmp_path, params_before()))
    assert snap["schema"] == public.SCHEMA and list(snap["queue"]) == ["length", "files_checked_at", "next", "line", "pending", "files"]
    assert snap["queue"]["pending"] == [] and snap["queue"]["line"][:50] == snap["queue"]["next"]
    assert all(tuple(row) == ROW_OF_NEXT == publisher.QUEUE_COLUMNS for row in snap["queue"]["next"])
    assert all(tuple(row) == ROW for row in snap["queue"]["files"]["verdicts"])


def test_the_id_of_the_params_is_the_same(tmp_path):
    # The defaults of that time (``data/params_default_a5c2b6a6.json``). Later defaults differ: other caps, a cap on the update.
    assert Params.from_json(PARAMS_BEFORE.read_bytes()).params_hash == PARAMS_HASH_BEFORE, "this change adds no param and changes none"
    world = line_world(tmp_path)
    snap = last(world)
    assert snap["params_hash"] == world.params.params_hash == Params.from_dict(snap["params"]).params_hash
    assert snap["params"]["max_stages"] == 5 and snap["params"]["recipe"]["sft_weight"] == 1.0
    assert {key: value for key, value in snap["params"].items() if key not in ("max_stages", "recipe")} == {
        key: value for key, value in world.params.to_dict().items() if key not in ("max_stages", "recipe")
    }, "queue.files is no part of the params"
    assert {key: value for key, value in snap["params"]["recipe"].items() if key != "sft_weight"} == world.params.recipe.to_dict()


def test_nothing_else_about_a_waiting_file_or_a_seat_is_published(tmp_path):
    world = line_world(tmp_path)
    snap = last(world)
    files = snap["queue"]["files"]
    text = json.dumps(files)
    stored = [sha256_hex(world.store.get(layout.miner_object(f"hk{uid}"))) for uid in range(1, SEATS + 1) if kind(uid) != NOT_JUDGED]
    assert len(stored) == 40 and all(sha in text for sha in stored), "only the explicitly authorized stored-file SHA"
    assert not any(key in text for key in ("rule", "reason", "position", "registration_block", "front", "demoted", "held", "uri", "url", "train"))
    assert [row["uid"] for row in files["verdicts"]] == list(range(1, SEATS + 1)), "UID order"
    assert all(row["batch_sha256"] is None and row["checked_at"] is None for row in files["verdicts"] if row["file_verdict"] == NOT_JUDGED)
    assert all(row["batch_sha256"] is None and row["checked_at"] is None for row in snap["queue"]["line"] if row["file_verdict"] == NOT_JUDGED)
    publisher.refuse_private(snap["queue"])


def test_mining_json_has_the_list_and_llms_txt_uses_no_value_of_it(tmp_path):
    world = line_world(tmp_path)
    snap = last(world)
    mining, llms = publisher.render(snap, "https://example.org", "https://example.org/links")
    doc = json.loads(mining)
    assert doc["queue"]["files"] == snap["queue"]["files"] and doc["queue"]["next"] == snap["queue"]["next"]
    assert publisher.check_pair(mining, llms, world.params) == []
    text = llms.decode()
    words = " ".join(text.split())
    assert "under `queue.files`" in words and "`file_verdict` is `valid`, `not_valid` or `not_judged`" in words
    assert "| position | uid | hotkey | registration_block | file_valid | file_verdict | batch_sha256 | checked_at | front | demoted | held | valid_ahead | eta_rounds |" in text
    assert "Waiting-file publication is SHA256 + check time only." in words and "queue.pending" in words
    assert "ceil(valid_ahead / 4)" in words and "not a scheduler prediction" in words
    assert "The validator does not publish the sha256 of a waiting file" not in words

    # Without the key, only the digest of the snapshot differs in llms.txt.
    without = {**snap, "queue": {key: value for key, value in snap["queue"].items() if key != "files"}}
    old_mining, old_llms = publisher.render(without, "https://example.org", "https://example.org/links")
    digest, old_digest = doc["snapshot_digest"], json.loads(old_mining)["snapshot_digest"]
    assert digest != old_digest and text.replace(digest, old_digest) == old_llms.decode()
