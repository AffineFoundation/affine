"""``queue.files`` of the public snapshot: the verdict on the stored file of every hotkey that has a waiting seat.

``queue.next`` shows the first 50 seats. A reader that counts the valid files of the whole line needs the other
seats too. ``queue.files`` has one row for each hotkey, and the counts.

What these tests hold:

* the counts over the whole line, against files that are valid, not valid and not there;
* a file that is replaced after the check shows the verdict of the check until the next check;
* the time of the list is the start of the check, and no check runs inside a round;
* every key that the snapshot had before is the same, byte for byte;
* the id of the params is the same, and nothing else about a waiting file or a seat is published.
"""

from __future__ import annotations

import json
from pathlib import Path

from sn120_v2 import layout, public, publisher
from sn120_v2.admission import StoreAdmission
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.params import Params, Recipe
from sn120_v2.public import NOT_JUDGED, NOT_VALID, VALID, file_verdict, files_view
from sn120_v2.queue import waiting
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.store import LocalStore
from sn120_v2.types import Verdict

SEATS = 60  # more than ``public.QUEUE_PREVIEW``
ROW = ("uid", "hotkey", "file_valid", "file_verdict")
FILES = ("checked_at", "seats", "hotkeys", VALID, NOT_VALID, NOT_JUDGED, "verdicts")
ROW_OF_NEXT = ("position", "uid", "hotkey", "registration_block", "file_valid", "front", "demoted", "held")

# Both values are from the code of v2 at 580aee91, the commit before ``queue.files``. This change adds one key and
# changes nothing else. A later change that changes a param or the snapshot on purpose puts its new value here.
PARAMS_HASH_BEFORE = "65ce6ef3bc867172eb87673bdb39461d03a8c85aac743c61e8b3d233b9f26d54"
PARAMS_BEFORE = Path(__file__).parent / "data" / "params_default_a5c2b6a6.json"
SNAPSHOT_BEFORE = "a5ffb03f2dd84bfc420f222fb685871452c21c6957546520e4b690d934909021"  # sha256 of the snapshot of ``line_world``


def kind(uid: int) -> str:
    """The file of a seat in ``line_world``: one seat in three has a valid file, one a file that is not valid, one no file."""
    return (NOT_JUDGED, VALID, NOT_VALID)[uid % 3]


def params_before() -> Params:
    """The sim params with the defaults of that time: the 4 caps of a file, no cap on the update, the 7 bounds of a miner."""
    bounds = tuple(bound for bound in Recipe().miner_bounds if bound.key not in ("minibatches", "global_norm_cap"))
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
    assert files["checked_at"] == queue["files_checked_at"] is not None


def test_the_three_verdicts():
    assert file_verdict(None) == NOT_JUDGED == "not_judged"
    assert file_verdict(Verdict("hk", True)) == VALID == "valid"
    assert file_verdict(Verdict("hk", False, "breaks a rule", rule="json")) == NOT_VALID == "not_valid"
    # A fault of our machine shows as file_valid shows it today: not valid. Which rule a file breaks is not published.
    assert file_verdict(Verdict("hk", False, "the grader gave no answer", rule="grader_fault")) == NOT_VALID


def test_the_rows_are_in_uid_order_and_show_no_place_and_no_flag_of_a_seat(build):
    """The queue order puts a seat at the front first and a demoted seat last. The rows do not follow it."""
    line = waiting(build.queue(build.seat(7, front=True), build.seat(2, demoted=True), build.seat(5, held=True), build.seat(3)))
    assert [seat.reg.uid for seat in line] == [7, 3, 5, 2], "the queue order"
    flagged = files_view(line, build.verdicts("hk3", "hk7"), 9)
    plain = files_view(waiting(build.queue(*(build.seat(uid) for uid in (7, 2, 5, 3)))), build.verdicts("hk3", "hk7"), 9)
    assert [row["uid"] for row in flagged["verdicts"]] == [2, 3, 5, 7]
    assert canonical_json(flagged) == canonical_json(plain), "the flags and the places of the seats change no byte of queue.files"
    assert counts(flagged) == (2, 0, 2) and flagged["checked_at"] == 9


def test_a_hotkey_with_two_waiting_seats_has_one_row_and_its_file_is_counted_once(build):
    line = waiting(build.queue(build.seat(4, hotkey="same"), build.seat(1, hotkey="same"), build.seat(3)))
    files = files_view(line, build.verdicts("same"), 9)
    assert files["seats"] == 3 and files["hotkeys"] == 2 and counts(files) == (1, 0, 1)
    assert files["verdicts"] == [
        {"uid": 1, "hotkey": "same", "file_valid": True, "file_verdict": VALID},
        {"uid": 3, "hotkey": "hk3", "file_valid": False, "file_verdict": NOT_JUDGED},
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
    assert counts(after) == (0, 3, 1) and after["verdicts"][0] == {"uid": 1, "hotkey": "hk1", "file_valid": False, "file_verdict": NOT_VALID}

    text = json.dumps(files) + json.dumps(after)
    assert not any(verdict.batch_sha256 in text for verdict in checked.values() if verdict.batch_sha256), "no sha256 of a waiting file"
    assert checked["hk2"].rule not in text.replace("hotkey", "") and "reason" not in text


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


def test_every_key_that_the_snapshot_had_before_is_the_same_byte_for_byte(tmp_path):
    world = line_world(tmp_path, params_before())  # the hash below is of a snapshot with the params of that time
    snap = json.loads(json.dumps(last(world)))
    files = snap["queue"].pop("files")
    assert list(snap["queue"]) == ["length", "files_checked_at", "next"]
    assert sha256_hex(canonical_json(snap)) == SNAPSHOT_BEFORE, "the snapshot without queue.files is the snapshot of the code before"

    # The same for the queue alone, from the state: the rows as the code before made them.
    valid = {row["hotkey"]: row["file_verdict"] == VALID for row in files["verdicts"]}
    before = [
        {
            "position": position,
            "uid": seat.reg.uid,
            "hotkey": seat.reg.hotkey,
            "registration_block": seat.reg.registration_block,
            "file_valid": valid[seat.reg.hotkey],
            "front": seat.front,
            "demoted": seat.demoted,
            "held": seat.held,
        }
        for position, seat in enumerate(waiting(world.controller.state.queue)[: public.QUEUE_PREVIEW])
    ]
    assert json.dumps(snap["queue"]["next"]) == json.dumps(before)
    assert all(tuple(row) == ROW_OF_NEXT == publisher.QUEUE_COLUMNS for row in snap["queue"]["next"]), "no new key in a row of queue.next"


def test_the_id_of_the_params_is_the_same(tmp_path):
    # The defaults of that time (``data/params_default_a5c2b6a6.json``). Later defaults differ: other caps, a cap on the update.
    assert Params.from_json(PARAMS_BEFORE.read_bytes()).params_hash == PARAMS_HASH_BEFORE, "this change adds no param and changes none"
    world = line_world(tmp_path)
    snap = last(world)
    assert snap["params_hash"] == world.params.params_hash == Params.from_dict(snap["params"]).params_hash
    assert snap["params"] == world.params.to_dict(), "queue.files is no part of the params"


def test_nothing_else_about_a_waiting_file_or_a_seat_is_published(tmp_path):
    world = line_world(tmp_path)
    files = last(world)["queue"]["files"]
    text = json.dumps(files)
    stored = [sha256_hex(world.store.get(layout.miner_object(f"hk{uid}"))) for uid in range(1, SEATS + 1) if kind(uid) != NOT_JUDGED]
    assert len(stored) == 40 and not any(sha in text for sha in stored), "the sha256 of a waiting file is not public"
    assert not any(key in text for key in ("batch_sha256", "rule", "reason", "position", "registration_block", "front", "demoted", "held"))
    assert [row["uid"] for row in files["verdicts"]] == list(range(1, SEATS + 1)), "UID order"


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
    assert "| position | uid | hotkey | registration_block | file_valid | front | demoted | held |" in text, "the table of llms.txt has the columns of before"
    assert "file_verdict |" not in text

    # Without the key, only the digest of the snapshot differs in llms.txt.
    without = {**snap, "queue": {key: value for key, value in snap["queue"].items() if key != "files"}}
    old_mining, old_llms = publisher.render(without, "https://example.org", "https://example.org/links")
    digest, old_digest = doc["snapshot_digest"], json.loads(old_mining)["snapshot_digest"]
    assert digest != old_digest and text.replace(digest, old_digest) == old_llms.decode()
