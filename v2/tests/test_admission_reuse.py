"""The two rules on reused rollouts: ``public_reuse`` and ``seat_reuse``.

* What makes two rollouts the same, which change of a copy the rules catch, and the one that they do not catch.
* The share: a file passes at the limit, and is refused one rollout above it.
* The validator and a miner compute the same verdict from the same public files.
* A new winner file makes a waiting copy not valid, with no new check of that copy.
* Between seats of one round, the seat ahead keeps its entry.
* The rules act on a file that is checked or frozen from now on. Nothing of a past round changes.
"""

from __future__ import annotations

import json
import shutil

import pytest

from sn120_v2 import assembly, layout, publisher, replicate
from sn120_v2.admission import PUBLIC_REUSE, RULES, SEAT_REUSE, Invalid, StoreAdmission, check_batch, freeze_log, verdict_note
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import ConfigError, InfraError
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.params import Params
from sn120_v2.reuse import KEY_BYTES, WinnerRollouts, file_keys, rollout_key, rollout_keys, round_name, store_reader
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import load
from sn120_v2.types import Verdict

THINK = "step " * 40  # 40 reasoning tokens: above the least
SHARE = 0.10


class WordTokens:
    calls = 0

    def prompt_tokens(self, prompt: str) -> int:
        self.calls += 1
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


class Store(dict):
    """The ``ObjectStore`` calls that admission makes, over a dict. ``reads`` holds the key of each ``get``."""

    put = dict.__setitem__

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reads: list[str] = []

    def get(self, key: str) -> bytes:
        self.reads.append(key)
        return self[key]

    def delete(self, key: str) -> None:
        self.pop(key, None)

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


def groups(tag: str, count: int = 10, start: int = 0) -> list[dict]:
    """``count`` prompts with 2 rollouts each, one right and one wrong. ``tag`` is in each output: two tags give no equal rollout."""
    return [
        {"prompt": f"double {n}", "rollouts": [{"output": f"{THINK}{tag} \\boxed{{{2 * n}}}", "reward": 1.0}, {"output": f"{THINK}{tag} \\boxed{{-1}}", "reward": 0.0}]}
        for n in range(start, start + count)
    ]


def file(hotkey: str, rows: list[dict], **more) -> bytes:
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": rows} | more).encode()


def winners(*files: bytes | None) -> dict[str, bytes]:
    """Winner files by round id, from round 1. None: that round has no winner."""
    return {round_name(number): raw for number, raw in enumerate(files, 1) if raw is not None}


def known(*files: bytes | None):
    return WinnerRollouts(winners(*files).get).known(len(files))


def refusal(raw: bytes, hotkey: str, params: Params, public) -> tuple[str, str]:
    """The rule code and the reason of the check, or two empty texts for a valid file."""
    try:
        batch = check_batch(raw, hotkey, params, tokens=WordTokens(), public=public)
    except Invalid as exc:
        return exc.rule, str(exc)
    assert batch.checked == ("format", "tokens", "reuse")
    return "", ""


def admission(build, store: Store, public: bool = True, **changes) -> StoreAdmission:
    reader = WinnerRollouts(store_reader(store)) if public else None
    return StoreAdmission(store, build.params(**changes), tokens=WordTokens(), grader=None, clock=lambda: 77.9, public=reader)


def upload(store: Store, hotkey: str, rows: list[dict]) -> bytes:
    raw = file(hotkey, rows)
    store[layout.miner_object(hotkey)] = raw
    return raw


def freeze_lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("freeze: ")]


# ------------------------------------------------------------ what "the same rollout" is


def test_the_key_of_a_rollout_is_its_prompt_and_its_output_with_no_white_space():
    assert len(rollout_key("p", "o")) == KEY_BYTES
    assert rollout_key("What is 1 + 1?", "It is\n\n  2.\t\\boxed{2}") == rollout_key(" Whatis1+1 ?\n", "Itis2.\\boxed{2}"), "white space is no part of the key"
    assert rollout_key("ab", "c") != rollout_key("a", "bc"), "the two texts are kept apart"
    assert rollout_key("p", "answer") != rollout_key("p", "answer."), "each other change gives another key"
    assert rollout_key("p", "answer") != rollout_key("p", "Answer") and rollout_key("p", "o") != rollout_key("q", "o")


def test_the_hotkey_the_version_train_the_reward_and_the_order_are_no_part_of_the_key():
    rows = groups("w")
    raw = file("5Winner", rows)
    edited = [{"prompt": row["prompt"], "rollouts": [{"output": r["output"], "reward": 1.0 - r["reward"]} for r in reversed(row["rollouts"])]} for row in reversed(rows)]
    other = json.dumps({"version": "affine-batch-v2", "hotkey": "5Copier", "train": {"rank": 16}, "groups": edited}).encode()
    assert sorted(file_keys(other)) == sorted(file_keys(raw)) and len(set(file_keys(raw))) == 20
    batch = check_batch(raw, "5Winner", Params(), tokens=WordTokens())
    assert rollout_keys(batch.groups) == file_keys(raw), "the checker and the reader of a published file give the same keys"


# ------------------------------------------------------------ what the rule stops, and what passes it


def test_a_copy_of_a_winner_file_is_refused_and_new_rollouts_on_the_same_prompts_pass(build):
    params, first, second = build.params(), groups("w1"), groups("w2", start=100)
    public = known(file("5King1", first), None, file("5King2", second))
    assert params.max_reused_rollout_frac == SHARE == Params().max_reused_rollout_frac
    limit = "; the limit is a share of 0.1"
    whole = (PUBLIC_REUSE, "20 of 20 rollouts are copies of rollouts in the published winner files, 20 of them in the file of round r000001" + limit)

    # 1. Only the hotkey is changed: what UID 961 and UID 15 did.
    assert refusal(file("5Copier", first), "5Copier", params, public) == whole
    # 2. The copy has other white space, other rewards, another order of its groups and its own ``train``.
    spaced = [{"prompt": row["prompt"] + "\n", "rollouts": [{"output": r["output"].replace(" ", "\n\n"), "reward": r["reward"]} for r in row["rollouts"]]} for row in reversed(first)]
    assert check_batch(file("5Copier", spaced, train={"rank": 16}), "5Copier", params) and refusal(file("5Copier", spaced, train={"rank": 16}), "5Copier", params, public) == whole
    # 3. One group of ten is new: 18 of 20 rollouts are copies.
    assert refusal(file("5Copier", first[:9] + groups("mine", 1, 9)), "5Copier", params, public)[1].startswith("18 of 20 rollouts are copies")
    # 4. Half of one winner file and half of another: all rounds count together.
    mixed = (PUBLIC_REUSE, "20 of 20 rollouts are copies of rollouts in the published winner files, 10 of them in the file of round r000001" + limit)
    assert refusal(file("5Copier", first[:5] + second[:5]), "5Copier", params, public) == mixed
    # 5. A second hotkey of the winner, and the winner's own hotkey: the file of a winner counts for every hotkey.
    assert refusal(file("5King1", first), "5King1", params, public) == whole, "no file wins two times"

    # What passes: the same prompts with rollouts of your own.
    assert refusal(file("5Honest", groups("mine")), "5Honest", params, public) == ("", "")
    assert check_batch(file("5Copier", first), "5Copier", params, tokens=WordTokens()).checked == ("format", "tokens"), "without the public files the rule does not run"


def test_the_known_limit_one_changed_character_in_each_rollout_passes(build):
    """The rules compare whole texts. A copy with one more character in each output, not white space, is a new file for them."""
    first = groups("w1")
    public = known(file("5King1", first))
    dotted = [{"prompt": row["prompt"], "rollouts": [{"output": r["output"] + ".", "reward": r["reward"]} for r in row["rollouts"]]} for row in first]
    assert refusal(file("5Copier", first), "5Copier", build.params(), public)[0] == PUBLIC_REUSE
    assert refusal(file("5Copier", dotted), "5Copier", build.params(), public) == ("", "")


def test_a_file_passes_at_the_share_and_is_refused_one_rollout_above_it(build):
    first = groups("w1")
    public = known(file("5King1", first))
    own = groups("mine")
    two = file("5M", first[:1] + own[1:])  # 2 of 20 rollouts are copies: the share is 0.10
    half = {"prompt": first[1]["prompt"], "rollouts": [first[1]["rollouts"][0], own[1]["rollouts"][1]]}  # one copied rollout, one new
    three = file("5M", first[:1] + [half] + own[2:])
    assert refusal(two, "5M", build.params(), public) == ("", "")
    assert refusal(three, "5M", build.params(), public) == (PUBLIC_REUSE, "3 of 20 rollouts are copies of rollouts in the published winner files, 3 of them in the file of round r000001; the limit is a share of 0.1")
    assert refusal(two, "5M", build.params(max_reused_rollout_frac=0.0), public)[0] == PUBLIC_REUSE, "a share of 0: no copied rollout at all"
    assert refusal(file("5M", own), "5M", build.params(max_reused_rollout_frac=0.0), public) == ("", "")
    assert refusal(file("5M", first), "5M", build.params(max_reused_rollout_frac=1.0), public) == ("", ""), "a share of 1 turns the rule off"
    for bad in (-0.01, 1.01):
        with pytest.raises(ConfigError, match="max_reused_rollout_frac"):
            build.params(max_reused_rollout_frac=bad)


def test_a_file_at_the_caps_can_hold_this_many_copied_rollouts():
    """A stated limit: copies below the share pass. With the default caps that is 200 rollouts in a file of 2000."""
    params = Params()
    assert int(params.max_reused_rollout_frac * params.max_rollouts) == 200


# ------------------------------------------------------------ the validator: the live check


def test_a_new_winner_file_makes_a_waiting_copy_not_valid_with_no_new_check(build):
    store = Store()
    king, copy = upload(store, "5King", groups("w1")), upload(store, "5Copier", groups("w1"))
    upload(store, "5Honest", groups("mine"))
    adm = admission(build, store)
    before = adm.verdicts(["5King", "5Copier", "5Honest"], rounds=0)
    assert all(verdict.valid for verdict in before.values()), "no winner file yet: the three files are valid"
    calls, reads = adm.tokens.calls, list(store.reads)

    store[layout.winner_batch("r000001")] = king  # round 1 closed, and 5King won it with this file
    after = adm.verdicts(["5King", "5Copier", "5Honest"], rounds=1)
    reason = "20 of 20 rollouts are copies of rollouts in the published winner files, 20 of them in the file of round r000001; the limit is a share of 0.1"
    for hotkey, raw in (("5Copier", copy), ("5King", king)):
        assert after[hotkey] == Verdict(hotkey, False, reason, batch_sha256=sha256_hex(raw), checked_at=77, rule=PUBLIC_REUSE)
    assert after["5Honest"] is before["5Honest"], "a valid file keeps its verdict"
    assert adm.tokens.calls == calls and store.reads == reads + [layout.winner_batch("r000001")], "one read of the winner file, and no file of a miner is read or checked again"

    assert adm.verdicts(["5King", "5Copier", "5Honest"], rounds=1) == after and store.reads == reads + [layout.winner_batch("r000001")], "a round is read once"
    upload(store, "5Copier", groups("new"))
    assert adm.verdicts(["5Copier"], rounds=1)["5Copier"].valid, "a new file of the same hotkey is judged for itself"


def test_the_live_check_without_the_winner_files_or_with_a_share_of_1_runs_no_reuse_rule(build):
    store = Store({layout.winner_batch("r000001"): file("5King", groups("w1"))})
    upload(store, "5Copier", groups("w1"))
    upload(store, "5Twin", groups("w1"))
    for adm in (admission(build, store, public=False), admission(build, store, max_reused_rollout_frac=1.0)):
        store.reads.clear()
        got = adm.verdicts(["5Copier", "5Twin"], rounds=1)
        assert got["5Copier"].valid and got["5Twin"].valid and layout.winner_batch("r000001") not in store.reads
    assert adm.freeze(build.reg(1, hotkey="5Copier"), "r000002") and adm.freeze(build.reg(2, hotkey="5Twin"), "r000002"), "a share of 1 turns the rule between seats off too"


def test_a_winner_file_that_cannot_be_read_stops_the_check_as_our_failure(build):
    store = Store({layout.winner_batch("r000001"): b"<html>not found</html>"})
    upload(store, "5Honest", groups("mine"))
    with pytest.raises(InfraError, match="the published winner file of round r000001 is not a batch file"):
        admission(build, store).verdicts(["5Honest"], rounds=1)
    store.put(layout.winner_batch("r000001"), json.dumps({"format": 1, "groups": [{"prompt": "a prompt with no rollouts"}]}).encode())
    with pytest.raises(InfraError, match="round r000001 is not a batch file: KeyError"):
        admission(build, store).verdicts(["5Honest"], rounds=1)


def test_a_published_json_object_with_no_groups_holds_no_rollout(build):
    """The winner file of a simulated store (``sim.SimAdmission``) is such an object. The validator publishes none."""
    simulated = json.dumps({"hotkey": "5King", "quality": 0.5, "cap_rate": 0.0, "train": {}, "valid": True}).encode()
    assert file_keys(simulated) == [] and len(file_keys(file("5King", groups("w1")))) == 20
    store = Store({layout.winner_batch("r000001"): simulated, layout.winner_batch("r000002"): file("5King", groups("w2"))})
    upload(store, "5Honest", groups("mine"))
    upload(store, "5Copier", groups("w2"))
    got = admission(build, store).verdicts(["5Honest", "5Copier"], rounds=2)
    assert got["5Honest"].valid and got["5Copier"].rule == PUBLIC_REUSE, "the round with rollouts still counts"


# ------------------------------------------------------------ the validator: the freeze, between seats of one round


def test_the_seat_ahead_keeps_its_entry_and_the_copy_behind_it_is_passed_over(build, capsys):
    store = Store()
    first, second, third = build.reg(1, hotkey="5First"), build.reg(2, hotkey="5Second"), build.reg(3, hotkey="5Third")
    upload(store, "5First", groups("shared"))
    copy = upload(store, "5Second", groups("shared"))
    upload(store, "5Third", groups("mine"))
    adm = admission(build, store)
    got = adm.verdicts(["5First", "5Second", "5Third"], rounds=0)
    assert all(verdict.valid for verdict in got.values()), "the live check does not compare the files of two seats: file_valid is yes for both"

    assert adm.freeze(first, "r000001").hotkey == "5First" and freeze_lines(capsys) == []
    assert adm.freeze(second, "r000001") is None and layout.frozen_batch("r000001", "5Second") not in store
    reason = "20 of 20 rollouts are copies of rollouts of a seat ahead in this round; the limit is a share of 0.1"
    refused, start = Verdict("5Second", False, reason, batch_sha256=sha256_hex(copy), rule=SEAT_REUSE), sha256_hex(copy)[:12]
    assert freeze_lines(capsys) == [freeze_log(2, "r000001", f"{start} valid, memory tag {start}", verdict_note(refused))], "the validator's log has the plain reason"
    assert verdict_note(refused) == f"{start} not valid, rule seat_reuse: {reason}"
    assert adm.freeze(third, "r000001").hotkey == "5Third"
    assert adm.freeze(first, "r000001").hotkey == "5First", "a second freeze of one hotkey is not compared with its own file"

    # The next walk of the queue starts with no seat ahead: the first seat was served, the copy has the first turn.
    assert adm.verdicts(["5Second", "5Third"], rounds=1)["5Second"].valid
    assert adm.freeze(second, "r000002").hotkey == "5Second"


def test_a_file_that_the_freeze_refuses_is_not_compared_with_the_seats_behind_it(build):
    store = Store()
    upload(store, "5First", groups("shared"))
    upload(store, "5Second", groups("shared") + groups("b", 4, 50))  # 20 of 28 rollouts are copies of the first seat
    upload(store, "5Third", groups("b", 4, 50) + groups("mine"))  # 8 of 28 rollouts are equal to rollouts of the second seat
    adm = admission(build, store)
    adm.verdicts(["5First", "5Second", "5Third"], rounds=0)
    taken = [adm.freeze(build.reg(uid, hotkey=hotkey), "r000001") is not None for uid, hotkey in enumerate(("5First", "5Second", "5Third"), 1)]
    assert taken == [True, False, True]


def test_the_freeze_runs_the_public_rule_on_the_bytes_that_it_copies(build, capsys):
    store = Store({layout.winner_batch("r000001"): file("5King", groups("w1"))})
    upload(store, "5Late", groups("mine"))
    adm = admission(build, store)
    assert adm.verdicts(["5Late"], rounds=1)["5Late"].valid
    upload(store, "5Late", groups("w1"))  # the miner wrote a copy of the winner file after the check
    assert adm.freeze(build.reg(4, hotkey="5Late"), "r000002") is None and layout.frozen_batch("r000002", "5Late") not in store
    (line,) = freeze_lines(capsys)
    assert f"rule {PUBLIC_REUSE}: 20 of 20 rollouts are copies of rollouts in the published winner files" in line


def test_a_seat_passed_over_for_seat_reuse_keeps_its_place_and_the_public_files_do_not_name_the_rule(tmp_path, capsys):
    """What a miner can read after rule ``seat_reuse`` passed his seat over: the round took a seat behind his, and his file is still valid."""
    world = SimWorld(tmp_path, sim_params())
    world.admission = StoreAdmission(world.store, world.params, tokens=WordTokens(), grader=None, public=WinnerRollouts(store_reader(world.store)))
    controller = world.reopen()
    for uid in (1, 2, 3):
        world.register(uid, None)
    for hotkey, tag in (("hk1", "shared"), ("hk2", "shared"), ("hk3", "mine")):
        world.store.put(layout.miner_object(hotkey), file(hotkey, groups(tag)))
    capsys.readouterr()
    (ticket,) = controller.open_round(world.clock(), "entropy")
    assert [entry.reg.hotkey for entry in ticket.entries] == ["hk1", "hk3"], "the seat ahead keeps its entry; the round goes on to the seat behind the copy"
    (line,) = freeze_lines(capsys)
    assert line.startswith("freeze: UID 2 gets no seat in round r000001.") and f"rule {SEAT_REUSE}: 20 of 20 rollouts are copies of rollouts of a seat ahead in this round" in line

    state = controller.state
    seat = next(seat for seat in state.queue.values() if seat.reg.hotkey == "hk2")
    assert state.round.skipped == (seat.reg.key,) and not seat.leased and not seat.demoted and not seat.held, "not charged: the seat waits at its place"
    snap = world.publisher.snapshots[-1]
    assert snap["round"]["round_id"] == "r000001" and snap["round"]["skipped"] == 1 and [entry["hotkey"] for entry in snap["round"]["entries"]] == ["hk1", "hk3"]
    (row,) = [row for row in snap["queue"]["next"] if row["hotkey"] == "hk2"]
    assert row["file_valid"] and row["position"] == 0, "the public queue: his file is valid and he is first"
    assert SEAT_REUSE not in json.dumps(snap) and "copies of rollouts" not in json.dumps(snap), "the rule and the reason of a waiting file are not published"


# ------------------------------------------------------------ the public files as an input: one read, a cache, the same keys


def test_the_winner_files_are_read_once_and_the_cache_only_saves_the_reading(tmp_path):
    files, asked = winners(file("5King1", groups("w1")), None, file("5King2", groups("w1")[:3] + groups("w3", 2, 100))), []

    def read(round_id: str) -> bytes | None:
        asked.append(round_id)
        return files.get(round_id)

    index = WinnerRollouts(read, tmp_path / "cache")
    got = dict(index.known(3))
    assert asked == ["r000001", "r000002", "r000003"] and len(got) == 24
    assert set(got.values()) == {"r000001", "r000003"} and list(got.values()).count("r000001") == 20, "a rollout belongs to the first round that published it"
    assert dict(index.known(3)) == got and dict(index.known(2)) == got and asked == ["r000001", "r000002", "r000003"]
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == ["r000001.keys", "r000002.keys", "r000003.keys"]
    assert (tmp_path / "cache" / "r000001.keys").read_bytes() == b"".join(file_keys(files["r000001"])) and (tmp_path / "cache" / "r000002.keys").read_bytes() == b""

    assert dict(WinnerRollouts(read, tmp_path / "cache").known(3)) == got and len(asked) == 3, "a new process reads the cache"
    (tmp_path / "cache" / "r000003.keys").write_bytes(b"damaged")
    assert dict(WinnerRollouts(read, tmp_path / "cache").known(3)) == got and asked[3:] == ["r000003"], "a cache file that is not whole keys is read again"
    shutil.rmtree(tmp_path / "cache")
    assert dict(WinnerRollouts(read, tmp_path / "cache").known(3)) == got and asked[4:] == ["r000001", "r000002", "r000003"], "the store is the record: the cache is rebuilt from it"
    assert dict(WinnerRollouts(read).known(3)) == got, "with no cache"
    (tmp_path / "no-directory").write_bytes(b"a file where the cache directory must be")
    assert dict(WinnerRollouts(read, tmp_path / "no-directory").known(3)) == got, "a cache that cannot be written stops nothing"


def test_a_reader_that_can_be_behind_asks_again_for_a_new_round_with_no_file(tmp_path):
    files, asked = winners(file("5King1", groups("w1"))), []

    def read(round_id: str) -> bytes | None:
        asked.append(round_id)
        return files.get(round_id)

    assert len(WinnerRollouts(read, tmp_path).known(3, settled=1)) == 20
    assert sorted(path.name for path in tmp_path.iterdir()) == ["r000001.keys"], "a round above ``settled`` with no file is not kept as a round with no winner"
    files |= {"r000003": file("5King3", groups("w3"))}  # the site has the file now
    assert len(WinnerRollouts(read, tmp_path).known(3, settled=1)) == 40 and asked == ["r000001", "r000002", "r000003", "r000002", "r000003"]


# ------------------------------------------------------------ llms.txt and the rule table


def test_llms_txt_states_the_rule_with_its_share_and_the_pair_check_passes(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    mining, llms = publisher.render(world.publisher.snapshots[-1], "https://site.example", "https://site.example/links")
    text = llms.decode()
    assert publisher.check_pair(mining, llms) == [] and publisher.check_pair(mining, llms, world.params) == []
    assert json.loads(mining)["params"]["max_reused_rollout_frac"] == SHARE
    assert "- Reuse: the published file of each round winner (section 9) is public, and a file that reuses its rollouts is\n  refused." in text
    assert "At most 0.1 of your rollouts may be the same as rollouts in the published winner files" in " ".join(text.split())
    assert "no file wins two times" in text
    for code in (PUBLIC_REUSE, SEAT_REUSE):
        assert f"| `{code}` | {RULES[code]} |" in text, "the table of the rule codes has the plain meaning"
    assert list(RULES).index("pool") < list(RULES).index(PUBLIC_REUSE) < list(RULES).index(SEAT_REUSE) < list(RULES).index("check_fault")

    off = SimWorld(tmp_path / "off", sim_params(max_reused_rollout_frac=1.0))
    mining, llms = publisher.render(off.publisher.snapshots[-1], "https://site.example", "https://site.example/links")
    assert publisher.check_pair(mining, llms) == []
    assert "- Reuse: no rule on reused rollouts is in force now (`max_reused_rollout_frac` is 1)." in llms.decode() and "no file wins two times" not in llms.decode()
    edited = text.replace("At most 0.1 of your rollouts", "At most 0.5 of your rollouts")
    assert edited != text and publisher.check_pair(json.dumps(json.loads(mining) | {}).encode(), edited.encode()) != [], "a text with another share is not the text of these params"


# ------------------------------------------------------------ not backward: nothing of a past round changes


def test_the_rule_changes_nothing_of_a_round_that_ran_before_it(tmp_path, capsys):
    """Three closed rounds with a king each. The three winner files hold the same rollouts, as r000008, r000011 and r000014 do.

    Then a validator with the rule in force starts on the same journal and the same store, checks the waiting files
    and opens a round. The journal of the three rounds, each of their files and what a rebuild reads from them are the same.
    """
    world = SimWorld(tmp_path, sim_params())
    for uid, quality in ((1, 0.3), (2, 0.6), (3, 0.9)):
        world.register(uid, quality)
        assert world.round().outcome == "applied"
    state = world.controller.state
    assert [king.round_id for king in state.kings] == ["r000001", "r000002", "r000003"] and state.rounds_closed == 3
    repeated, hotkeys = groups("repeated"), {"r000001": "hk1", "r000002": "hk2", "r000003": "hk3"}
    for round_id, hotkey in hotkeys.items():  # the past as it is: three winner files, the hotkey is the one difference
        world.store.put(layout.winner_batch(round_id), file(hotkey, repeated))

    def past() -> dict[str, bytes]:
        keys = [key for round_id in hotkeys for zone in ("public", "private") for key in world.store.list(f"{zone}/rounds/{round_id}/")]
        return {key: world.store.get(key) for key in keys}

    journal = tmp_path / JOURNAL_FILE
    lines, before, files = journal.read_bytes(), load(journal), past()
    assert len(files) >= 9 and all(layout.winner_batch(round_id) in files for round_id in hotkeys)

    # The validator starts with the rule in force. A copy of the repeated file and a file of new rollouts wait.
    live = StoreAdmission(world.store, world.params, tokens=WordTokens(), grader=None, public=WinnerRollouts(store_reader(world.store), tmp_path / "winner_rollouts"))
    world.admission = live
    controller = world.reopen()
    for uid in (4, 5):
        world.register(uid, None)
    world.store.put(layout.miner_object("hk4"), file("hk4", repeated))
    world.store.put(layout.miner_object("hk5"), file("hk5", groups("new")))
    capsys.readouterr()
    (ticket,) = controller.open_round(world.clock(), "entropy")
    assert [entry.reg.hotkey for entry in ticket.entries] == ["hk5"], "the copy gets no seat from now on"
    assert f"UID 4 {sha256_hex(file('hk4', repeated))[:12]} not valid, rule {PUBLIC_REUSE}: 20 of 20 rollouts are copies" in capsys.readouterr().err

    after = load(journal)
    assert journal.read_bytes().startswith(lines), "the records of the past rounds are the same bytes"
    assert after.kings == before.kings and after.rounds[:3] == before.rounds and after.heads[: len(before.heads)] == before.heads
    assert [record.params_hash for record in after.rounds[:3]] == [record.params_hash for record in before.rounds]
    assert past() == files, "each public and private file of the three rounds is the same bytes"

    # What a trainer and a rebuild of a past round read from a frozen file: the groups, with no reuse rule.
    for round_id, hotkey in hotkeys.items():
        raw = world.store.get(layout.winner_batch(round_id))
        read = check_batch(raw, hotkey, world.params)
        assert read.checked == ("format",) and rollout_keys(read.groups) == file_keys(file("any", repeated))
        assert replicate._read_groups(raw, hotkey, world.params) == assembly._read_groups(raw, hotkey, world.params) == read.groups


def test_params_of_a_journal_from_before_the_rule_take_the_default_share():
    old = {name: value for name, value in Params().to_dict().items() if name != "max_reused_rollout_frac"}
    assert Params.from_dict(old) == Params() and Params.from_dict(old).max_reused_rollout_frac == SHARE
    assert Params.from_dict(old).recipe_id == Params().recipe_id, "the recipe and its id are the same: a rebuild trains as before"
