"""The two rules on reused rollouts: ``public_reuse`` and ``seat_reuse``.

* What makes two rollouts the same, which change of a copy the rules catch, and the one that they do not catch.
* The share: a file passes at the limit, and is refused one rollout above it.
* The validator and a miner compute the same verdict from the same public files.
* A new winner file makes a waiting copy not valid, with no new check of that copy.
* Between seats of one round, the seat ahead keeps its entry.
* The rules act on a file that is checked or frozen from now on. Nothing of a past round changes.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

from sn120_v2 import assembly, layout, publisher, replicate
from sn120_v2.admission import PUBLIC_REUSE, RULES, SEAT_REUSE, Invalid, StoreAdmission, check_batch, freeze_log, verdict_note
from sn120_v2.canonical import ABSENT_AT_DEFAULT, Record, sha256_hex, written_fields
from sn120_v2.cli import changed_params
from sn120_v2.errors import ConfigError, ContractError, InfraError
from sn120_v2.journal import JOURNAL_FILE, read_journal
from sn120_v2.params import Params
from sn120_v2.public import CUT_LOGIN, CUT_QUERY, public_text
from sn120_v2.reuse import KEY_BYTES, KEY_FORM, REASON_LIMIT, REMOVED, WinnerRollouts, file_keys, normal, rollout_key, rollout_keys, round_name, store_reader
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import REUSE_LEFT_OUT, RoundRecord, load
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


def rule_on(build, **changes) -> Params:
    """The small params of the tests with the reuse rules in force. The default of the code is a share of 1: no rule."""
    return build.params(**({"max_reused_rollout_frac": SHARE} | changes))


def admission(build, store: Store, public: bool = True, **changes) -> StoreAdmission:
    reader = WinnerRollouts(store_reader(store)) if public else None
    return StoreAdmission(store, rule_on(build, **changes), tokens=WordTokens(), grader=None, clock=lambda: 77.9, public=reader)


def upload(store: Store, hotkey: str, rows: list[dict]) -> bytes:
    raw = file(hotkey, rows)
    store[layout.miner_object(hotkey)] = raw
    return raw


def freeze_lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("freeze: ")]


# ------------------------------------------------------------ what "the same rollout" is


SAME_TEXT = {
    "white space of each kind, also between two characters": ("It is 2", " I t\tis\n\n2\u00a0\u2003\u2028\u3000"),
    "case": ("It is 2", "IT IS 2"),
    "case folding, more than lower case": ("strasse", "Stra\u00dfe"),
    "a full-width letter and digit (NFKC makes them equal)": ("abc1", "\uff21\uff22\uff43\uff11"),
    "a ligature and a superscript (NFKC makes them equal)": ("fi x2", "\ufb01 x\u00b2"),
    "a zero-width space": ("answer", "an\u200bswer"),
    "each other format character: joiner, word joiner, soft hyphen, byte order mark, direction mark": ("answer", "\ufeffa\u200dn\u2060s\u00adw\u200eer"),
    "a control character": ("answer", "an\x00sw\x1fer"),
    "an accent, composed or not": ("e", "\u00e9"),
    "a format character between a letter and its accent, which stops the composition of NFKC": ("\u00e9", "e\u200b\u0301"),
    "a full stop at the end": ("answer", "answer."),
    "punctuation and symbols at the end, with white space": ("answer", "answer \u2026 !! $ \u3002"),
}
OTHER_TEXT = {
    "another letter": ("answer", "answers"),
    "another digit": ("x = 12", "x = 13"),
    "punctuation that is not at the end": ("a.b", "ab"),
    "a sign that is not at the end": ("-1", "1"),
}


@pytest.mark.parametrize("why", SAME_TEXT)
def test_the_normal_form_makes_these_two_texts_equal(why):
    first, second = SAME_TEXT[why]
    assert first != second and normal(first) == normal(second), why
    assert rollout_key(first, "o") == rollout_key(second, "o") and rollout_key("p", first) == rollout_key("p", second), "the prompt and the output have the same form"


@pytest.mark.parametrize("why", OTHER_TEXT)
def test_the_normal_form_keeps_these_two_texts_apart(why):
    first, second = OTHER_TEXT[why]
    assert normal(first) != normal(second) and rollout_key("p", first) != rollout_key("p", second), why


def test_the_key_of_a_rollout_is_the_normal_form_of_its_prompt_and_of_its_output():
    assert len(rollout_key("p", "o")) == KEY_BYTES
    assert normal("What is 1 + 1?") == "whatis1+1" and normal("It is\n\n  2.\t\\boxed{2}") == "itis2.\\boxed{2"
    assert rollout_key("What is 1 + 1?", "It is\n\n  2.\t\\boxed{2}") == rollout_key(" whatis1+1 ?\n", "ITIS2.\\boxed{2}.")
    assert rollout_key("ab", "c") != rollout_key("a", "bc"), "the two texts are kept apart"
    assert rollout_key("p", "o") != rollout_key("q", "o") and rollout_key("p", "o") != rollout_key("p", "q")
    assert normal(" . ") == normal("") == "" and len(rollout_key("", "")) == KEY_BYTES, "a text with no character left has a key too"
    assert rollout_key("\ud800", "o") != rollout_key("\ud801", "o"), "a lone surrogate, which JSON can hold, does not stop the key"


def test_the_form_has_a_stated_limit_a_character_of_another_category_that_shows_nothing():
    """Not white space, not a format character and not a mark for Unicode, so the form keeps them. A copy with one of them inside each text passes."""
    for blank in ("\u3164", "\u115f", "\u2800"):  # Hangul filler, Hangul choseong filler, Braille blank
        assert normal(f"an{blank}swer") != normal("answer")
    assert normal("answer\u2800") == normal("answer"), "the Braille blank is a symbol: it goes from the end, as each symbol does"


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
    params, first, second = rule_on(build), groups("w1"), groups("w2", start=100)
    public = known(file("5King1", first), None, file("5King2", second))
    assert params.max_reused_rollout_frac == SHARE
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


def edited(rows: list[dict], prompt=str, output=str) -> list[dict]:
    """The groups with one edit of each prompt and one edit of each output."""
    return [{"prompt": prompt(row["prompt"]), "rollouts": [{"output": output(r["output"]), "reward": r["reward"]} for r in row["rollouts"]]} for row in rows]


def inside(mark: str):
    """Put ``mark`` in the middle of a text and after its first word."""
    return lambda text: text[: len(text) // 2].replace(" ", f" {mark}", 1) + mark + text[len(text) // 2 :]


TRIVIAL_EDITS = {
    "a full stop at the end of each output": dict(output=lambda text: text + "."),
    "a zero-width space in each output": dict(output=inside("\u200b")),
    "a space and a full stop at the end of each prompt": dict(prompt=lambda text: text + " ."),
    "another case in each output": dict(output=str.upper),
    "another case in each prompt": dict(prompt=str.title),
    "all of them, and a full-width digit in each prompt": dict(prompt=lambda text: text.upper().replace("0", "\uff10") + " !", output=lambda text: inside("\u2060")(text).swapcase() + " ..."),
}


@pytest.mark.parametrize("edit", TRIVIAL_EDITS)
def test_a_copy_with_a_trivial_edit_of_each_rollout_is_refused(build, edit):
    """The four edits that passed the first form of the rule, which removed white space only, and their sum."""
    first = groups("w1")
    public, copy = known(file("5King1", first)), file("5Copier", edited(first, **TRIVIAL_EDITS[edit]))
    assert copy != file("5Copier", first) and len(set(file_keys(copy))) == 20
    assert check_batch(copy, "5Copier", rule_on(build), tokens=WordTokens()).checked == ("format", "tokens"), "the edited file is a valid file without the rule"
    whole = (PUBLIC_REUSE, "20 of 20 rollouts are copies of rollouts in the published winner files, 20 of them in the file of round r000001; the limit is a share of 0.1")
    assert refusal(copy, "5Copier", rule_on(build), public) == whole


def test_the_known_limit_one_visible_edit_inside_each_rollout_passes(build):
    """The rules compare whole texts in the normal form. A copy with one more letter inside each output is a new file for them."""
    first = groups("w1")
    public = known(file("5King1", first))
    assert refusal(file("5Copier", first), "5Copier", rule_on(build), public)[0] == PUBLIC_REUSE
    assert refusal(file("5Copier", edited(first, output=inside("x"))), "5Copier", rule_on(build), public) == ("", "")


def test_a_file_passes_at_the_share_and_is_refused_one_rollout_above_it(build):
    first = groups("w1")
    public = known(file("5King1", first))
    own = groups("mine")
    two = file("5M", first[:1] + own[1:])  # 2 of 20 rollouts are copies: the share is 0.10
    half = {"prompt": first[1]["prompt"], "rollouts": [first[1]["rollouts"][0], own[1]["rollouts"][1]]}  # one copied rollout, one new
    three = file("5M", first[:1] + [half] + own[2:])
    assert refusal(two, "5M", rule_on(build), public) == ("", "")
    assert refusal(three, "5M", rule_on(build), public) == (PUBLIC_REUSE, "3 of 20 rollouts are copies of rollouts in the published winner files, 3 of them in the file of round r000001; the limit is a share of 0.1")
    assert refusal(two, "5M", build.params(max_reused_rollout_frac=0.0), public)[0] == PUBLIC_REUSE, "a share of 0: no copied rollout at all"
    assert refusal(file("5M", own), "5M", build.params(max_reused_rollout_frac=0.0), public) == ("", "")
    assert refusal(file("5M", first), "5M", build.params(), public) == ("", "") and build.params().max_reused_rollout_frac == 1.0, "the default share of 1: no rule"
    for bad in (-0.01, 1.01):
        with pytest.raises(ConfigError, match="max_reused_rollout_frac"):
            build.params(max_reused_rollout_frac=bad)


def test_a_file_at_the_caps_can_hold_this_many_copied_rollouts():
    """A stated limit: copies below the share pass. With the default caps and a share of 0.10 that is 200 rollouts in a file of 2000."""
    params = Params(max_reused_rollout_frac=SHARE)
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


# ------------------------------------------------------------ liveness: a winner file that cannot be read holds nothing


class Flaky:
    """The reader of a store. The read of a round in ``down`` fails with the error given there. ``asked`` holds each round that was asked."""

    def __init__(self, store) -> None:
        self.read, self.down, self.asked = store_reader(store), {}, []

    def __call__(self, round_id: str) -> bytes | None:
        self.asked.append(round_id)
        if round_id in self.down:
            raise self.down[round_id]
        return self.read(round_id)


def reuse_lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("reuse: ")]


def cache_files(directory) -> list[str]:
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


NOT_A_FILE = "ValueError: the bytes are not a batch file (JSONDecodeError)"
ASKED_AGAIN = "The check goes on without it and asks for it again at the next check"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"<html>bad gateway</html>", "ValueError: the bytes are not a batch file (JSONDecodeError)"),
        (b"\x80 not UTF-8", "ValueError: the bytes are not a batch file (UnicodeDecodeError)"),
        (b"[1, 2]", "ValueError: the bytes are not a batch file (TypeError)"),
        (json.dumps({"hotkey": "5King", "quality": 0.5}).encode(), "ValueError: the bytes are not a batch file (KeyError)"),
        (json.dumps({"groups": [{"prompt": "a prompt with no rollouts"}]}).encode(), "ValueError: the bytes are not a batch file (KeyError)"),
        (json.dumps({"groups": [{"prompt": 7, "rollouts": [{"output": "o"}]}]}).encode(), "ValueError: the bytes are not a batch file (TypeError)"),
    ],
)
def test_bytes_that_are_not_a_batch_file_are_never_read_as_a_file_with_no_rollout(tmp_path, raw, reason):
    """A JSON object with no ``groups`` is one of them. The winner file of a simulated store (``sim.SimAdmission``) is such an object."""
    with pytest.raises(ValueError, match="the bytes are not a batch file"):
        file_keys(raw)
    index = WinnerRollouts({"r000001": raw, "r000002": file("5King", groups("w2"))}.get, tmp_path)
    assert len(index.known(2)) == 20 and index.left_out == (("r000001", reason),), "the round with rollouts still counts"
    assert cache_files(tmp_path) == [f"r000002.form{KEY_FORM}.keys"]


def test_a_winner_file_that_cannot_be_read_is_left_out_and_counts_again_from_the_check_that_reads_it(build, tmp_path, capsys):
    """The check never stops on a winner file. It leaves the file out for that pass, says so, and keeps nothing of it."""
    store = Store({layout.winner_batch("r000001"): file("5King1", groups("w1")), layout.winner_batch("r000002"): b"<html>bad gateway</html>", layout.winner_batch("r000003"): file("5King3", groups("w3"))})
    hotkeys = ["5Copy1", "5Copy2", "5Copy3", "5Honest"]
    for hotkey, tag in zip(hotkeys, ("w1", "w2", "w3", "mine")):
        upload(store, hotkey, groups(tag))
    reader, cache = Flaky(store), tmp_path / "cache"
    reader.down["r000001"] = InfraError("the store answered 503")
    adm = StoreAdmission(store, rule_on(build), tokens=WordTokens(), grader=None, public=WinnerRollouts(reader, cache))
    valid = lambda: {hotkey: verdict.valid for hotkey, verdict in adm.verdicts(hotkeys, rounds=3).items()}

    assert valid() == {"5Copy1": True, "5Copy2": True, "5Copy3": False, "5Honest": True}, "the file that was read counts; a copy of a file that was not read passes for now"
    assert adm.left_out() == (("r000001", "InfraError: the store answered 503"), ("r000002", NOT_A_FILE))
    assert reuse_lines(capsys) == [
        f"reuse: the winner file of round r000001 is left out of rule public_reuse: InfraError: the store answered 503. {ASKED_AGAIN}",
        f"reuse: the winner file of round r000002 is left out of rule public_reuse: {NOT_A_FILE}. {ASKED_AGAIN}",
    ]
    assert cache_files(cache) == [f"r000003.form{KEY_FORM}.keys"], "nothing is kept of a file that was not read: not as an empty file, not at all"

    asked = len(reader.asked)
    assert valid()["5Copy1"] and reader.asked[asked:] == ["r000001", "r000002"] and reuse_lines(capsys) == [], "each check asks again; the log names a file once"
    reader.down["r000001"] = TimeoutError("timed out")
    assert valid()["5Copy1"] and adm.left_out()[0] == ("r000001", "TimeoutError: timed out")
    assert reuse_lines(capsys) == [f"reuse: the winner file of round r000001 is left out of rule public_reuse: TimeoutError: timed out. {ASKED_AGAIN}"], "another reason is a new line"
    assert dict(WinnerRollouts(reader, cache).known(3)) == dict(adm.public.known(3)) and len(adm.public.known(3)) == 20, "a new process has the same view: the two files are not in the cache as empty"

    # The two files read now. Their rollouts count from this check on, with no new check of a miner file.
    reader.down.clear()
    store[layout.winner_batch("r000002")] = file("5King2", groups("w2"))
    calls = adm.tokens.calls
    assert valid() == {"5Copy1": False, "5Copy2": False, "5Copy3": False, "5Honest": True} and adm.left_out() == () and adm.tokens.calls == calls
    read_now = "is read now: rule public_reuse counts its rollouts again"
    assert reuse_lines(capsys) == [f"reuse: the winner file of round r000001 {read_now}", f"reuse: the winner file of round r000002 {read_now}"]
    assert cache_files(cache) == [f"r00000{number}.form{KEY_FORM}.keys" for number in (1, 2, 3)]
    asked = len(reader.asked)
    assert valid()["5Honest"] and reader.asked[asked:] == [] and reuse_lines(capsys) == [], "a file that was read is not asked again"


def test_a_rollout_of_a_file_that_reads_late_belongs_to_the_first_round_that_published_it():
    rows = groups("repeated")
    files = winners(file("5King1", rows), file("5King2", rows))
    reader = Flaky(Store({layout.winner_batch(round_id): raw for round_id, raw in files.items()}))
    reader.down["r000001"] = InfraError("down")
    index = WinnerRollouts(reader)
    assert set(index.known(2).values()) == {"r000002"} and index.left_out == (("r000001", "InfraError: down"),)
    reader.down.clear()
    assert set(index.known(2).values()) == {"r000001"} and index.left_out == (), "the same answer as with no failure"


def test_a_reason_is_one_line_of_a_bounded_length():
    index = WinnerRollouts(Flaky(Store()))
    index.read.down["r000001"] = InfraError("a long\nreason " * 100)
    index.known(1)
    ((round_id, reason),) = index.left_out
    assert round_id == "r000001" and len(reason) == REASON_LIMIT and "\n" not in reason and reason.startswith("InfraError: a long reason a long reason")


def test_a_round_opens_without_a_winner_file_that_cannot_be_read_and_its_records_name_the_file(tmp_path, capsys):
    """Liveness. A winner file whose read fails, and one whose bytes are not a batch file, do not hold the loop.

    The round opens. The journal record ``round_opened`` holds each file that was left out with the full reason,
    beside the round record and not in it. The public round file holds the same list with the safe form of each
    reason. The next round, which read the files, has no such key.
    """
    world = SimWorld(tmp_path, sim_params(max_reused_rollout_frac=SHARE))
    for uid, quality in ((1, 0.3), (2, 0.6)):
        world.register(uid, quality)
        assert world.round().outcome == "applied"
    store, journal = world.store, tmp_path / JOURNAL_FILE
    store.put(layout.winner_batch("r000001"), file("hk1", groups("w1")))
    store.put(layout.winner_batch("r000002"), b"<html>bad gateway</html>")
    reader = Flaky(store)
    reader.down["r000001"] = InfraError("GET https://user:password@store.example/batch.json?X-Amz-Signature=abc answered 503")
    full = f"InfraError: {reader.down['r000001']}"
    world.admission = StoreAdmission(store, world.params, tokens=WordTokens(), grader=None, public=WinnerRollouts(reader, tmp_path / "winner_rollouts"))
    controller = world.reopen()
    for uid in (3, 4, 5):
        world.register(uid, None)
    store.put(layout.miner_object("hk3"), file("hk3", groups("w1")))  # a copy of the winner file that cannot be read
    store.put(layout.miner_object("hk4"), file("hk4", groups("mine")))
    capsys.readouterr()

    def opened() -> dict:
        return [record.data for record in read_journal(journal) if record.kind == "round_opened"][-1]

    def end_round() -> dict:
        """Close the open round with no result, as ``end-round`` does, and give its public round file."""
        round_id = controller.state.round.round_id
        while (record := controller.state.round) is not None and record.decision is None:
            controller.fail_lease(next(lease for lease in record.leases if lease.status == "open").lease_id, "ended by the test", world.clock())
        return json.loads(store.get(layout.round_file(round_id)))

    (ticket,) = controller.open_round(world.clock(), "entropy")
    assert [entry.reg.hotkey for entry in ticket.entries] == ["hk3", "hk4"], "the round opens, and the copy of the file that was not read has a seat in it"
    left_out = (("r000001", full), ("r000002", NOT_A_FILE))
    assert reuse_lines(capsys) == [f"reuse: the winner file of round {round_id} is left out of rule public_reuse: {reason}. {ASKED_AGAIN}" for round_id, reason in left_out]
    data = opened()
    assert set(data) == {"round", REUSE_LEFT_OUT} and data[REUSE_LEFT_OUT] == [list(pair) for pair in left_out], "the journal has the full reason, beside the round record"
    assert RoundRecord.from_dict(data["round"]) == controller.state.round, "the round record has no new field: code of before the rule reads it"
    assert controller.state.reuse_left_out == left_out == load(journal).reuse_left_out, "a controller that starts on this journal knows it too"

    report = end_round()
    safe = f"InfraError: GET https://{CUT_LOGIN}@store.example/batch.json?{CUT_QUERY} answered 503"
    assert report[REUSE_LEFT_OUT] == [{"round_id": "r000001", "reason": public_text(full)}, {"round_id": "r000002", "reason": NOT_A_FILE}]
    assert public_text(full) == safe and "password" not in json.dumps(report) and "X-Amz-Signature" not in json.dumps(report)
    assert controller.state.round is None and controller.state.reuse_left_out == () == load(journal).reuse_left_out

    # The two files read now. The round after that names no file: its records have the keys of before the rule.
    reader.down.clear()
    store.put(layout.winner_batch("r000002"), file("hk2", groups("w2")))
    store.put(layout.miner_object("hk5"), file("hk5", groups("w1")))
    world.now += world.params.apply_retry_s  # the wait after a failed lease
    (ticket,) = controller.open_round(world.clock(), "entropy")
    assert "hk5" not in [entry.reg.hotkey for entry in ticket.entries], "a copy of the same winner file gets no seat now"
    assert set(opened()) == {"round"} and REUSE_LEFT_OUT not in end_round()


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
    world = SimWorld(tmp_path, sim_params(max_reused_rollout_frac=SHARE))
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
    first, second, third = (tmp_path / "cache" / f"r00000{number}.form{KEY_FORM}.keys" for number in (1, 2, 3))
    assert cache_files(tmp_path / "cache") == [path.name for path in (first, second, third)] == ["r000001.form2.keys", "r000002.form2.keys", "r000003.form2.keys"]
    assert first.read_bytes() == b"".join(file_keys(files["r000001"])) and second.read_bytes() == b"", "a round with no winner, which no later read can change, is kept as no key"

    assert dict(WinnerRollouts(read, tmp_path / "cache").known(3)) == got and len(asked) == 3, "a new process reads the cache"
    third.write_bytes(b"damaged")
    assert dict(WinnerRollouts(read, tmp_path / "cache").known(3)) == got and asked[3:] == ["r000003"], "a cache file that is not whole keys is read again"
    shutil.rmtree(tmp_path / "cache")
    assert dict(WinnerRollouts(read, tmp_path / "cache").known(3)) == got and asked[4:] == ["r000001", "r000002", "r000003"], "the store is the record: the cache is rebuilt from it"
    assert dict(WinnerRollouts(read).known(3)) == got, "with no cache"
    (tmp_path / "no-directory").write_bytes(b"a file where the cache directory must be")
    assert dict(WinnerRollouts(read, tmp_path / "no-directory").known(3)) == got, "a cache that cannot be written stops nothing"


def test_a_cache_of_keys_of_another_form_is_never_read_as_keys_of_this_form(tmp_path):
    """The first form of the key removed white space only, and its cache file had the name ``<round>.keys``. The name of a cache file holds the form."""
    raw, asked = file("5King1", groups("w1")), []
    old_form = b"".join(sha256_hex(f"old key {n}".encode()).encode()[:KEY_BYTES] for n in range(20))
    (tmp_path / "r000001.keys").write_bytes(old_form)

    def read(round_id: str) -> bytes | None:
        asked.append(round_id)
        return raw

    got = WinnerRollouts(read, tmp_path).known(1)
    assert asked == ["r000001"] and sorted(got) == sorted(file_keys(raw)) and not set(got) & {old_form[at : at + KEY_BYTES] for at in range(0, len(old_form), KEY_BYTES)}
    assert cache_files(tmp_path) == ["r000001.form2.keys", "r000001.keys"] and (tmp_path / "r000001.keys").read_bytes() == old_form, "the old file is not read and not changed"
    assert KEY_FORM == 2, "a change of ``normal`` or of ``rollout_key`` takes a new number here and in ``reuse.KEY_FORM``"


def test_a_reader_that_can_be_behind_asks_again_for_a_new_round_with_no_file(tmp_path):
    files, asked = winners(file("5King1", groups("w1"))), []

    def read(round_id: str) -> bytes | None:
        asked.append(round_id)
        return files.get(round_id)

    assert len(WinnerRollouts(read, tmp_path).known(3, settled=1)) == 20
    assert cache_files(tmp_path) == ["r000001.form2.keys"], "a round above ``settled`` with no file is not kept as a round with no winner"
    files |= {"r000003": file("5King3", groups("w3"))}  # the site has the file now
    assert len(WinnerRollouts(read, tmp_path).known(3, settled=1)) == 40 and asked == ["r000001", "r000002", "r000003", "r000002", "r000003"]


# ------------------------------------------------------------ llms.txt and the rule table


def rendered(directory, **changes) -> tuple[bytes, bytes, Params]:
    world = SimWorld(directory, sim_params(**changes))
    world.register(1, 0.3)
    return *publisher.render(world.publisher.snapshots[-1], "https://site.example", "https://site.example/links"), world.params


def test_llms_txt_states_the_rule_with_its_share_and_what_the_same_rollout_is(tmp_path):
    mining, llms, params = rendered(tmp_path, max_reused_rollout_frac=SHARE)
    text = llms.decode()
    flat = " ".join(text.split())
    assert publisher.check_pair(mining, llms) == [] and publisher.check_pair(mining, llms, params) == []
    assert json.loads(mining)["params"]["max_reused_rollout_frac"] == SHARE and "| `max_reused_rollout_frac` | 0.1 |" in text
    assert "- Reuse: the published file of each round winner (section 9) is public, and a file that reuses its rollouts is\n  refused." in text
    assert "At most 0.1 of your rollouts may be the same as rollouts in the published winner files" in flat
    assert "no file wins two times" in text
    # What is matched, as ``reuse.normal`` does it. The test of the form itself is above.
    assert "Unicode NFKD, then case folding (`str.casefold` of Python), then NFKD again." in flat
    assert "each character of the Unicode categories Zs, Zl, Zp, Cc, Cf, Mn, Me is removed" in flat and tuple(REMOVED) == ("Zs", "Zl", "Zp", "Cc", "Cf", "Mn", "Me")
    assert "punctuation and symbols (each category P and S) are removed from the end of the text" in flat
    assert "The function is `normal` in `v2/sn120_v2/reuse.py`" in flat and "after all white space is removed" not in flat
    assert f"its public round file names the file and the reason under `{REUSE_LEFT_OUT}`" in flat, "a winner file that cannot be read"
    assert "A sign of it, which other causes can also give:" in flat, "the sign of seat_reuse is not said to be unique"
    for code in (PUBLIC_REUSE, SEAT_REUSE):
        assert f"| `{code}` | {RULES[code]} |" in text, "the table of the rule codes has the plain meaning"
    assert list(RULES).index("pool") < list(RULES).index(PUBLIC_REUSE) < list(RULES).index(SEAT_REUSE) < list(RULES).index("check_fault")
    edited = text.replace("At most 0.1 of your rollouts", "At most 0.5 of your rollouts")
    assert edited != text and publisher.check_pair(mining, edited.encode()) != [], "a text with another share is not the text of these params"


def test_with_the_default_share_the_public_files_have_no_word_of_the_rule(tmp_path):
    """New code alone puts no rule in force and changes no public file: the default share is 1, and the files are those of before the rule."""
    mining, llms, params = rendered(tmp_path)
    text = llms.decode()
    assert params.max_reused_rollout_frac == 1.0 and publisher.check_pair(mining, llms) == [] and publisher.check_pair(mining, llms, params) == []
    assert "max_reused_rollout_frac" not in mining.decode() and "reuse" not in mining.decode()
    for word in ("max_reused_rollout_frac", "Reuse", PUBLIC_REUSE, SEAT_REUSE, "no file wins two times", "NFKD"):
        assert word not in text
    on = rendered(tmp_path / "on", max_reused_rollout_frac=SHARE)[1].decode()
    assert len(on.splitlines()) > len(text.splitlines()), "the words of the rule come with the rule"


# ------------------------------------------------------------ not backward: nothing of a past round changes


def test_the_rule_changes_nothing_of_a_round_that_ran_before_it(tmp_path, capsys):
    """Three closed rounds with a king each. The three winner files hold the same rollouts, as r000008, r000011 and r000014 do.

    Then a validator with the rule in force starts on the same journal and the same store, checks the waiting files
    and opens a round. The journal of the three rounds, each of their files and what a rebuild reads from them are the same.
    """
    world = SimWorld(tmp_path, sim_params())  # the three rounds run with no rule: the default
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

    # The operator puts the rule in force with ``set-params``, and the validator starts with it. A copy of the repeated file and a file of new rollouts wait.
    assert world.params.max_reused_rollout_frac == 1.0
    world.params = sim_params(max_reused_rollout_frac=SHARE)
    live = StoreAdmission(world.store, world.params, tokens=WordTokens(), grader=None, public=WinnerRollouts(store_reader(world.store), tmp_path / "winner_rollouts"))
    world.admission = live
    world.reopen().set_params(world.params, world.clock())
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


# ------------------------------------------------------------ old readers: what code of before the rule can still read


@dataclasses.dataclass(frozen=True)
class Before(Record):
    """A record as code of before a new field knows it."""

    count: int = 1


@dataclasses.dataclass(frozen=True)
class After(Record):
    """The same record with a field that came later and is absent at its default."""

    count: int = 1
    share: float = dataclasses.field(default=1.0, metadata=ABSENT_AT_DEFAULT)


def test_a_field_that_is_absent_at_its_default_gives_the_record_of_before_until_it_has_another_value():
    assert After().to_dict() == Before().to_dict() == {"count": 1} and After().to_json() == Before().to_json() and After().digest() == Before().digest()
    assert Before.from_dict(After(count=7).to_dict()) == Before(count=7), "code of before the field reads the record"
    assert After.from_dict(Before(count=7).to_dict()) == After(count=7) and After.from_dict({"count": 1, "share": 1.0}) == After()
    moved = After(share=0.5)
    assert moved.to_dict() == {"count": 1, "share": 0.5} and After.from_dict(moved.to_dict()) == moved and moved.digest() != After().digest()
    with pytest.raises(ContractError, match="unknown field"):
        Before.from_dict(moved.to_dict())
    assert [f.name for f in written_fields(After())] == ["count"] and [f.name for f in written_fields(moved)] == ["count", "share"]


def test_params_with_no_rule_are_the_params_of_before_the_field_and_set_params_puts_the_rule_in_force(tmp_path):
    """What new code writes on its own: the params of before the rule, byte for byte. So a checkout alone stops no old reader.

    From the ``set-params`` that puts the rule in force, each record with params holds the field, and code of
    before the rule refuses that record (``canonical.from_jsonable`` takes no unknown field).
    """
    name = "max_reused_rollout_frac"
    assert Params().max_reused_rollout_frac == 1.0 and name not in Params().to_dict(), "the default: no rule, and no such key"
    assert Params.from_dict(Params().to_dict()) == Params() == Params.from_dict(Params().to_dict() | {name: 1.0})
    on = Params(max_reused_rollout_frac=SHARE)
    assert on.to_dict() == Params().to_dict() | {name: SHARE} and Params.from_dict(on.to_dict()) == on
    assert on.params_hash != Params().params_hash and on.recipe_id == Params().recipe_id, "the rule is a part of the params id; the recipe and its id are the same"

    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    world.round()
    journal = tmp_path / JOURNAL_FILE
    written = lambda: [record.data["params"] for record in read_journal(journal) if record.kind == "params_set"]
    ticket = lambda: json.loads(world.store.get(layout.ticket("r000001", world.controller.state.rounds[-1].leases[0].lease_id)))["params"]
    assert name not in written()[-1] and name not in world.publisher.snapshots[-1]["params"] and name not in ticket()
    assert world.controller.state.rounds[-1].params_hash == sim_params().params_hash

    with_rule = sim_params(max_reused_rollout_frac=SHARE)
    assert changed_params(world.controller.params, with_rule) == {name: {"before": 1.0, "after": SHARE}}, "what set-params prints"
    world.controller.set_params(with_rule, world.clock())
    assert written()[-1][name] == SHARE and world.controller.params == with_rule == load(journal).params
    world.controller.publish(world.clock())
    assert world.publisher.snapshots[-1]["params"][name] == SHARE
    # A params file with no such key is a file with no rule: set-params replaces every value.
    assert changed_params(with_rule, sim_params()) == {name: {"before": SHARE, "after": 1.0}}
    world.controller.set_params(sim_params(), world.clock())
    assert name not in written()[-1] and load(journal).params == sim_params()
