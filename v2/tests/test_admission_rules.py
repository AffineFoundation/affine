"""Admission rules that v1 did not have: ``train`` bounds, the content rules, public verdicts and the freeze."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from sn120_v2 import layout
from sn120_v2.admission import GraderTimeLimit, Invalid, StoreAdmission, check_batch, env_grader, final_answer, verdict
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import InfraError, VerifyError

HOTKEY = "5MinerA"
THINK = "step " * 40  # 40 reasoning tokens: above the least of 32


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


def grade(prompt: str, output: str, capped: bool) -> float | None:
    """A toy official grader: the task "double N" has the answer 2N. Any other prompt is not in the pool."""
    if not prompt.startswith("double "):
        return None
    return float(not capped and final_answer(output) == str(2 * int(prompt.split()[1])))


def solved(n: int, k: int = 0) -> dict:
    return {"output": f"{THINK}{k} \\boxed{{{2 * n}}}", "reward": 1}


def file(groups=None, train=None, version="affine-batch-v3", hotkey=HOTKEY) -> bytes:
    groups = [{"prompt": f"double {n}", "rollouts": [solved(n, 0), solved(n, 1)]} for n in range(10)] if groups is None else groups
    return json.dumps({"version": version, "hotkey": hotkey, "groups": groups} | ({} if train is None else {"train": train})).encode()


def rule(raw: bytes, params, **inputs) -> str:
    """The code of the rule that fails, or "" for a valid file."""
    try:
        check_batch(raw, HOTKEY, params, tokens=WordTokens(), **inputs)
    except Invalid as exc:
        return exc.rule
    return ""


def flat(rollouts: list[dict]) -> bytes:
    """One rollout per prompt, so a test controls every output of the file."""
    return file([{"prompt": f"double {n}", "rollouts": [rollout]} for n, rollout in enumerate(rollouts)])


# --- train: only the published keys, only inside the published bounds, never clamped ---


def edges(bound) -> tuple[list, list]:
    """Values on the two ends of a bound (valid), and the nearest values outside it (invalid)."""
    if bound.choices:
        return [[bound.choices[0]], list(bound.choices)], [[], ["not_a_module"], [bound.choices[0]] * 2, [[bound.choices[0]]], bound.choices[0]]
    if bound.integer:
        return [int(bound.low), int(bound.high)], [int(bound.low) - 1, int(bound.high) + 1, bound.low + 0.5, float(bound.low)]
    return [bound.low, bound.high], [math.nextafter(bound.low, -math.inf), math.nextafter(bound.high, math.inf)]


def test_train_accepts_each_bound_end_and_keeps_the_values_as_given(build):
    params = build.params()
    assert len(params.recipe.miner_bounds) >= 5
    for bound in params.recipe.miner_bounds:
        for value in edges(bound)[0]:
            batch = check_batch(file(train={bound.key: value}), HOTKEY, params)
            assert batch.train == {bound.key: value}
            assert getattr(params.recipe.for_miner(batch.train), bound.key) in (value, tuple(value) if bound.choices else value)
    everything = {bound.key: edges(bound)[0][0] for bound in params.recipe.miner_bounds}
    assert check_batch(file(train=everything, version="affine-batch-v2"), HOTKEY, params).train == everything


def test_train_value_outside_its_bound_makes_the_file_invalid(build):
    params = build.params()
    for bound in params.recipe.miner_bounds:
        for value in edges(bound)[1] + [True, None, "1", 10**400, {"a": 1}]:
            assert rule(file(train={bound.key: value}), params) == "train", (bound.key, value)


@pytest.mark.parametrize("key", ["lr", "kl", "step_mult", "loss_agg", "base_lr", "image_digests", "miner_bounds", ""])
def test_train_key_that_is_not_settable_makes_the_file_invalid(key, build):
    assert rule(file(train={key: 1}), build.params()) == "train"


def test_train_reason_is_public(build):
    low = build.params().recipe.miner_bounds[0]
    for train in ({low.key: 0.123456789}, {"private_key_name": 1}, {low.key: 10**400}):
        reason = verdict(file(train=train), HOTKEY, build.params()).reason
        assert reason and "0.123456789" not in reason and "private_key_name" not in reason and "1000" not in reason


# --- content rules ---


def test_good_file_passes_every_rule_family(build):
    batch = check_batch(file(), HOTKEY, build.params(), tokens=WordTokens(), grader=grade)
    assert (batch.n_prompts, batch.n_rollouts, batch.checked) == (10, 20, ("format", "tokens", "grader"))
    assert check_batch(file(), HOTKEY, build.params()).checked == ("format",)


def test_reward_must_be_the_official_grade(build):
    spoofed = flat([solved(n) for n in range(19)] + [{"output": THINK + "\\boxed{7}", "reward": 1}])
    assert rule(spoofed, build.params(), grader=grade) == "grader"
    assert rule(spoofed, build.params()) == ""  # without a grader the reward is not checked
    near = flat([solved(n) | {"reward": 0.995} for n in range(20)])
    assert rule(near, build.params(), grader=grade) == "grader"
    assert rule(near, build.params(reward_tolerance=0.01), grader=grade) == ""


def test_output_at_the_token_cap_is_graded_as_capped(build):
    params = build.params(max_output_tokens=42, max_context_tokens=60)  # solved() writes 42 tokens: exactly the cap
    assert rule(flat([solved(n) for n in range(20)]), params, grader=grade) == "grader"  # a capped row earns 0, the file says 1
    assert rule(flat([solved(n) | {"reward": 0} for n in range(20)]), params, grader=grade) == ""
    assert rule(flat([solved(n) for n in range(20)]), build.params(max_output_tokens=43, max_context_tokens=60), grader=grade) == ""


def test_env_grader_joins_the_pool_and_the_grade_function(build):
    pool = SimpleNamespace(task_for_prompt=lambda prompt: int(prompt.split()[1]) if prompt.startswith("double ") else None)
    official = env_grader(pool, lambda task, output, capped: SimpleNamespace(reward=grade(f"double {task}", output, capped)))
    assert (official("double 2", "\\boxed{4}", False), official("double 2", "\\boxed{4}", True), official("other", "x", False)) == (1.0, 0.0, None)
    assert rule(file(), build.params(), grader=official) == ""


def test_row_the_grader_cannot_grade_in_time_is_not_allowed_and_not_remembered(build):
    seen, slow = [], {"double 3"}

    def timed(prompt: str, output: str, capped: bool) -> float | None:
        seen.append(prompt)
        if prompt in slow:
            raise GraderTimeLimit
        return grade(prompt, output, capped)

    for reward in (0, 1):  # whatever reward the file gives the row
        seen.clear()
        raw = flat([solved(n) | {"reward": reward if n in (3, 7) else 1} for n in range(20)])
        assert rule(raw, build.params(), grader=timed) == "grader_time_limit"
        assert seen == ["double 0", "double 1", "double 2", "double 3"]  # the check stops at the first such row
    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=timed)
    adm.store.put(layout.miner_object(HOTKEY), file())
    assert "groups[3].rollouts[0]" in adm.verdicts([HOTKEY])[HOTKEY].reason and "time limit" in adm.verdicts([HOTKEY])[HOTKEY].reason
    assert adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001") is None and list(adm.store) == [layout.miner_object(HOTKEY)]  # a skip
    slow.clear()  # the machine is fast again: the same bytes are checked again and pass
    assert adm.verdicts([HOTKEY])[HOTKEY].valid and adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001") is not None


def test_failure_of_the_grader_itself_is_an_infra_error(build):
    def broken(prompt: str, output: str, capped: bool) -> float | None:
        raise RuntimeError("the grader process is gone")

    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=broken)
    adm.store.put(layout.miner_object(HOTKEY), file())
    for call in (lambda: rule(file(), build.params(), grader=broken), lambda: adm.verdicts([HOTKEY]), lambda: adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001")):
        with pytest.raises(InfraError, match="RuntimeError"):
            call()


def test_prompts_outside_the_pool_follow_the_pool_share(build):
    mixed = file([{"prompt": f"double {n}", "rollouts": [solved(n)]} for n in range(3)] + [{"prompt": "my own task", "rollouts": [solved(1)]}])
    assert rule(mixed, build.params(), grader=grade) == ""  # the default share is 0
    assert rule(mixed, build.params(min_pool_prompt_frac=0.75), grader=grade) == ""
    assert rule(mixed, build.params(min_pool_prompt_frac=0.8), grader=grade) == "pool"


def test_bare_answers_are_limited(build):
    bare = {"output": "\\boxed{4}", "reward": 1}
    short = {"output": "step " * 31 + "\\boxed{4}", "reward": 1}  # 31 reasoning tokens: one under the least
    just = {"output": "step " * 32 + "\\boxed{4}", "reward": 1}
    assert rule(flat([solved(n) for n in range(19)] + [bare]), build.params()) == ""  # 1 of 20 is the 5% limit
    assert rule(flat([solved(n) for n in range(18)] + [bare, short]), build.params()) == "answer_only"
    assert rule(flat([solved(n) for n in range(18)] + [bare, just]), build.params()) == ""
    assert rule(flat([solved(n) for n in range(18)] + [bare, short]), build.params(max_answer_only_frac=0.1)) == ""


def test_reasoning_must_be_the_larger_part_of_the_output(build):
    tail = [{"output": THINK + "\\boxed{4} " + "after " * 60, "reward": 1}]
    assert rule(flat(tail), build.params()) == "reasoning_share"  # 40 of 101 tokens come before the answer
    assert rule(flat(tail), build.params(reasoning_token_share=0.25)) == ""


def test_one_answer_may_not_fill_the_file(build):
    def same(n: int, total: int = 40) -> bytes:
        return flat([{"output": f"{THINK}{k} \\boxed{{ 4 }}" if k % 2 else f"{THINK}{k} \\boxed{{4}}", "reward": 0} for k in range(n)]
                    + [solved(100 + k) for k in range(total - n)])

    count = build.params(max_answer_repeat_count=3)  # 10% of 40 rollouts is 4: the looser limit wins
    assert (rule(same(4), count), rule(same(5), count)) == ("", "answer_repeat")
    share = build.params(max_answer_repeat_count=6, max_answer_repeat_frac=0.1)  # now the count is looser
    assert (rule(same(6), share), rule(same(7), share)) == ("", "answer_repeat")
    assert rule(same(33), build.params()) == "answer_repeat"  # the default count is 32; it holds without a tokenizer too
    with pytest.raises(Invalid, match="33 rollouts"):
        check_batch(same(33), HOTKEY, build.params())


def test_final_answer():
    assert final_answer("a \\boxed{1} b \\boxed{\\frac{1}{ 2 }} c") == "\\frac{1}{2}"
    assert final_answer("no answer") is None and final_answer("\\boxed{1") is None and final_answer("\\boxed 1") is None


# --- public verdicts and the freeze ---


class Store(dict):
    """The four ``ObjectStore`` calls that admission makes, over a dict."""

    put = dict.__setitem__
    get = dict.__getitem__
    delete = dict.pop

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


class Counting(WordTokens):
    calls = 0

    def prompt_tokens(self, prompt: str) -> int:
        self.calls += 1
        return super().prompt_tokens(prompt)


def admission(build, store=None, **changes) -> StoreAdmission:
    return StoreAdmission(Store() if store is None else store, build.params(**changes), tokens=Counting(), grader=grade, clock=lambda: 77.9)


def test_verdicts_report_the_format_only(build):
    adm = admission(build)
    secret = file([{"prompt": "double 3", "rollouts": [{"output": "my secret text <|im_end|>", "reward": 1}]}], train={"rank": 16})
    adm.store.put(layout.miner_object("good"), file(hotkey="good", train={"rank": 16}))
    adm.store.put(layout.miner_object(HOTKEY), secret)
    adm.store.put(layout.miner_object("thief"), file(hotkey="good"))  # a copy of another miner's file
    got = adm.verdicts(["good", HOTKEY, "thief", "absent"])
    assert list(got) == ["good", HOTKEY, "thief"]  # no stored file, no verdict
    good = got["good"]
    assert (good.valid, good.reason, good.format, good.n_prompts, good.n_rollouts, good.checked_at) == (True, "", "affine-batch-v3", 10, 20, 77)
    assert good.batch_sha256 == adm.store.sha256(layout.miner_object("good"))
    bad = got[HOTKEY]
    assert (bad.valid, bad.n_prompts, bad.batch_sha256) == (False, 0, sha256_hex(secret))
    assert "control token" in bad.reason and "secret" not in bad.reason
    assert not got["thief"].valid and "hotkey" in got["thief"].reason


def test_verdicts_check_a_file_again_only_when_it_changes(build):
    adm = admission(build)
    adm.store.put(layout.miner_object(HOTKEY), file())
    first = adm.verdicts([HOTKEY])[HOTKEY]
    calls = adm.tokens.calls
    assert adm.verdicts([HOTKEY])[HOTKEY] is first and adm.tokens.calls == calls
    adm.store.put(layout.miner_object(HOTKEY), b"{}")  # the miner overwrites the file
    assert not adm.verdicts([HOTKEY])[HOTKEY].valid
    adm.store.put(layout.miner_object(HOTKEY), file())
    adm.params = build.params(max_rollouts=1)  # new parameters: the same bytes are checked again
    assert not adm.verdicts([HOTKEY])[HOTKEY].valid
    del adm.store[layout.miner_object(HOTKEY)]
    assert adm.verdicts([HOTKEY]) == {}


def test_oversized_object_is_never_downloaded(build):
    class NoDownload(Store):
        def get(self, key):
            raise AssertionError("downloaded")

        sha256 = get

    adm = admission(build, NoDownload(), max_batch_bytes=100)
    adm.store.put(layout.miner_object(HOTKEY), file())
    assert "100 bytes" in adm.verdicts([HOTKEY])[HOTKEY].reason
    assert adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001") is None


def test_freeze_copies_the_checked_bytes_and_is_safe_to_repeat(build):
    adm, reg = admission(build), build.reg(1, hotkey=HOTKEY)
    raw = file(train={"rank": 16, "target_modules": ["q_proj"]})
    adm.store.put(layout.miner_object(HOTKEY), raw)
    frozen = adm.freeze(reg, "r000001")
    assert frozen.uri == layout.frozen_batch("r000001", HOTKEY) and frozen.uri.startswith("private/")
    assert adm.store[frozen.uri] == raw and frozen.batch_sha256 == sha256_hex(raw)
    assert (frozen.hotkey, frozen.format, frozen.n_prompts, frozen.n_rollouts) == (HOTKEY, "affine-batch-v3", 10, 20)
    assert frozen.train == {"rank": 16, "target_modules": ["q_proj"]}
    adm.store.put(layout.miner_object(HOTKEY), b"{}")  # a later overwrite does not touch the frozen round
    assert adm.freeze(reg, "r000001") == frozen and adm.store[frozen.uri] == raw
    assert adm.freeze(reg, "r000002") is None  # the next round reads the new, invalid file: a skip


def test_freeze_skips_a_missing_or_invalid_file_and_writes_nothing(build):
    adm = admission(build)
    assert adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001") is None
    adm.store.put(layout.miner_object(HOTKEY), file(train={"rank": 10**6}))
    assert adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001") is None
    assert list(adm.store) == [layout.miner_object(HOTKEY)]


def test_freeze_raises_on_a_damaged_copy_or_a_store_failure(build):
    class Damaging(Store):
        def put(self, key, data):
            self[key] = data + b" " if "/rounds/" in key else data

    class Down(Store):
        def size(self, key):
            raise InfraError("store is down")

    adm = admission(build, Damaging())
    adm.store.put(layout.miner_object(HOTKEY), file())
    with pytest.raises(VerifyError):
        adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001")
    assert list(adm.store) == [layout.miner_object(HOTKEY)]  # the damaged copy is gone, so a replay starts clean
    for call in (lambda: admission(build, Down()).freeze(build.reg(1, hotkey=HOTKEY), "r000001"), lambda: admission(build, Down()).verdicts([HOTKEY])):
        with pytest.raises(InfraError):
            call()
