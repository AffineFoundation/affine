"""The output token counts of one file taken many texts at a time: the same verdict, rule and reason as one text at a time."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import TOKENIZER_FAULT, HFTokens, Invalid, StoreAdmission, check_batch
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import InfraError

HOTKEY = "5MinerA"
THINK = "step " * 40
BROKEN = "<break>"  # a text with this word makes the tokenizers of this file fail


class WordTokens:
    """One text at a time, as before. ``asked`` holds every output text that was counted, in order."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        self.asked.append(text)
        if BROKEN in text:
            raise ValueError("this text breaks the tokenizer")
        return len(text.split())


class ManyTokens(WordTokens):
    """The same counts, and many texts in one call. ``calls`` holds the texts of each call."""

    def __init__(self, spoil=None) -> None:
        super().__init__()
        self.calls: list[list[str]] = []
        self.spoil = spoil

    def output_tokens_many(self, texts):
        self.calls.append(list(texts))
        counts = [len(text.split()) for text in texts]
        if any(BROKEN in text for text in texts):
            raise ValueError("a text of this call breaks the tokenizer")
        return counts if self.spoil is None else self.spoil(counts)


class Store(dict):
    put = dict.__setitem__
    get = dict.__getitem__

    def delete(self, key: str) -> None:
        self.pop(key, None)

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


def row(answer, reward, think: str = THINK) -> dict:
    return {"output": f"{think}\\boxed{{{answer}}}", "reward": reward}


def file(groups=None, hotkey: str = HOTKEY) -> bytes:
    """By default 10 prompts "double N", each with a right and a wrong rollout."""
    groups = [{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0)]} for n in range(10)] if groups is None else groups
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": groups}).encode()


def with_output(output: str, at: int = 7) -> bytes:
    """The default file with one other output text in rollout 1 of group ``at``."""
    return file([{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), {"output": output, "reward": 0} if n == at else row(-1, 0)]} for n in range(10)])


def outcome(raw: bytes, params, tokens):
    """What the check of one file gives: the batch, or the rule, the reason and whether the fault is proven."""
    try:
        return check_batch(raw, HOTKEY, params, tokens=tokens)
    except Invalid as exc:
        return exc.rule, str(exc), exc.proven
    except InfraError as exc:
        return "InfraError", str(exc)


def small(build, **changes):
    return build.params(**(dict(max_output_tokens=60, max_context_tokens=70, max_answer_only_frac=0.2, reasoning_token_share=0.8, max_answer_repeat_count=12, max_answer_repeat_frac=0.1) | changes))


CASES = {
    "valid": file(),
    "output_tokens": with_output("word " * 61 + "\\boxed{1}"),
    "context_tokens": file([{"prompt": "double 1 " + "and " * 20, "rollouts": [row(2, 1, "step " * 50), row(-1, 0)]}] + [{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0)]} for n in range(2, 10)]),
    "answer_only": file([{"prompt": f"double {n}", "rollouts": [row(2 * n, 1, "so "), row(-1, 0)]} for n in range(10)]),
    "reasoning_share": file([{"prompt": f"double {n}", "rollouts": [{"output": f"{THINK}\\boxed{{{n}}} " + "after " * 15, "reward": 0}, {"output": f"{THINK}\\boxed{{{n + 100}}} " + "after " * 15, "reward": 1}]} for n in range(10)]),
    "tokenizer_fault": with_output(THINK + BROKEN + "\\boxed{1}"),
    "tokenizer_fault_at_the_first_group": with_output(THINK + BROKEN + "\\boxed{1}", at=0),
    "control_token": with_output(THINK + "<|im_end|>\\boxed{1}"),
    "no_boxed_answer": with_output("step " * 45),
    "two_boxed_answers": with_output(THINK + "\\boxed{1} or \\boxed{2}"),
    "answer_repeat": file([{"prompt": f"double {n}", "rollouts": [row(4, 1), row(4, 0)]} for n in range(10)]),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("ahead", [1, 300, 2_000_000])
def test_many_texts_in_one_call_give_the_verdict_and_the_reason_of_one_text_at_a_time(name, ahead, build, monkeypatch):
    monkeypatch.setattr(admission, "AHEAD_CHARS", ahead)
    params = small(build)
    one, many = WordTokens(), ManyTokens()
    assert outcome(CASES[name], params, many) == outcome(CASES[name], params, one)
    assert many.calls, "the texts are counted many at a time"
    if BROKEN not in CASES[name].decode():
        assert many.asked == [], "no text is counted twice"


def test_the_cases_cover_each_token_rule(build):
    found = {name: got[0] if isinstance(got, tuple) else "" for name, raw in CASES.items() for got in [outcome(raw, small(build), ManyTokens())]}
    assert found == {name: name if name in admission.RULES else "" for name in CASES} | {"tokenizer_fault_at_the_first_group": TOKENIZER_FAULT}
    assert outcome(CASES["tokenizer_fault"], small(build), ManyTokens()) == (TOKENIZER_FAULT, "groups[7].rollouts[1].output: the tokenizer fails on this text", True)


def test_random_files_get_the_same_outcome_with_any_size_of_a_call(build, monkeypatch):
    rng = random.Random(120)
    params = small(build, max_answer_only_frac=0.1, reasoning_token_share=0.9, max_answer_repeat_count=50)
    rules = set()
    for _ in range(300):
        groups = []
        for n in range(rng.randint(1, 8)):
            rollouts = []
            for reward in (1, 0, 0)[: rng.randint(2, 3)]:
                think = "step " * rng.choice([1, 33, 40, 40, 40, 40, 40, 56, 70]) + rng.choice([""] * 30 + [BROKEN + " "])
                tail = rng.choice([f"\\boxed{{{n + reward}}}"] * 6 + [f"\\boxed{{{n}}} then " + "more " * rng.randint(0, 16) + f"\\boxed{{{n + reward}}}", "no answer"])
                rollouts.append({"output": think + tail, "reward": reward})
            groups.append({"prompt": f"double {n}", "rollouts": rollouts})
        raw = file(groups)
        want = outcome(raw, params, WordTokens())
        rules.add(want[0] if isinstance(want, tuple) else "")
        for ahead in (1, rng.randint(2, 2000), 2_000_000):
            monkeypatch.setattr(admission, "AHEAD_CHARS", ahead)
            assert outcome(raw, params, ManyTokens()) == want
    assert {"", "output_tokens", "answer_only", "reasoning_share", TOKENIZER_FAULT} <= rules


def test_one_call_holds_a_bounded_count_of_characters_and_the_walk_asks_in_order(build, monkeypatch):
    monkeypatch.setattr(admission, "AHEAD_CHARS", 1000)
    one, many = WordTokens(), ManyTokens()
    assert outcome(file(), build.params(), many) == outcome(file(), build.params(), one)
    assert [text for call in many.calls for text in call] == one.asked, "the same texts in the same order"
    assert len(many.calls) > 5
    for call in many.calls:
        assert sum(map(len, call[:-1])) < 1000, "a call ends with the text that reaches the bound"
    big = "step " * 500
    assert outcome(with_output(big + "\\boxed{1}"), build.params(), many := ManyTokens()) == outcome(with_output(big + "\\boxed{1}"), build.params(), WordTokens())
    assert [call for call in many.calls if big in call[-1]] == [[big + "\\boxed{1}"], [big]], "a text over the bound is a call of its own"


def test_a_file_that_fails_at_its_first_row_costs_one_call(build, monkeypatch):
    monkeypatch.setattr(admission, "AHEAD_CHARS", 500)
    many = ManyTokens()
    raw = file([{"prompt": "double 0", "rollouts": [row(0, 1, "word " * 61), row(-1, 0)]}] + [{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0)]} for n in range(1, 10)])
    assert outcome(raw, small(build), many)[0] == "output_tokens"
    assert len(many.calls) == 1 and sum(map(len, many.calls[0])) < 1000


class Panic(BaseException):
    """What a Rust tokenizer raises when it panics."""


@pytest.mark.parametrize("cause", [ZeroDivisionError("x"), MemoryError(), OSError("no thread"), Panic("rust"), InfraError("ours")])
def test_a_call_that_raises_is_followed_by_one_text_at_a_time_with_the_same_verdict(cause, build):
    def spoil(counts):
        raise cause

    many = ManyTokens(spoil)
    for name, raw in CASES.items():
        many.calls.clear()
        assert outcome(raw, small(build), many) == outcome(raw, small(build), WordTokens()), name
        assert len(many.calls) == 1, "no other call for this file"


@pytest.mark.parametrize("spoil", [
    lambda counts: counts[:-1],
    lambda counts: counts + [1],
    lambda counts: [None] * len(counts),
    lambda counts: [-1] + counts[1:],
    lambda counts: [float(n) for n in counts],
    lambda counts: [True] * len(counts),
    lambda counts: None,
    lambda counts: 7,
])
def test_a_call_that_does_not_give_one_count_for_each_text_is_not_used(spoil, build):
    many, one = ManyTokens(spoil), WordTokens()
    assert outcome(file(), build.params(), many) == outcome(file(), build.params(), one)
    assert len(many.calls) == 1 and many.asked == one.asked


def test_a_stop_by_the_operator_is_not_caught(build):
    def stop(counts):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        check_batch(file(), HOTKEY, build.params(), tokens=ManyTokens(stop))


def test_a_tokenizer_that_fails_on_every_text_is_our_failure_as_before(build):
    class Dead(ManyTokens):
        def prompt_tokens(self, prompt):
            raise RuntimeError("no tokenizer")

    assert outcome(file(), build.params(), Dead()) == ("InfraError", "the tokenizer fails on its known-good case: RuntimeError")


def test_a_file_that_breaks_the_call_does_not_change_the_verdict_of_another_file(build):
    files = {"first": file(hotkey="first"), "bad": with_output(THINK + BROKEN + "\\boxed{1}").replace(HOTKEY.encode(), b"bad"), "last": file(hotkey="last")}
    got = {}
    for tokens in (WordTokens(), ManyTokens()):
        adm = StoreAdmission(Store(), small(build), tokens=tokens, grader=None)
        for hotkey, raw in files.items():
            adm.store.put(layout.miner_object(hotkey), raw)
        got[type(tokens)] = [(hotkey, verdict.valid, verdict.rule, verdict.reason) for hotkey, verdict in adm.verdicts(list(files)).items()]
        assert [(valid, rule) for _, valid, rule, _ in got[type(tokens)]] == [(True, ""), (False, TOKENIZER_FAULT), (True, "")]
    assert got[ManyTokens] == got[WordTokens]
    assert [BROKEN in "".join(call) for call in tokens.calls] == [False, True, False], "the file after the broken one is counted many at a time again"


def test_without_a_tokenizer_nothing_is_counted(build):
    assert check_batch(file(), HOTKEY, build.params()).checked == ("format",)


def test_the_tokenizer_of_the_head_hands_the_call_on():
    assembly = pytest.importorskip("sn120_v2.assembly", reason="install the miner extra: uv pip install -e '.[dev,miner]'")
    tokens = assembly.HeadTokens(Store(), None)
    tokens._tokens = ManyTokens()
    assert tokens.output_tokens_many(["a b", "c"]) == [2, 1] and tokens._tokens.calls == [["a b", "c"]]


def test_a_fast_tokenizer_counts_each_text_of_a_call_as_it_counts_the_text_alone(build):
    transformers = pytest.importorskip("transformers", reason="install the miner extra: uv pip install -e '.[dev,miner]'")
    tokenizers = pytest.importorskip("tokenizers")
    rng = random.Random(7)
    words = ["step", "so", "x", "=", "12", "\\boxed", "{", "}", "+", "frac", "\n", "é", "数", "<think>"]
    texts = ["".join(rng.choice(words) + rng.choice(["", " ", "  "]) for _ in range(rng.randint(0, 400))) for _ in range(200)] + ["", " ", "a" * 5000]
    core = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.train_from_iterator(texts, tokenizers.trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "[PAD]", "<think>"]))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", model_max_length=16)
    tokens = HFTokens(fast)
    alone = [tokens.output_tokens(text) for text in texts]
    assert tokens.output_tokens_many(texts) == alone and max(alone) > 16, "no text is cut at the length of the model"
    assert tokens.output_tokens_many([]) == [] and tokens.output_tokens_many(texts[:1]) == alone[:1]
    raw = file([{"prompt": f"double {n}", "rollouts": [row(2 * n, 1, texts[n] + " "), row(-1, 0, texts[n + 50] + " ")]} for n in range(40)])
    params = replace(build.params(), max_answer_only_frac=1.0, reasoning_token_share=0.0)

    class Alone:
        prompt_tokens = staticmethod(lambda prompt: 5)
        output_tokens = tokens.output_tokens

    class Many(Alone):
        output_tokens_many = tokens.output_tokens_many

    assert check_batch(raw, HOTKEY, params, tokens=Many()) == check_batch(raw, HOTKEY, params, tokens=Alone())
