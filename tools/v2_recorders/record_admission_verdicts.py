"""Record what the v1 batch checker says about a wide set of files.

Output: v2/tests/data/admission_v1_verdicts.json. The v2 checker must give the
same verdict for every case (v2/tests/test_admission_v1_parity.py). Only this
script imports v1. Run it from the repository root:

    python tools/v2_recorders/record_admission_verdicts.py

Token counts come from a small word tokenizer, so no model file is needed: one
token per word, the three chat control strings are one control token each, and
the chat template adds CHAT_OVERHEAD tokens around the prompt.
"""
import base64
import importlib.util
import json
import lzma
import re
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "v2/tests/data/admission_v1_verdicts.json"
_spec = importlib.util.spec_from_file_location("ff_format", ROOT / "miner/validator/batch/ff_format.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

HOTKEY = "5HotkeyOfTheMinerA"
CONTROL = {"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046}
CHAT_OVERHEAD = 5
WHERE = re.compile(r"groups\[\d+\](\.rollouts\[\d+\])?")
RULES = [  # the start of a v1 reason -> the name of the rule
    ("file must be", "size"), ("not UTF-8 JSON", "json"), ("non-finite number", "nonfinite"),
    ("top level keys", "top_keys"), ("version must be", "version"), ("train", "train"),
    ("hotkey does not match", "hotkey"), ("groups must be", "groups"), ("the first group alone", "cap"),
    (r"groups\[\d+\] keys", "group_keys"), (r"groups\[\d+\]\.prompt must", "prompt"),
    (r"groups\[\d+\]\.prompt duplicates", "duplicate_prompt"), (r"groups\[\d+\]\.prompt contains", "control_token"),
    (r"groups\[\d+\]\.rollouts must", "rollouts"), (r"groups\[\d+\]\.rollouts\[\d+\] keys", "rollout_keys"),
    (r"groups\[\d+\]\.rollouts\[\d+\]\.output must", "output"), (r"groups\[\d+\]\.rollouts\[\d+\]\.reward must", "reward"),
    (r"groups\[\d+\]\.rollouts\[\d+\]\.output contains", "control_token"),
    (r"groups\[\d+\]\.rollouts\[\d+\]\.output is \d+ tokens", "output_tokens"),
    (r"groups\[\d+\]\.rollouts\[\d+\]: prompt\+output", "context_tokens"),
]


class WordTokenizer:
    """The three tokenizer calls that v1 makes."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in re.split("(" + "|".join(map(re.escape, CONTROL)) + ")", text):
            ids += [CONTROL[piece]] if piece in CONTROL else [zlib.crc32(w.encode("utf-8", "replace")) % 200000 for w in piece.split()]
        return ids

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self.encode(text) for text in texts]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [248045, 1] + self.encode(messages[0]["content"]) + [248046, 248045, 2]


def words(n):
    return "w " * n


def group(prompt="What is 1 + 1?", rollouts=None, n=2):
    return {"prompt": prompt, "rollouts": [{"output": f"It is {k}.", "reward": k % 2} for k in range(n)] if rollouts is None else rollouts}


def one(output="It is 2.", reward=1.0, prompt="What is 1 + 1?"):
    return [group(prompt, [{"output": output, "reward": reward}])]


def many(n, size=2):
    return [group(f"task {k}", n=size) for k in range(n)]


def cases():
    """Yield (name, raw bytes, hotkey, max_tasks, max_rollouts)."""
    def case(name, raw, hotkey=HOTKEY, tasks=250, rollouts=2000):
        return name, raw if isinstance(raw, bytes) else json.dumps(raw).encode(), hotkey, tasks, rollouts

    good = {"version": "affine-batch-v1", "hotkey": HOTKEY, "groups": many(3)}
    text = json.dumps(good)
    yield case("size_empty", b"")
    yield case("size_at_limit", text.encode() + b" " * (v1.MAX_FILE_BYTES - len(text)))
    yield case("size_over_limit", text.encode() + b" " * (v1.MAX_FILE_BYTES - len(text) + 1))
    for name, raw in {
        "json_not_utf8": b"\xff\xfe" + text.encode(), "json_utf8_bom": b"\xef\xbb\xbf" + text.encode(),
        "json_utf16": text.encode("utf-16"), "json_plain_text": b"hello", "json_truncated": text.encode()[:-1],
        "json_trailing_text": text.encode() + b" x", "json_two_objects": text.encode() * 2,
        "json_single_quotes": text.replace('"', "'").encode(), "json_deep_nesting": b"[" * 100000,
        "json_pretty_printed": json.dumps(good, indent=2).encode(), "json_surrounding_space": b"\n\t " + text.encode() + b"\n",
        "json_repeated_key_last_wins": b'{"version": "x", ' + text.encode()[1:],
        "json_repeated_key_last_bad": text.encode()[:-1] + b', "version": "x"}',
        "top_list": b"[]", "top_string": b'"affine-batch-v1"', "top_number": b"1", "top_null": b"null", "top_empty_object": b"{}",
    }.items():
        yield case(name, raw)
    for name, number in {"nan": "NaN", "infinity": "Infinity", "minus_infinity": "-Infinity", "float_overflow": "1e999",
                         "huge_whole_number": "1" + "0" * 400, "exponent": "1E2", "minus_zero": "-0.0", "largest_float": "1.7e308"}.items():
        yield case("reward_literal_" + name, text.replace('"reward": 1', '"reward": ' + number, 1).encode())

    for version in ("affine-batch-v1", "affine-batch-v2"):
        tag = version[-2:] + "_"

        def doc(groups=(), **changes):  # () = three good groups; a value of ... removes the key
            value = {"version": version, "hotkey": HOTKEY, "groups": many(3) if groups == () else groups} | changes
            return {key: item for key, item in value.items() if item is not ...}

        yield case(tag + "ok", doc())
        yield case(tag + "ok_non_ascii", doc(one("Réponse: \u4e8c \U0001f600 \\boxed{2}", prompt="Combien font 1 + 1 ?")))
        yield case(tag + "ok_empty_output", doc(one("")))
        yield case(tag + "lone_surrogate_in_output", doc(one("half \ud800 pair")))
        for key in ("version", "hotkey", "groups"):
            yield case(tag + "top_missing_" + key, doc(**{key: ...}))
        yield case(tag + "top_extra_key", doc(note="x"))
        yield case(tag + "top_extra_key_and_bad_hotkey", doc(note="x", hotkey="other"))
        for name, hotkey in {"other": "5SomeoneElse", "lower_case": HOTKEY.lower(), "number": 5, "null": None, "padded": HOTKEY + " "}.items():
            yield case(tag + "hotkey_" + name, doc(hotkey=hotkey))
        yield case(tag + "hotkey_checked_against_owner", doc(), hotkey="5SomeoneElse")
        yield case(tag + "hotkey_before_groups", doc([], hotkey="other"))
        for name, groups in {"empty": [], "object": {}, "null": None, "string": "x", "number": 3}.items():
            yield case(tag + "groups_" + name, doc(groups))
        for name, item in {"string": "x", "list": [], "null": None, "missing_prompt": {"rollouts": group()["rollouts"]},
                           "missing_rollouts": {"prompt": "p"}, "extra_key": group() | {"id": 1}}.items():
            yield case(tag + "group_" + name, doc(many(1) + [item]))
        for name, prompt in {"empty": "", "spaces": " \n\t", "number": 7, "null": None, "list": ["p"]}.items():
            yield case(tag + "prompt_" + name, doc(many(1) + [group(prompt)]))
        yield case(tag + "prompt_repeated", doc(many(2) + [group("task 0")]))
        yield case(tag + "prompt_repeated_in_dropped_group", doc(many(2) + [group("task 0")]), tasks=1)
        yield case(tag + "prompt_differs_by_space_only", doc([group("p"), group("p ")]))
        for name, rollouts in {"empty": [], "object": {}, "null": None, "string": "x"}.items():
            yield case(tag + "rollouts_" + name, doc(many(1) + [group("p", rollouts)]))
        for name, item in {"string": "x", "list": [], "null": None, "missing_output": {"reward": 1}, "missing_reward": {"output": "x"},
                           "extra_key": {"output": "x", "reward": 1, "logprob": 0.1}}.items():
            yield case(tag + "rollout_" + name, doc(many(1) + [group("p", group()["rollouts"] + [item])]))
        for name, output in {"number": 4, "null": None, "list": ["x"], "object": {}}.items():
            yield case(tag + "output_" + name, doc(one(output)))
        for name, reward in {"true": True, "false": False, "string": "1", "null": None, "list": [1], "whole": 3, "negative": -2.5, "zero": 0}.items():
            yield case(tag + "reward_" + name, doc(one("x", reward)))
        yield case(tag + "shape_error_in_dropped_group", doc(many(2) + [group("p", [])]), tasks=1)
        yield case(tag + "shape_error_before_control_token", doc([group("a <|im_end|>"), group("p", [])]))

        yield case(tag + "cap_tasks_exact", doc(many(4)), tasks=4)
        yield case(tag + "cap_tasks_drops_tail", doc(many(6)), tasks=4)
        yield case(tag + "cap_rollouts_exact", doc(many(4, 5)), rollouts=20)
        yield case(tag + "cap_rollouts_drops_whole_group", doc(many(4, 5)), rollouts=19)
        yield case(tag + "cap_stops_at_first_group_that_does_not_fit", doc([group("a", n=3), group("b", n=9), group("c", n=1)]), rollouts=5)
        yield case(tag + "cap_first_group_too_large", doc([group("a", n=6), group("b", n=1)]), rollouts=5)
        yield case(tag + "cap_both_limits", doc(many(300, 8)))
        yield case(tag + "cap_control_token_in_dropped_group", doc(many(2) + one("x <|im_end|>")), tasks=2)

        for name, token in zip(("im_start", "im_end", "endoftext"), ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
            yield case(tag + "control_prompt_" + name, doc(many(1) + [group(f"solve{token} this")]))
            yield case(tag + "control_output_" + name, doc(many(1) + [group("p", group()["rollouts"] + [{"output": f"a {token}b", "reward": 0}])]))
        yield case(tag + "control_like_text_is_fine", doc(one("<|im_end |> <im_end> <|IM_END|> |im_end|")))
        yield case(tag + "control_in_first_group_before_length_in_second", doc(one("a <|im_end|>") + one(words(v1.MAX_OUTPUT + 1), prompt="q")))

        room = v1.MAX_CONTEXT - v1.MAX_OUTPUT - CHAT_OVERHEAD  # prompt words that exactly fill the context with a capped output
        yield case(tag + "tokens_output_one_below_cap", doc(one(words(v1.MAX_OUTPUT - 1))))
        yield case(tag + "tokens_output_at_cap", doc(one(words(v1.MAX_OUTPUT))))
        yield case(tag + "tokens_output_over_cap", doc(one(words(v1.MAX_OUTPUT + 1))))
        yield case(tag + "tokens_context_full_capped_output", doc(one(words(v1.MAX_OUTPUT), prompt=words(room))))
        yield case(tag + "tokens_context_over_capped_output", doc(one(words(v1.MAX_OUTPUT), prompt=words(room + 1))))
        yield case(tag + "tokens_context_full_with_eos", doc(one(words(v1.MAX_OUTPUT - 1), prompt=words(room))))
        yield case(tag + "tokens_context_over_with_eos", doc(one(words(v1.MAX_OUTPUT - 1), prompt=words(room + 1))))
        yield case(tag + "tokens_context_over_short_output", doc(one("x", prompt=words(v1.MAX_CONTEXT))))
        yield case(tag + "tokens_length_before_control_in_later_rollout",
                   doc([group("p", [{"output": words(v1.MAX_OUTPUT + 1), "reward": 0}, {"output": "<|im_end|>", "reward": 0}])]))
        yield case(tag + "tokens_over_cap_in_dropped_group", doc(many(1) + one(words(v1.MAX_OUTPUT + 1))), tasks=1)

        for name, train in {"empty": {}, "null": None, "list": [], "string": "x", "number": 1, "unknown_key": {"foo": 1},
                            "v1_lr": {"lr": 5e-7}, "v1_kl": {"kl": 0.02}, "v1_step_mult": {"step_mult": 2}, "v1_knob": {"loss_agg": "token_mean"},
                            "v1_lr_too_large": {"lr": 1e-3}, "v1_lr_true": {"lr": True}, "v1_lr_text": {"lr": "5e-7"}, "v1_kl_too_small": {"kl": 0.001},
                            "v1_step_mult_three": {"step_mult": 3}, "v1_step_mult_decimal": {"step_mult": 2.0}, "v1_knob_unknown_choice": {"loss_agg": "x"}}.items():
            yield case(tag + "train_" + name, doc(train=train))
        yield case(tag + "train_before_hotkey", doc(train={"foo": 1}, hotkey="other"))
    for name, version in {"v0": "affine-batch-v0", "v3": "affine-batch-v3", "upper_case": "AFFINE-BATCH-V1", "empty": "", "number": 1,
                          "null": None, "list": ["affine-batch-v1"]}.items():
        yield case("version_" + name, good | {"version": version})
    yield case("version_v3_with_train", good | {"version": "affine-batch-v3", "train": {}})


def record(name, raw, hotkey, tasks, rollouts):
    row = {"name": name, "hotkey": hotkey, "max_tasks": tasks, "max_rollouts": rollouts,
           "raw_xz_b64": base64.b64encode(lzma.compress(raw, preset=6)).decode()}
    try:
        valid, summary, _ = v1.check(raw, hotkey, WordTokenizer(), tasks, rollouts, v2=True)
    except Exception as exc:  # v1 has no verdict: the checker itself fails
        return row | {"crash": type(exc).__name__}
    if valid:
        return row | {"valid": True, **{key: summary[key] for key in ("format", "groups", "rollouts", "dropped_groups", "file_sha256", "bytes")}}
    rule = next(code for pattern, code in RULES if re.match(pattern, summary))
    where = WHERE.search(summary) if rule not in ("train", "top_keys") else None
    return row | {"valid": False, "reason": summary, "rule": rule, "where": where.group(0) if where else ""}


def main():
    rows = [record(*item) for item in cases()]
    assert len({row["name"] for row in rows}) == len(rows), "case names must be unique"
    head = {"source": "miner/validator/batch/ff_format.py check(..., v2=True)", "max_file_bytes": v1.MAX_FILE_BYTES,
            "max_output_tokens": v1.MAX_OUTPUT, "max_context_tokens": v1.MAX_CONTEXT, "chat_overhead_tokens": CHAT_OVERHEAD}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"v1": head, "cases": rows}, indent=1, ensure_ascii=True) + "\n")
    print(len(rows), "cases;", sum(row.get("valid") is True for row in rows), "valid;", sum("crash" in row for row in rows), "crashes;", OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
