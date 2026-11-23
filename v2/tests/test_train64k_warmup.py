"""Pure tests of the opt-in 64k Triton warm-up tool; no GPU is required."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sn120_v2.gpu.batch import END_OF_TURN, fit_tokens
from sn120_v2.params import Recipe

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("train64k_warmup", ROOT / "tools" / "train64k" / "warmup.py")
assert SPEC and SPEC.loader
warmup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(warmup)


class StubTokenizer:
    unk_token_id = -1

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(len(text.split())))

    def convert_tokens_to_ids(self, token):
        assert token == END_OF_TURN
        return 999


def test_inclusive_lengths_and_kernel_classes():
    assert warmup.lengths(4, 10, 2) == [4, 6, 8, 10]
    assert warmup.lengths(32768, 69632, 2048)[-1] == 69632
    assert len(warmup.lengths()) == 19
    assert warmup.lengths(4, 5, 2) == [4]
    with pytest.raises(ValueError):
        warmup.lengths(5, 4, 1)
    with pytest.raises(ValueError):
        warmup.lengths(1, 4, 0)
    assert [warmup.l2norm_class(n) for n in (0, 4095, 4096, 4097, 69632)] == [0, 1, 1, 2, 17]


def test_rows_have_exact_requested_token_counts_and_nonzero_advantages():
    tokenizer = StubTokenizer()
    rows = warmup.warmup_rows(tokenizer, [12, 16], prompt_tokens=4)
    assert len(rows) == 4
    assert [len(tokenizer.encode(row["output"])) for row in rows] == [8, 8, 12, 12]
    assert [row["advantage"] for row in rows] == [0.5, -0.5, 0.5, -0.5]
    assert all(row["reward"] in (0.0, 1.0) for row in rows)


def test_production_fit_preserves_exact_long_row_counts():
    tokenizer = StubTokenizer()
    rows = warmup.warmup_rows(tokenizer, [12, 16], prompt_tokens=4)
    recipe = Recipe(train_max_output_tokens=65536, train_max_context_tokens=69632)
    fitted = [fit_tokens(4, tokenizer.encode(row["output"]), 999,
                         recipe.train_max_output_tokens, recipe.train_max_context_tokens) for row in rows]
    assert [len(tokens) for tokens, _capped, _cut in fitted] == [9, 9, 13, 13]
    assert all(tokens[-1] == 999 and not capped and not cut for tokens, capped, cut in fitted)


def test_compute_process_conflicts_and_fail_closed_parsing():
    gpus = "0, GPU-aaa\n1, GPU-bbb\n"
    assert warmup.compute_processes(gpus, "", [0, 1]) == {}
    processes = "GPU-bbb, 123, python, 456\nGPU-ccc, 124, other, 789\n"
    # An unknown process UUID is malformed, never silently treated as idle.
    with pytest.raises(ValueError, match="compute-process"):
        warmup.compute_processes(gpus, processes, [0, 1])
    known = "GPU-bbb, 123, python, 456\nGPU-aaa, 124, python, 789\n"
    assert warmup.compute_processes(gpus, known, [1]) == {1: ["123,python,456"]}
    with pytest.raises(ValueError, match="GPU row"):
        warmup.compute_processes("not-a-row", "", [0])
