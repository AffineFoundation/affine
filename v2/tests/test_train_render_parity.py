"""Training and admission use the exact prompt tokens of the scoring request."""

from __future__ import annotations

import pytest

from sn120_v2.admission import HFTokens
from sn120_v2.envs.math_pool import MANDATE, load_pool
from sn120_v2.gpu.batch import render_prompt_ids
from sn120_v2.params import GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, Params, Recipe

train_job = pytest.importorskip("sn120_v2.gpu.train_job")
from sn120_v2.gpu.evaluator import chat_body  # noqa: E402


class FakeTokenizer:
    """One token per character, and distinct system/user/thinking template markers."""

    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert add_generation_prompt and not tokenize
        text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        return text + ("<assistant><think>" if kwargs.get("enable_thinking") else "<assistant>")

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


def scoring_ids(tokenizer, task) -> list[int]:
    body = chat_body("head", task.messages(), 5, Params())
    text = tokenizer.apply_chat_template(
        body["messages"], add_generation_prompt=True, tokenize=False, **body["chat_template_kwargs"]
    )
    return tokenizer.encode(text, add_special_tokens=False)


def check_parity(tokenizer) -> None:
    pool = load_pool()
    tasks = [pool.task(task_id) for task_id in ("math:0", "math:7", "dapo:3", "dapo:5")]
    assert all("\\" in task.problem and "\n" in task.problem for task in (tasks[0], tasks[2], tasks[3]))
    assert any(not task.problem.isascii() for task in tasks)
    for task in tasks:
        expected = scoring_ids(tokenizer, task)
        for prompt in (task.batch_prompt(), task.problem):
            rows, dropped = train_job.encode(tokenizer, [{"prompt": prompt, "output": "answer", "advantage": 1.0}], Recipe())
            assert dropped == 0 and rows[0]["prompt_ids"] == expected
            assert HFTokens(tokenizer).prompt_tokens(prompt) == len(expected)
        text = tokenizer.decode(expected, skip_special_tokens=False)
        assert text.count(MANDATE) == 1 and text.index(MANDATE) < text.index(task.problem)
    problem = "An arbitrary math prompt with π and a line break\nnext"
    rows, dropped = train_job.encode(tokenizer, [{"prompt": problem, "output": "answer", "advantage": 1.0}], Recipe())
    assert dropped == 0 and rows[0]["prompt_ids"] == render_prompt_ids(tokenizer, problem)
    text = tokenizer.decode(rows[0]["prompt_ids"], skip_special_tokens=False)
    assert text.count(MANDATE) == 1 and text.index(MANDATE) < text.index(problem)


def test_pool_batch_prompts_match_scoring_with_a_distinct_fake_template():
    check_parity(FakeTokenizer())


def test_pool_batch_prompts_match_scoring_with_the_real_head_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            GENESIS_MODEL_REPO, revision=GENESIS_MODEL_REVISION, local_files_only=True
        )
    except (OSError, ValueError) as error:
        pytest.skip(f"the head's tokenizer is not available locally offline: {type(error).__name__}")
    check_parity(tokenizer)
