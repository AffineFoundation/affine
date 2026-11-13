"""The v2 checker gives the verdict that the v1 checker gave, file by file.

``data/admission_v1_verdicts.json`` holds the files and what v1 said about each
(recorded by tools/v2_recorders/record_admission_verdicts.py, outside v2).
"""

from __future__ import annotations

import base64
import json
import lzma
import re
from pathlib import Path

import pytest

from sn120_v2 import admission
from sn120_v2.admission import Invalid, check_batch

RECORDED = json.loads((Path(__file__).parent / "data/admission_v1_verdicts.json").read_text())
V1 = RECORDED["v1"]
WHERE = re.compile(r"groups\[\d+\](\.rollouts\[\d+\])?")

# Every file where v2 differs from v1 on purpose: name -> the v2 rule that fails ("" = v2 accepts the file).
ON_PURPOSE = {
    "reward_literal_huge_whole_number": "reward",  # v1 gives no verdict: its checker stops with OverflowError
    "v1_lone_surrogate_in_output": "text",  # half of a surrogate pair is not text a tokenizer can read
    "v2_lone_surrogate_in_output": "text",
    "version_v3": "",  # the spec adds affine-batch-v3
    "version_v3_with_train": "",
    "v2_train_v1_lr": "train",  # the v1 train keys are not settable in v2
    "v2_train_v1_kl": "train",
    "v2_train_v1_step_mult": "train",
    "v2_train_v1_knob": "train",
}


class WordTokens:
    """The recorder's tokenizer: one token per word, and a fixed number of chat template tokens around a prompt."""

    def prompt_tokens(self, prompt: str) -> int:
        return V1["chat_overhead_tokens"] + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.fixture(autouse=True)
def v1_rules_only(monkeypatch):
    """v1 had no training-signal rule and no cost cap. Many recorded files give every rollout of a prompt the same reward."""
    monkeypatch.setattr(admission, "_trains", lambda *inputs: 0)


def judge(case, build) -> tuple:
    """The v2 verdict under the v1 limits, with the content rules that only v2 has switched off."""
    params = build.params(
        max_batch_bytes=V1["max_file_bytes"], max_output_tokens=V1["max_output_tokens"], max_context_tokens=V1["max_context_tokens"],
        max_prompts=case["max_tasks"], max_rollouts=case["max_rollouts"],
        min_reasoning_tokens=0, reasoning_token_share=0.0, max_answer_only_frac=1.0, max_answer_repeat_frac=1.0,
    )
    raw = lzma.decompress(base64.b64decode(case["raw_xz_b64"]))
    try:
        batch = check_batch(raw, case["hotkey"], params, tokens=WordTokens())
    except Invalid as exc:
        where = WHERE.search(str(exc))
        return False, exc.rule, where.group(0) if where else ""
    return True, batch.format, batch.n_prompts, batch.n_rollouts, batch.dropped_groups, batch.sha256, batch.n_bytes


@pytest.mark.parametrize("case", [case for case in RECORDED["cases"] if case["name"] not in ON_PURPOSE], ids=lambda case: case["name"])
def test_same_verdict_as_v1(case, build):
    if case["valid"]:
        want = (True, *(case[key] for key in ("format", "groups", "rollouts", "dropped_groups", "file_sha256", "bytes")))
    else:
        want = (False, case["rule"], case["where"])
    assert judge(case, build) == want


@pytest.mark.parametrize("name", ON_PURPOSE)
def test_difference_from_v1_on_purpose(name, build):
    case = next(case for case in RECORDED["cases"] if case["name"] == name)
    got = judge(case, build)
    assert (got[0], "" if got[0] else got[1]) == (ON_PURPOSE[name] == "", ON_PURPOSE[name])
