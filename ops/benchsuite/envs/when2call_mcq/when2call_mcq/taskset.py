"""when2call-mcq: NVIDIA When2Call, TEST split, multiple-choice form.

When2Call (https://github.com/NVIDIA/When2Call) scores tool-calling
*decisions*: given a user message and a tool list, should the model call a
tool, answer directly, ask for missing information, or say it cannot help?
Each test row carries one candidate reply per class in `answers`
(`direct`, `tool_call`, `request_for_info`, `cannot_answer`) and the gold
class in `correct_answer`.

NVIDIA's reference MCQ eval (lm-evaluation-harness `output_type:
multiple_choice`) picks the candidate with the highest log-likelihood. An
OpenAI-compatible chat endpoint does not expose that, so this taskset asks
the model to *choose the letter* instead: same system prompt, same tool
rendering, same four candidates, one letter answer, deterministic
extraction (the GPQA regex). Numbers are therefore comparable across our
kings but not with the paper's table.

Held out from D. The datagen worker folds the TRAIN split (`train/*.jsonl`)
into D; only `test/when2call_test_mcq.jsonl` is read here, at a pinned Hub
revision, and its sha256 is checked before any task is yielded. The three
split hashes are recorded in `suite.lock.json`.
"""

import hashlib
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from gpqa.mcq import extract_mcq_answer
from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/When2Call"
REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
TEST_FILE = "test/when2call_test_mcq.jsonl"
TEST_SHA256 = "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c"
TRAIN_FILES = {
    "train/when2call_train_sft.jsonl": "3eb20258557513579995ff55c09fcc33fabf2cd2004dea49dc3a0ba9880e631c",
    "train/when2call_train_pref.jsonl": "d90637f108fabf1b097493c5818c3692e1d73140259d2f8e490536255e765bc4",
}

CLASSES = ("direct", "tool_call", "request_for_info", "cannot_answer")
LETTERS = ("A", "B", "C", "D")
# Per-row option shuffle seed; fixed so letters are stable across loads and reigns.
SHUFFLE_SEED = 0

# Verbatim from evaluation/mcq/lm_eval_harness/when2call/utils.py.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. \n"
    "You have access to the following tools described in <tool></tool> which you can use to answer the user's questions.\n"
    "Only use a tool if it directly answers the user's question.\n"
)
TOOL_USE_INSTRUCTIONS = (
    "To use a tool, return JSON in the following format:\n"
    '{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2", ...}}\n'
)
# Only the choice instruction is added; the decision rule stays NVIDIA's one
# line in the system prompt ("Only use a tool if it directly answers ...").
MCQ_PROMPT = (
    "Below are four candidate replies to the user's message. Choose the best one. The last line "
    "of your response should be of the following format: 'Answer: $LETTER' (without quotes) "
    "where LETTER is one of ABCD."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def format_tools(tools: list[str]) -> str:
    if not tools:
        return "(no tools available)"
    return "\n\n".join(f"<tool>{t}</tool>" for t in tools)


class When2CallData(vf.TaskData):
    network_allow: list[str] = []
    uuid: str
    answer: str
    """Gold option letter."""
    correct_class: str
    """Gold class: direct | tool_call | request_for_info | cannot_answer."""
    letter_to_class: dict[str, str]
    n_tools: int
    source: str


def acted_tool_call(reply: str) -> bool:
    """True when the visible reply IS a tool call: one JSON object with `name`
    and `arguments` (optionally inside a ``` fence). Seen on 20% of reign 11's
    tool_call rows: the model picks the tool option in its reasoning, then
    obeys the system prompt's "to use a tool, return JSON" and emits the call
    instead of the letter. Decision-wise that is the tool_call class."""
    s = reply.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        obj = json.loads(s)
    except ValueError:
        return False
    return isinstance(obj, dict) and "name" in obj and "arguments" in obj


class When2CallTask(vf.Task[When2CallData]):
    def predicted_class(self, trace: vf.Trace) -> str | None:
        letter = extract_mcq_answer(trace.last_reply)
        if letter:
            return self.data.letter_to_class.get(letter)
        return "tool_call" if acted_tool_call(trace.last_reply) else None

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return 1.0 if self.predicted_class(trace) == self.data.correct_class else 0.0

    @vf.metric
    async def letter_found(self, trace: vf.Trace) -> float:
        return 1.0 if extract_mcq_answer(trace.last_reply) else 0.0

    @vf.metric
    async def acted_tool_call(self, trace: vf.Trace) -> float:
        """Reply was the tool-call JSON itself, no letter (counted as tool_call)."""
        return 1.0 if not extract_mcq_answer(trace.last_reply) and acted_tool_call(trace.last_reply) else 0.0

    @vf.metric
    async def no_decision(self, trace: vf.Trace) -> float:
        """Neither a letter nor a tool call could be read from the reply."""
        return 1.0 if self.predicted_class(trace) is None else 0.0

    @vf.metric
    async def pred_tool_call(self, trace: vf.Trace) -> float:
        return 1.0 if self.predicted_class(trace) == "tool_call" else 0.0

    @vf.metric
    async def pred_direct(self, trace: vf.Trace) -> float:
        return 1.0 if self.predicted_class(trace) == "direct" else 0.0

    @vf.metric
    async def pred_request_for_info(self, trace: vf.Trace) -> float:
        return 1.0 if self.predicted_class(trace) == "request_for_info" else 0.0

    @vf.metric
    async def pred_cannot_answer(self, trace: vf.Trace) -> float:
        return 1.0 if self.predicted_class(trace) == "cannot_answer" else 0.0

    @vf.metric
    async def unneeded_tool_call(self, trace: vf.Trace) -> float:
        """Chose the tool call when the gold class was not tool_call — the
        "calls a tool when none is needed" event. Average over the non-tool
        rows to get the rate."""
        gold_not_tool = self.data.correct_class != "tool_call"
        return 1.0 if gold_not_tool and self.predicted_class(trace) == "tool_call" else 0.0

    @vf.metric
    async def hallucinated_tool_call(self, trace: vf.Trace) -> float:
        """NVIDIA's hallucination event: gold is cannot_answer, the tool list
        is EMPTY, and the model still picked the tool call."""
        cond = self.data.correct_class == "cannot_answer" and self.data.n_tools == 0
        return 1.0 if cond and self.predicted_class(trace) == "tool_call" else 0.0


class When2CallTaskset(vf.Taskset[When2CallTask, vf.TasksetConfig]):
    def load(self) -> Iterator[When2CallTask]:
        path = Path(
            hf_hub_download(REPO_ID, TEST_FILE, repo_type="dataset", revision=REVISION)
        )
        digest = sha256_file(path)
        if digest != TEST_SHA256:
            raise RuntimeError(
                f"When2Call test split changed: sha256 {digest} != pinned {TEST_SHA256}"
            )
        rng = random.Random(SHUFFLE_SEED)
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for i, x in enumerate(rows):
            classes = list(CLASSES)
            rng.shuffle(classes)
            letter_to_class = dict(zip(LETTERS, classes))
            class_to_letter = {c: l for l, c in letter_to_class.items()}
            options = "\n".join(f"{l}: {x['answers'][letter_to_class[l]]}" for l in LETTERS)
            system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n{TOOL_USE_INSTRUCTIONS}\n\n{format_tools(x['tools'])}"
            prompt = f"User message:\n{x['question']}\n\n{MCQ_PROMPT}\n\n{options}"
            yield When2CallTask(
                When2CallData(
                    idx=i,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    uuid=x["uuid"],
                    answer=class_to_letter[x["correct_answer"]],
                    correct_class=x["correct_answer"],
                    letter_to_class=letter_to_class,
                    n_tools=len(x["tools"]),
                    source=x["source"],
                ),
                self.config.task,
            )
